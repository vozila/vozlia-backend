"""VOZLIA FILE PURPOSE
Purpose: Background worker to deliver scheduled reports (scheduled_deliveries).
Hot path: no (worker/cron).
Public interfaces: main(), tick().
Reads/Writes: scheduled_deliveries, web_search_skills (and optionally db_query_skills).
Feature flags: DBQUERY_SCHEDULE_ENABLED (enable dbquery_* schedules).
Failure mode: delivery failures reschedule with backoff; missing skills disable schedule.
Last touched: 2026-01-31 (extend to execute dbquery_* schedules via DBQuerySkill)
"""

# workers/scheduled_deliveries_worker.py
from __future__ import annotations

import os
import time
from datetime import datetime, timedelta

from sqlalchemy.orm import Session

from core.logging import logger
from db import Base, engine, SessionLocal
from models import ScheduledDelivery, WebSearchSkill, DBQuerySkill, DeliveryChannel
from services.web_search_service import run_web_search
from services.db_query_service import run_db_query
from services.web_search_skill_store import compute_next_run_at
from services.notification_service import send_sms, send_whatsapp, send_email, make_phone_call


def _dbquery_schedule_enabled() -> bool:
    v = (os.getenv("DBQUERY_SCHEDULE_ENABLED") or "0").strip().lower()
    return v in ("1", "true", "yes", "on")


POLL_S = float((os.getenv("SCHEDULED_DELIVERY_POLL_S") or "30").strip() or "30")
MAX_PER_TICK = int((os.getenv("SCHEDULED_DELIVERY_MAX_PER_TICK") or "20").strip() or "20")


def _now_utc() -> datetime:
    return datetime.utcnow()


def _parse_time_of_day(s: str) -> tuple[int, int] | None:
    try:
        parts = (s or "").strip().split(":")
        if len(parts) != 2:
            return None
        h = int(parts[0])
        m = int(parts[1])
        if 0 <= h <= 23 and 0 <= m <= 59:
            return (h, m)
    except Exception:
        return None
    return None


def _deliver(channel: DeliveryChannel, destination: str, *, subject: str, body: str) -> str:
    if channel == DeliveryChannel.sms:
        return send_sms(destination, body)
    if channel == DeliveryChannel.whatsapp:
        return send_whatsapp(destination, body)
    if channel == DeliveryChannel.email:
        send_email(destination, subject, body)
        return "ok"
    if channel == DeliveryChannel.call:
        return make_phone_call(destination, body)
    raise RuntimeError(f"Unsupported channel: {channel}")


def _format_report(name: str, body: str) -> tuple[str, str]:
    today = _now_utc().strftime("%Y-%m-%d")
    nm = (name or "Report").strip() or "Report"
    subject = f"{nm} — {today}"
    b = (body or "").strip()
    if not b:
        b = "I couldn't find a clear answer today."
    return subject, b


def tick(db: Session) -> int:
    now = _now_utc()
    rows = (
        db.query(ScheduledDelivery)
        .filter(ScheduledDelivery.enabled == True)  # noqa: E712
        .filter(ScheduledDelivery.next_run_at != None)  # noqa: E711
        .filter(ScheduledDelivery.next_run_at <= now)
        .order_by(ScheduledDelivery.next_run_at.asc())
        .limit(MAX_PER_TICK)
        .all()
    )

    if not rows:
        return 0

    processed = 0
    for r in rows:
        # Determine polymorphic skill key (back-compat with older rows)
        skill_key = (getattr(r, "skill_key", None) or "").strip()
        if not skill_key and getattr(r, "web_search_skill_id", None):
            skill_key = f"websearch_{r.web_search_skill_id}"

        if not skill_key:
            r.enabled = False
            r.updated_at = now
            db.add(r)
            db.commit()
            continue

        latency_ms: float | None = None

        # -----------------------------
        # WebSearch schedules
        # -----------------------------
        if skill_key.startswith("websearch_"):
            skill = None
            try:
                if getattr(r, "web_search_skill_id", None):
                    skill = db.query(WebSearchSkill).filter(WebSearchSkill.id == r.web_search_skill_id).first()
            except Exception:
                skill = None

            if not skill or not bool(getattr(skill, "enabled", True)):
                r.enabled = False
                r.updated_at = now
                db.add(r)
                db.commit()
                continue

            res = run_web_search(skill.query)
            subject, body = _format_report(skill.name, res.answer)

            try:
                delivery_id = _deliver(r.channel, r.destination, subject=subject, body=body)
                logger.info(
                    "SCHEDULED_DELIVERY_SENT id=%s skill_key=%s channel=%s dest=%s delivery_id=%s",
                    r.id,
                    skill_key,
                    getattr(r.channel, "value", str(r.channel)),
                    r.destination,
                    delivery_id,
                )
                try:
                    latency_ms = float(getattr(res, "latency_ms", None)) if getattr(res, "latency_ms", None) is not None else None
                except Exception:
                    latency_ms = None
            except Exception as e:
                logger.exception("SCHEDULED_DELIVERY_SEND_FAIL id=%s err=%s", r.id, e)
                r.next_run_at = now + timedelta(minutes=15)
                r.updated_at = now
                db.add(r)
                db.commit()
                continue

        # -----------------------------
        # DBQuery schedules (optional)
        # -----------------------------
        elif skill_key.startswith("dbquery_"):
            if not _dbquery_schedule_enabled():
                # Fail safe: disable row so it doesn't spin.
                r.enabled = False
                r.updated_at = now
                db.add(r)
                db.commit()
                continue

            raw_id = skill_key.replace("dbquery_", "", 1).strip()
            qskill = None
            try:
                from uuid import UUID
                qskill = db.query(DBQuerySkill).filter(DBQuerySkill.id == UUID(raw_id)).first()
            except Exception:
                qskill = db.query(DBQuerySkill).filter(DBQuerySkill.id == raw_id).first()

            if not qskill or not bool(getattr(qskill, "enabled", True)):
                r.enabled = False
                r.updated_at = now
                db.add(r)
                db.commit()
                continue

            t0 = time.time()
            qres = run_db_query(db, tenant_uuid=str(r.tenant_id), spec=(qskill.spec or {}))
            latency_ms = (time.time() - t0) * 1000.0

            subject, body = _format_report(qskill.name, qres.spoken_summary or "")

            try:
                delivery_id = _deliver(r.channel, r.destination, subject=subject, body=body)
                logger.info(
                    "SCHEDULED_DELIVERY_SENT id=%s skill_key=%s channel=%s dest=%s delivery_id=%s",
                    r.id,
                    skill_key,
                    getattr(r.channel, "value", str(r.channel)),
                    r.destination,
                    delivery_id,
                )
            except Exception as e:
                logger.exception("SCHEDULED_DELIVERY_SEND_FAIL id=%s err=%s", r.id, e)
                r.next_run_at = now + timedelta(minutes=15)
                r.updated_at = now
                db.add(r)
                db.commit()
                continue

        else:
            # Unknown skill type (future-proof): disable.
            r.enabled = False
            r.updated_at = now
            db.add(r)
            db.commit()
            continue

        # Update next_run_at
        hm = _parse_time_of_day(r.time_of_day)
        if hm:
            h, m = hm
            r.last_run_at = now
            r.last_latency_ms = float(latency_ms) if latency_ms is not None else None
            r.next_run_at = compute_next_run_at(hour=h, minute=m, timezone=r.timezone or "America/New_York", now_utc=now)
            r.updated_at = now
            db.add(r)
            db.commit()
        else:
            r.enabled = False
            r.updated_at = now
            db.add(r)
            db.commit()

        processed += 1

    return processed


def main() -> None:
    logger.info("SCHEDULED_DELIVERIES_WORKER_START poll_s=%s max_per_tick=%s", POLL_S, MAX_PER_TICK)
    Base.metadata.create_all(bind=engine)

    while True:
        try:
            with SessionLocal() as db:
                n = tick(db)
            if n:
                logger.info("SCHEDULED_DELIVERIES_TICK processed=%s", n)
        except Exception as e:
            logger.exception("SCHEDULED_DELIVERIES_TICK_FAIL err=%s", e)

        time.sleep(POLL_S)


if __name__ == "__main__":
    main()

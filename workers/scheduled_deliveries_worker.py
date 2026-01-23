# workers/scheduled_deliveries_worker.py
from __future__ import annotations

import os
import time
from datetime import datetime, timedelta

from sqlalchemy.orm import Session

from core.logging import logger
from db import Base, engine, SessionLocal
from models import ScheduledDelivery, WebSearchSkill, DeliveryChannel
from services.web_search_service import run_web_search
from services.web_search_skill_store import compute_next_run_at
from services.notification_service import send_sms, send_whatsapp, send_email, make_phone_call


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


def _format_report(skill: WebSearchSkill, answer: str) -> tuple[str, str]:
    today = _now_utc().strftime("%Y-%m-%d")
    name = (skill.name or "Report").strip()
    subject = f"{name} — {today}"
    body = (answer or "").strip()
    if not body:
        body = "I couldn't find a clear answer today."
    return subject, body


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
        skill = db.query(WebSearchSkill).filter(WebSearchSkill.id == r.web_search_skill_id).first()
        if not skill or not bool(skill.enabled):
            r.enabled = False
            r.updated_at = now
            db.add(r)
            db.commit()
            continue

        res = run_web_search(skill.query)
        subject, body = _format_report(skill, res.answer)

        try:
            delivery_id = _deliver(r.channel, r.destination, subject=subject, body=body)
            logger.info(
                "SCHEDULED_DELIVERY_SENT id=%s skill_id=%s channel=%s dest=%s delivery_id=%s",
                r.id,
                r.web_search_skill_id,
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

        hm = _parse_time_of_day(r.time_of_day)
        if hm:
            h, m = hm
            r.last_run_at = now
            r.last_latency_ms = res.latency_ms
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

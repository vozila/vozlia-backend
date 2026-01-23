# services/web_search_skill_store.py
from __future__ import annotations

from datetime import datetime, timedelta
from typing import List, Tuple

from sqlalchemy.orm import Session
from zoneinfo import ZoneInfo

from core.logging import logger
from models import User, WebSearchSkill, ScheduledDelivery, DeliveryChannel, DeliveryCadence
from services.settings_service import get_skills_config, get_skills_priority_order, set_setting, set_skills_priority_order


def _now_utc() -> datetime:
    return datetime.utcnow()


def parse_time_of_day(text: str) -> Tuple[int, int] | None:
    """Parse a time-of-day from text.

    Accepts:
      - 8am, 8 AM, 8:30am
      - 20:15, 08:15
      - 8 (assume 8:00)
    Returns (hour, minute) in 24h.
    """
    import re

    t = (text or "").strip().lower()
    if not t:
        return None

    m = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", t)
    if m:
        h = int(m.group(1))
        mm = int(m.group(2) or "0")
        ap = m.group(3)
        if h == 12:
            h = 0
        if ap == "pm":
            h += 12
        if 0 <= h <= 23 and 0 <= mm <= 59:
            return (h, mm)

    m = re.search(r"\b(\d{1,2}):(\d{2})\b", t)
    if m:
        h = int(m.group(1))
        mm = int(m.group(2))
        if 0 <= h <= 23 and 0 <= mm <= 59:
            return (h, mm)

    m = re.search(r"\b(\d{1,2})\b", t)
    if m:
        h = int(m.group(1))
        if 0 <= h <= 23:
            return (h, 0)

    return None


def compute_next_run_at(*, hour: int, minute: int, timezone: str, now_utc: datetime | None = None) -> datetime:
    now_utc = now_utc or _now_utc()
    tz = ZoneInfo(timezone)
    now_local = now_utc.replace(tzinfo=ZoneInfo("UTC")).astimezone(tz)

    candidate_local = now_local.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if candidate_local <= now_local:
        candidate_local = candidate_local + timedelta(days=1)

    candidate_utc = candidate_local.astimezone(ZoneInfo("UTC")).replace(tzinfo=None)
    return candidate_utc


def _skill_key_for(skill: WebSearchSkill) -> str:
    return f"websearch_{skill.id}"


def _register_skill_in_skills_config(db: Session, user: User, skill: WebSearchSkill) -> None:
    key = _skill_key_for(skill)
    cfg = get_skills_config(db, user) or {}
    if not isinstance(cfg, dict):
        cfg = {}

    greeting_line = f"I can run '{skill.name}'. Just ask: {(skill.triggers[0] if skill.triggers else skill.query)}"

    cfg[key] = {
        "enabled": True,
        "label": skill.name,
        "type": "web_search",
        "query": skill.query,
        "engagement_phrases": skill.triggers or [],
        "add_to_greeting": False,
        "auto_execute_after_greeting": False,
        "greeting_line": greeting_line,
        "web_search_skill_id": str(skill.id),
    }

    set_setting(db, user, "skills_config", cfg)

    order = get_skills_priority_order(db, user) or []
    if key not in order:
        order = list(order) + [key]
        set_skills_priority_order(db, user, order)


def create_web_search_skill(
    db: Session,
    user: User,
    *,
    name: str,
    query: str,
    triggers: List[str] | None = None,
) -> WebSearchSkill:
    skill = WebSearchSkill(
        tenant_id=user.id,
        name=(name or "").strip() or "Web Search",
        query=(query or "").strip(),
        triggers=[t.strip() for t in (triggers or []) if isinstance(t, str) and t.strip()],
        enabled=True,
    )
    db.add(skill)
    db.commit()
    db.refresh(skill)

    try:
        _register_skill_in_skills_config(db, user, skill)
        db.commit()
    except Exception as e:
        logger.exception("WEBSEARCH_SKILL_REGISTER_FAIL tenant_id=%s skill_id=%s err=%s", user.id, skill.id, e)

    logger.info("WEBSEARCH_SKILL_CREATED tenant_id=%s skill_id=%s skill_key=%s", user.id, skill.id, _skill_key_for(skill))
    return skill


def list_web_search_skills(db: Session, user: User) -> List[WebSearchSkill]:
    return (
        db.query(WebSearchSkill)
        .filter(WebSearchSkill.tenant_id == user.id)
        .order_by(WebSearchSkill.created_at.desc())
        .all()
    )


def get_web_search_skill(db: Session, user: User, skill_id) -> WebSearchSkill | None:
    return (
        db.query(WebSearchSkill)
        .filter(WebSearchSkill.tenant_id == user.id, WebSearchSkill.id == skill_id)
        .first()
    )


def delete_web_search_skill(db: Session, user: User, skill_id) -> bool:
    skill = get_web_search_skill(db, user, skill_id)
    if not skill:
        return False
    db.delete(skill)
    db.commit()
    logger.info("WEBSEARCH_SKILL_DELETED tenant_id=%s skill_id=%s", user.id, skill_id)
    return True


def upsert_daily_schedule(
    db: Session,
    user: User,
    *,
    web_search_skill_id,
    hour: int,
    minute: int,
    timezone: str,
    channel: DeliveryChannel,
    destination: str,
) -> ScheduledDelivery:
    timezone = (timezone or "").strip() or "America/New_York"
    time_of_day = f"{hour:02d}:{minute:02d}"

    row = (
        db.query(ScheduledDelivery)
        .filter(
            ScheduledDelivery.tenant_id == user.id,
            ScheduledDelivery.web_search_skill_id == web_search_skill_id,
        )
        .first()
    )

    next_run_at = compute_next_run_at(hour=hour, minute=minute, timezone=timezone)

    if row:
        row.enabled = True
        row.cadence = DeliveryCadence.daily
        row.time_of_day = time_of_day
        row.timezone = timezone
        row.channel = channel
        row.destination = destination
        row.next_run_at = next_run_at
        row.updated_at = _now_utc()
        db.add(row)
        db.commit()
        db.refresh(row)
        logger.info("SCHEDULED_DELIVERY_UPDATED id=%s tenant_id=%s next=%s", row.id, user.id, row.next_run_at)
        return row

    row = ScheduledDelivery(
        tenant_id=user.id,
        web_search_skill_id=web_search_skill_id,
        enabled=True,
        cadence=DeliveryCadence.daily,
        time_of_day=time_of_day,
        timezone=timezone,
        channel=channel,
        destination=destination,
        next_run_at=next_run_at,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    logger.info("SCHEDULED_DELIVERY_CREATED id=%s tenant_id=%s next=%s", row.id, user.id, row.next_run_at)
    return row


def disable_all_schedules(db: Session, user: User) -> int:
    rows = db.query(ScheduledDelivery).filter(ScheduledDelivery.tenant_id == user.id, ScheduledDelivery.enabled == True).all()  # noqa: E712
    for r in rows:
        r.enabled = False
        r.updated_at = _now_utc()
        db.add(r)
    if rows:
        db.commit()
    return len(rows)


def disable_schedule_for_skill(db: Session, user: User, web_search_skill_id) -> bool:
    row = (
        db.query(ScheduledDelivery)
        .filter(
            ScheduledDelivery.tenant_id == user.id,
            ScheduledDelivery.web_search_skill_id == web_search_skill_id,
            ScheduledDelivery.enabled == True,  # noqa: E712
        )
        .first()
    )
    if not row:
        return False
    row.enabled = False
    row.updated_at = _now_utc()
    db.add(row)
    db.commit()
    logger.info("SCHEDULED_DELIVERY_DISABLED id=%s tenant_id=%s", row.id, user.id)
    return True


def list_schedules(db: Session, user: User) -> list[ScheduledDelivery]:
    return (
        db.query(ScheduledDelivery)
        .filter(ScheduledDelivery.tenant_id == user.id)
        .order_by(ScheduledDelivery.created_at.desc())
        .all()
    )

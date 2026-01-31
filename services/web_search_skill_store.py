"""VOZLIA FILE PURPOSE
Purpose: CRUD + dynamic registration for WebSearch skills and their scheduled deliveries.
Hot path: no
Public interfaces: create_web_search_skill, delete_web_search_skill, upsert_daily_schedule, list_schedules.
Reads/Writes: web_search_skills, scheduled_deliveries, user_settings (skills_config, skills_priority_order).
Feature flags: n/a (called by routers and intent_v2 when enabled).
Failure mode: raises on invalid skill IDs; scheduling failures return handled errors upstream.
Last touched: 2026-01-31 (backfill scheduled_deliveries.skill_key for websearch schedules)
"""

# services/web_search_skill_store.py
from __future__ import annotations

from datetime import datetime, timedelta
from typing import List, Tuple, Optional

from sqlalchemy.orm import Session
from zoneinfo import ZoneInfo
from uuid import UUID

from core.logging import logger
from models import (
    User,
    WebSearchSkill,
    ScheduledDelivery,
    DeliveryChannel,
    DeliveryCadence,
)
from services.settings_service import (
    get_skills_config,
    get_skills_priority_order,
    set_setting,
    set_skills_priority_order,
)


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

    # 8:30am / 8am / 8:30 am
    m = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", t)
    if m:
        h = int(m.group(1))
        mm = int(m.group(2) or "0")
        ap = (m.group(3) or "").lower()
        if h == 12:
            h = 0
        if ap == "pm":
            h += 12
        if 0 <= h <= 23 and 0 <= mm <= 59:
            return (h, mm)

    # 20:15 / 08:15
    m = re.search(r"\b(\d{1,2}):(\d{2})\b", t)
    if m:
        h = int(m.group(1))
        mm = int(m.group(2))
        if 0 <= h <= 23 and 0 <= mm <= 59:
            return (h, mm)

    # 8 → 8:00
    m = re.search(r"\b(\d{1,2})\b", t)
    if m:
        h = int(m.group(1))
        if 0 <= h <= 23:
            return (h, 0)

    return None


def compute_next_run_at(
    *,
    hour: int,
    minute: int,
    timezone: str,
    now_utc: datetime | None = None,
) -> datetime:
    """Compute the next run time (UTC) for a given local time-of-day."""
    now_utc = now_utc or _now_utc()
    tz = ZoneInfo(timezone)
    now_local = now_utc.replace(tzinfo=ZoneInfo("UTC")).astimezone(tz)

    candidate_local = now_local.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if candidate_local <= now_local:
        candidate_local = candidate_local + timedelta(days=1)

    candidate_utc = candidate_local.astimezone(ZoneInfo("UTC")).replace(tzinfo=None)
    return candidate_utc


# ----------------------------
# WebSearch skills
# ----------------------------

def list_web_search_skills(db: Session, user: User) -> List[WebSearchSkill]:
    return (
        db.query(WebSearchSkill)
        .filter(WebSearchSkill.tenant_id == user.id)
        .order_by(WebSearchSkill.created_at.asc())
        .all()
    )


def create_web_search_skill(
    db: Session,
    user: User,
    *,
    name: str,
    query: str,
    triggers: Optional[List[str]] = None,
) -> WebSearchSkill:
    skill = WebSearchSkill(
        tenant_id=user.id,
        name=(name or "").strip() or "Web Search",
        query=(query or "").strip(),
        triggers=[t for t in (triggers or []) if isinstance(t, str) and t.strip()],
        enabled=True,
    )
    db.add(skill)
    db.commit()
    db.refresh(skill)

    # Register in skills_config so Intent V2 / dynamic runtime can see it.
    skill_key = f"websearch_{skill.id}"
    cfg = get_skills_config(db, user) or {}
    if not isinstance(cfg, dict):
        cfg = {}
    if skill_key not in cfg:
        cfg[skill_key] = {
            "type": "web_search",
            "label": skill.name,
            "query": skill.query,
            "enabled": True,
            "category": (cfg.get("category") if isinstance(cfg, dict) else None) or "",
            "engagement_phrases": skill.triggers or [],
            "add_to_greeting": False,
            "auto_execute_after_greeting": False,
            "web_search_skill_id": str(skill.id),
        }
        try:
            set_setting(db, user, "skills_config", cfg)
        except Exception:
            logger.exception("WEBSEARCH_SKILL_CREATE_CFG_FAIL skill_id=%s", skill.id)

        # Append to priority order (keep existing ordering).
        try:
            order = get_skills_priority_order(db, user) or []
            if not isinstance(order, list):
                order = []
            if skill_key not in order:
                order = list(order) + [skill_key]
                set_skills_priority_order(db, user, order)
        except Exception:
            logger.exception("WEBSEARCH_SKILL_CREATE_PRIORITY_FAIL skill_id=%s", skill.id)

    logger.info(
        "WEBSEARCH_SKILL_CREATED id=%s tenant_id=%s name=%r triggers=%s",
        skill.id,
        user.id,
        skill.name,
        skill.triggers,
    )
    return skill


def delete_web_search_skill(db: Session, user: User, skill_id: str) -> bool:
    """
    Delete a WebSearchSkill AND its associated schedules + dynamic skill config.

    Accepts either:
      - raw UUID string (e.g., '1bb87827-52ed-4ec0-844e-e4716ec96222')
      - or skill_key style 'websearch_<uuid>'.

    Behavior:
      1) Resolve the skill for this tenant.
      2) Delete all ScheduledDelivery rows for this skill.
      3) Remove its dynamic config entry from skills_config + priority order.
      4) Delete the WebSearchSkill row itself.

    Returns:
      True if a skill was found and deleted; False otherwise.
    """
    raw = (skill_id or "").strip()
    if not raw:
        return False

    # Support both plain UUID and 'websearch_<uuid>'.
    if raw.startswith("websearch_"):
        raw = raw.replace("websearch_", "", 1).strip()

    try:
        sid = UUID(raw)
    except Exception:
        logger.warning("WEBSEARCH_SKILL_DELETE_BAD_ID skill_id=%r", skill_id)
        return False

    skill = (
        db.query(WebSearchSkill)
        .filter(
            WebSearchSkill.tenant_id == user.id,
            WebSearchSkill.id == sid,
        )
        .first()
    )
    if not skill:
        return False

    # 1) Delete schedules for this skill (double safety; FK has ON DELETE CASCADE too).
    schedules_q = (
        db.query(ScheduledDelivery)
        .filter(
            ScheduledDelivery.tenant_id == user.id,
            ScheduledDelivery.web_search_skill_id == skill.id,
        )
    )
    deleted_schedules = schedules_q.delete(synchronize_session=False)

    # 2) Remove from skills_config + priority order.
    key = f"websearch_{skill.id}"
    cfg = get_skills_config(db, user) or {}
    if not isinstance(cfg, dict):
        cfg = {}

    cfg_changed = False
    if key in cfg:
        cfg.pop(key, None)
        cfg_changed = True

    if cfg_changed:
        try:
            set_setting(db, user, "skills_config", cfg)
        except Exception:
            logger.exception("WEBSEARCH_SKILL_DELETE_CFG_FAIL skill_id=%s", skill.id)

    try:
        order = get_skills_priority_order(db, user) or []
        if not isinstance(order, list):
            order = []
        if key in order:
            order = [k for k in order if k != key]
            set_skills_priority_order(db, user, order)
    except Exception:
        logger.exception("WEBSEARCH_SKILL_DELETE_PRIORITY_FAIL skill_id=%s", skill.id)

    # 3) Delete the skill row itself.
    db.delete(skill)
    db.commit()
    logger.info(
        "WEBSEARCH_SKILL_DELETED id=%s tenant_id=%s schedules_deleted=%s",
        skill.id,
        user.id,
        deleted_schedules,
    )
    return True


# ----------------------------
# Schedules
# ----------------------------

def list_schedules(db: Session, user: User) -> List[ScheduledDelivery]:
    return (
        db.query(ScheduledDelivery)
        .filter(ScheduledDelivery.tenant_id == user.id)
        .filter(ScheduledDelivery.web_search_skill_id != None)  # noqa: E711
        .order_by(ScheduledDelivery.created_at.asc())
        .all()
    )


def upsert_daily_schedule(
    db: Session,
    user: User,
    *,
    web_search_skill_id: str | UUID,
    hour: int,
    minute: int,
    timezone: str,
    channel: DeliveryChannel,
    destination: str,
) -> ScheduledDelivery:
    """
    Create or update a daily ScheduledDelivery for a WebSearch skill.

    If a schedule already exists for (tenant, web_search_skill_id), it will be updated.
    """
    # Normalize skill id to UUID
    sid: UUID
    if isinstance(web_search_skill_id, UUID):
        sid = web_search_skill_id
    else:
        raw = str(web_search_skill_id or "").strip()
        if raw.startswith("websearch_"):
            raw = raw.replace("websearch_", "", 1).strip()
        sid = UUID(raw)

    # Ensure the skill exists + belongs to this tenant.
    skill = (
        db.query(WebSearchSkill)
        .filter(WebSearchSkill.tenant_id == user.id, WebSearchSkill.id == sid)
        .first()
    )
    if not skill:
        raise ValueError(f"WebSearchSkill not found for id={sid}")

    time_of_day = f"{int(hour):02d}:{int(minute):02d}"
    tz = (timezone or "America/New_York").strip() or "America/New_York"
    next_run_at = compute_next_run_at(hour=int(hour), minute=int(minute), timezone=tz)

    row = (
        db.query(ScheduledDelivery)
        .filter(
            ScheduledDelivery.tenant_id == user.id,
            ScheduledDelivery.web_search_skill_id == skill.id,
        )
        .first()
    )
    if row:
        row.enabled = True
        row.cadence = DeliveryCadence.daily
        row.time_of_day = time_of_day
        row.timezone = tz
        row.channel = channel
        row.destination = destination
        row.next_run_at = next_run_at
        row.updated_at = _now_utc()
    else:
        row = ScheduledDelivery(
            tenant_id=user.id,
            web_search_skill_id=skill.id,
            enabled=True,
            cadence=DeliveryCadence.daily,
            time_of_day=time_of_day,
            timezone=tz,
            channel=channel,
            destination=destination,
            next_run_at=next_run_at,
        )
        db.add(row)

    # Always maintain the polymorphic key for back-compat + future multi-skill scheduling.
    try:
        row.skill_key = f"websearch_{skill.id}"
    except Exception:
        pass

    db.commit()
    db.refresh(row)

    logger.info(
        "WEBSEARCH_SCHEDULE_UPSERT id=%s tenant_id=%s skill_id=%s time_of_day=%s tz=%s channel=%s dest=%s next_run_at=%s",
        row.id,
        user.id,
        skill.id,
        row.time_of_day,
        row.timezone,
        row.channel.value if hasattr(row.channel, "value") else str(row.channel),
        row.destination,
        row.next_run_at.isoformat() if row.next_run_at else None,
    )
    return row


def disable_all_schedules(db: Session, user: User) -> int:
    """
    Disable all ScheduledDelivery rows for this tenant.

    Returns:
      number of rows updated.
    """
    q = db.query(ScheduledDelivery).filter(
        ScheduledDelivery.tenant_id == user.id,
        ScheduledDelivery.enabled == True,  # noqa
    )
    count = q.count()
    q.update({"enabled": False, "updated_at": _now_utc()}, synchronize_session=False)
    db.commit()
    logger.info("WEBSEARCH_SCHEDULES_DISABLED tenant_id=%s count=%s", user.id, count)
    return count

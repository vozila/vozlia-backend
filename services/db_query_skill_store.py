"""VOZLIA FILE PURPOSE
Purpose: Dynamic DBQuery skill persistence + registration in skills_config (and lightweight scheduling helpers).
Hot path: no
Public interfaces: create_db_query_skill, list_db_query_skills, delete_db_query_skill, upsert_daily_schedule_dbquery.
Reads/Writes: db_query_skills, scheduled_deliveries, user_settings (skills_config, skills_priority_order).
Feature flags: DBQUERY_SCHEDULE_ENABLED (scheduling path).
Failure mode: raises on invalid skill IDs; scheduling errors are handled upstream by callers.
Last touched: 2026-01-31 (add upsert scheduling helper for dbquery_* skills)
"""

# NOTE (LEGACY / SLATED FOR REMOVAL)
# ---------------------------------
# This module was part of the earlier DBQuery/Wizard experimentation.
# It is kept for backwards compatibility and troubleshooting only.
# The current direction is to route natural-language intent via an LLM
# (schema-validated) and execute deterministically via the skill engines.
#
# When Intent Router V2 is fully cut over, we should remove or archive this
# DBQuery v1 path (do NOT delete until an env-var gated rollback exists).

# services/db_query_skill_store.py
from __future__ import annotations

from datetime import datetime
from uuid import UUID

from typing import Any, List

from sqlalchemy.orm import Session

from models import DBQuerySkill, ScheduledDelivery, User, DeliveryCadence, DeliveryChannel
from services.settings_service import (
    get_skills_config,
    get_skills_priority_order,
    set_setting,
    set_skills_priority_order,
)
from core.logging import logger
from services.web_search_skill_store import compute_next_run_at


def _skill_key_for(skill: DBQuerySkill) -> str:
    return f"dbquery_{skill.id}"


def _register_skill_in_skills_config(db: Session, user: User, skill: DBQuerySkill) -> None:
    key = _skill_key_for(skill)
    cfg = get_skills_config(db, user) or {}
    if not isinstance(cfg, dict):
        cfg = {}

    trigger_hint = ""
    try:
        trigger_hint = (skill.triggers[0] if skill.triggers else skill.name)
    except Exception:
        trigger_hint = skill.name

    greeting_line = f"I can run '{skill.name}'. Just ask: {trigger_hint}"

    cfg[key] = {
        "enabled": True,
        "label": skill.name,
        "type": "db_query",
        "entity": skill.entity,
        "spec": skill.spec or {},
        "engagement_phrases": skill.triggers or [],
        "add_to_greeting": False,
        "auto_execute_after_greeting": False,
        "greeting_line": greeting_line,
        "db_query_skill_id": str(skill.id),
    }

    set_setting(db, user, "skills_config", cfg)

    # Append to skill priority list (keep existing order)
    order = get_skills_priority_order(db, user) or []
    if key not in order:
        order = list(order) + [key]
        set_skills_priority_order(db, user, order)


def create_db_query_skill(
    db: Session,
    user: User,
    *,
    name: str,
    entity: str,
    spec: dict[str, Any],
    triggers: List[str] | None = None,
) -> DBQuerySkill:
    skill = DBQuerySkill(
        tenant_id=user.id,
        name=(name or "").strip() or "DB Query",
        entity=(entity or "").strip() or "caller_memory_events",
        spec=(spec or {}),
        triggers=(triggers or []),
        enabled=True,
    )
    db.add(skill)
    db.commit()
    db.refresh(skill)

    try:
        _register_skill_in_skills_config(db, user, skill)
    except Exception:
        logger.exception("DBQUERY_SKILL_REGISTER_CONFIG_FAIL id=%s", skill.id)

    return skill


def list_db_query_skills(db: Session, user: User) -> list[DBQuerySkill]:
    return (
        db.query(DBQuerySkill)
        .filter(DBQuerySkill.tenant_id == user.id)
        .order_by(DBQuerySkill.created_at.desc())
        .all()
    )


def delete_db_query_skill(db: Session, user: User, skill_id: str) -> bool:
    sid: object = skill_id
    try:
        from uuid import UUID
        sid = UUID(str(skill_id))
    except Exception:
        sid = skill_id

    row = (
        db.query(DBQuerySkill)
        .filter(DBQuerySkill.tenant_id == user.id)
        .filter(DBQuerySkill.id == sid)
        .first()
    )
    if not row:
        return False
    db.delete(row)
    db.commit()
    # Note: we intentionally do not mutate skills_config here (fail-open).
    return True



def _now_utc() -> datetime:
    return datetime.utcnow()


def upsert_daily_schedule_dbquery(
    db: Session,
    user: User,
    *,
    db_query_skill_id: str | UUID,
    hour: int,
    minute: int,
    timezone: str,
    channel: DeliveryChannel,
    destination: str,
) -> ScheduledDelivery:
    """
    Create or update a daily ScheduledDelivery for a DBQuery skill.

    Upsert semantics:
      - If a schedule already exists for (tenant, skill_key=dbquery_<id>), it will be updated.
      - Schedules are stored in scheduled_deliveries with skill_key="dbquery_<uuid>".
    """
    # Normalize skill id to UUID
    sid: UUID
    if isinstance(db_query_skill_id, UUID):
        sid = db_query_skill_id
    else:
        raw = str(db_query_skill_id or "").strip()
        if raw.startswith("dbquery_"):
            raw = raw.replace("dbquery_", "", 1).strip()
        sid = UUID(raw)

    # Ensure the skill exists + belongs to this tenant.
    skill = (
        db.query(DBQuerySkill)
        .filter(DBQuerySkill.tenant_id == user.id, DBQuerySkill.id == sid)
        .first()
    )
    if not skill:
        raise ValueError(f"DBQuerySkill not found for id={sid}")

    skill_key = f"dbquery_{skill.id}"
    time_of_day = f"{int(hour):02d}:{int(minute):02d}"
    tz = (timezone or "America/New_York").strip() or "America/New_York"
    next_run_at = compute_next_run_at(hour=int(hour), minute=int(minute), timezone=tz)

    row = (
        db.query(ScheduledDelivery)
        .filter(ScheduledDelivery.tenant_id == user.id)
        .filter(ScheduledDelivery.skill_key == skill_key)
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
            web_search_skill_id=None,
            skill_key=skill_key,
            enabled=True,
            cadence=DeliveryCadence.daily,
            time_of_day=time_of_day,
            timezone=tz,
            channel=channel,
            destination=destination,
            next_run_at=next_run_at,
        )
        db.add(row)

    db.commit()
    db.refresh(row)

    logger.info(
        "DBQUERY_SCHEDULE_UPSERT id=%s tenant_id=%s skill_id=%s time_of_day=%s tz=%s channel=%s dest=%s next_run_at=%s",
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


def list_dbquery_schedules(db: Session, user: User) -> list[ScheduledDelivery]:
    """List only dbquery_* schedules for this tenant."""
    return (
        db.query(ScheduledDelivery)
        .filter(ScheduledDelivery.tenant_id == user.id)
        .filter(ScheduledDelivery.skill_key.like("dbquery_%"))
        .order_by(ScheduledDelivery.created_at.asc())
        .all()
    )

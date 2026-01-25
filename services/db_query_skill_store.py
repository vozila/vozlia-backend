# services/db_query_skill_store.py
from __future__ import annotations

from typing import Any, List

from sqlalchemy.orm import Session

from models import DBQuerySkill, User
from services.settings_service import (
    get_skills_config,
    get_skills_priority_order,
    set_setting,
    set_skills_priority_order,
)
from core.logging import logger


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


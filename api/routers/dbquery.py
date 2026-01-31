"""VOZLIA FILE PURPOSE
Purpose: Admin endpoints for DBQuery (metrics) skill CRUD + safe execution helpers.
Hot path: no
Public interfaces: /admin/dbquery/skills, /admin/dbquery/run, /admin/dbquery/schedules.
Reads/Writes: db_query_skills, scheduled_deliveries.
Feature flags: DBQUERY_SCHEDULE_ENABLED (scheduling enable in runtime paths).
Failure mode: returns HTTP 4xx/5xx with safe messages; no side effects without admin key.
Last touched: 2026-01-31 (add DBQuery schedule list/upsert endpoints)
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

# api/routers/dbquery.py
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from api.deps.admin_key import require_admin_key
from deps import get_db
from services.user_service import get_or_create_primary_user
from services.db_query_service import DBQuerySpec, DBQueryResult, run_db_query, supported_entities
from models import DeliveryChannel
from services.db_query_skill_store import (
    create_db_query_skill,
    delete_db_query_skill,
    list_db_query_skills,
    upsert_daily_schedule_dbquery,
    list_dbquery_schedules,
)


router = APIRouter(prefix="/admin/dbquery", tags=["admin-dbquery"], dependencies=[Depends(require_admin_key)])


class DBQueryRunIn(BaseModel):
    spec: DBQuerySpec


class DBQuerySkillCreateIn(BaseModel):
    name: str = Field(..., min_length=1)
    entity: str = Field(..., min_length=1)
    spec: dict = Field(default_factory=dict)
    triggers: list[str] | None = None


class DBQuerySkillOut(BaseModel):
    id: str
    skill_key: str
    name: str
    entity: str
    spec: dict
    triggers: list[str] = []
    enabled: bool = True


@router.get("/entities")
def admin_dbquery_entities():
    return {"entities": supported_entities()}


@router.post("/run", response_model=DBQueryResult)
def admin_dbquery_run(payload: DBQueryRunIn, db: Session = Depends(get_db)):
    user = get_or_create_primary_user(db)
    return run_db_query(db, tenant_uuid=str(user.id), spec=payload.spec)


@router.get("/skills", response_model=list[DBQuerySkillOut])
def admin_dbquery_list_skills(db: Session = Depends(get_db)):
    user = get_or_create_primary_user(db)
    skills = list_db_query_skills(db, user)
    return [
        {
            "id": str(s.id),
            "skill_key": f"dbquery_{s.id}",
            "name": s.name,
            "entity": s.entity,
            "spec": s.spec or {},
            "triggers": s.triggers or [],
            "enabled": bool(s.enabled),
        }
        for s in skills
    ]


@router.post("/skills", response_model=DBQuerySkillOut)
def admin_dbquery_create_skill(payload: DBQuerySkillCreateIn, db: Session = Depends(get_db)):
    user = get_or_create_primary_user(db)
    skill = create_db_query_skill(
        db,
        user,
        name=payload.name,
        entity=payload.entity,
        spec=payload.spec or {},
        triggers=payload.triggers,
    )
    return {
        "id": str(skill.id),
        "skill_key": f"dbquery_{skill.id}",
        "name": skill.name,
        "entity": skill.entity,
        "spec": skill.spec or {},
        "triggers": skill.triggers or [],
        "enabled": bool(skill.enabled),
    }


@router.delete("/skills/{skill_id}")
def admin_dbquery_delete_skill(skill_id: str, db: Session = Depends(get_db)):
    user = get_or_create_primary_user(db)
    ok = delete_db_query_skill(db, user, skill_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Skill not found")
    return {"ok": True}


class DBQueryScheduleUpsertIn(BaseModel):
    db_query_skill_id: str
    hour: int = Field(..., ge=0, le=23)
    minute: int = Field(..., ge=0, le=59)
    timezone: str = "America/New_York"
    channel: DeliveryChannel
    destination: str = Field(..., min_length=3)


class DBQueryScheduledDeliveryOut(BaseModel):
    id: str
    db_query_skill_id: str
    skill_key: str
    enabled: bool
    cadence: str
    time_of_day: str
    timezone: str
    channel: str
    destination: str
    next_run_at: str | None = None
    last_run_at: str | None = None


@router.get("/schedules", response_model=list[DBQueryScheduledDeliveryOut])
def admin_dbquery_list_schedules(db: Session = Depends(get_db)):
    user = get_or_create_primary_user(db)
    rows = list_dbquery_schedules(db, user)
    out: list[dict] = []
    for r in rows:
        sk = (r.skill_key or "").strip()
        raw_id = sk.replace("dbquery_", "", 1) if sk.startswith("dbquery_") else ""
        out.append(
            {
                "id": str(r.id),
                "db_query_skill_id": raw_id,
                "skill_key": sk,
                "enabled": bool(r.enabled),
                "cadence": getattr(r.cadence, "value", str(r.cadence)),
                "time_of_day": r.time_of_day,
                "timezone": r.timezone,
                "channel": getattr(r.channel, "value", str(r.channel)),
                "destination": r.destination,
                "next_run_at": (r.next_run_at.isoformat() if r.next_run_at else None),
                "last_run_at": (r.last_run_at.isoformat() if r.last_run_at else None),
            }
        )
    return out


@router.post("/schedules", response_model=DBQueryScheduledDeliveryOut)
def admin_dbquery_upsert_schedule(payload: DBQueryScheduleUpsertIn, db: Session = Depends(get_db)):
    user = get_or_create_primary_user(db)
    row = upsert_daily_schedule_dbquery(
        db,
        user,
        db_query_skill_id=payload.db_query_skill_id,
        hour=payload.hour,
        minute=payload.minute,
        timezone=payload.timezone,
        channel=payload.channel,
        destination=payload.destination,
    )
    sk = (row.skill_key or "").strip()
    raw_id = sk.replace("dbquery_", "", 1) if sk.startswith("dbquery_") else ""
    return {
        "id": str(row.id),
        "db_query_skill_id": raw_id,
        "skill_key": sk,
        "enabled": bool(row.enabled),
        "cadence": getattr(row.cadence, "value", str(row.cadence)),
        "time_of_day": row.time_of_day,
        "timezone": row.timezone,
        "channel": getattr(row.channel, "value", str(row.channel)),
        "destination": row.destination,
        "next_run_at": (row.next_run_at.isoformat() if row.next_run_at else None),
        "last_run_at": (row.last_run_at.isoformat() if row.last_run_at else None),
    }

# api/routers/dbquery.py
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from core.logging import logger

from api.deps.admin_key import require_admin_key
from deps import get_db
from services.user_service import get_or_create_primary_user
from services.db_query_service import DBQuerySpec, DBQueryResult, run_db_query, supported_entities
from services.db_query_skill_store import (
    create_db_query_skill,
    delete_db_query_skill,
    list_db_query_skills,
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
    try:
        return run_db_query(db, tenant_uuid=str(user.id), spec=payload.spec)
    except Exception as e:
        # Never 500 for user-facing analytics; return a safe error envelope.
        logger.exception("ADMIN_DBQUERY_RUN_FAIL")
        ent = "unknown"
        try:
            ent = str(payload.spec.entity)
        except Exception:
            pass
        return DBQueryResult(ok=False, entity=ent, count=0, rows=[], aggregates=None, spoken_summary=f"DB query failed: {e}")


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

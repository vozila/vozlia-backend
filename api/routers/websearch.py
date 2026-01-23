# api/routers/websearch.py
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from api.deps.admin_key import require_admin_key
from deps import get_db
from services.user_service import get_or_create_primary_user
from services.web_search_service import run_web_search
from services.web_search_skill_store import (
    create_web_search_skill,
    delete_web_search_skill,
    list_web_search_skills,
    list_schedules,
    upsert_daily_schedule,
)
from models import DeliveryChannel


router = APIRouter(prefix="/admin/websearch", tags=["admin-websearch"], dependencies=[Depends(require_admin_key)])


class WebSearchRunIn(BaseModel):
    query: str = Field(..., min_length=1)
    model: str | None = None


class WebSearchRunOut(BaseModel):
    query: str
    answer: str
    sources: list[dict] = []
    latency_ms: float | None = None
    model: str | None = None


class WebSearchSkillCreateIn(BaseModel):
    name: str = Field(..., min_length=1)
    query: str = Field(..., min_length=1)
    triggers: list[str] | None = None


class WebSearchSkillOut(BaseModel):
    id: str
    skill_key: str
    name: str
    query: str
    triggers: list[str] = []
    enabled: bool = True


class ScheduleUpsertIn(BaseModel):
    web_search_skill_id: str
    hour: int = Field(..., ge=0, le=23)
    minute: int = Field(..., ge=0, le=59)
    timezone: str = "America/New_York"
    channel: DeliveryChannel
    destination: str = Field(..., min_length=3)


class ScheduledDeliveryOut(BaseModel):
    id: str
    web_search_skill_id: str
    enabled: bool
    cadence: str
    time_of_day: str
    timezone: str
    channel: str
    destination: str
    next_run_at: str | None = None
    last_run_at: str | None = None


@router.post("/search", response_model=WebSearchRunOut)
def admin_websearch_run(payload: WebSearchRunIn, db: Session = Depends(get_db)):
    # Tenant resolution: current MVP is primary user
    user = get_or_create_primary_user(db)
    res = run_web_search(payload.query, model=payload.model)
    return {
        "query": res.query,
        "answer": res.answer,
        "sources": [{"title": s.title, "url": s.url, "snippet": s.snippet} for s in (res.sources or [])],
        "latency_ms": res.latency_ms,
        "model": res.model,
    }


@router.get("/skills", response_model=list[WebSearchSkillOut])
def admin_websearch_list_skills(db: Session = Depends(get_db)):
    user = get_or_create_primary_user(db)
    skills = list_web_search_skills(db, user)
    return [
        {
            "id": str(s.id),
            "skill_key": f"websearch_{s.id}",
            "name": s.name,
            "query": s.query,
            "triggers": s.triggers or [],
            "enabled": bool(s.enabled),
        }
        for s in skills
    ]


@router.post("/skills", response_model=WebSearchSkillOut)
def admin_websearch_create_skill(payload: WebSearchSkillCreateIn, db: Session = Depends(get_db)):
    user = get_or_create_primary_user(db)
    skill = create_web_search_skill(db, user, name=payload.name, query=payload.query, triggers=payload.triggers)
    return {
        "id": str(skill.id),
        "skill_key": f"websearch_{skill.id}",
        "name": skill.name,
        "query": skill.query,
        "triggers": skill.triggers or [],
        "enabled": bool(skill.enabled),
    }


@router.delete("/skills/{skill_id}")
def admin_websearch_delete_skill(skill_id: str, db: Session = Depends(get_db)):
    user = get_or_create_primary_user(db)
    ok = delete_web_search_skill(db, user, skill_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Skill not found")
    return {"ok": True}


@router.get("/schedules", response_model=list[ScheduledDeliveryOut])
def admin_websearch_list_schedules(db: Session = Depends(get_db)):
    user = get_or_create_primary_user(db)
    rows = list_schedules(db, user)
    return [
        {
            "id": str(r.id),
            "web_search_skill_id": str(r.web_search_skill_id),
            "enabled": bool(r.enabled),
            "cadence": getattr(r.cadence, "value", str(r.cadence)),
            "time_of_day": r.time_of_day,
            "timezone": r.timezone,
            "channel": getattr(r.channel, "value", str(r.channel)),
            "destination": r.destination,
            "next_run_at": (r.next_run_at.isoformat() if r.next_run_at else None),
            "last_run_at": (r.last_run_at.isoformat() if r.last_run_at else None),
        }
        for r in rows
    ]


@router.post("/schedules", response_model=ScheduledDeliveryOut)
def admin_websearch_upsert_schedule(payload: ScheduleUpsertIn, db: Session = Depends(get_db)):
    user = get_or_create_primary_user(db)
    row = upsert_daily_schedule(
        db,
        user,
        web_search_skill_id=payload.web_search_skill_id,
        hour=payload.hour,
        minute=payload.minute,
        timezone=payload.timezone,
        channel=payload.channel,
        destination=payload.destination,
    )
    return {
        "id": str(row.id),
        "web_search_skill_id": str(row.web_search_skill_id),
        "enabled": bool(row.enabled),
        "cadence": getattr(row.cadence, "value", str(row.cadence)),
        "time_of_day": row.time_of_day,
        "timezone": row.timezone,
        "channel": getattr(row.channel, "value", str(row.channel)),
        "destination": row.destination,
        "next_run_at": (row.next_run_at.isoformat() if row.next_run_at else None),
        "last_run_at": (row.last_run_at.isoformat() if row.last_run_at else None),
    }

# api/routers/metrics.py
from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import Any, Dict, Optional
from sqlalchemy.orm import Session

from api.deps.admin_key import require_admin_key
from deps import get_db
from services.user_service import get_or_create_primary_user
from services.metrics_service import maybe_answer_metrics

router = APIRouter(prefix="/admin/metrics", tags=["admin-metrics"], dependencies=[Depends(require_admin_key)])


class MetricsRunIn(BaseModel):
    question: str
    timezone: Optional[str] = None


@router.post("/run")
def metrics_run(body: MetricsRunIn, db: Session = Depends(get_db)) -> Dict[str, Any]:
    user = get_or_create_primary_user(db)
    tz = body.timezone or "America/New_York"
    out = None
    try:
        out = maybe_answer_metrics(db, tenant_id=str(user.id), text=body.question or "", default_tz=tz)
    except Exception:
        out = None

    if not out:
        return {"ok": False, "spoken_summary": "I can’t compute that metric yet from the current database.", "data": None}

    return {"ok": True, "spoken_summary": out.get("spoken_reply"), "data": out.get("data"), "key": out.get("key")}


@router.get("/capabilities")
def metrics_caps() -> Dict[str, Any]:
    return {
        "ok": True,
        "supports": [
            "calls count (today/yesterday/this week/last week/this month/last month)",
            "unique callers count (same timeframes)",
            "top callers",
            "email summary requests and skill executions",
            "top skills executed (best-effort)",
        ],
        "notes": [
            "Metrics are derived from caller_memory_events until call_sessions/skill_invocations tables are added.",
        ],
    }

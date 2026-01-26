# api/routers/metrics.py
from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from api.deps.admin_key import require_admin_key
from deps import get_db
from services.user_service import get_or_create_primary_user
from services.metrics_service import run_metrics_question, capabilities

router = APIRouter(prefix="/admin/metrics", tags=["metrics"])


class MetricsRunRequest(BaseModel):
    question: str
    timezone: str | None = None


@router.get("/health")
def metrics_health():
    return {"ok": True, "capabilities": capabilities()}


@router.get("/capabilities", dependencies=[Depends(require_admin_key)])
def metrics_capabilities():
    return capabilities()


@router.post("/run", dependencies=[Depends(require_admin_key)])
def metrics_run(payload: MetricsRunRequest, db: Session = Depends(get_db)):
    user = get_or_create_primary_user(db)
    tenant_id = str(user.id)

    out = run_metrics_question(
        db,
        tenant_id=tenant_id,
        question=payload.question,
        timezone=payload.timezone,
    )
    # Ensure version field exists
    out.setdefault("version", capabilities().get("version"))
    return out

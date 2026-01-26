# api/routers/metrics.py
from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from api.deps.admin_key import require_admin_key
from deps import get_db
from services.user_service import get_or_create_primary_user
from services.metrics_service import run_metrics_question, capabilities

router = APIRouter(prefix="/admin/metrics", tags=["admin-metrics"])


class MetricsRunIn(BaseModel):
    question: str = Field(..., description="Natural language metric question")
    timezone: str = Field(default="America/New_York")


@router.post("/run", dependencies=[Depends(require_admin_key)])
def run_metrics(payload: MetricsRunIn, db: Session = Depends(get_db)):
    user = get_or_create_primary_user(db)
    tenant_id = str(user.id)

    out = run_metrics_question(
        db,
        tenant_id=tenant_id,
        question=payload.question,
        timezone=payload.timezone,
    )
    if isinstance(out, dict):
        out.setdefault("version", capabilities().get("version"))
    return out


@router.get("/capabilities", dependencies=[Depends(require_admin_key)])
def get_capabilities():
    return capabilities()

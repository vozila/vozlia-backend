from __future__ import annotations

import os
import re
from typing import Any, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from api.deps.admin_key import require_admin_key
from api.deps.db import get_db
from api.deps.primary_user import get_or_create_primary_user
from services.db_query_service import DBQuerySpec, run_db_query

# Optional ORM models for lightweight lookups
from models import ConceptAssignment


router = APIRouter(
    prefix="/admin/metrics",
    tags=["admin_metrics"],
    dependencies=[Depends(require_admin_key)],
)


class MetricsRunIn(BaseModel):
    question: str = Field(..., min_length=1)
    timezone: Optional[str] = None


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _parse_timeframe(question: str, tz: str) -> Optional[dict[str, Any]]:
    q = _norm(question)

    # Common phrases
    if "today" in q:
        return {"preset": "today", "timezone": tz}
    if "yesterday" in q:
        return {"preset": "yesterday", "timezone": tz}

    if "this week" in q:
        return {"preset": "this_week", "timezone": tz}
    if "last week" in q:
        return {"preset": "last_week", "timezone": tz}

    # "last month" usually means "last 30 days" for most users.
    if "last month" in q:
        return {"preset": "last_30_days", "timezone": tz}
    if "this month" in q:
        return {"preset": "this_month", "timezone": tz}

    # Explicit ranges
    m = re.search(r"last\s+(\d+)\s+days?", q)
    if m:
        n = int(m.group(1))
        if n <= 1:
            return {"preset": "today", "timezone": tz}
        if n <= 7:
            return {"preset": "last_7_days", "timezone": tz}
        if n <= 30:
            return {"preset": "last_30_days", "timezone": tz}

    # Default: no timeframe
    return None


def _best_concept_for_question(db: Session, tenant_id: str, question: str) -> Optional[str]:
    q = _norm(question)
    # Cheap keyword extraction
    tokens = set(re.findall(r"[a-z0-9]{3,}", q))
    if not tokens:
        return None

    rows = (
        db.query(ConceptAssignment.concept_code)
        .filter(ConceptAssignment.tenant_id == tenant_id)
        .distinct()
        .all()
    )
    concept_codes = [r[0] for r in rows if r and r[0]]

    if not concept_codes:
        return None

    best = None
    best_score = 0
    for code in concept_codes:
        c = _norm(code)
        # Split code "menu.steak" -> {"menu","steak"}
        parts = set([p for p in re.split(r"[\W_]+", c) if p])
        score = len(tokens.intersection(parts))
        if score > best_score:
            best_score = score
            best = code

    # Require at least one matching token
    return best if best_score >= 1 else None


@router.post("/run")
def metrics_run(payload: MetricsRunIn, db: Session = Depends(get_db)) -> dict[str, Any]:
    """Answer simple metric questions deterministically.

    This endpoint exists primarily so the control-plane wizard can fastpath
    common "how many ..." questions without relying on an LLM plan.
    """
    user = get_or_create_primary_user(db)
    tz = payload.timezone or os.getenv("DEFAULT_TIMEZONE", "UTC")
    q = _norm(payload.question)

    # 1) Unique callers
    if "caller" in q and ("unique" in q or "how many" in q):
        tf = _parse_timeframe(payload.question, tz) or {"preset": "last_30_days", "timezone": tz}
        spec = DBQuerySpec(
            entity="caller_memory_events",
            timeframe=tf,
            aggregations=[
                {"op": "count_distinct", "field": "caller_id", "as_name": "unique_callers"}
            ],
            filters=[
                {"field": "kind", "op": "eq", "value": "turn"},
            ],
            limit=1,
        )
        res = run_db_query(db, user, spec)
        out = res.model_dump() if hasattr(res, "model_dump") else res.dict()  # pydantic v2/v1
        out["ok"] = True
        out["metric_kind"] = "unique_callers"
        return out

    # 2) Menu/Kb concept counts (e.g., "how many items on the menu has steak")
    if ("menu" in q or "dish" in q or "dishes" in q) and ("how many" in q or "count" in q or "number" in q):
        concept = _best_concept_for_question(db, str(user.id), payload.question)
        if concept:
            spec = DBQuerySpec(
                entity="kb_chunks",
                aggregations=[
                    {"op": "count_distinct", "field": "id", "as_name": "matching_chunks"}
                ],
                filters=[
                    {"field": "id", "op": "has_concept", "value": concept},
                ],
                limit=1,
            )
            res = run_db_query(db, user, spec)
            out = res.model_dump() if hasattr(res, "model_dump") else res.dict()
            out["ok"] = True
            out["metric_kind"] = "kb_concept_count"
            out["concept_code"] = concept
            return out

        # Fallback: basic keyword text match (first non-trivial token)
        kws = [t for t in re.findall(r"[a-z0-9]{4,}", q) if t not in {"menu", "dishes", "dish", "items"}]
        if kws:
            kw = kws[0]
            spec = DBQuerySpec(
                entity="kb_chunks",
                aggregations=[
                    {"op": "count_distinct", "field": "id", "as_name": "matching_chunks"}
                ],
                filters=[
                    {"field": "text", "op": "ilike", "value": kw},
                ],
                limit=1,
            )
            res = run_db_query(db, user, spec)
            out = res.model_dump() if hasattr(res, "model_dump") else res.dict()
            out["ok"] = True
            out["metric_kind"] = "kb_text_count"
            out["keyword"] = kw
            return out

    # Default: we don't know how to answer deterministically
    return {
        "ok": False,
        "spoken_summary": "I couldn't compute that metric deterministically yet.",
    }
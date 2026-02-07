"""api/routers/metrics.py

Purpose:
- Deterministic “metrics” endpoint used by the Control Plane wizard fast-path.
- Avoids LLM numeric hallucinations by translating common metric questions into DBQuerySpec and executing
  via services.db_query_service.run_db_query.

Notes:
- This is an admin endpoint (requires ADMIN_API_KEY).
- Keep it conservative: only support a small, explicit set of metric patterns for now.
"""

from __future__ import annotations

import re
from typing import Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from api.deps.admin_key import require_admin_key
from deps import get_db
from services.db_query_service import (
    DBQueryResult,
    DBQuerySpec,
    DBTimeframe,
    DBFilter,
    DBAggregation,
    TimePreset,
    run_db_query,
)
from services.user_service import get_or_create_primary_user


router = APIRouter(
    prefix="/admin/metrics",
    tags=["admin-metrics"],
    dependencies=[Depends(require_admin_key)],
)


class MetricsRunIn(BaseModel):
    question: str = Field(..., min_length=1)
    timezone: str = "America/New_York"


def _infer_time_preset(q: str) -> Optional[TimePreset]:
    s = q.lower()

    # Exact preset keywords
    if "today" in s:
        return "today"
    if "yesterday" in s:
        return "yesterday"

    # Week / month-ish
    if "this week" in s or "this_week" in s:
        return "this_week"
    if "last week" in s or "last_week" in s:
        return "last_week"
    if "this month" in s or "this_month" in s:
        return "this_month"

    # Rolling windows
    if "last 7 days" in s or "past 7 days" in s or "last_7_days" in s:
        return "last_7_days"
    if "last 30 days" in s or "past 30 days" in s or "last_30_days" in s:
        return "last_30_days"

    # Common phrasing (MVP mapping)
    if "last month" in s:
        # We don't have last_month; approximate as rolling 30 days for now.
        return "last_30_days"

    return None


def _looks_like_unique_callers(q: str) -> bool:
    s = q.lower()
    if "caller" in s and ("unique" in s or "how many" in s or "count" in s):
        return True
    return False


def _looks_like_unique_calls(q: str) -> bool:
    s = q.lower()
    if "call" in s and ("unique" in s or "how many" in s or "count" in s):
        # Avoid matching "caller" (callers handled above) unless explicitly "calls"
        if "caller" in s and "calls" not in s:
            return False
        return True
    return False


def _looks_like_menu_steak_question(q: str) -> bool:
    s = q.lower()
    if "steak" in s and ("menu" in s or "dish" in s or "dishes" in s):
        return True
    return False


def _extract_steak_dish_lines(text: str) -> list[str]:
    """Heuristic extraction of menu item lines mentioning steak.

    This is intentionally simple and deterministic. We'll replace with concept enrichment + structured KB later.
    """
    dish_lines: list[str] = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        if "steak" not in line.lower():
            continue
        # Common menu patterns: "Item Name — $X" or "Item Name - $X"
        if "—" in line or " - " in line or "$" in line:
            # Strip trailing price-ish fragments to get a stable title
            title = re.split(r"\s+—\s+|\s+-\s+|\s+\$", line, maxsplit=1)[0].strip()
            if title:
                dish_lines.append(title)
        else:
            dish_lines.append(line)
    # Deduplicate while preserving order
    seen = set()
    out = []
    for d in dish_lines:
        key = d.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(d)
    return out


@router.post("/run", response_model=DBQueryResult)
def admin_metrics_run(payload: MetricsRunIn, db: Session = Depends(get_db)):
    """Compute a small set of deterministic metrics from the DB."""
    user = get_or_create_primary_user(db)
    tenant_uuid = str(user.id)

    q = (payload.question or "").strip()
    tz = payload.timezone or "America/New_York"
    preset = _infer_time_preset(q)

    # 1) Unique callers metric (voice + portal)
    if _looks_like_unique_callers(q):
        spec = DBQuerySpec(
            entity="caller_memory_events",
            timeframe=DBTimeframe(preset=preset, timezone=tz) if preset else DBTimeframe(preset="last_30_days", timezone=tz),
            aggregations=[DBAggregation(op="count_distinct", field="caller_id", as_name="unique_callers")],
            limit=1,
        )
        return run_db_query(db, tenant_uuid=tenant_uuid, spec=spec)

    # 2) Unique calls metric
    if _looks_like_unique_calls(q):
        spec = DBQuerySpec(
            entity="caller_memory_events",
            timeframe=DBTimeframe(preset=preset, timezone=tz) if preset else DBTimeframe(preset="last_30_days", timezone=tz),
            aggregations=[DBAggregation(op="count_distinct", field="call_sid", as_name="unique_calls")],
            limit=1,
        )
        return run_db_query(db, tenant_uuid=tenant_uuid, spec=spec)

    # 3) Menu steak questions (MVP heuristic using KB text search)
    if _looks_like_menu_steak_question(q):
        # Pull all chunks that mention "steak" and do a deterministic extraction.
        chunks = run_db_query(
            db,
            tenant_uuid=tenant_uuid,
            spec=DBQuerySpec(
                entity="kb_chunks",
                select=["id", "file_id", "chunk_index", "text"],
                filters=[DBFilter(field="text", op="icontains", value="steak")],
                order_by=[{"field": "chunk_index", "direction": "asc"}],
                limit=200,
            ),
        )

        dish_titles: list[str] = []
        for row in chunks.rows:
            dish_titles.extend(_extract_steak_dish_lines(str(row.get("text") or "")))

        count = len(dish_titles)
        preview = dish_titles[:10]

        spoken = (
            f"I found {count} menu line(s) that mention steak."
            + (f" Examples: {', '.join(preview)}." if preview else "")
        )

        return DBQueryResult(
            ok=True,
            entity="kb_chunks",
            count=count,
            rows=[{"dish": d} for d in preview],
            aggregates={"steak_dish_count": count},
            spoken_summary=spoken,
        )

    # Default: return a safe message (keeps wizard stable)
    return DBQueryResult(
        ok=True,
        entity="metrics",
        count=0,
        rows=[],
        aggregates=None,
        spoken_summary="I can’t compute that metric yet from the current database.",
    )

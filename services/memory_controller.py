# services/memory_controller.py
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from core.logging import logger

try:
    from models import CallerMemoryEvent  # type: ignore
except Exception:
    CallerMemoryEvent = None  # type: ignore


DEBUG_MEMORY = (os.getenv("VOZLIA_DEBUG_MEMORY", "0") or "0").strip() == "1"


@dataclass
class MemoryQuery:
    start_ts: Optional[datetime] = None
    end_ts: Optional[datetime] = None
    skill_key: Optional[str] = None
    keywords: list[str] = None  # type: ignore


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def parse_memory_query(text: str) -> MemoryQuery:
    t = (text or "").lower()
    now = _utcnow()

    start = None
    end = now

    # Time parsing (minimal, expandable)
    if "yesterday" in t:
        start = now - timedelta(days=1)
    elif "last week" in t:
        start = now - timedelta(days=7)
    elif "last month" in t:
        start = now - timedelta(days=30)
    elif "last day" in t or "past day" in t:
        start = now - timedelta(days=1)
    elif "past" in t and "hours" in t:
        m = re.search(r"past\s+(\d{1,2})\s+hours", t)
        if m:
            start = now - timedelta(hours=int(m.group(1)))

    # Topic/skill inference (simple)
    skill = None
    if any(w in t for w in ["weather", "forecast", "temperature", "rain", "snow"]):
        skill = "weather"
    elif any(w in t for w in ["email", "inbox", "gmail"]):
        skill = "gmail_summary"

    # Keywords: simple token extraction
    toks = re.findall(r"[a-zA-Z0-9_\-']{3,}", t)
    stop = {"what", "did", "say", "about", "that", "this", "last", "week", "yesterday", "previous", "call", "remind", "me", "we", "talked", "past", "hours", "month", "day"}
    keywords = [x for x in toks if x not in stop][:12]

    return MemoryQuery(start_ts=start, end_ts=end, skill_key=skill, keywords=keywords)


def search_memory_events(
    db: Any,
    *,
    tenant_id: str,
    caller_id: str,
    q: MemoryQuery,
    limit: int = 8,
) -> list[dict[str, Any]]:
    if CallerMemoryEvent is None:
        return []
    if not (tenant_id and caller_id):
        return []

    now = _utcnow()
    qry = (
        db.query(CallerMemoryEvent)
        .filter(CallerMemoryEvent.tenant_id == tenant_id)
        .filter(CallerMemoryEvent.caller_id == caller_id)
        .filter((CallerMemoryEvent.expires_at.is_(None)) | (CallerMemoryEvent.expires_at >= now))
    )

    if q.start_ts is not None:
        qry = qry.filter(CallerMemoryEvent.created_at >= q.start_ts)
    if q.end_ts is not None:
        qry = qry.filter(CallerMemoryEvent.created_at <= q.end_ts)

    if q.skill_key:
        # Match either exact skill_key or tag-ish matches like "weather:*"
        qry = qry.filter(CallerMemoryEvent.skill_key.ilike(f"%{q.skill_key}%"))

    # Keyword filters (ILIKE AND). Cheap and good enough for MVP.
    if q.keywords:
        for kw in q.keywords:
            qry = qry.filter(CallerMemoryEvent.text.ilike(f"%{kw}%"))

    rows = qry.order_by(CallerMemoryEvent.created_at.desc()).limit(int(limit or 8)).all() or []

    out: list[dict[str, Any]] = []
    for r in rows:
        out.append(
            {
                "id": str(getattr(r, "id", "")),
                "created_at": (r.created_at or now).astimezone(timezone.utc).isoformat(),
                "skill_key": r.skill_key,
                "text": r.text,
                "call_sid": r.call_sid,
                "data_json": r.data_json,
                "tags": r.tags_json,
            }
        )

    if DEBUG_MEMORY:
        logger.info(
            "MEMORY_SEARCH_OK tenant_id=%s caller_id=%s hits=%s skill=%s kws=%s",
            tenant_id,
            caller_id,
            len(out),
            q.skill_key,
            len(q.keywords or []),
        )

    return out

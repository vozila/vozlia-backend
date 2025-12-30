# services/memory_controller.py
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Optional

from core.logging import logger
from models import CallerMemoryEvent


@dataclass
class MemoryQuery:
    start_ts: datetime
    end_ts: datetime
    skill_key: str | None
    keywords: list[str]
    raw_text: str


_TIME_PATTERNS = [
    # (regex, days_back)
    (re.compile(r"\b(yesterday)\b", re.I), 1),
    (re.compile(r"\b(last\s+week)\b", re.I), 7),
    (re.compile(r"\b(last\s+month)\b", re.I), 30),
]

_REL_RX = [
    (re.compile(r"\b(\d+)\s*(minute|minutes|min|mins)\s+ago\b", re.I), "minutes"),
    (re.compile(r"\b(\d+)\s*(hour|hours|hr|hrs)\s+ago\b", re.I), "hours"),
    (re.compile(r"\b(\d+)\s*(day|days)\s+ago\b", re.I), "days"),
]

def _utcnow() -> datetime:
    return datetime.now(timezone.utc)

def _parse_time_window(raw: str) -> tuple[datetime, datetime]:
    now = _utcnow()
    start = now - timedelta(days=30)
    end = now

    # last N days
    m = re.search(r"\b(last)\s+(\d+)\s+days\b", raw, re.I)
    if m:
        n = int(m.group(2))
        start = now - timedelta(days=max(1, min(n, 365)))
        return start, end

    # N minutes/hours/days ago => narrow to that window (± 10% or min 2 minutes)
    for rx, unit in _REL_RX:
        mm = rx.search(raw)
        if not mm:
            continue
        n = int(mm.group(1))
        if unit == "minutes":
            delta = timedelta(minutes=max(1, min(n, 60 * 24 * 31)))
        elif unit == "hours":
            delta = timedelta(hours=max(1, min(n, 24 * 31)))
        else:
            delta = timedelta(days=max(1, min(n, 365)))
        # Window: [now - delta - pad, now - delta + pad]
        pad = max(timedelta(minutes=2), timedelta(seconds=int(delta.total_seconds() * 0.10)))
        center = now - delta
        start = center - pad
        end = center + pad
        return start, end

    # Simple patterns
    for rx, days in _TIME_PATTERNS:
        if rx.search(raw):
            start = now - timedelta(days=days)
            return start, end

    return start, end

def parse_memory_query(text: str) -> MemoryQuery:
    raw = (text or "").strip()
    start, end = _parse_time_window(raw)
    low = raw.lower()

    # Topic/skill inference (expand later)
    skill = None
    if any(w in low for w in ["weather", "forecast", "temperature", "rain", "snow"]):
        skill = "weather"
    elif any(w in low for w in ["email", "inbox", "gmail"]):
        skill = "gmail_summary"

    # Keyword extraction (cheap) — also remove obvious memory-question filler
    tokens = re.findall(r"[a-zA-Z0-9_]{3,}", low)
    stop = {
        "what","did","say","about","that","this","last","week","month","yesterday","minutes","minute","hours","hour",
        "remind","me","we","talked","previous","call","report","ago","tell","told","remember",
        "favorite","color"  # queried explicitly
    }
    kws = [t for t in tokens if t not in stop][:16]

    return MemoryQuery(start_ts=start, end_ts=end, skill_key=skill, keywords=kws, raw_text=raw)

def search_memory_events(
    db: Any,
    *,
    tenant_id: str,
    caller_id: str,
    q: MemoryQuery,
    limit: int = 12,
) -> list[CallerMemoryEvent]:
    # Defensive: if DB missing, return empty
    if not tenant_id or not caller_id:
        return []

    # Base query
    qq = (
        db.query(CallerMemoryEvent)
        .filter(CallerMemoryEvent.tenant_id == tenant_id)
        .filter(CallerMemoryEvent.caller_id == caller_id)
        .filter(CallerMemoryEvent.created_at >= q.start_ts.replace(tzinfo=None))
        .filter(CallerMemoryEvent.created_at <= q.end_ts.replace(tzinfo=None))
    )

    if q.skill_key:
        qq = qq.filter(CallerMemoryEvent.skill_key == q.skill_key)

    # MVP keyword filter: OR over ILIKE
    if q.keywords:
        ors = []
        for kw in q.keywords[:8]:
            ors.append(CallerMemoryEvent.text.ilike(f"%{kw}%"))
        from sqlalchemy import or_
        qq = qq.filter(or_(*ors))

    rows = qq.order_by(CallerMemoryEvent.created_at.desc()).limit(limit).all()
    return rows

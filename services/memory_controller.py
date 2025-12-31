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
    (re.compile(r"\b(last\s+\d+)\s+days\b", re.I), None),
]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def parse_memory_query(text: str) -> MemoryQuery:
    raw = (text or "").strip()
    now = _utcnow()
    start = now - timedelta(days=30)
    end = now


    # Minutes / hours parsing (MVP)
    m = re.search(r"\b(last)\s+(\d+)\s+minutes\b", raw, re.I)
    if m:
        n = int(m.group(2))
        start = now - timedelta(minutes=min(max(n, 1), 60*24*7))
    else:
        m = re.search(r"\b(\d+)\s+minutes?\s+ago\b", raw, re.I)
        if m:
            n = int(m.group(1))
            start = now - timedelta(minutes=min(max(n * 3, 10), 60*24*7))
        else:
            m = re.search(r"\b(last)\s+(\d+)\s+hours\b", raw, re.I)
            if m:
                n = int(m.group(2))
                start = now - timedelta(hours=min(max(n, 1), 24*30))
            else:
                m = re.search(r"\b(\d+)\s+hours?\s+ago\b", raw, re.I)
                if m:
                    n = int(m.group(1))
                    start = now - timedelta(hours=min(max(n * 2, 2), 24*30))

    # Time parsing (MVP)
    m = re.search(r"\b(last)\s+(\d+)\s+days\b", raw, re.I)
    if m:
        n = int(m.group(2))
        start = now - timedelta(days=max(1, min(n, 365)))
    else:
        for rx, days in _TIME_PATTERNS:
            if rx.search(raw):
                if days is not None:
                    start = now - timedelta(days=days)
                break

    # Topic inference (MVP)
    skill = None
    low = raw.lower()
    if any(w in low for w in ["weather", "forecast", "temperature", "rain", "snow"]):
        skill = "weather"
    elif any(w in low for w in ["email", "inbox", "gmail"]):
        skill = "gmail_summary"

    # Keyword extraction (cheap)
    tokens = re.findall(r"[a-zA-Z0-9_]{3,}", low)
    stop = {"what","did","say","about","that","this","last","week","yesterday","remind","me","we","talked","previous","call","report"}
    kws = [t for t in tokens if t not in stop][:12]

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

# services/memory_controller.py
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from core.logging import logger
from models import CallerMemoryEvent

# Broad triggers for memory queries
_MEM_Q_TRIG = (
    "what did i say", "what did i tell", "remind me", "remember", "last time",
    "previous call", "yesterday", "last week", "five minutes", "minutes ago", "hours ago",
    "what were we talking about", "what was i talking about",
)

_COLOR_WORDS = {
    "red","orange","yellow","green","blue","purple","violet","pink","brown","black","white","gray","grey",
    "gold","silver","beige","tan","navy","teal","cyan","magenta","maroon","olive",
}


@dataclass
class MemoryQuery:
    start_ts: datetime
    end_ts: datetime
    skill_key: Optional[str] = None
    keywords: Optional[List[str]] = None


def looks_like_memory_question(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return False
    if any(x in t for x in _MEM_Q_TRIG):
        return True
    if "?" in t and ("ago" in t or "yesterday" in t or "last" in t):
        return True
    return False


def parse_memory_query(text: str, *, now_utc: Optional[datetime] = None) -> MemoryQuery:
    """
    Heuristic fallback parser (kept for backwards compatibility).
    The new preferred path is services.memory_llm.plan_memory_request.
    """
    now = now_utc or datetime.now(timezone.utc)
    t = (text or "").strip().lower()

    # default 24h
    lookback = timedelta(hours=24)

    # quick relative matches
    m = re.search(r"\b(\d+)\s*(minute|minutes)\b", t)
    if m:
        lookback = timedelta(minutes=int(m.group(1)))
    m = re.search(r"\b(\d+)\s*(hour|hours)\b", t)
    if m:
        lookback = timedelta(hours=int(m.group(1)))
    m = re.search(r"\b(\d+)\s*(day|days)\b", t)
    if m:
        lookback = timedelta(days=int(m.group(1)))

    if "yesterday" in t:
        # yesterday midnight->midnight (UTC)
        end = datetime(now.year, now.month, now.day, tzinfo=timezone.utc)
        start = end - timedelta(days=1)
        return MemoryQuery(start_ts=start, end_ts=end, keywords=["yesterday"])

    start_ts = now - lookback
    end_ts = now

    # keywords: keep broad and include obvious nouns
    kws: List[str] = []
    for w in re.findall(r"[a-z0-9_']+", t):
        if len(w) < 3:
            continue
        if w in ("what","did","say","tell","about","that","this","last","time","previous","call","remind","remember"):
            continue
        kws.append(w)

    # if mentions a color, include it
    for c in _COLOR_WORDS:
        if re.search(rf"\b{re.escape(c)}\b", t):
            kws.append(c)

    # dedupe
    seen = set()
    keywords = []
    for k in kws:
        k = k.strip().lower()
        if k and k not in seen:
            seen.add(k)
            keywords.append(k)

    return MemoryQuery(start_ts=start_ts, end_ts=end_ts, keywords=keywords[:12])


def search_memory_events(
    db,
    *,
    tenant_uuid: str,
    caller_id: str,
    q: MemoryQuery,
    limit: int = 50,
) -> List[CallerMemoryEvent]:
    qq = db.query(CallerMemoryEvent).filter(CallerMemoryEvent.tenant_id == str(tenant_uuid))
    if caller_id:
        qq = qq.filter(CallerMemoryEvent.caller_id == str(caller_id))

    qq = qq.filter(CallerMemoryEvent.created_at >= q.start_ts).filter(CallerMemoryEvent.created_at <= q.end_ts)

    if q.skill_key:
        qq = qq.filter(CallerMemoryEvent.skill_key == q.skill_key)

    if q.keywords:
        ors = []
        from sqlalchemy import or_
        for kw in q.keywords[:12]:
            ors.append(CallerMemoryEvent.text.ilike(f"%{kw}%"))
        qq = qq.filter(or_(*ors))

    rows = qq.order_by(CallerMemoryEvent.created_at.desc()).limit(limit).all()
    return rows or []

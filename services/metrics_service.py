# services/metrics_service.py
"""Deterministic, tenant-scoped metrics derived from existing DB tables.

Design goals:
- Never hallucinate numeric answers: metrics must come from DB queries.
- Works for both voice and portal troubleshooting (shared engine).
- Avoids schema churn: implements a small set of common metric intents using current tables.

Current backing data:
- caller_memory_events: call_sid, caller_id, kind, created_at, skill_key, text

Limitations:
- This is not a full BI layer. Without canonical call_sessions / skill_invocations tables,
  some metrics are best-effort and we say so in the spoken reply.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from typing import Any, Dict, Optional, Tuple

from sqlalchemy.orm import Session
from sqlalchemy import func, distinct

from models import CallerMemoryEvent


_METRIC_HINTS = (
    "how many",
    "number of",
    "how often",
    "times",
    "count",
    "most",
    "top",
    "least",
)


def looks_like_metric_question(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return False
    return any(h in t for h in _METRIC_HINTS)


def _parse_timeframe(text: str, default_tz: str) -> Tuple[Optional[datetime], Optional[datetime], str]:
    """Return (start_utc_naive, end_utc_naive, scope_label).

    DB stores created_at as naive UTC in this codebase; we therefore compare against naive UTC.
    """
    t = (text or "").lower()
    tz = ZoneInfo(default_tz or os.getenv("APP_TZ", "America/New_York"))
    now_utc = datetime.now(timezone.utc)
    now_local = now_utc.astimezone(tz)

    # default: last 7 days (conservative) if user implies timeframe but doesn't specify
    start_local = None
    end_local = None
    scope = ""

    def day_start(dt):
        return dt.replace(hour=0, minute=0, second=0, microsecond=0)

    if "today" in t:
        start_local = day_start(now_local)
        end_local = start_local + timedelta(days=1)
        scope = "today"
    elif "yesterday" in t:
        end_local = day_start(now_local)
        start_local = end_local - timedelta(days=1)
        scope = "yesterday"
    elif "this week" in t:
        start_local = day_start(now_local - timedelta(days=now_local.weekday()))
        end_local = start_local + timedelta(days=7)
        scope = "this week"
    elif "last week" in t or "past week" in t:
        end_local = day_start(now_local - timedelta(days=now_local.weekday()))
        start_local = end_local - timedelta(days=7)
        scope = "last week"
    elif "this month" in t:
        start_local = now_local.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        # next month start
        if start_local.month == 12:
            end_local = start_local.replace(year=start_local.year + 1, month=1)
        else:
            end_local = start_local.replace(month=start_local.month + 1)
        scope = "this month"
    elif "last month" in t or "past month" in t:
        this_month_start = now_local.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        end_local = this_month_start
        # previous month start
        if this_month_start.month == 1:
            start_local = this_month_start.replace(year=this_month_start.year - 1, month=12)
        else:
            start_local = this_month_start.replace(month=this_month_start.month - 1)
        scope = "last month"
    elif "this year" in t:
        start_local = now_local.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
        end_local = start_local.replace(year=start_local.year + 1)
        scope = "this year"
    elif "last 30" in t or "past 30" in t:
        start_utc = now_utc - timedelta(days=30)
        return (start_utc.replace(tzinfo=None), None, "past 30 days")
    elif "last 7" in t or "past 7" in t:
        start_utc = now_utc - timedelta(days=7)
        return (start_utc.replace(tzinfo=None), None, "past 7 days")

    if start_local is None and end_local is None:
        return (None, None, "")

    start_utc = start_local.astimezone(timezone.utc).replace(tzinfo=None)
    end_utc = end_local.astimezone(timezone.utc).replace(tzinfo=None) if end_local else None
    return (start_utc, end_utc, scope)


def _base_q(db: Session, tenant_id: str, start_utc: Optional[datetime], end_utc: Optional[datetime]):
    q = db.query(CallerMemoryEvent).filter(CallerMemoryEvent.tenant_id == str(tenant_id))
    if start_utc is not None:
        q = q.filter(CallerMemoryEvent.created_at >= start_utc)
    if end_utc is not None:
        q = q.filter(CallerMemoryEvent.created_at < end_utc)
    return q


def _count_distinct_calls(db: Session, tenant_id: str, start_utc: Optional[datetime], end_utc: Optional[datetime]) -> int:
    q = _base_q(db, tenant_id, start_utc, end_utc).filter(CallerMemoryEvent.call_sid.isnot(None))
    return int(q.with_entities(func.count(distinct(CallerMemoryEvent.call_sid))).scalar() or 0)


def _count_distinct_callers(db: Session, tenant_id: str, start_utc: Optional[datetime], end_utc: Optional[datetime]) -> int:
    q = _base_q(db, tenant_id, start_utc, end_utc)
    return int(q.with_entities(func.count(distinct(CallerMemoryEvent.caller_id))).scalar() or 0)


def _top_callers(db: Session, tenant_id: str, start_utc: Optional[datetime], end_utc: Optional[datetime], limit: int = 5):
    q = _base_q(db, tenant_id, start_utc, end_utc).filter(CallerMemoryEvent.call_sid.isnot(None))
    rows = (
        q.with_entities(
            CallerMemoryEvent.caller_id,
            func.count(distinct(CallerMemoryEvent.call_sid)).label("calls"),
        )
        .group_by(CallerMemoryEvent.caller_id)
        .order_by(func.count(distinct(CallerMemoryEvent.call_sid)).desc())
        .limit(limit)
        .all()
    )
    return [(r[0], int(r[1] or 0)) for r in rows]


def _count_skill_invocations(db: Session, tenant_id: str, skill_key: str, start_utc: Optional[datetime], end_utc: Optional[datetime]) -> int:
    q = _base_q(db, tenant_id, start_utc, end_utc).filter(
        CallerMemoryEvent.kind == "skill",
        CallerMemoryEvent.skill_key == skill_key,
    )
    return int(q.count() or 0)


def _count_turn_mentions(db: Session, tenant_id: str, phrase: str, start_utc: Optional[datetime], end_utc: Optional[datetime]) -> int:
    # best-effort: count distinct call_sids where user turns mention a phrase
    q = _base_q(db, tenant_id, start_utc, end_utc).filter(
        CallerMemoryEvent.kind == "turn",
        CallerMemoryEvent.call_sid.isnot(None),
        CallerMemoryEvent.text.ilike(f"%{phrase}%"),
    )
    return int(q.with_entities(func.count(distinct(CallerMemoryEvent.call_sid))).scalar() or 0)


def maybe_answer_metrics(db: Session, *, tenant_id: str, text: str, default_tz: str) -> Optional[Dict[str, Any]]:
    """Return {{spoken_reply, key, data}} if we can answer deterministically; else None."""
    t = (text or "").strip().lower()
    if not t:
        return None
    if not looks_like_metric_question(t):
        return None

    start_utc, end_utc, scope = _parse_timeframe(t, default_tz)
    scope_phrase = (f" ({scope})" if scope else "")

    # Calls received / callers
    if "call" in t and ("how many" in t or "number of" in t or "count" in t):
        calls = _count_distinct_calls(db, tenant_id, start_utc, end_utc)
        # Clarify that this is derived from memory events
        spoken = f"Calls received{scope_phrase}: {calls}." if scope else f"Calls received: {calls}."
        return {"key": "calls_count", "spoken_reply": spoken, "data": {"calls": calls, "scope": scope}}

    if ("customer" in t or "caller" in t) and ("how many" in t or "number of" in t or "count" in t):
        callers = _count_distinct_callers(db, tenant_id, start_utc, end_utc)
        spoken = f"Unique callers{scope_phrase}: {callers}." if scope else f"Unique callers: {callers}."
        return {"key": "callers_count", "spoken_reply": spoken, "data": {"unique_callers": callers, "scope": scope}}

    # Top caller
    if ("most calls" in t) or ("top caller" in t) or ("who" in t and "most" in t and "call" in t):
        top = _top_callers(db, tenant_id, start_utc, end_utc, limit=3)
        if not top:
            return {"key": "top_callers", "spoken_reply": f"I don’t see any calls{scope_phrase}.", "data": {"top": []}}
        parts = [f"{cid} ({cnt})" for cid, cnt in top]
        spoken = f"Top callers{scope_phrase}: " + ", ".join(parts) + "."
        return {"key": "top_callers", "spoken_reply": spoken, "data": {"top": top, "scope": scope}}

    # Email summaries / gmail summary usage or requests
    if ("email" in t or "gmail" in t) and ("summary" in t or "summaries" in t):
        # canonical skill_key from env defaults
        skill_key = "gmail_summary"
        inv = _count_skill_invocations(db, tenant_id, skill_key, start_utc, end_utc)

        # best-effort request intent mentions
        mention = _count_turn_mentions(db, tenant_id, "email summar", start_utc, end_utc)
        # If the user asked for "requested", prefer intent mentions; otherwise mention both
        if "request" in t or "requested" in t or "asking" in t:
            spoken = f"Calls where callers asked for email summaries{scope_phrase}: {mention}. (Skill executions: {inv}.)"
            return {"key": "email_summary_requests", "spoken_reply": spoken, "data": {"request_calls": mention, "skill_executions": inv, "scope": scope}}
        spoken = f"Email summary skill executions{scope_phrase}: {inv}." + (f" Calls asking for summaries: {mention}." if mention else "")
        return {"key": "email_summary_usage", "spoken_reply": spoken, "data": {"skill_executions": inv, "request_calls": mention, "scope": scope}}

    # Skills requested on calls (best-effort): top skill keys in time window
    if ("what skills" in t or "which skills" in t) and ("requested" in t or "used" in t or "on those calls" in t):
        q = _base_q(db, tenant_id, start_utc, end_utc).filter(CallerMemoryEvent.kind == "skill")
        rows = (
            q.with_entities(CallerMemoryEvent.skill_key, func.count(CallerMemoryEvent.id).label("n"))
            .group_by(CallerMemoryEvent.skill_key)
            .order_by(func.count(CallerMemoryEvent.id).desc())
            .limit(10)
            .all()
        )
        if not rows:
            return {"key": "skills_requested", "spoken_reply": f"I don’t see any skill executions{scope_phrase}.", "data": {"skills": []}}
        skills = [(r[0], int(r[1] or 0)) for r in rows]
        spoken = f"Top skills executed{scope_phrase}: " + ", ".join([f"{k} ({n})" for k, n in skills]) + "."
        return {"key": "skills_requested", "spoken_reply": spoken, "data": {"skills": skills, "scope": scope}}

    return None

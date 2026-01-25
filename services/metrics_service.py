# services/metrics_service.py
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, time
from typing import Any, Dict, List, Optional, Tuple

from zoneinfo import ZoneInfo

from sqlalchemy.orm import Session
from sqlalchemy import func, and_, distinct

from models import CallerMemoryEvent


METRICS_VERSION = "2026-01-25-metrics-recent-callers-v2"

# -----------------------------
# Helpers
# -----------------------------

@dataclass
class TimeRange:
    start_utc: datetime
    end_utc: datetime
    timezone: str


def _local_day_bounds_utc(day_local, tz: ZoneInfo) -> Tuple[datetime, datetime]:
    """Return UTC [start,end) bounds for a local date."""
    start_local = datetime.combine(day_local, time(0, 0, 0), tzinfo=tz)
    end_local = start_local + timedelta(days=1)
    return start_local.astimezone(ZoneInfo("UTC")).replace(tzinfo=None), end_local.astimezone(ZoneInfo("UTC")).replace(tzinfo=None)


def resolve_time_range(preset: str, timezone: str) -> TimeRange:
    tz = ZoneInfo(timezone)
    now_local = datetime.now(tz)
    preset = (preset or "today").lower().strip()

    if preset in ("today", "tod"):
        day = now_local.date()
        start_utc, end_utc = _local_day_bounds_utc(day, tz)
    elif preset in ("yesterday", "yday"):
        day = (now_local - timedelta(days=1)).date()
        start_utc, end_utc = _local_day_bounds_utc(day, tz)
    elif preset in ("this_week", "week", "thisweek"):
        # Week starts Monday
        day = now_local.date()
        start_of_week = day - timedelta(days=day.weekday())
        start_local = datetime.combine(start_of_week, time(0, 0, 0), tzinfo=tz)
        end_local = start_local + timedelta(days=7)
        start_utc = start_local.astimezone(ZoneInfo("UTC")).replace(tzinfo=None)
        end_utc = end_local.astimezone(ZoneInfo("UTC")).replace(tzinfo=None)
    elif preset in ("last_week", "lastweek"):
        day = now_local.date()
        start_of_this_week = day - timedelta(days=day.weekday())
        start_of_last_week = start_of_this_week - timedelta(days=7)
        start_local = datetime.combine(start_of_last_week, time(0, 0, 0), tzinfo=tz)
        end_local = start_local + timedelta(days=7)
        start_utc = start_local.astimezone(ZoneInfo("UTC")).replace(tzinfo=None)
        end_utc = end_local.astimezone(ZoneInfo("UTC")).replace(tzinfo=None)
    else:
        # Default to today
        day = now_local.date()
        start_utc, end_utc = _local_day_bounds_utc(day, tz)

    return TimeRange(start_utc=start_utc, end_utc=end_utc, timezone=timezone)


def _is_metric_question(q: str) -> bool:
    q = (q or "").lower()
    return any(x in q for x in ["how many", "count", "number of", "most", "top", "list", "recent"])


def _extract_limit(q: str, default: int = 25, max_limit: int = 200) -> int:
    m = re.search(r"\b(\d{1,3})\b", q)
    if not m:
        return default
    n = int(m.group(1))
    if n < 1:
        return default
    return min(n, max_limit)


# -----------------------------
# Metric computations (MVP)
# -----------------------------

def metric_calls_count(db: Session, tenant_id: str, tr: TimeRange) -> Dict[str, Any]:
    # "Calls received" approximated as distinct call_sid with at least one 'turn' event.
    q = (
        db.query(func.count(distinct(CallerMemoryEvent.call_sid)))
        .filter(
            CallerMemoryEvent.tenant_id == tenant_id,
            CallerMemoryEvent.created_at >= tr.start_utc,
            CallerMemoryEvent.created_at < tr.end_utc,
            CallerMemoryEvent.call_sid.isnot(None),
            CallerMemoryEvent.kind == "turn",
        )
    )
    calls = int(q.scalar() or 0)
    return {"calls": calls}


def metric_unique_callers(db: Session, tenant_id: str, tr: TimeRange) -> Dict[str, Any]:
    q = (
        db.query(func.count(distinct(CallerMemoryEvent.caller_id)))
        .filter(
            CallerMemoryEvent.tenant_id == tenant_id,
            CallerMemoryEvent.created_at >= tr.start_utc,
            CallerMemoryEvent.created_at < tr.end_utc,
            CallerMemoryEvent.kind == "turn",
        )
    )
    callers = int(q.scalar() or 0)
    return {"unique_callers": callers}


def metric_recent_callers(db: Session, tenant_id: str, tr: TimeRange, limit: int) -> Dict[str, Any]:
    # List callers sorted by last seen time within range (using any event kind, but must have call_sid or turn)
    last_seen = func.max(CallerMemoryEvent.created_at).label("last_seen")
    calls_cnt = func.count(distinct(CallerMemoryEvent.call_sid)).label("calls_in_range")

    rows = (
        db.query(CallerMemoryEvent.caller_id.label("caller_id"), last_seen, calls_cnt)
        .filter(
            CallerMemoryEvent.tenant_id == tenant_id,
            CallerMemoryEvent.created_at >= tr.start_utc,
            CallerMemoryEvent.created_at < tr.end_utc,
            CallerMemoryEvent.caller_id.isnot(None),
        )
        .group_by(CallerMemoryEvent.caller_id)
        .order_by(last_seen.desc())
        .limit(limit)
        .all()
    )

    items = []
    for r in rows:
        items.append(
            {
                "caller_id": r.caller_id,
                "last_seen_utc": r.last_seen.isoformat() if r.last_seen else None,
                "calls_in_range": int(r.calls_in_range or 0),
            }
        )
    return {"callers": items, "limit": limit}


# -----------------------------
# Question → metric router
# -----------------------------

def run_metrics_question(db: Session, tenant_id: str, question: str, timezone: str = "America/New_York") -> Dict[str, Any]:
    q = (question or "").strip()
    ql = q.lower()

    if not _is_metric_question(ql):
        return {"ok": False, "reason": "not_metric_question"}

    # Time presets
    if "yesterday" in ql:
        preset = "yesterday"
    elif "this week" in ql or "thisweek" in ql:
        preset = "this_week"
    elif "last week" in ql:
        preset = "last_week"
    else:
        preset = "today"

    tr = resolve_time_range(preset, timezone)

    # Recent callers list
    if ("recent" in ql or "most recent" in ql or "latest" in ql) and ("caller" in ql or "callers" in ql):
        limit = _extract_limit(ql, default=25)
        data = metric_recent_callers(db, tenant_id, tr, limit=limit)
        return {
            "ok": True,
            "metric": "calls.recent_callers",
            "timeframe": preset,
            "data": data,
            "spoken_summary": _spoken_recent_callers(data, preset, timezone),
        }

    # Calls count
    if "call" in ql and any(x in ql for x in ["how many", "count", "number of"]):
        data = metric_calls_count(db, tenant_id, tr)
        return {
            "ok": True,
            "metric": "calls.count",
            "timeframe": preset,
            "data": data,
            "spoken_summary": f"Calls received {preset.replace('_',' ')}: {data['calls']}.",
        }

    # Unique callers count
    if any(x in ql for x in ["caller", "callers", "customers"]) and any(x in ql for x in ["how many", "count", "number of"]):
        data = metric_unique_callers(db, tenant_id, tr)
        return {
            "ok": True,
            "metric": "calls.unique_callers",
            "timeframe": preset,
            "data": data,
            "spoken_summary": f"Unique callers {preset.replace('_',' ')}: {data['unique_callers']}.",
        }

    return {"ok": False, "reason": "unsupported_metric"}


def _spoken_recent_callers(data: Dict[str, Any], preset: str, timezone: str) -> str:
    callers = data.get("callers") or []
    if not callers:
        return f"No callers found {preset.replace('_',' ')}."
    # Keep it short for chat/voice
    top = callers[:10]
    lines = [f"Most recent callers {preset.replace('_',' ')} (showing {len(top)} of {len(callers)}):"]
    for i, c in enumerate(top, 1):
        cid_full = c.get("caller_id")
        cid = ("***" + cid_full[-4:]) if isinstance(cid_full, str) and len(cid_full) >= 4 else str(cid_full)
        calls_in_range = c.get("calls_in_range", 0)
        lines.append(f"{i}. {cid} ({calls_in_range} call session(s))")
    return "\n".join(lines)


def capabilities() -> dict:
    return {
        "version": METRICS_VERSION,
        "supported": [
            {
                "metric": "calls.recent_callers",
                "examples": [
                    "list the 25 most recent callers yesterday",
                    "show 10 latest callers today",
                    "most recent callers this week"
                ]
            },
            {
                "metric": "calls.count",
                "examples": [
                    "how many calls did we receive today?",
                    "call count this week",
                    "number of calls yesterday"
                ]
            },
            {
                "metric": "callers.count_unique",
                "examples": [
                    "how many unique callers today?",
                    "how many customers called yesterday?"
                ]
            }
        ]
    }

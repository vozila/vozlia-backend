# services/metrics_service.py
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, time
from typing import Any, Dict, List, Optional, Tuple

from zoneinfo import ZoneInfo

from sqlalchemy.orm import Session
from sqlalchemy import func, distinct, or_

from models import CallerMemoryEvent


METRICS_VERSION = "2026-01-26-metrics-email-summaries-v3"

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
    return (
        start_local.astimezone(ZoneInfo("UTC")).replace(tzinfo=None),
        end_local.astimezone(ZoneInfo("UTC")).replace(tzinfo=None),
    )


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
    elif preset in ("this_month", "month", "thismonth"):
        day = now_local.date()
        start_local = datetime.combine(day.replace(day=1), time(0, 0, 0), tzinfo=tz)
        # end: first day of next month
        if day.month == 12:
            next_month = day.replace(year=day.year + 1, month=1, day=1)
        else:
            next_month = day.replace(month=day.month + 1, day=1)
        end_local = datetime.combine(next_month, time(0, 0, 0), tzinfo=tz)
        start_utc = start_local.astimezone(ZoneInfo("UTC")).replace(tzinfo=None)
        end_utc = end_local.astimezone(ZoneInfo("UTC")).replace(tzinfo=None)
    elif preset in ("last_month", "lastmonth"):
        day = now_local.date()
        # start: first day of last month
        if day.month == 1:
            start_month = day.replace(year=day.year - 1, month=12, day=1)
        else:
            start_month = day.replace(month=day.month - 1, day=1)
        start_local = datetime.combine(start_month, time(0, 0, 0), tzinfo=tz)
        # end: first day of this month
        this_month = day.replace(day=1)
        end_local = datetime.combine(this_month, time(0, 0, 0), tzinfo=tz)
        start_utc = start_local.astimezone(ZoneInfo("UTC")).replace(tzinfo=None)
        end_utc = end_local.astimezone(ZoneInfo("UTC")).replace(tzinfo=None)
    else:
        # Default to today
        day = now_local.date()
        start_utc, end_utc = _local_day_bounds_utc(day, tz)

    return TimeRange(start_utc=start_utc, end_utc=end_utc, timezone=timezone)


def _normalize(q: str) -> str:
    q = (q or "").strip().lower()
    q = re.sub(r"[^a-z0-9\s']", " ", q)
    q = re.sub(r"\s+", " ", q).strip()
    return q


def _is_metric_question(q: str) -> bool:
    q = (q or "").lower()
    return any(x in q for x in ["how many", "count", "number of", "most", "top", "list", "recent", "latest"])


def _extract_limit(q: str, default: int = 25, max_limit: int = 200) -> int:
    m = re.search(r"\b(\d{1,3})\b", q)
    if not m:
        return default
    try:
        n = int(m.group(1))
    except Exception:
        return default
    if n < 1:
        return default
    return min(n, max_limit)


def _mask_caller_id(caller_id: str | None) -> str:
    if not caller_id:
        return "***----"
    s = str(caller_id)
    if len(s) >= 4:
        return "***" + s[-4:]
    return "***" + s


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

    items: List[Dict[str, Any]] = []
    for r in rows:
        items.append(
            {
                "caller_id": r.caller_id,
                "last_seen_utc": r.last_seen.isoformat() if r.last_seen else None,
                "calls_in_range": int(r.calls_in_range or 0),
            }
        )
    return {"callers": items, "limit": limit}


_EMAIL_SUMMARY_SQL_PATTERNS: List[str] = [
    "%email%summary%",
    "%email%summaries%",
    "%gmail%summary%",
    "%summariz%email%",
    "%summar%email%",
    "%email%digest%",
    "%email%report%",
    "%check%email%",
    "%check%my%email%",
    "%summarize%my%emails%",
    "%summarize%emails%",
]


def metric_email_summary_requests(db: Session, tenant_id: str, tr: TimeRange) -> Dict[str, Any]:
    """
    Approximate "email summaries requested" using:
      A) user turns that mention email summaries (requests)
      B) gmail_summary skill executions (serves)
    """
    # A) Request mentions (user turns)
    req_base = db.query(CallerMemoryEvent).filter(
        CallerMemoryEvent.tenant_id == tenant_id,
        CallerMemoryEvent.created_at >= tr.start_utc,
        CallerMemoryEvent.created_at < tr.end_utc,
        CallerMemoryEvent.kind == "turn",
        CallerMemoryEvent.skill_key == "turn_user",
    )
    req_filter = or_(*[CallerMemoryEvent.text.ilike(p) for p in _EMAIL_SUMMARY_SQL_PATTERNS])
    req_q = req_base.filter(req_filter)

    requests_mentions = int(req_q.count() or 0)
    calls_with_request = int(
        db.query(func.count(distinct(CallerMemoryEvent.call_sid)))
        .filter(
            CallerMemoryEvent.tenant_id == tenant_id,
            CallerMemoryEvent.created_at >= tr.start_utc,
            CallerMemoryEvent.created_at < tr.end_utc,
            CallerMemoryEvent.kind == "turn",
            CallerMemoryEvent.skill_key == "turn_user",
            CallerMemoryEvent.call_sid.isnot(None),
            req_filter,
        )
        .scalar()
        or 0
    )
    unique_callers_requested = int(
        db.query(func.count(distinct(CallerMemoryEvent.caller_id)))
        .filter(
            CallerMemoryEvent.tenant_id == tenant_id,
            CallerMemoryEvent.created_at >= tr.start_utc,
            CallerMemoryEvent.created_at < tr.end_utc,
            CallerMemoryEvent.kind == "turn",
            CallerMemoryEvent.skill_key == "turn_user",
            req_filter,
        )
        .scalar()
        or 0
    )

    # B) Skill executions (gmail_summary)
    exec_q = db.query(CallerMemoryEvent).filter(
        CallerMemoryEvent.tenant_id == tenant_id,
        CallerMemoryEvent.created_at >= tr.start_utc,
        CallerMemoryEvent.created_at < tr.end_utc,
        CallerMemoryEvent.kind == "skill",
        CallerMemoryEvent.skill_key == "gmail_summary",
    )
    executions = int(exec_q.count() or 0)
    calls_with_execution = int(
        db.query(func.count(distinct(CallerMemoryEvent.call_sid)))
        .filter(
            CallerMemoryEvent.tenant_id == tenant_id,
            CallerMemoryEvent.created_at >= tr.start_utc,
            CallerMemoryEvent.created_at < tr.end_utc,
            CallerMemoryEvent.kind == "skill",
            CallerMemoryEvent.skill_key == "gmail_summary",
            CallerMemoryEvent.call_sid.isnot(None),
        )
        .scalar()
        or 0
    )
    unique_callers_executed = int(
        db.query(func.count(distinct(CallerMemoryEvent.caller_id)))
        .filter(
            CallerMemoryEvent.tenant_id == tenant_id,
            CallerMemoryEvent.created_at >= tr.start_utc,
            CallerMemoryEvent.created_at < tr.end_utc,
            CallerMemoryEvent.kind == "skill",
            CallerMemoryEvent.skill_key == "gmail_summary",
        )
        .scalar()
        or 0
    )

    return {
        "requests_mentions": requests_mentions,
        "calls_with_request": calls_with_request,
        "unique_callers_requested": unique_callers_requested,
        "executions": executions,
        "calls_with_execution": calls_with_execution,
        "unique_callers_executed": unique_callers_executed,
    }


def _spoken_recent_callers(data: Dict[str, Any], preset: str) -> str:
    callers = data.get("callers") or []
    if not callers:
        return f"No callers found {preset.replace('_',' ')}."
    top = callers[:10]
    lines = [f"Most recent callers {preset.replace('_',' ')} (showing {len(top)} of {len(callers)}):"]
    for i, c in enumerate(top, 1):
        cid = _mask_caller_id(c.get("caller_id"))
        calls_in_range = c.get("calls_in_range", 0)
        lines.append(f"{i}. {cid} ({calls_in_range} call session(s))")
    return "\n".join(lines)


def _spoken_email_summary(metric_data: Dict[str, Any], preset: str, intent: str) -> str:
    # intent: "requested" | "executed" | "both"
    preset_label = preset.replace("_", " ")
    rm = int(metric_data.get("requests_mentions") or 0)
    cwr = int(metric_data.get("calls_with_request") or 0)
    ucr = int(metric_data.get("unique_callers_requested") or 0)
    ex = int(metric_data.get("executions") or 0)
    cwe = int(metric_data.get("calls_with_execution") or 0)
    uce = int(metric_data.get("unique_callers_executed") or 0)

    if intent == "executed":
        if ex == 0:
            return f"Email summaries were not executed {preset_label}."
        return f"Email summaries executed {preset_label}: {ex} time(s) across {cwe} call session(s) by {uce} caller(s)."

    if intent == "requested":
        if rm == 0 and cwr == 0:
            return f"No one requested email summaries {preset_label}."
        # Prefer calls-based phrasing when user says "called asking"
        if cwr > 0:
            return f"Email summaries requested {preset_label}: on {cwr} call session(s) by {ucr} caller(s) ({rm} request mention(s))."
        return f"Email summaries requested {preset_label}: {rm} request mention(s) by {ucr} caller(s)."

    # both
    if rm == 0 and ex == 0:
        return f"No email summary activity found {preset_label}."
    parts = []
    if rm or cwr:
        parts.append(f"requested on {cwr} call(s) ({rm} mention(s), {ucr} caller(s))")
    if ex:
        parts.append(f"executed {ex} time(s) ({cwe} call(s), {uce} caller(s))")
    return f"Email summaries {preset_label}: " + "; ".join(parts) + "."


# -----------------------------
# Question → metric router
# -----------------------------

def run_metrics_question(db: Session, tenant_id: str, question: str, timezone: str = "America/New_York") -> Dict[str, Any]:
    q = (question or "").strip()
    ql = _normalize(q)

    if not _is_metric_question(ql):
        return {"ok": False, "reason": "not_metric_question", "spoken_summary": "Not a metric question."}

    # Time presets
    if "yesterday" in ql:
        preset = "yesterday"
    elif "this week" in ql or "thisweek" in ql:
        preset = "this_week"
    elif "last week" in ql or "lastweek" in ql:
        preset = "last_week"
    elif "this month" in ql or "thismonth" in ql:
        preset = "this_month"
    elif "last month" in ql or "lastmonth" in ql:
        preset = "last_month"
    else:
        # Default for metrics with no explicit timeframe:
        preset = "today"

    tr = resolve_time_range(preset, timezone)

    # 1) Email summaries (requests/usage) — must be checked BEFORE calls.count because query may include "call"
    if ("email" in ql or "gmail" in ql) and any(x in ql for x in ["summary", "summaries", "digest", "report"]):
        data = metric_email_summary_requests(db, tenant_id, tr)
        # Determine intent
        intent = "both"
        if any(x in ql for x in ["executed", "execute", "ran", "run", "used", "usage"]):
            intent = "executed"
        elif any(x in ql for x in ["requested", "request", "asking", "asked", "call asking", "called asking"]):
            intent = "requested"
        return {
            "ok": True,
            "metric": "gmail_summary.activity",
            "timeframe": preset,
            "data": data,
            "spoken_summary": _spoken_email_summary(data, preset, intent=intent),
        }

    # 2) Recent callers list
    if any(x in ql for x in ["recent", "most recent", "latest"]) and ("caller" in ql or "callers" in ql):
        limit = _extract_limit(ql, default=25)
        data = metric_recent_callers(db, tenant_id, tr, limit=limit)
        return {
            "ok": True,
            "metric": "calls.recent_callers",
            "timeframe": preset,
            "data": data,
            "spoken_summary": _spoken_recent_callers(data, preset),
        }

    # 3) Calls count
    if "call" in ql and any(x in ql for x in ["how many", "count", "number of"]):
        data = metric_calls_count(db, tenant_id, tr)
        return {
            "ok": True,
            "metric": "calls.count",
            "timeframe": preset,
            "data": data,
            "spoken_summary": f"Calls received {preset.replace('_',' ')}: {data['calls']}.",
        }

    # 4) Unique callers count
    if any(x in ql for x in ["caller", "callers", "customers"]) and any(x in ql for x in ["how many", "count", "number of"]):
        data = metric_unique_callers(db, tenant_id, tr)
        return {
            "ok": True,
            "metric": "calls.unique_callers",
            "timeframe": preset,
            "data": data,
            "spoken_summary": f"Unique callers {preset.replace('_',' ')}: {data['unique_callers']}.",
        }

    return {
        "ok": False,
        "reason": "unsupported_metric",
        "spoken_summary": "I can’t compute that metric yet from the current database.",
        "data": None,
    }


def capabilities() -> dict:
    return {
        "version": METRICS_VERSION,
        "supported": [
            {
                "metric": "gmail_summary.activity",
                "examples": [
                    "how many times was email summary requested?",
                    "how many people called asking for email summaries yesterday?",
                    "how many times was gmail summary used this week?",
                ],
                "notes": "Reports request mentions in user turns and gmail_summary executions (best-effort).",
            },
            {
                "metric": "calls.recent_callers",
                "examples": [
                    "list the 25 most recent callers yesterday",
                    "show 10 latest callers today",
                    "most recent callers this week",
                ],
            },
            {
                "metric": "calls.count",
                "examples": [
                    "how many calls did we receive today?",
                    "call count this week",
                    "number of calls yesterday",
                ],
            },
            {
                "metric": "calls.unique_callers",
                "examples": [
                    "how many unique callers today?",
                    "how many customers called yesterday?",
                ],
            },
        ],
    }


# -----------------------------
# Backwards-compatible wrappers
# (some versions of assistant_service import these)
# -----------------------------

def looks_like_metric_question(text: str) -> bool:
    """Return True if the utterance appears to request a quantitative metric."""
    try:
        return _is_metric_question(text or "")
    except Exception:
        return False


def maybe_answer_metrics(
    db: Session,
    *,
    tenant_id: str,
    text: str,
    timezone: str = "America/New_York",
) -> str | None:
    """Return a short spoken answer for a metric question, or None.

    Kept for compatibility with assistant_service, which expects a deterministic
    string reply (and should never hallucinate numeric values).
    """
    try:
        out = run_metrics_question(db, tenant_id, text, timezone=timezone)
    except Exception:
        return None
    if not isinstance(out, dict) or not out.get("ok"):
        return None
    s = out.get("spoken_summary")
    if isinstance(s, str) and s.strip():
        return s.strip()
    return None

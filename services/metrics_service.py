# services/metrics_service.py
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Optional
from uuid import UUID

from zoneinfo import ZoneInfo
from sqlalchemy import and_, or_, func, distinct, desc
from sqlalchemy.orm import Session

from core.logging import logger
from models import (
    CallerMemoryEvent,
    WebSearchSkill,
    ScheduledDelivery,
    CallerSkillCache,
    KBDocument,
    KBDocumentStatus,
    Task,
    TaskStatus,
    EmailAccount,
    UserSetting,
    DeliveryChannel,
)

# IMPORTANT:
# - This module is used by both Portal troubleshooting chat and Voice "DB lookup" fast-path.
# - Keep it deterministic; do not call LLMs from here.
# - Avoid heavy work in audio hot paths; queries should be indexed and bounded.

METRICS_VERSION = "2026-01-26-metrics-500q-v1"

# -----------------------------------------------------------------------------
# Parsing helpers
# -----------------------------------------------------------------------------

_TIMEFRAME_PATTERNS: list[tuple[str, str]] = [
    (r"\btoday\b", "today"),
    (r"\byesterday\b", "yesterday"),
    (r"\bthis\s+week\b", "this_week"),
    (r"\blast\s+week\b", "last_week"),
    (r"\bthis\s+month\b", "this_month"),
    (r"\blast\s+30\s+days\b", "last_30_days"),
    (r"\blast\s+90\s+days\b", "last_90_days"),
]

_COMPARE_RE = re.compile(r"\bcompare\b", re.I)
_VS_RE = re.compile(r"\b(vs\.?|versus)\b", re.I)

# Extract limits like: "top 10", "list the 25", "list 50"
_LIMIT_RE = re.compile(r"\b(?:top|list(?:\s+the)?)\s+(\d+)\b", re.I)

# Extract explicit skill keys like: skill 'gmail_summary'
_SKILL_QUOTED_RE = re.compile(r"\bskill\s+['\"]([a-zA-Z0-9_\-]+)['\"]\b", re.I)

# Friendly-to-skill_key mapping
_SKILL_SYNONYMS: dict[str, str] = {
    "email summaries": "gmail_summary",
    "email summary": "gmail_summary",
    "gmail summary": "gmail_summary",
    "investment report": "investment_reporting",
    "investment reporting": "investment_reporting",
    "web search": "web_search",
    "websearch": "web_search",
    "kb": "kb_query",
    "knowledge base": "kb_query",
    "lead capture": "lead_capture",
    "appointment request": "appointment_request",
}

# Keywords that indicate the user is asking for a metric (not chitchat)
_METRIC_HINTS = (
    "how many",
    "count",
    "list",
    "show",
    "top",
    "most recent",
    "compare",
    "versus",
    "vs",
    "invoked",
    "invocations",
    "created",
    "enabled",
    "disabled",
    "pending",
    "completed",
    "due to run",
)

_EMAIL_SUMMARY_REQUEST_HINTS = (
    "email summaries",
    "email summary",
    "gmail summary",
    "summarize my email",
    "summarize my emails",
)

@dataclass
class TimeWindow:
    start_utc: datetime  # naive UTC
    end_utc: datetime    # naive UTC
    preset: str


def _normalize_q(q: str) -> str:
    t = (q or "").strip().lower()
    t = re.sub(r"[^\w\s'\"@:/\-\.]", "", t)  # keep quotes, @, etc.
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _safe_tz(tz_name: str | None) -> str:
    tz_name = (tz_name or "").strip() or "America/New_York"
    try:
        ZoneInfo(tz_name)
        return tz_name
    except Exception:
        return "America/New_York"


def _utcnow_naive() -> datetime:
    # App convention: DB stores naive timestamps representing UTC.
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _resolve_time_window(preset: str | None, tz_name: str, *, now_utc_naive: datetime | None = None) -> TimeWindow:
    tz_name = _safe_tz(tz_name)
    tz = ZoneInfo(tz_name)

    now_utc_naive = now_utc_naive or _utcnow_naive()
    now_utc = now_utc_naive.replace(tzinfo=timezone.utc)
    now_local = now_utc.astimezone(tz)

    start_of_today = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
    start_of_week = start_of_today - timedelta(days=start_of_today.weekday())  # Monday
    start_of_month = start_of_today.replace(day=1)

    def _local_to_utc_naive(dt_local: datetime) -> datetime:
        return dt_local.astimezone(timezone.utc).replace(tzinfo=None)

    preset = (preset or "today").strip()

    if preset == "today":
        start_local = start_of_today
        end_local = now_local
    elif preset == "yesterday":
        start_local = start_of_today - timedelta(days=1)
        end_local = start_of_today
    elif preset == "this_week":
        start_local = start_of_week
        end_local = now_local
    elif preset == "last_week":
        start_local = start_of_week - timedelta(days=7)
        end_local = start_of_week
    elif preset == "this_month":
        start_local = start_of_month
        end_local = now_local
    elif preset == "last_30_days":
        start_local = now_local - timedelta(days=30)
        end_local = now_local
    elif preset == "last_90_days":
        start_local = now_local - timedelta(days=90)
        end_local = now_local
    else:
        # default safely to today
        start_local = start_of_today
        end_local = now_local
        preset = "today"

    return TimeWindow(start_utc=_local_to_utc_naive(start_local), end_utc=_local_to_utc_naive(end_local), preset=preset)


def _extract_timeframe_preset(q_norm: str) -> str | None:
    for pat, preset in _TIMEFRAME_PATTERNS:
        if re.search(pat, q_norm, re.I):
            return preset
    return None


def _extract_limit(q_norm: str, default: int) -> int:
    m = _LIMIT_RE.search(q_norm)
    if not m:
        return default
    try:
        n = int(m.group(1))
        if 1 <= n <= 500:
            return n
    except Exception:
        pass
    return default


def _tenant_uuid(tenant_id: str) -> UUID | None:
    try:
        return UUID(str(tenant_id))
    except Exception:
        return None


def _mask_phone(s: str) -> str:
    t = (s or "").strip()
    if not t:
        return t
    # simple phone mask: keep last 4 digits
    digits = re.sub(r"\D", "", t)
    if len(digits) >= 4:
        return f"***{digits[-4:]}"
    return "***"


def _local_time_expr(col, tz_name: str):
    # created_at is naive timestamp representing UTC.
    # Convert: (created_at AT TIME ZONE 'UTC') AT TIME ZONE 'America/New_York'
    # In SQLAlchemy: timezone(tz, timezone('UTC', col))
    return func.timezone(tz_name, func.timezone("UTC", col))


def looks_like_metric_question(text: str) -> bool:
    q = _normalize_q(text)
    if not q:
        return False
    return any(h in q for h in _METRIC_HINTS) or any(w in q for w in ("calls", "callers", "schedules", "skills", "tasks", "kb"))


def maybe_answer_metrics(
    db: Session,
    *,
    tenant_id: str,
    question: str,
    timezone: str | None = None,
    **_: Any,
) -> dict[str, Any] | None:
    """Backwards-compatible entrypoint used by assistant_service.

    Returns:
      - dict response if this is a metric question (ok true/false)
      - None if not a metric question
    """
    if not looks_like_metric_question(question):
        return None
    return run_metrics_question(db, tenant_id=tenant_id, question=question, timezone=timezone or "America/New_York")


def capabilities() -> dict[str, Any]:
    return {
        "version": METRICS_VERSION,
        "supported_metrics": [
            # calls / memory
            "calls.count",
            "calls.unique_callers",
            "memory.events_count",
            "calls.top_callers",
            "calls.recent_callers",
            "calls.recent_sessions",
            "calls.sessions_timeseries",
            "calls.unique_callers_timeseries",
            # skills
            "skills.invocations",
            "skills.distinct_callers",
            "skills.top_invoked",
            "skills.invocations_timeseries",
            "skills.recent_invocations",
            "skills.email_summaries_requested",
            # websearch & schedules
            "websearch.skills_count",
            "websearch.skills_list",
            "websearch.skills_enabled_count",
            "websearch.skills_disabled_count",
            "websearch.skills_enabled_list",
            "websearch.skills_disabled_list",
            "schedules.count_created",
            "schedules.enabled_count",
            "schedules.disabled_count",
            "schedules.recent_updates",
            "schedules.due_next",
            "schedules.by_channel",
            "schedules.by_timezone",
            "schedules.by_destination_domain",
            "schedules.never_run",
            # cache
            "cache.writes_count",
            "cache.recent_writes",
            "cache.by_skill_key",
            "cache.by_caller",
            # kb
            "kb.count_created",
            "kb.by_status",
            "kb.list_by_status",
            # tasks
            "tasks.count_created",
            "tasks.count_completed",
            "tasks.pending_count",
            "tasks.list_pending",
            "tasks.list_created",
            "tasks.list_completed",
            # email accounts
            "email.accounts_count",
            "email.accounts_list",
            "email.accounts_active_count",
            "email.accounts_active_list",
            "email.default_sender",
            # settings
            "settings.recent_keys",
            "settings.changed_keys_count",
            # comparisons
            "compare.calls",
            "compare.unique_callers",
            "compare.skill_invocations",
        ],
        "notes": {
            "call_sessions_definition": "count(distinct call_sid) from caller_memory_events within timeframe",
            "skill_invocations_definition": "count rows where caller_memory_events.kind='skill' within timeframe",
            "pii_masking": "caller_id masked in spoken_summary",
        },
    }


def _ok(metric: str, timeframe: str, data: dict[str, Any], spoken_summary: str) -> dict[str, Any]:
    return {
        "ok": True,
        "metric": metric,
        "timeframe": timeframe,
        "data": data,
        "spoken_summary": spoken_summary,
        "version": METRICS_VERSION,
    }


def _fail(spoken_summary: str) -> dict[str, Any]:
    return {
        "ok": False,
        "spoken_summary": spoken_summary,
        "data": None,
        "version": METRICS_VERSION,
    }

# -----------------------------------------------------------------------------
# Query builders
# -----------------------------------------------------------------------------

def _mem_q(db: Session, tenant_id: str, w: TimeWindow):
    return (
        db.query(CallerMemoryEvent)
        .filter(CallerMemoryEvent.tenant_id == str(tenant_id))
        .filter(CallerMemoryEvent.created_at >= w.start_utc)
        .filter(CallerMemoryEvent.created_at < w.end_utc)
    )


def _count_call_sessions(db: Session, tenant_id: str, w: TimeWindow) -> int:
    q = (
        db.query(func.count(distinct(CallerMemoryEvent.call_sid)))
        .filter(CallerMemoryEvent.tenant_id == str(tenant_id))
        .filter(CallerMemoryEvent.created_at >= w.start_utc)
        .filter(CallerMemoryEvent.created_at < w.end_utc)
        .filter(CallerMemoryEvent.call_sid.isnot(None))
        .filter(CallerMemoryEvent.call_sid != "")
    )
    return int(q.scalar() or 0)


def _count_unique_callers(db: Session, tenant_id: str, w: TimeWindow) -> int:
    q = (
        db.query(func.count(distinct(CallerMemoryEvent.caller_id)))
        .filter(CallerMemoryEvent.tenant_id == str(tenant_id))
        .filter(CallerMemoryEvent.created_at >= w.start_utc)
        .filter(CallerMemoryEvent.created_at < w.end_utc)
        .filter(CallerMemoryEvent.call_sid.isnot(None))
        .filter(CallerMemoryEvent.call_sid != "")
    )
    return int(q.scalar() or 0)


def _count_memory_events(db: Session, tenant_id: str, w: TimeWindow) -> int:
    q = (
        db.query(func.count(CallerMemoryEvent.id))
        .filter(CallerMemoryEvent.tenant_id == str(tenant_id))
        .filter(CallerMemoryEvent.created_at >= w.start_utc)
        .filter(CallerMemoryEvent.created_at < w.end_utc)
    )
    return int(q.scalar() or 0)


def _recent_callers(db: Session, tenant_id: str, w: TimeWindow, tz_name: str, limit: int):
    last_seen = func.max(CallerMemoryEvent.created_at).label("last_seen_utc")
    calls_in_range = func.count(distinct(CallerMemoryEvent.call_sid)).label("calls_in_range")
    rows = (
        db.query(CallerMemoryEvent.caller_id, last_seen, calls_in_range)
        .filter(CallerMemoryEvent.tenant_id == str(tenant_id))
        .filter(CallerMemoryEvent.created_at >= w.start_utc)
        .filter(CallerMemoryEvent.created_at < w.end_utc)
        .filter(CallerMemoryEvent.call_sid.isnot(None))
        .filter(CallerMemoryEvent.call_sid != "")
        .group_by(CallerMemoryEvent.caller_id)
        .order_by(desc(last_seen))
        .limit(limit)
        .all()
    )
    out = []
    for r in rows:
        out.append(
            {
                "caller_id": r[0],
                "last_seen_utc": (r[1].isoformat() if r[1] else None),
                "calls_in_range": int(r[2] or 0),
            }
        )
    return out


def _recent_call_sessions(db: Session, tenant_id: str, w: TimeWindow, limit: int):
    last_seen = func.max(CallerMemoryEvent.created_at).label("last_seen_utc")
    caller_any = func.max(CallerMemoryEvent.caller_id).label("caller_id")
    rows = (
        db.query(CallerMemoryEvent.call_sid, caller_any, last_seen)
        .filter(CallerMemoryEvent.tenant_id == str(tenant_id))
        .filter(CallerMemoryEvent.created_at >= w.start_utc)
        .filter(CallerMemoryEvent.created_at < w.end_utc)
        .filter(CallerMemoryEvent.call_sid.isnot(None))
        .filter(CallerMemoryEvent.call_sid != "")
        .group_by(CallerMemoryEvent.call_sid)
        .order_by(desc(last_seen))
        .limit(limit)
        .all()
    )
    out = []
    for call_sid, caller_id, last_seen_dt in rows:
        out.append(
            {
                "call_sid": call_sid,
                "caller_id": caller_id,
                "last_seen_utc": (last_seen_dt.isoformat() if last_seen_dt else None),
            }
        )
    return out


def _top_callers(db: Session, tenant_id: str, w: TimeWindow, limit: int):
    call_sessions = func.count(distinct(CallerMemoryEvent.call_sid)).label("call_sessions")
    last_seen = func.max(CallerMemoryEvent.created_at).label("last_seen_utc")
    rows = (
        db.query(CallerMemoryEvent.caller_id, call_sessions, last_seen)
        .filter(CallerMemoryEvent.tenant_id == str(tenant_id))
        .filter(CallerMemoryEvent.created_at >= w.start_utc)
        .filter(CallerMemoryEvent.created_at < w.end_utc)
        .filter(CallerMemoryEvent.call_sid.isnot(None))
        .filter(CallerMemoryEvent.call_sid != "")
        .group_by(CallerMemoryEvent.caller_id)
        .order_by(desc(call_sessions), desc(last_seen))
        .limit(limit)
        .all()
    )
    out = []
    for caller_id, cs, last_seen_dt in rows:
        out.append(
            {
                "caller_id": caller_id,
                "call_sessions": int(cs or 0),
                "last_seen_utc": (last_seen_dt.isoformat() if last_seen_dt else None),
            }
        )
    return out


def _timeseries_call_sessions(db: Session, tenant_id: str, w: TimeWindow, tz_name: str, bucket: str):
    bucket_expr = func.date_trunc(bucket, _local_time_expr(CallerMemoryEvent.created_at, tz_name)).label("bucket_local")
    call_sessions = func.count(distinct(CallerMemoryEvent.call_sid)).label("call_sessions")
    rows = (
        db.query(bucket_expr, call_sessions)
        .filter(CallerMemoryEvent.tenant_id == str(tenant_id))
        .filter(CallerMemoryEvent.created_at >= w.start_utc)
        .filter(CallerMemoryEvent.created_at < w.end_utc)
        .filter(CallerMemoryEvent.call_sid.isnot(None))
        .filter(CallerMemoryEvent.call_sid != "")
        .group_by(bucket_expr)
        .order_by(bucket_expr.asc())
        .all()
    )
    out = []
    for bkt, cs in rows:
        out.append({"bucket_local": (bkt.isoformat() if bkt else None), "call_sessions": int(cs or 0)})
    return out


def _timeseries_unique_callers(db: Session, tenant_id: str, w: TimeWindow, tz_name: str, bucket: str):
    bucket_expr = func.date_trunc(bucket, _local_time_expr(CallerMemoryEvent.created_at, tz_name)).label("bucket_local")
    callers = func.count(distinct(CallerMemoryEvent.caller_id)).label("unique_callers")
    rows = (
        db.query(bucket_expr, callers)
        .filter(CallerMemoryEvent.tenant_id == str(tenant_id))
        .filter(CallerMemoryEvent.created_at >= w.start_utc)
        .filter(CallerMemoryEvent.created_at < w.end_utc)
        .filter(CallerMemoryEvent.call_sid.isnot(None))
        .filter(CallerMemoryEvent.call_sid != "")
        .group_by(bucket_expr)
        .order_by(bucket_expr.asc())
        .all()
    )
    out = []
    for bkt, n in rows:
        out.append({"bucket_local": (bkt.isoformat() if bkt else None), "unique_callers": int(n or 0)})
    return out


def _skill_key_from_question(q_norm: str) -> str | None:
    m = _SKILL_QUOTED_RE.search(q_norm)
    if m:
        return m.group(1).strip()
    # friendly synonyms
    for k, v in _SKILL_SYNONYMS.items():
        if k in q_norm:
            return v
    return None


def _skills_q(db: Session, tenant_id: str, w: TimeWindow):
    return (
        db.query(CallerMemoryEvent)
        .filter(CallerMemoryEvent.tenant_id == str(tenant_id))
        .filter(CallerMemoryEvent.created_at >= w.start_utc)
        .filter(CallerMemoryEvent.created_at < w.end_utc)
        .filter(CallerMemoryEvent.kind == "skill")
    )


def _count_skill_invocations(db: Session, tenant_id: str, w: TimeWindow, skill_key: str | None) -> int:
    q = _skills_q(db, tenant_id, w)
    if skill_key:
        q = q.filter(CallerMemoryEvent.skill_key == skill_key)
    return int(q.with_entities(func.count(CallerMemoryEvent.id)).scalar() or 0)


def _count_skill_distinct_callers(db: Session, tenant_id: str, w: TimeWindow, skill_key: str | None) -> int:
    q = _skills_q(db, tenant_id, w)
    if skill_key:
        q = q.filter(CallerMemoryEvent.skill_key == skill_key)
    return int(q.with_entities(func.count(distinct(CallerMemoryEvent.caller_id))).scalar() or 0)


def _skills_timeseries(db: Session, tenant_id: str, w: TimeWindow, tz_name: str, bucket: str, skill_key: str | None):
    bucket_expr = func.date_trunc(bucket, _local_time_expr(CallerMemoryEvent.created_at, tz_name)).label("bucket_local")
    inv = func.count(CallerMemoryEvent.id).label("invocations")
    q = (
        db.query(bucket_expr, inv)
        .filter(CallerMemoryEvent.tenant_id == str(tenant_id))
        .filter(CallerMemoryEvent.created_at >= w.start_utc)
        .filter(CallerMemoryEvent.created_at < w.end_utc)
        .filter(CallerMemoryEvent.kind == "skill")
    )
    if skill_key:
        q = q.filter(CallerMemoryEvent.skill_key == skill_key)
    rows = q.group_by(bucket_expr).order_by(bucket_expr.asc()).all()
    out = []
    for bkt, n in rows:
        out.append({"bucket_local": (bkt.isoformat() if bkt else None), "invocations": int(n or 0)})
    return out


def _skills_recent_invocations(db: Session, tenant_id: str, w: TimeWindow, limit: int):
    rows = (
        db.query(CallerMemoryEvent)
        .filter(CallerMemoryEvent.tenant_id == str(tenant_id))
        .filter(CallerMemoryEvent.created_at >= w.start_utc)
        .filter(CallerMemoryEvent.created_at < w.end_utc)
        .filter(CallerMemoryEvent.kind == "skill")
        .order_by(CallerMemoryEvent.created_at.desc())
        .limit(limit)
        .all()
    )
    out = []
    for e in rows:
        out.append(
            {
                "created_at_utc": (e.created_at.isoformat() if e.created_at else None),
                "caller_id": e.caller_id,
                "call_sid": e.call_sid,
                "skill_key": e.skill_key,
                "text": (e.text or "")[:240],
            }
        )
    return out


def _skills_top(db: Session, tenant_id: str, w: TimeWindow, limit: int, by_distinct_callers: bool = False):
    if by_distinct_callers:
        metric = func.count(distinct(CallerMemoryEvent.caller_id)).label("distinct_callers")
    else:
        metric = func.count(CallerMemoryEvent.id).label("invocations")
    rows = (
        db.query(CallerMemoryEvent.skill_key, metric)
        .filter(CallerMemoryEvent.tenant_id == str(tenant_id))
        .filter(CallerMemoryEvent.created_at >= w.start_utc)
        .filter(CallerMemoryEvent.created_at < w.end_utc)
        .filter(CallerMemoryEvent.kind == "skill")
        .group_by(CallerMemoryEvent.skill_key)
        .order_by(desc(metric))
        .limit(limit)
        .all()
    )
    out = []
    for sk, n in rows:
        out.append({"skill_key": sk, ("distinct_callers" if by_distinct_callers else "invocations"): int(n or 0)})
    return out


def _email_summaries_requested(db: Session, tenant_id: str, w: TimeWindow) -> dict[str, int]:
    # 1) Skill executed
    invocations = _count_skill_invocations(db, tenant_id, w, "gmail_summary")
    # 2) "Requested" approximated by text search in turn events
    q = (
        db.query(CallerMemoryEvent)
        .filter(CallerMemoryEvent.tenant_id == str(tenant_id))
        .filter(CallerMemoryEvent.created_at >= w.start_utc)
        .filter(CallerMemoryEvent.created_at < w.end_utc)
        .filter(CallerMemoryEvent.kind == "turn")
    )
    like_terms = []
    for s in _EMAIL_SUMMARY_REQUEST_HINTS:
        like_terms.append(CallerMemoryEvent.text.ilike(f"%{s}%"))
    if like_terms:
        q = q.filter(or_(*like_terms))
    request_turns = int(q.with_entities(func.count(CallerMemoryEvent.id)).scalar() or 0)
    request_call_sessions = int(
        q.with_entities(func.count(distinct(CallerMemoryEvent.call_sid)))
        .filter(CallerMemoryEvent.call_sid.isnot(None))
        .filter(CallerMemoryEvent.call_sid != "")
        .scalar()
        or 0
    )
    return {"skill_invocations": invocations, "request_turns": request_turns, "request_call_sessions": request_call_sessions}


# -----------------------------------------------------------------------------
# Non-memory tables metrics
# -----------------------------------------------------------------------------

def _websearch_skills_base(db: Session, tenant_uuid: UUID):
    return db.query(WebSearchSkill).filter(WebSearchSkill.tenant_id == tenant_uuid)


def _schedules_base(db: Session, tenant_uuid: UUID):
    return db.query(ScheduledDelivery).filter(ScheduledDelivery.tenant_id == tenant_uuid)


def _cache_base(db: Session, tenant_uuid: UUID):
    return db.query(CallerSkillCache).filter(CallerSkillCache.tenant_id == tenant_uuid)


def _kb_base(db: Session, tenant_uuid: UUID):
    return db.query(KBDocument).filter(KBDocument.tenant_id == tenant_uuid)


def _tasks_base(db: Session, tenant_uuid: UUID):
    return db.query(Task).filter(Task.user_id == tenant_uuid)


def _email_accounts_base(db: Session, tenant_uuid: UUID):
    return db.query(EmailAccount).filter(EmailAccount.user_id == tenant_uuid)


def _settings_base(db: Session, tenant_uuid: UUID):
    return db.query(UserSetting).filter(UserSetting.user_id == tenant_uuid)


# -----------------------------------------------------------------------------
# Main entry
# -----------------------------------------------------------------------------

def run_metrics_question(
    db: Session,
    *,
    tenant_id: str,
    question: str,
    timezone: str | None = None,
) -> dict[str, Any]:
    q_norm = _normalize_q(question)
    tz_name = _safe_tz(timezone)

    if not q_norm:
        return _fail("Please provide a metric question.")

    explicit_preset = _extract_timeframe_preset(q_norm)
    preset = explicit_preset or "today"
    w = _resolve_time_window(preset, tz_name)

    # -----------------------------
    # Comparisons (e.g. "compare X today versus yesterday")
    # -----------------------------
    if _COMPARE_RE.search(q_norm) and _VS_RE.search(q_norm):
        return _handle_compare(db, tenant_id=tenant_id, q_norm=q_norm, tz_name=tz_name)

    # -----------------------------
    # Calls / callers / memory
    # -----------------------------
    if "most recent callers" in q_norm or ("recent callers" in q_norm and "skills" not in q_norm):
        limit = _extract_limit(q_norm, default=25)
        callers = _recent_callers(db, tenant_id, w, tz_name, limit)
        spoken = f"Most recent callers {w.preset.replace('_',' ')} (showing {len(callers)} of {limit}):"
        for i, c in enumerate(callers[: min(limit, 10)], 1):
            spoken += f"\n{i}. {_mask_phone(c['caller_id'])} ({c['calls_in_range']} call session(s))"
        return _ok(
            "calls.recent_callers",
            w.preset,
            {"callers": callers, "limit": limit},
            spoken,
        )

    if "most recent call sessions" in q_norm or ("recent call sessions" in q_norm):
        limit = _extract_limit(q_norm, default=25)
        sessions = _recent_call_sessions(db, tenant_id, w, limit)
        spoken = f"Most recent call sessions {w.preset.replace('_',' ')} (showing {len(sessions)} of {limit}):"
        for i, s in enumerate(sessions[: min(limit, 10)], 1):
            spoken += f"\n{i}. {s['call_sid']} from {_mask_phone(s.get('caller_id') or '')}"
        return _ok("calls.recent_sessions", w.preset, {"sessions": sessions, "limit": limit}, spoken)

    if "top" in q_norm and "callers" in q_norm:
        limit = _extract_limit(q_norm, default=10)
        rows = _top_callers(db, tenant_id, w, limit)
        spoken = f"Top callers {w.preset.replace('_',' ')} (showing {len(rows)} of {limit}):"
        for i, r in enumerate(rows[: min(limit, 10)], 1):
            spoken += f"\n{i}. {_mask_phone(r['caller_id'])}: {r['call_sessions']} call session(s)"
        return _ok("calls.top_callers", w.preset, {"callers": rows, "limit": limit}, spoken)

    if "call sessions by day" in q_norm:
        rows = _timeseries_call_sessions(db, tenant_id, w, tz_name, bucket="day")
        spoken = f"Call sessions by day ({w.preset.replace('_',' ')}, {tz_name})"
        return _ok("calls.sessions_timeseries", w.preset, {"bucket": "day", "rows": rows}, spoken)

    if "call sessions by hour" in q_norm:
        rows = _timeseries_call_sessions(db, tenant_id, w, tz_name, bucket="hour")
        spoken = f"Call sessions by hour ({w.preset.replace('_',' ')}, {tz_name})"
        return _ok("calls.sessions_timeseries", w.preset, {"bucket": "hour", "rows": rows}, spoken)

    if "unique callers by day" in q_norm or "distinct callers by day" in q_norm:
        rows = _timeseries_unique_callers(db, tenant_id, w, tz_name, bucket="day")
        spoken = f"Unique callers by day ({w.preset.replace('_',' ')}, {tz_name})"
        return _ok("calls.unique_callers_timeseries", w.preset, {"bucket": "day", "rows": rows}, spoken)

    if "unique callers by hour" in q_norm or "distinct callers by hour" in q_norm:
        rows = _timeseries_unique_callers(db, tenant_id, w, tz_name, bucket="hour")
        spoken = f"Unique callers by hour ({w.preset.replace('_',' ')}, {tz_name})"
        return _ok("calls.unique_callers_timeseries", w.preset, {"bucket": "hour", "rows": rows}, spoken)

    if "memory events" in q_norm:
        n = _count_memory_events(db, tenant_id, w)
        return _ok("memory.events_count", w.preset, {"memory_events": n}, f"Memory events {w.preset.replace('_',' ')}: {n}.")

    # "distinct call sessions occurred" OR "calls received"
    if ("call sessions" in q_norm and ("how many" in q_norm or "count" in q_norm)) or ("calls received" in q_norm) or (q_norm.startswith("how many calls")):
        n = _count_call_sessions(db, tenant_id, w)
        return _ok("calls.count", w.preset, {"call_sessions": n}, f"Calls received {w.preset.replace('_',' ')}: {n}.")

    if ("distinct callers" in q_norm or "unique callers" in q_norm) and ("how many" in q_norm or "count" in q_norm):
        n = _count_unique_callers(db, tenant_id, w)
        return _ok("calls.unique_callers", w.preset, {"unique_callers": n}, f"Unique callers {w.preset.replace('_',' ')}: {n}.")

    # -----------------------------
    # Skills metrics
    # -----------------------------
    if any(h in q_norm for h in ("email summaries requested", "email summary requested", "someone request email", "emails requested")) or ("email summaries" in q_norm and "requested" in q_norm):
        stats = _email_summaries_requested(db, tenant_id, w)
        spoken = (
            f"Email summaries {w.preset.replace('_',' ')}:\n"
            f"- requested in {stats['request_call_sessions']} call session(s)\n"
            f"- {stats['request_turns']} matching turn(s)\n"
            f"- skill executed {stats['skill_invocations']} time(s)"
        )
        return _ok("skills.email_summaries_requested", w.preset, stats, spoken)

    if "invoked" in q_norm and "skill" in q_norm:
        sk = _skill_key_from_question(q_norm)
        if not sk:
            return _fail("Which skill key do you mean? Example: skill 'gmail_summary'.")
        inv = _count_skill_invocations(db, tenant_id, w, sk)
        return _ok("skills.invocations", w.preset, {"skill_key": sk, "invocations": inv}, f"Skill '{sk}' invocations {w.preset.replace('_',' ')}: {inv}.")

    if "distinct callers invoked" in q_norm or "distinct callers invoked skill" in q_norm:
        sk = _skill_key_from_question(q_norm)
        if not sk:
            return _fail("Which skill key do you mean? Example: skill 'gmail_summary'.")
        n = _count_skill_distinct_callers(db, tenant_id, w, sk)
        return _ok("skills.distinct_callers", w.preset, {"skill_key": sk, "distinct_callers": n}, f"Distinct callers invoking '{sk}' {w.preset.replace('_',' ')}: {n}.")

    if ("skill invocations by day" in q_norm) or (("invocations by day" in q_norm) and ("skill" in q_norm)):
        sk = _skill_key_from_question(q_norm)
        rows = _skills_timeseries(db, tenant_id, w, tz_name, bucket="day", skill_key=sk)
        label = f"Skill '{sk}' invocations by day" if sk else "Skill invocations by day"
        return _ok("skills.invocations_timeseries", w.preset, {"bucket": "day", "skill_key": sk, "rows": rows}, f"{label} ({w.preset.replace('_',' ')}, {tz_name}).")

    if ("skill invocations by hour" in q_norm) or (("invocations by hour" in q_norm) and ("skill" in q_norm)):
        sk = _skill_key_from_question(q_norm)
        rows = _skills_timeseries(db, tenant_id, w, tz_name, bucket="hour", skill_key=sk)
        label = f"Skill '{sk}' invocations by hour" if sk else "Skill invocations by hour"
        return _ok("skills.invocations_timeseries", w.preset, {"bucket": "hour", "skill_key": sk, "rows": rows}, f"{label} ({w.preset.replace('_',' ')}, {tz_name}).")

    if "most recent skill invocations" in q_norm or ("recent skill invocations" in q_norm):
        limit = _extract_limit(q_norm, default=50)
        rows = _skills_recent_invocations(db, tenant_id, w, limit)
        spoken = f"Most recent skill invocations {w.preset.replace('_',' ')} (showing {min(len(rows),10)} of {limit})."
        return _ok("skills.recent_invocations", w.preset, {"invocations": rows, "limit": limit}, spoken)

    if "top" in q_norm and (("skills invoked" in q_norm) or ("skills by invocation" in q_norm)):
        limit = _extract_limit(q_norm, default=20)
        by_distinct = "distinct callers" in q_norm
        rows = _skills_top(db, tenant_id, w, limit, by_distinct_callers=by_distinct)
        spoken = f"Top skills {w.preset.replace('_',' ')} (showing {min(len(rows),10)} of {limit})."
        return _ok("skills.top_invoked", w.preset, {"top": rows, "limit": limit, "by_distinct_callers": by_distinct}, spoken)

    # "How many times was skill 'X' invoked ..." (without the word "skill" sometimes)
    if "how many times was" in q_norm and "invoked" in q_norm:
        sk = _skill_key_from_question(q_norm)
        if sk:
            inv = _count_skill_invocations(db, tenant_id, w, sk)
            return _ok("skills.invocations", w.preset, {"skill_key": sk, "invocations": inv}, f"Skill '{sk}' invocations {w.preset.replace('_',' ')}: {inv}.")

    # -----------------------------
    # Websearch skills
    # -----------------------------
    if "websearch skills" in q_norm or "web search skills" in q_norm:
        tenant_uuid = _tenant_uuid(tenant_id)
        if not tenant_uuid:
            return _fail("Tenant scope not available for websearch skills.")
        base = _websearch_skills_base(db, tenant_uuid)

        if "top" in q_norm and "creation time" in q_norm:
            limit = _extract_limit(q_norm, default=5)
            rows = (
                base.filter(WebSearchSkill.created_at >= w.start_utc)
                .filter(WebSearchSkill.created_at < w.end_utc)
                .order_by(WebSearchSkill.created_at.desc())
                .limit(limit)
                .all()
            )
            data = [{"id": str(s.id), "name": s.name, "created_at": (s.created_at.isoformat() if s.created_at else None)} for s in rows]
            return _ok("websearch.skills_list", w.preset, {"skills": data, "limit": limit}, f"Top websearch skills by creation time {w.preset.replace('_',' ')}: {len(data)}.")

        if "currently enabled" in q_norm:
            n = int(base.filter(WebSearchSkill.enabled.is_(True)).with_entities(func.count(WebSearchSkill.id)).scalar() or 0)
            return _ok("websearch.skills_enabled_count", "all", {"enabled_websearch_skills": n}, f"Enabled websearch skills: {n}.")
        if "currently disabled" in q_norm:
            n = int(base.filter(WebSearchSkill.enabled.is_(False)).with_entities(func.count(WebSearchSkill.id)).scalar() or 0)
            return _ok("websearch.skills_disabled_count", "all", {"disabled_websearch_skills": n}, f"Disabled websearch skills: {n}.")
        if "list disabled" in q_norm:
            rows = base.filter(WebSearchSkill.enabled.is_(False)).order_by(WebSearchSkill.created_at.desc()).limit(200).all()
            data = [{"id": str(s.id), "name": s.name, "created_at": s.created_at.isoformat() if s.created_at else None} for s in rows]
            return _ok("websearch.skills_disabled_list", "all", {"skills": data}, f"Disabled websearch skills: {len(data)}.")
        if "list enabled" in q_norm:
            rows = base.filter(WebSearchSkill.enabled.is_(True)).order_by(WebSearchSkill.created_at.desc()).limit(200).all()
            data = [{"id": str(s.id), "name": s.name, "created_at": s.created_at.isoformat() if s.created_at else None} for s in rows]
            return _ok("websearch.skills_enabled_list", "all", {"skills": data}, f"Enabled websearch skills: {len(data)}.")

        if "were created" in q_norm or "created" in q_norm:
            n = int(
                base.filter(WebSearchSkill.created_at >= w.start_utc)
                .filter(WebSearchSkill.created_at < w.end_utc)
                .with_entities(func.count(WebSearchSkill.id))
                .scalar()
                or 0
            )
            if "how many" in q_norm:
                return _ok("websearch.skills_count", w.preset, {"created_websearch_skills": n}, f"Websearch skills created {w.preset.replace('_',' ')}: {n}.")
            # list created
            rows = (
                base.filter(WebSearchSkill.created_at >= w.start_utc)
                .filter(WebSearchSkill.created_at < w.end_utc)
                .order_by(WebSearchSkill.created_at.desc())
                .limit(200)
                .all()
            )
            data = [{"id": str(s.id), "name": s.name, "created_at": s.created_at.isoformat() if s.created_at else None} for s in rows]
            return _ok("websearch.skills_list", w.preset, {"skills": data}, f"Websearch skills created {w.preset.replace('_',' ')}: {len(data)}.")

    # -----------------------------
    # Schedules
    # -----------------------------
    if "schedules" in q_norm:
        tenant_uuid = _tenant_uuid(tenant_id)
        if not tenant_uuid:
            return _fail("Tenant scope not available for schedules.")
        base = _schedules_base(db, tenant_uuid)

        if "how many schedules were created" in q_norm:
            n = int(
                base.filter(ScheduledDelivery.created_at >= w.start_utc)
                .filter(ScheduledDelivery.created_at < w.end_utc)
                .with_entities(func.count(ScheduledDelivery.id))
                .scalar()
                or 0
            )
            return _ok("schedules.count_created", w.preset, {"created_schedules": n}, f"Schedules created {w.preset.replace('_',' ')}: {n}.")

        if "currently enabled" in q_norm:
            n = int(base.filter(ScheduledDelivery.enabled.is_(True)).with_entities(func.count(ScheduledDelivery.id)).scalar() or 0)
            return _ok("schedules.enabled_count", "all", {"enabled_schedules": n}, f"Enabled schedules: {n}.")
        if "currently disabled" in q_norm:
            n = int(base.filter(ScheduledDelivery.enabled.is_(False)).with_entities(func.count(ScheduledDelivery.id)).scalar() or 0)
            return _ok("schedules.disabled_count", "all", {"disabled_schedules": n}, f"Disabled schedules: {n}.")

        if "most recently updated schedules" in q_norm or "recently updated schedules" in q_norm:
            limit = _extract_limit(q_norm, default=50)
            rows = (
                base.filter(ScheduledDelivery.updated_at >= w.start_utc)
                .filter(ScheduledDelivery.updated_at < w.end_utc)
                .order_by(ScheduledDelivery.updated_at.desc())
                .limit(limit)
                .all()
            )
            data = [
                {
                    "id": str(s.id),
                    "enabled": bool(s.enabled),
                    "channel": (getattr(s.channel, "value", None) or str(s.channel)),
                    "timezone": s.timezone,
                    "destination": s.destination,
                    "next_run_at": (s.next_run_at.isoformat() if s.next_run_at else None),
                    "last_run_at": (s.last_run_at.isoformat() if s.last_run_at else None),
                    "updated_at": (s.updated_at.isoformat() if s.updated_at else None),
                }
                for s in rows
            ]
            return _ok("schedules.recent_updates", w.preset, {"schedules": data, "limit": limit}, f"Most recently updated schedules {w.preset.replace('_',' ')}: {len(data)}.")

        if "top" in q_norm and ("next run time" in q_norm or "next run" in q_norm):
            # This is an "upcoming schedules" view (primarily useful operationally).
            # Timeframe words in the question are treated as labeling only (next_run_at is typically in the future).
            limit = _extract_limit(q_norm, default=10)
            rows = (
                base.filter(ScheduledDelivery.next_run_at.isnot(None))
                .order_by(ScheduledDelivery.next_run_at.asc())
                .limit(limit)
                .all()
            )
            data = [
                {
                    "id": str(s.id),
                    "channel": (getattr(s.channel, "value", None) or str(s.channel)),
                    "timezone": s.timezone,
                    "destination": s.destination,
                    "next_run_at": (s.next_run_at.isoformat() if s.next_run_at else None),
                    "web_search_skill_id": str(s.web_search_skill_id),
                }
                for s in rows
            ]
            return _ok("schedules.due_next", w.preset, {"schedules": data, "limit": limit}, f"Top schedules by next run time (showing {len(data)} of {limit}).")

        if "due to run next" in q_norm or "run next" in q_norm or "next run" in q_norm:
            limit = _extract_limit(q_norm, default=50)
            rows = (
                base.filter(ScheduledDelivery.next_run_at.isnot(None))
                .filter(ScheduledDelivery.next_run_at >= w.start_utc)
                .filter(ScheduledDelivery.next_run_at < w.end_utc)
                .order_by(ScheduledDelivery.next_run_at.asc())
                .limit(limit)
                .all()
            )
            data = [
                {
                    "id": str(s.id),
                    "channel": (getattr(s.channel, "value", None) or str(s.channel)),
                    "timezone": s.timezone,
                    "destination": s.destination,
                    "next_run_at": (s.next_run_at.isoformat() if s.next_run_at else None),
                    "web_search_skill_id": str(s.web_search_skill_id),
                }
                for s in rows
            ]
            return _ok("schedules.due_next", w.preset, {"schedules": data, "limit": limit}, f"Schedules due next {w.preset.replace('_',' ')}: {len(data)}.")

        if "never run" in q_norm:
            limit = _extract_limit(q_norm, default=200)
            rows = (
                base.filter(ScheduledDelivery.last_run_at.is_(None))
                .order_by(ScheduledDelivery.created_at.desc())
                .limit(limit)
                .all()
            )
            data = [{"id": str(s.id), "destination": s.destination, "channel": (getattr(s.channel, "value", None) or str(s.channel)), "created_at": (s.created_at.isoformat() if s.created_at else None)} for s in rows]
            return _ok("schedules.never_run", "all", {"schedules": data, "limit": limit}, f"Schedules that have never run: {len(data)}.")

        if "by delivery channel" in q_norm:
            rows = (
            (base.filter(ScheduledDelivery.created_at >= w.start_utc).filter(ScheduledDelivery.created_at < w.end_utc) if explicit_preset else base)
            .with_entities(ScheduledDelivery.channel, func.count(ScheduledDelivery.id))
                .group_by(ScheduledDelivery.channel)
                .order_by(desc(func.count(ScheduledDelivery.id)))
                .all()
            )
            data = [{"channel": (getattr(ch, "value", None) or str(ch)), "count": int(n or 0)} for ch, n in rows]
            return _ok("schedules.by_channel", "all", {"rows": data}, "Schedules by delivery channel.")

        if "by timezone" in q_norm:
            rows = (
            (base.filter(ScheduledDelivery.created_at >= w.start_utc).filter(ScheduledDelivery.created_at < w.end_utc) if explicit_preset else base)
            .with_entities(ScheduledDelivery.timezone, func.count(ScheduledDelivery.id))
                .group_by(ScheduledDelivery.timezone)
                .order_by(desc(func.count(ScheduledDelivery.id)))
                .all()
            )
            data = [{"timezone": tz, "count": int(n or 0)} for tz, n in rows]
            return _ok("schedules.by_timezone", "all", {"rows": data}, "Schedules by timezone.")

        if "by destination domain" in q_norm:
            # Only meaningful for email schedules.
            dom = func.split_part(ScheduledDelivery.destination, "@", 2).label("domain")
            rows = (
                (base.filter(ScheduledDelivery.created_at >= w.start_utc).filter(ScheduledDelivery.created_at < w.end_utc) if explicit_preset else base)
                .filter(ScheduledDelivery.channel == DeliveryChannel.email)
                .with_entities(dom, func.count(ScheduledDelivery.id))
                .group_by(dom)
                .order_by(desc(func.count(ScheduledDelivery.id)))
                .all()
            )
            data = [{"domain": (d or ""), "count": int(n or 0)} for d, n in rows]
            return _ok("schedules.by_destination_domain", "all", {"rows": data}, "Email schedules by destination domain.")

    # -----------------------------
    # Cache metrics
    # -----------------------------
    if "cached" in q_norm and (("skill results" in q_norm) or ("cached results" in q_norm)):
        tenant_uuid = _tenant_uuid(tenant_id)
        if not tenant_uuid:
            return _fail("Tenant scope not available for cache metrics.")
        base = _cache_base(db, tenant_uuid)

        if "how many cached skill results were written" in q_norm:
            n = int(
                base.filter(CallerSkillCache.created_at >= w.start_utc)
                .filter(CallerSkillCache.created_at < w.end_utc)
                .with_entities(func.count(CallerSkillCache.id))
                .scalar()
                or 0
            )
            return _ok("cache.writes_count", w.preset, {"cache_writes": n}, f"Cached skill results written {w.preset.replace('_',' ')}: {n}.")

        if "list the" in q_norm and "most recent cached skill results" in q_norm:
            limit = _extract_limit(q_norm, default=50)
            rows = (
                base.filter(CallerSkillCache.created_at >= w.start_utc)
                .filter(CallerSkillCache.created_at < w.end_utc)
                .order_by(CallerSkillCache.created_at.desc())
                .limit(limit)
                .all()
            )
            data = [
                {
                    "created_at": (r.created_at.isoformat() if r.created_at else None),
                    "caller_id": r.caller_id,
                    "skill_key": r.skill_key,
                    "expires_at": (r.expires_at.isoformat() if r.expires_at else None),
                }
                for r in rows
            ]
            return _ok("cache.recent_writes", w.preset, {"rows": data, "limit": limit}, f"Recent cache writes {w.preset.replace('_',' ')}: {len(data)}.")

        if "cached results by skill key" in q_norm:
            rows = (
                base.filter(CallerSkillCache.created_at >= w.start_utc)
                .filter(CallerSkillCache.created_at < w.end_utc)
                .with_entities(CallerSkillCache.skill_key, func.count(CallerSkillCache.id))
                .group_by(CallerSkillCache.skill_key)
                .order_by(desc(func.count(CallerSkillCache.id)))
                .all()
            )
            data = [{"skill_key": sk, "count": int(n or 0)} for sk, n in rows]
            return _ok("cache.by_skill_key", w.preset, {"rows": data}, "Cache writes by skill key.")

        if "cached results by caller" in q_norm:
            rows = (
                base.filter(CallerSkillCache.created_at >= w.start_utc)
                .filter(CallerSkillCache.created_at < w.end_utc)
                .with_entities(CallerSkillCache.caller_id, func.count(CallerSkillCache.id))
                .group_by(CallerSkillCache.caller_id)
                .order_by(desc(func.count(CallerSkillCache.id)))
                .limit(50)
                .all()
            )
            data = [{"caller_id": caller, "count": int(n or 0)} for caller, n in rows]
            return _ok("cache.by_caller", w.preset, {"rows": data}, "Cache writes by caller.")

    # -----------------------------
    # KB document metrics
    # -----------------------------
    if "kb documents" in q_norm:
        tenant_uuid = _tenant_uuid(tenant_id)
        if not tenant_uuid:
            return _fail("Tenant scope not available for KB metrics.")
        base = _kb_base(db, tenant_uuid)

        if "were uploaded" in q_norm or "uploaded" in q_norm:
            n = int(
                base.filter(KBDocument.created_at >= w.start_utc)
                .filter(KBDocument.created_at < w.end_utc)
                .with_entities(func.count(KBDocument.id))
                .scalar()
                or 0
            )
            if "how many" in q_norm:
                return _ok("kb.count_created", w.preset, {"kb_documents": n}, f"KB documents created {w.preset.replace('_',' ')}: {n}.")
            rows = (
                base.filter(KBDocument.created_at >= w.start_utc)
                .filter(KBDocument.created_at < w.end_utc)
                .order_by(KBDocument.created_at.desc())
                .limit(200)
                .all()
            )
            data = [{"id": str(d.id), "status": str(d.status), "created_at": d.created_at.isoformat() if d.created_at else None, "filename": d.filename} for d in rows]
            return _ok("kb.count_created", w.preset, {"documents": data}, f"KB documents created {w.preset.replace('_',' ')}: {len(data)}.")

        if "status" in q_norm and ("failed" in q_norm or "active" in q_norm):
            want_list = q_norm.startswith("list") or " list " in f" {q_norm} "
            if "failed" in q_norm:
                q = base.filter(KBDocument.status == KBDocumentStatus.failed)
                if want_list:
                    rows = q.order_by(KBDocument.created_at.desc()).limit(200).all()
                    data = [{"id": str(d.id), "status": str(d.status), "created_at": (d.created_at.isoformat() if d.created_at else None), "filename": d.filename} for d in rows]
                    return _ok("kb.list_by_status", "all", {"status": "failed", "documents": data, "limit": 200}, f"KB documents failed: {len(data)} (showing up to 200).")
                n = int(q.with_entities(func.count(KBDocument.id)).scalar() or 0)
                return _ok("kb.by_status", "all", {"status": "failed", "count": n}, f"KB documents failed: {n}.")
            # define "active" as not failed
            q = base.filter(KBDocument.status != KBDocumentStatus.failed)
            if want_list:
                rows = q.order_by(KBDocument.created_at.desc()).limit(200).all()
                data = [{"id": str(d.id), "status": str(d.status), "created_at": (d.created_at.isoformat() if d.created_at else None), "filename": d.filename} for d in rows]
                return _ok("kb.list_by_status", "all", {"status": "active", "documents": data, "limit": 200}, f"KB documents active: {len(data)} (showing up to 200).")
            n = int(q.with_entities(func.count(KBDocument.id)).scalar() or 0)
            return _ok("kb.by_status", "all", {"status": "active", "count": n}, f"KB documents active: {n}.")

    # -----------------------------
    # Task metrics
    # -----------------------------
    if "tasks" in q_norm:
        tenant_uuid = _tenant_uuid(tenant_id)
        if not tenant_uuid:
            return _fail("Tenant scope not available for task metrics.")
        base = _tasks_base(db, tenant_uuid)

        if "how many tasks were created" in q_norm:
            n = int(
                base.filter(Task.created_at >= w.start_utc)
                .filter(Task.created_at < w.end_utc)
                .with_entities(func.count(Task.id))
                .scalar()
                or 0
            )
            return _ok("tasks.count_created", w.preset, {"tasks_created": n}, f"Tasks created {w.preset.replace('_',' ')}: {n}.")

        if "how many tasks were completed" in q_norm:
            n = int(
                base.filter(Task.status == TaskStatus.COMPLETED)
                .filter(Task.updated_at >= w.start_utc)
                .filter(Task.updated_at < w.end_utc)
                .with_entities(func.count(Task.id))
                .scalar()
                or 0
            )
            return _ok("tasks.count_completed", w.preset, {"tasks_completed": n}, f"Tasks completed {w.preset.replace('_',' ')}: {n}.")

        if "currently pending" in q_norm or "are currently pending" in q_norm:
            pending_statuses = (
                TaskStatus.PENDING,
                TaskStatus.COLLECTING_INPUT,
                TaskStatus.READY,
                TaskStatus.EXECUTING,
                TaskStatus.WAITING,
            )
            n = int(base.filter(Task.status.in_(pending_statuses)).with_entities(func.count(Task.id)).scalar() or 0)
            return _ok("tasks.pending_count", "all", {"pending_tasks": n}, f"Pending tasks: {n}.")

        if "list pending tasks" in q_norm:
            pending_statuses = (
                TaskStatus.PENDING,
                TaskStatus.COLLECTING_INPUT,
                TaskStatus.READY,
                TaskStatus.EXECUTING,
                TaskStatus.WAITING,
            )
            rows = base.filter(Task.status.in_(pending_statuses)).order_by(Task.created_at.desc()).limit(200).all()
            data = [{"id": str(t.id), "status": str(t.status), "created_at": (t.created_at.isoformat() if t.created_at else None), "type": str(t.type)} for t in rows]
            return _ok("tasks.list_pending", "all", {"tasks": data}, f"Pending tasks: {len(data)}.")

        if "list tasks created" in q_norm:
            rows = base.filter(Task.created_at >= w.start_utc).filter(Task.created_at < w.end_utc).order_by(Task.created_at.desc()).limit(200).all()
            data = [{"id": str(t.id), "status": str(t.status), "created_at": (t.created_at.isoformat() if t.created_at else None), "type": str(t.type)} for t in rows]
            return _ok("tasks.list_created", w.preset, {"tasks": data}, f"Tasks created {w.preset.replace('_',' ')}: {len(data)}.")

        if "list tasks completed" in q_norm:
            rows = base.filter(Task.status == TaskStatus.COMPLETED).filter(Task.updated_at >= w.start_utc).filter(Task.updated_at < w.end_utc).order_by(Task.updated_at.desc()).limit(200).all()
            data = [{"id": str(t.id), "updated_at": (t.updated_at.isoformat() if t.updated_at else None), "type": str(t.type)} for t in rows]
            return _ok("tasks.list_completed", w.preset, {"tasks": data}, f"Tasks completed {w.preset.replace('_',' ')}: {len(data)}.")

    # -----------------------------
    # Email account metrics
    # -----------------------------
    if "email account" in q_norm or "email accounts" in q_norm:
        tenant_uuid = _tenant_uuid(tenant_id)
        if not tenant_uuid:
            return _fail("Tenant scope not available for email account metrics.")
        base = _email_accounts_base(db, tenant_uuid)

        if "how many email accounts are configured" in q_norm:
            n = int(base.with_entities(func.count(EmailAccount.id)).scalar() or 0)
            return _ok("email.accounts_count", "all", {"email_accounts": n}, f"Configured email accounts: {n}.")

        if "list configured email accounts" in q_norm:
            rows = base.order_by(EmailAccount.created_at.desc()).limit(200).all()
            data = [{"id": str(a.id), "email_address": a.email_address, "is_active": bool(a.is_active), "is_primary": bool(a.is_primary)} for a in rows]
            return _ok("email.accounts_list", "all", {"accounts": data}, f"Configured email accounts: {len(data)}.")

        if "how many email accounts are active" in q_norm:
            n = int(base.filter(EmailAccount.is_active.is_(True)).with_entities(func.count(EmailAccount.id)).scalar() or 0)
            return _ok("email.accounts_active_count", "all", {"active_email_accounts": n}, f"Active email accounts: {n}.")

        if "list active email accounts" in q_norm:
            rows = base.filter(EmailAccount.is_active.is_(True)).order_by(EmailAccount.created_at.desc()).limit(200).all()
            data = [{"id": str(a.id), "email_address": a.email_address, "is_primary": bool(a.is_primary)} for a in rows]
            return _ok("email.accounts_active_list", "all", {"accounts": data}, f"Active email accounts: {len(data)}.")

        if "default sender" in q_norm or "default" in q_norm:
            acct = base.filter(EmailAccount.is_primary.is_(True)).first()
            if not acct:
                return _fail("No primary email sender is configured.")
            return _ok("email.default_sender", "all", {"email_address": acct.email_address, "id": str(acct.id)}, f"Default sender: {acct.email_address}.")

    # -----------------------------
    # Settings metrics
    # -----------------------------
    if "settings" in q_norm and "key" in q_norm:
        tenant_uuid = _tenant_uuid(tenant_id)
        if not tenant_uuid:
            return _fail("Tenant scope not available for settings metrics.")
        base = _settings_base(db, tenant_uuid)

        if "most recently updated" in q_norm:
            limit = _extract_limit(q_norm, default=50)
            rows = (
                base.filter(UserSetting.updated_at >= w.start_utc)
                .filter(UserSetting.updated_at < w.end_utc)
                .order_by(UserSetting.updated_at.desc())
                .limit(limit)
                .all()
            )
            data = [{"key": r.key, "updated_at": r.updated_at.isoformat() if r.updated_at else None} for r in rows]
            return _ok("settings.recent_keys", w.preset, {"keys": data, "limit": limit}, f"Recently updated settings keys {w.preset.replace('_',' ')}: {len(data)}.")

        if "how many settings keys changed" in q_norm:
            n = int(
                base.filter(UserSetting.updated_at >= w.start_utc)
                .filter(UserSetting.updated_at < w.end_utc)
                .with_entities(func.count(distinct(UserSetting.key)))
                .scalar()
                or 0
            )
            return _ok("settings.changed_keys_count", w.preset, {"changed_keys": n}, f"Settings keys changed {w.preset.replace('_',' ')}: {n}.")

    # Fallback: not recognized
    return _fail("I can’t compute that metric yet from the current database.")


# -----------------------------------------------------------------------------
# Compare handler
# -----------------------------------------------------------------------------

def _parse_compare_timeframes(q_norm: str) -> tuple[str | None, str | None]:
    # Look for explicit presets in the string. The first match is A, second is B.
    found: list[str] = []
    for _, preset in _TIMEFRAME_PATTERNS:
        if preset in ("today", "yesterday", "this_week", "last_week", "this_month", "last_30_days", "last_90_days"):
            # check by keywords, not preset tokens
            pass
    # we re-scan by patterns in order of appearance
    # We'll extract by searching the string sequentially for known tokens.
    tokens = [
        ("today", r"\btoday\b"),
        ("yesterday", r"\byesterday\b"),
        ("this_week", r"\bthis\s+week\b"),
        ("last_week", r"\blast\s+week\b"),
        ("this_month", r"\bthis\s+month\b"),
        ("last_30_days", r"\blast\s+30\s+days\b"),
        ("last_90_days", r"\blast\s+90\s+days\b"),
    ]
    tmp = q_norm
    idx = 0
    while idx < len(tmp):
        next_hit = None
        next_pos = None
        for preset, pat in tokens:
            m = re.search(pat, tmp[idx:], re.I)
            if m:
                pos = idx + m.start()
                if next_pos is None or pos < next_pos:
                    next_pos = pos
                    next_hit = preset
        if next_hit is None:
            break
        found.append(next_hit)
        idx = (next_pos or 0) + 1
        if len(found) >= 2:
            break
    a = found[0] if len(found) >= 1 else None
    b = found[1] if len(found) >= 2 else None
    return a, b


def _handle_compare(db: Session, *, tenant_id: str, q_norm: str, tz_name: str) -> dict[str, Any]:
    a_preset, b_preset = _parse_compare_timeframes(q_norm)
    if not a_preset or not b_preset:
        return _fail("Please specify two timeframes to compare (e.g., today versus yesterday).")

    w_a = _resolve_time_window(a_preset, tz_name)
    w_b = _resolve_time_window(b_preset, tz_name)

    def _pct(delta: int, base: int) -> float | None:
        if base == 0:
            return None
        return round((delta / base) * 100.0, 2)

    # Determine metric type
    if "call sessions" in q_norm or "calls" in q_norm:
        a = _count_call_sessions(db, tenant_id, w_a)
        b = _count_call_sessions(db, tenant_id, w_b)
        delta = a - b
        return _ok(
            "compare.calls",
            f"{a_preset}_vs_{b_preset}",
            {"a": {"timeframe": a_preset, "value": a}, "b": {"timeframe": b_preset, "value": b}, "delta": delta, "pct_change": _pct(delta, b)},
            f"Call sessions: {a_preset.replace('_',' ')}={a}, {b_preset.replace('_',' ')}={b}, delta={delta}.",
        )

    if "unique callers" in q_norm or "distinct callers" in q_norm:
        a = _count_unique_callers(db, tenant_id, w_a)
        b = _count_unique_callers(db, tenant_id, w_b)
        delta = a - b
        return _ok(
            "compare.unique_callers",
            f"{a_preset}_vs_{b_preset}",
            {"a": {"timeframe": a_preset, "value": a}, "b": {"timeframe": b_preset, "value": b}, "delta": delta, "pct_change": _pct(delta, b)},
            f"Unique callers: {a_preset.replace('_',' ')}={a}, {b_preset.replace('_',' ')}={b}, delta={delta}.",
        )

    if "skill invocations" in q_norm or "invocations" in q_norm:
        sk = _skill_key_from_question(q_norm)
        a = _count_skill_invocations(db, tenant_id, w_a, sk)
        b = _count_skill_invocations(db, tenant_id, w_b, sk)
        delta = a - b
        label = f"Skill '{sk}' invocations" if sk else "Total skill invocations"
        return _ok(
            "compare.skill_invocations",
            f"{a_preset}_vs_{b_preset}",
            {"skill_key": sk, "a": {"timeframe": a_preset, "value": a}, "b": {"timeframe": b_preset, "value": b}, "delta": delta, "pct_change": _pct(delta, b)},
            f"{label}: {a_preset.replace('_',' ')}={a}, {b_preset.replace('_',' ')}={b}, delta={delta}.",
        )

    return _fail("I can compare calls, unique callers, or skill invocations. Try: 'Compare call sessions today versus yesterday'.")

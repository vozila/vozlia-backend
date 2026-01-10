# services/memory_controller.py
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone, date
from zoneinfo import ZoneInfo
from typing import Any, Iterable, Optional

from core.logging import logger


def _truthy_env(name: str, default: str = "0") -> bool:
    """Parse env var as a boolean.

    Accepts 1/true/yes/on (case-insensitive) as True.
    """
    v = (os.getenv(name, default) or default).strip().lower()
    return v in ("1", "true", "yes", "on")
from models import CallerMemoryEvent
from sqlalchemy import text


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



def _resolve_time_window_v2(text: str, *, now_utc: datetime | None = None) -> tuple[datetime, datetime] | None:
    """Deterministic calendar-based time window parsing in APP_TZ.

    Returns (start_utc_naive, end_utc_naive) suitable for comparing against DB 'created_at'
    which is stored as naive UTC timestamps in this app.
    """
    raw = (text or "").strip().lower()
    if not raw:
        return None

    tz = ZoneInfo(os.getenv("APP_TZ", "America/New_York"))
    now_utc = now_utc or _utcnow()
    now_local = now_utc.replace(tzinfo=timezone.utc).astimezone(tz)

    start_of_today = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
    start_of_week = start_of_today - timedelta(days=start_of_today.weekday())  # Monday
    start_of_month = start_of_today.replace(day=1)

    def _day_window(day_local: datetime) -> tuple[datetime, datetime]:
        s_local = day_local.replace(hour=0, minute=0, second=0, microsecond=0)
        e_local = s_local + timedelta(days=1)
        return (s_local.astimezone(timezone.utc).replace(tzinfo=None), e_local.astimezone(timezone.utc).replace(tzinfo=None))

    # Word-number mapping (covers common voice utterances like "three days ago")
    word_nums = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    }

    # today
    if "today" in raw:
        return (start_of_today.astimezone(timezone.utc).replace(tzinfo=None), now_utc.replace(tzinfo=None))

    # yesterday / a day ago / day ago
    if "yesterday" in raw or "a day ago" in raw or re.search(r"\bday\s+ago\b", raw):
        return _day_window(start_of_today - timedelta(days=1))

    # couple/few days ago => 48h window ending at today's midnight (per your spec)
    if "couple days ago" in raw or "a couple days ago" in raw or "few days ago" in raw or "a few days ago" in raw:
        s_local = start_of_today - timedelta(days=2)
        e_local = start_of_today
        return (s_local.astimezone(timezone.utc).replace(tzinfo=None), e_local.astimezone(timezone.utc).replace(tzinfo=None))

    # a week ago / week ago => calendar day 7 days ago
    if "a week ago" in raw or re.search(r"\bweek\s+ago\b", raw):
        return _day_window(start_of_today - timedelta(days=7))

    # numeric N days ago
    m = re.search(r"\b(\d+)\s+days?\s+ago\b", raw)
    if m:
        n = int(m.group(1))
        if 1 <= n <= 60:
            return _day_window(start_of_today - timedelta(days=n))

    # word N days ago
    m = re.search(r"\b(one|two|three|four|five|six|seven|eight|nine|ten)\s+days?\s+ago\b", raw)
    if m:
        n = word_nums.get(m.group(1))
        if n:
            return _day_window(start_of_today - timedelta(days=n))

    # numeric N weeks ago (treated as day window at 7N days ago)
    m = re.search(r"\b(\d+)\s+weeks?\s+ago\b", raw)
    if m:
        n = int(m.group(1))
        if 1 <= n <= 26:
            return _day_window(start_of_today - timedelta(days=7 * n))

    # word N weeks ago
    m = re.search(r"\b(one|two|three|four|five|six|seven|eight|nine|ten)\s+weeks?\s+ago\b", raw)
    if m:
        n = word_nums.get(m.group(1))
        if n:
            return _day_window(start_of_today - timedelta(days=7 * n))

    # this week
    if "this week" in raw:
        exclude_today = _truthy_env("TIMEFRAME_THIS_WEEK_EXCLUDE_TODAY", "1")
        end_local = start_of_today if exclude_today else now_local
        return (start_of_week.astimezone(timezone.utc).replace(tzinfo=None), end_local.astimezone(timezone.utc).replace(tzinfo=None))

    # last week (previous Mon->Mon)
    if "last week" in raw:
        s_local = start_of_week - timedelta(days=7)
        e_local = start_of_week
        return (s_local.astimezone(timezone.utc).replace(tzinfo=None), e_local.astimezone(timezone.utc).replace(tzinfo=None))

    # this month
    if "this month" in raw:
        return (start_of_month.astimezone(timezone.utc).replace(tzinfo=None), now_utc.replace(tzinfo=None))

    # last month (calendar month)
    if "last month" in raw:
        first_this_month = start_of_month
        last_month_end = first_this_month
        # go back one day then set to first of that month
        last_month_start = (first_this_month - timedelta(days=1)).replace(day=1)
        return (last_month_start.astimezone(timezone.utc).replace(tzinfo=None), last_month_end.astimezone(timezone.utc).replace(tzinfo=None))

    # weekday names => most recent previous weekday
    weekday_map = {
        "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
        "friday": 4, "saturday": 5, "sunday": 6,
    }
    for name, idx in weekday_map.items():
        if name in raw:
            days_ago = (now_local.weekday() - idx) % 7
            if days_ago == 0:
                days_ago = 7
            return _day_window(start_of_today - timedelta(days=days_ago))

    return None

def _utcnow() -> datetime:
    return datetime.now(timezone.utc)

def parse_memory_query(text: str) -> MemoryQuery:
    raw = (text or "").strip()
    now = _utcnow()
    start = now - timedelta(days=30)
    end = now

    # Minutes / hours parsing (MVP, rolling UTC)
    m = re.search(r"\b(last)\s+(\d+)\s+minutes\b", raw, re.I)
    if m:
        n = int(m.group(2))
        start = now - timedelta(minutes=min(max(n, 1), 60 * 24 * 7))
    else:
        m = re.search(r"\b(\d+)\s+minutes?\s+ago\b", raw, re.I)
        if m:
            n = int(m.group(1))
            start = now - timedelta(minutes=min(max(n * 3, 10), 60 * 24 * 7))
        else:
            m = re.search(r"\b(last)\s+(\d+)\s+hours\b", raw, re.I)
            if m:
                n = int(m.group(2))
                start = now - timedelta(hours=min(max(n, 1), 24 * 30))
            else:
                m = re.search(r"\b(\d+)\s+hours?\s+ago\b", raw, re.I)
                if m:
                    n = int(m.group(1))
                    start = now - timedelta(hours=min(max(n * 2, 2), 24 * 30))

    # Time parsing (legacy rolling UTC)
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

    # Deterministic calendar parsing (Phase 2) — feature-flagged
    # Default ON so time questions stop falling back to the 30-day window.
    if _truthy_env("TIMEFRAME_PARSER_V2", "1"):
        win = _resolve_time_window_v2(raw, now_utc=now)
        if win:
            start, end = win
            if _truthy_env("MEMORY_TIME_WINDOW_TRACE", "0"):
                try:
                    logger.info(
                        "MEMORY_TIME_WINDOW_V2 raw=%r start=%s end=%s",
                        raw[:200],
                        start.isoformat(timespec="seconds"),
                        end.isoformat(timespec="seconds"),
                    )
                except Exception:
                    pass

    # Topic inference (MVP)
    skill = None
    low = raw.lower()
    if any(w in low for w in ["weather", "forecast", "temperature", "rain", "snow"]):
        skill = "weather"
    elif any(w in low for w in ["email", "inbox", "gmail"]):
        skill = "gmail_summary"

    # Keyword extraction (cheap)
    tokens = re.findall(r"[a-zA-Z0-9_]{3,}", low)
    stop = {
        "what",
        "did",
        "say",
        "about",
        "that",
        "this",
        "last",
        "time",
        "when",
        "where",
        "who",
        "how",
        "was",
        "were",
        "you",
        "i",
        "me",
        "we",
        "talked",
        "previous",
        "call",
        "report",
        "week",
        "month",
        "today",
        "yesterday",
        "ago",
        "days",
        "hours",
        "minutes",
        "few",
        "couple",
        "remind",
    }
    kws = [t for t in tokens if t not in stop][:12]

    return MemoryQuery(start_ts=start, end_ts=end, skill_key=skill, keywords=kws, raw_text=raw)


def search_memory_events(
    db: Any,
    *,
    tenant_id: str,
    caller_id: str,
    q: MemoryQuery,
    limit: int = 12,
    include_turns: bool = False,
    **kwargs: Any,
) -> list[CallerMemoryEvent]:
    """Search caller memory events within a time window.

    Default behavior intentionally excludes per-turn rows (kind == "turn") so recall
    returns coherent summaries/skill outputs instead of disjoint turn fragments.

    To explicitly include turns, pass a `skill_key` that starts with "turn_" (e.g.
    "turn_user", "turn_assistant").
    """
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
        # Guard against historical bad rows
        .filter(CallerMemoryEvent.text.isnot(None))
        .filter(CallerMemoryEvent.text != "")
        # Exclude internal trace/audit rows (not real caller memory)
        .filter(CallerMemoryEvent.skill_key.notin_(["memory_recall_audit", "memory_context_audit"]))
    )

    # include_turns param forces turn rows; a 'turn_*' skill_key also forces it.
    if q.skill_key:
        qq = qq.filter(CallerMemoryEvent.skill_key == q.skill_key)
        if str(q.skill_key).lower().startswith("turn_"):
            include_turns = True

    # Default: exclude conversational turn spam from "memory recall" lookups
    if not include_turns:
        qq = qq.filter(CallerMemoryEvent.kind != "turn")

    # MVP keyword filter: OR over ILIKE
    if q.keywords:
        ors = []
        for kw in q.keywords[:8]:
            ors.append(CallerMemoryEvent.text.ilike(f"%{kw}%"))
        from sqlalchemy import or_
        qq = qq.filter(or_(*ors))

    rows = qq.order_by(CallerMemoryEvent.created_at.desc()).limit(limit).all()
    return rows
def infer_fact_key(text: str) -> Optional[str]:
    """Map a user question to a durable fact key (MVP: favorite_color)."""
    raw = (text or "").strip()
    if not raw:
        return None
    patterns = globals().get("_FACT_PATTERNS", [])
    for rx, key in patterns:
        if rx.search(raw):
            return key
    return None

def fetch_fact_history(
    db: Any,
    *,
    tenant_id: str,
    caller_id: str,
    fact_key: str,
    start_ts: Optional[datetime] = None,
    end_ts: Optional[datetime] = None,
    limit: int = 8,
    scan_limit: int = 200,
) -> list[dict[str, Any]]:
    """Return most-recent fact values for a caller (deterministic; no LLM).

    Implementation note:
    - Uses a small scan of recent rows and parses JSON in Python.
    - Avoids brittle JSON/UUID casting differences across environments.
    """
    if not (tenant_id and caller_id and fact_key):
        return []

    # Use UTC-naive timestamps to match `timestamp without time zone` columns safely.
    now = _utcnow().replace(tzinfo=None)
    start = (start_ts or (now - timedelta(days=365))).replace(tzinfo=None)
    end = (end_ts or now).replace(tzinfo=None)

    try:
        qq = (
            db.query(CallerMemoryEvent)
            .filter(CallerMemoryEvent.tenant_id == tenant_id)
            .filter(CallerMemoryEvent.caller_id == caller_id)
            .filter(CallerMemoryEvent.created_at >= start)
            .filter(CallerMemoryEvent.created_at <= end)
            .order_by(CallerMemoryEvent.created_at.desc())
            .limit(int(scan_limit))
        )
        rows = list(qq.all())
    except Exception as e:
        logger.exception("FACT_HISTORY_QUERY_FAIL tenant_id=%s caller_id=%s key=%s err=%s", tenant_id, caller_id, fact_key, e)
        return []

    out: list[dict[str, Any]] = []
    for r in rows:
        try:
            dj = r.data_json or {}
            facts = dj.get("facts") if isinstance(dj, dict) else None
            if not isinstance(facts, dict):
                continue
            val = facts.get(fact_key)
            if not isinstance(val, str) or not val.strip():
                continue
            ts = r.created_at
            ts_iso = ts.isoformat(timespec="seconds") if hasattr(ts, "isoformat") else ""
            out.append(
                {
                    "value": val.strip(),
                    "created_at": ts,
                    "created_at_iso": ts_iso,
                    "skill_key": getattr(r, "skill_key", None),
                    "text": getattr(r, "text", None),
                    "call_sid": getattr(r, "call_sid", None),
                }
            )
            if len(out) >= int(limit):
                break
        except Exception:
            continue

    return out




def _vector_literal(v: list[float]) -> str:
    return "[" + ",".join(f"{x:.8f}" for x in v) + "]"


def vector_search_call_summaries(
    db: Any,
    *,
    tenant_id: str,
    caller_id: str,
    start_ts: datetime,
    end_ts: datetime,
    limit: int = 6,
    q_embedding: Optional[list[float]] = None,
    query_embedding: Optional[list[float]] = None,
) -> list[CallerMemoryEvent]:
    """Vector search over call_summary rows using pgvector.

    Notes:
      - Accepts either `q_embedding` (legacy) or `query_embedding` (newer call sites).
      - Returns events ordered by vector similarity (best match first).
      - Logs a single VECTOR_RECALL_OK line for easy production verification.
    """
    emb = q_embedding or query_embedding or []
    if not tenant_id or not caller_id or not emb:
        return []

    qv = _vector_literal(emb)
    sql = """
    SELECT id, (embedding <=> (:qv)::vector) AS dist
    FROM caller_memory_events
    WHERE tenant_id = :tenant_id
      AND caller_id = :caller_id
      AND skill_key = 'call_summary'
      AND embedding IS NOT NULL
      AND created_at >= :start_ts
      AND created_at <= :end_ts
    ORDER BY dist
    LIMIT :limit
    """

    try:
        res = db.execute(
            text(sql),
            {
                "tenant_id": tenant_id,
                "caller_id": caller_id,
                "qv": qv,
                "start_ts": start_ts.replace(tzinfo=None),
                "end_ts": end_ts.replace(tzinfo=None),
                "limit": int(limit),
            },
        ).fetchall()

        if not res:
            logger.info(
                "VECTOR_RECALL_OK tenant_id=%s caller_id=%s k=%s hits=0",
                tenant_id,
                caller_id,
                int(limit),
            )
            return []

        ids = [r[0] for r in res]
        best_dist = None
        try:
            best_dist = float(res[0][1])
        except Exception:
            best_dist = None

        if best_dist is None:
            logger.info(
                "VECTOR_RECALL_OK tenant_id=%s caller_id=%s k=%s hits=%s",
                tenant_id,
                caller_id,
                int(limit),
                len(ids),
            )
        else:
            # Smaller distance is better; for cosine distance, 0.0 is perfect match.
            logger.info(
                "VECTOR_RECALL_OK tenant_id=%s caller_id=%s k=%s hits=%s best_dist=%.4f",
                tenant_id,
                caller_id,
                int(limit),
                len(ids),
                best_dist,
            )

        # Fetch ORM rows and preserve similarity order.
        events = db.query(CallerMemoryEvent).filter(CallerMemoryEvent.id.in_(ids)).all()
        by_id = {e.id: e for e in events}
        ordered = [by_id[i] for i in ids if i in by_id]
        return ordered
    except Exception:
        logger.exception("VECTOR_SEARCH_SQL_FAIL tenant_id=%s caller_id=%s", tenant_id, caller_id)
        return []

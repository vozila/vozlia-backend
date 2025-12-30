# services/longterm_memory.py
from __future__ import annotations

import os
import random
from datetime import datetime, timedelta
from typing import Any, Optional, Any as _Any

from core.logging import logger

try:
    from models import CallerMemoryEvent  # type: ignore
except Exception as e:
    CallerMemoryEvent = None  # type: ignore
    logger.warning("LONGTERM_MEMORY_DISABLED (missing CallerMemoryEvent): %s", e)


def longterm_memory_enabled_for_tenant(tenant_uuid: str) -> bool:
    if CallerMemoryEvent is None:
        return False

    default_on = (os.getenv("LONGTERM_MEMORY_ENABLED_DEFAULT", "0") or "0").strip().lower() in ("1","true","yes","on")
    allowlist = (os.getenv("LONGTERM_MEMORY_TENANT_ALLOWLIST", "") or "").strip()
    blocklist = (os.getenv("LONGTERM_MEMORY_TENANT_BLOCKLIST", "") or "").strip()

    if blocklist:
        blocked = {t.strip() for t in blocklist.split(",") if t.strip() and "<" not in t and ">" not in t}
        if tenant_uuid in blocked:
            return False

    if allowlist:
        allowed = {t.strip() for t in allowlist.split(",") if t.strip() and "<" not in t and ">" not in t}
        return tenant_uuid in allowed

    return default_on


def _retention_days() -> int:
    try:
        return int((os.getenv("LONGTERM_MEMORY_RETENTION_DAYS", "30") or "30").strip())
    except Exception:
        return 30


def _purge_prob() -> float:
    try:
        return float((os.getenv("LONGTERM_MEMORY_PURGE_PROB", "0.03") or "0.03").strip())
    except Exception:
        return 0.03


def purge_expired_memory(db: Any, *, tenant_uuid: str) -> int:
    """Best-effort purge; returns deleted row count."""
    if CallerMemoryEvent is None:
        return 0
    if not longterm_memory_enabled_for_tenant(tenant_uuid):
        return 0

    days = _retention_days()
    if days <= 0:
        return 0

    cutoff = datetime.utcnow() - timedelta(days=days)
    try:
        q = db.query(CallerMemoryEvent).filter(CallerMemoryEvent.tenant_id == str(tenant_uuid)).filter(CallerMemoryEvent.created_at < cutoff)
        deleted = q.delete(synchronize_session=False)
        db.commit()
        logger.info("MEMORY_PURGE_OK tenant_id=%s deleted=%s retention_days=%s", tenant_uuid, deleted, days)
        return int(deleted or 0)
    except Exception as e:
        db.rollback()
        logger.exception("MEMORY_PURGE_FAIL tenant_id=%s err=%s", tenant_uuid, e)
        return 0


def write_memory_event(
    db: Any,
    *,
    tenant_uuid: str,
    caller_id: str,
    call_sid: str | None,
    skill_key: str,
    text: str,
    data_json: Optional[dict[str, _Any]] = None,
    tags_json: Optional[list[str]] = None,
) -> bool:
    """Best-effort insert. Never raise to callers."""
    if CallerMemoryEvent is None:
        return False
    tenant_id = str(tenant_uuid or "")
    if not tenant_id or not caller_id:
        return False
    if not longterm_memory_enabled_for_tenant(tenant_id):
        return False

    body = (text or "").strip()
    if not body:
        return False

    # Clamp to avoid bloat
    max_chars = int((os.getenv("LONGTERM_MEMORY_EVENT_MAX_CHARS", "1200") or "1200").strip() or 1200)
    if len(body) > max_chars:
        body = body[:max_chars] + "…"

    try:
        row = CallerMemoryEvent(
            tenant_id=tenant_id,
            caller_id=str(caller_id),
            call_sid=(str(call_sid) if call_sid else None),
            skill_key=str(skill_key or "unknown"),
            text=body,
            data_json=data_json,
            tags_json=tags_json,
        )
        db.add(row)
        db.commit()

        if (os.getenv("VOZLIA_DEBUG_MEMORY", "0") or "0").strip() == "1":
            logger.info(
                "MEMORY_WRITE_OK tenant_id=%s caller_id=%s call_sid=%s skill=%s chars=%s tags=%s",
                tenant_id, caller_id, call_sid or None, skill_key, len(body), (tags_json or []),
            )
        else:
            logger.info("MEMORY_WRITE_OK tenant_id=%s caller_id=%s skill=%s chars=%s", tenant_id, caller_id, skill_key, len(body))

        # Probabilistic purge
        if random.random() < _purge_prob():
            purge_expired_memory(db, tenant_uuid=tenant_id)

        return True
    except Exception as e:
        db.rollback()
        logger.exception("MEMORY_WRITE_FAIL tenant_id=%s caller_id=%s skill=%s err=%s", tenant_id, caller_id, skill_key, e)
        return False


def fetch_recent_memory_text(
    db: Any,
    *,
    tenant_uuid: str,
    caller_id: str,
    limit: int = 8,
) -> str:
    """Returns a compact memory block for prompt injection."""
    if CallerMemoryEvent is None:
        return ""
    tenant_id = str(tenant_uuid or "")
    if not tenant_id or not caller_id:
        return ""
    if not longterm_memory_enabled_for_tenant(tenant_id):
        return ""

    max_chars = int((os.getenv("LONGTERM_MEMORY_CONTEXT_MAX_CHARS", "1200") or "1200").strip() or 1200)

    try:
        rows = (
            db.query(CallerMemoryEvent)
            .filter(CallerMemoryEvent.tenant_id == tenant_id)
            .filter(CallerMemoryEvent.caller_id == str(caller_id))
            .order_by(CallerMemoryEvent.created_at.desc())
            .limit(int(limit or 8))
            .all()
        )

        lines = []
        for r in rows:
            ts = r.created_at.isoformat(timespec="seconds")
            lines.append(f"- [{ts}] ({r.skill_key}) {r.text}")

        block = "\n".join(lines).strip()
        if len(block) > max_chars:
            block = block[:max_chars] + "…"

        if (os.getenv("VOZLIA_DEBUG_MEMORY", "0") or "0").strip() == "1" and block:
            logger.info("MEMORY_CONTEXT_READY tenant_id=%s caller_id=%s rows=%s chars=%s", tenant_id, caller_id, len(rows), len(block))

        return block
    except Exception as e:
        logger.exception("MEMORY_CONTEXT_FAIL tenant_id=%s caller_id=%s err=%s", tenant_id, caller_id, e)
        return ""



# -------------------------
# Memory recall helpers
# -------------------------
from dataclasses import dataclass
from zoneinfo import ZoneInfo
import re as _re

NY_TZ = ZoneInfo(os.getenv("VOZLIA_TZ", "America/New_York") or "America/New_York")

_STOPWORDS = {
    "the","a","an","and","or","to","of","in","on","for","with","at","from","by","about","is","are","was","were",
    "i","me","my","you","your","we","our","it","that","this","what","did","say","ago","last","time","talked",
    "please","vozlia","hey","hi","hello","remind","remember",
}

@dataclass
class MemoryQuery:
    start_utc: datetime
    end_utc: datetime
    keywords: list[str]
    tag: str | None = None

def _utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=ZoneInfo("UTC"))
    return dt.astimezone(ZoneInfo("UTC"))

def parse_memory_query(text: str, *, now_utc: datetime | None = None) -> MemoryQuery:
    """
    Very lightweight parser: detects rough time windows + keywords.
    This is intentionally cheap (no heavy NLP) and safe for the voice hot path.
    """
    s = (text or "").strip()
    now = now_utc or datetime.utcnow().replace(tzinfo=ZoneInfo("UTC"))

    # Defaults (configurable)
    default_days = int((os.getenv("LONGTERM_MEMORY_DEFAULT_LOOKBACK_DAYS", "7") or "7").strip() or "7")
    start = now - timedelta(days=default_days)
    end = now

    # Relative times: "5 minutes ago", "2 hours ago"
    m = _re.search(r"\b(\d{1,3})\s*(minute|minutes|hour|hours|day|days)\s*ago\b", s, flags=_re.I)
    if m:
        n = int(m.group(1))
        unit = m.group(2).lower()
        if "minute" in unit:
            start = now - timedelta(minutes=n)
        elif "hour" in unit:
            start = now - timedelta(hours=n)
        elif "day" in unit:
            start = now - timedelta(days=n)

    # Named windows
    low = s.lower()
    if "yesterday" in low:
        local_now = now.astimezone(NY_TZ)
        y = (local_now - timedelta(days=1)).date()
        start_local = datetime(y.year, y.month, y.day, 0, 0, 0, tzinfo=NY_TZ)
        end_local = datetime(y.year, y.month, y.day, 23, 59, 59, tzinfo=NY_TZ)
        start = _utc(start_local)
        end = _utc(end_local)
    elif "today" in low:
        local_now = now.astimezone(NY_TZ)
        d = local_now.date()
        start_local = datetime(d.year, d.month, d.day, 0, 0, 0, tzinfo=NY_TZ)
        start = _utc(start_local)
        end = now
    elif "last week" in low:
        start = now - timedelta(days=7)
    elif "last month" in low:
        start = now - timedelta(days=30)

    # Topic tags (very small set for now)
    tag = None
    if "favorite color" in low or "favourite colour" in low or "favorite colour" in low:
        tag = "favorite_color"
    elif "weather" in low or "forecast" in low:
        tag = "weather"
    elif "email" in low or "gmail" in low or "inbox" in low:
        tag = "gmail_summary"

    # Keywords
    toks = [_t for _t in _re.split(r"[^a-zA-Z0-9_]+", low) if _t]
    kws: list[str] = []
    for t in toks:
        if len(t) < 3:
            continue
        if t in _STOPWORDS:
            continue
        kws.append(t)
    # De-dupe while preserving order
    seen=set()
    keywords=[]
    for k in kws:
        if k in seen:
            continue
        seen.add(k)
        keywords.append(k)

    return MemoryQuery(start_utc=_utc(start), end_utc=_utc(end), keywords=keywords[:12], tag=tag)

def search_memory_events(
    db: Any,
    *,
    tenant_uuid: str,
    caller_id: str,
    q: MemoryQuery,
    limit: int = 12,
):
    """Cheap SQLAlchemy query for relevant memory events (time + keywords + optional tag)."""
    if CallerMemoryEvent is None:
        return []

    qry = (
        db.query(CallerMemoryEvent)
        .filter(CallerMemoryEvent.tenant_id == str(tenant_uuid))
        .filter(CallerMemoryEvent.caller_id == str(caller_id))
        .filter(CallerMemoryEvent.created_at >= q.start_utc.replace(tzinfo=None))
        .filter(CallerMemoryEvent.created_at <= q.end_utc.replace(tzinfo=None))
    )

    if q.tag:
        # Prefer tag match; fall back to skill_key match
        try:
            qry = qry.filter(
                (CallerMemoryEvent.skill_key == q.tag) | (CallerMemoryEvent.tags_json.contains([q.tag]))
            )
        except Exception:
            qry = qry.filter(CallerMemoryEvent.skill_key == q.tag)

    if q.keywords:
        clauses = []
        for kw in q.keywords:
            clauses.append(CallerMemoryEvent.text.ilike(f"%{kw}%"))
        try:
            from sqlalchemy import or_
            qry = qry.filter(or_(*clauses))
        except Exception:
            pass

    return qry.order_by(CallerMemoryEvent.created_at.desc()).limit(int(limit)).all()

def extract_favorite_color(text: str) -> str | None:
    """Extracts a favorite color value from a user utterance."""
    s = (text or "").strip()
    m = _re.search(r"\bfavo(?:u)?rite\s+colou?r\s+(?:is|=)\s+([a-zA-Z]+)\b", s, flags=_re.I)
    if not m:
        m = _re.search(r"\bmy\s+favo(?:u)?rite\s+colou?r\s+(?:is|=)\s+([a-zA-Z]+)\b", s, flags=_re.I)
    if m:
        return m.group(1).strip().lower()
    return None


# -------------------------
# Backwards-compatible API
# -------------------------
def record_skill_result(
    db: Any,
    *,
    tenant_uuid: str,
    caller_id: str,
    skill_key: str,
    input_text: str | None = None,
    memory_text: str = "",
    data_json: Optional[dict[str, _Any]] = None,
    expires_in_s: Optional[int] = None,
) -> bool:
    """Compat shim used by assistant_service; persists a skill outcome as a memory event."""
    _ = input_text  # reserved for later
    _ = expires_in_s  # reserved for later (future per-event TTL)
    return write_memory_event(
        db,
        tenant_uuid=str(tenant_uuid),
        caller_id=str(caller_id),
        call_sid=None,
        skill_key=str(skill_key or "skill"),
        text=(memory_text or "").strip(),
        data_json=data_json,
        tags_json=[str(skill_key or "skill")],
    )

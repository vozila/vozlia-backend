# services/greeting_memory.py
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy.orm import Session

from core.logging import logger
from models import CallerMemoryEvent
from services.caller_cache import normalize_caller_id


def _env_int(name: str, default: int) -> int:
    try:
        return int((os.getenv(name, str(default)) or str(default)).strip())
    except Exception:
        return default


def build_prev_call_preface(
    db: Session,
    *,
    tenant_id: str,
    caller_id: str,
    current_call_sid: str | None,
) -> Optional[str]:
    """Return a short preface based on the most recent previous call_summary for this caller."""
    enabled = (os.getenv("GREETING_PREV_CALL_ENABLED", "1") or "1").strip().lower() in ("1", "true", "yes", "on")
    if not enabled:
        return None

    caller_norm = normalize_caller_id(caller_id)
    if not tenant_id or not caller_norm:
        return None

    lookback_days = _env_int("GREETING_PREV_CALL_LOOKBACK_DAYS", 60)
    max_chars = _env_int("GREETING_PREV_CALL_MAX_CHARS", 180)

    end_utc = datetime.now(timezone.utc).replace(tzinfo=None)
    start_utc = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).replace(tzinfo=None)

    q = (
        db.query(CallerMemoryEvent)
        .filter(CallerMemoryEvent.tenant_id == str(tenant_id))
        .filter(CallerMemoryEvent.caller_id == str(caller_norm))
        .filter(CallerMemoryEvent.skill_key == "call_summary")
        .filter(CallerMemoryEvent.created_at >= start_utc)
        .filter(CallerMemoryEvent.created_at <= end_utc)
    )

    if current_call_sid:
        q = q.filter(CallerMemoryEvent.call_sid != str(current_call_sid))

    row = q.order_by(CallerMemoryEvent.created_at.desc()).limit(1).one_or_none()
    if not row:
        return None

    raw = (getattr(row, "text", "") or "").strip()
    if not raw:
        return None

    # Light cleanup for voice friendliness (optional)
    s = raw.replace("\n", " ").strip()
    if len(s) > max_chars:
        s = s[:max_chars].rstrip() + "…"

    # Avoid overly robotic phrasing
    # Example: "The caller said..." -> "we talked about..."
    s = s.replace("The caller", "we").replace("caller", "we")

    return f"Welcome back. Last time, {s}"

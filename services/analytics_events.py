# services/analytics_events.py
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Optional

from core.logging import logger

try:
    from db import SessionLocal
    from models import AnalyticsEvent  # type: ignore
except Exception as e:  # pragma: no cover
    SessionLocal = None  # type: ignore
    AnalyticsEvent = None  # type: ignore
    logger.warning("ANALYTICS_EVENTS_DISABLED (imports failed): %s", e)


def analytics_events_enabled() -> bool:
    return (os.getenv("ANALYTICS_EVENTS_ENABLED", "0") or "0").strip().lower() in ("1", "true", "yes", "on")


def _utcnow_naive() -> datetime:
    # App convention: DB stores naive timestamps representing UTC.
    return datetime.now(timezone.utc).replace(tzinfo=None)


def emit_analytics_event(
    *,
    tenant_id: str,
    event_type: str,
    caller_id: str | None = None,
    call_sid: str | None = None,
    skill_key: str | None = None,
    payload: dict[str, Any] | None = None,
    tags: list[str] | None = None,
) -> None:
    """Best-effort analytics event write. Never raises.

    Important safety properties:
    - Uses its own DB session so it cannot accidentally commit pending work in the caller's session.
    - Fail-open: if the table is missing or the DB is down, Vozlia keeps working.
    """
    if AnalyticsEvent is None or SessionLocal is None:
        return
    if not analytics_events_enabled():
        return

    tenant_id = (tenant_id or "").strip()
    event_type = (event_type or "").strip()
    if not tenant_id or not event_type:
        return

    # Clamp payload sizes to avoid bloating Postgres + indexes.
    def _clamp_str(s: str, n: int) -> str:
        s = (s or "").strip()
        return s if len(s) <= n else (s[:n] + "…")

    safe_payload: dict[str, Any] | None = None
    if isinstance(payload, dict) and payload:
        safe_payload = {}
        for k, v in payload.items():
            if v is None:
                continue
            if isinstance(v, str):
                safe_payload[str(k)[:80]] = _clamp_str(v, 600)
            elif isinstance(v, (int, float, bool)):
                safe_payload[str(k)[:80]] = v
            else:
                # Avoid huge nested objects by default
                safe_payload[str(k)[:80]] = _clamp_str(str(v), 600)

    safe_tags: list[str] | None = None
    if isinstance(tags, list) and tags:
        safe_tags = []
        for t in tags[:40]:
            if not isinstance(t, str):
                continue
            tt = t.strip()
            if tt:
                safe_tags.append(_clamp_str(tt, 120))

    try:
        db = SessionLocal()
        row = AnalyticsEvent(
            tenant_id=str(tenant_id),
            caller_id=(str(caller_id) if caller_id else None),
            call_sid=(str(call_sid) if call_sid else None),
            event_type=str(event_type),
            skill_key=(str(skill_key) if skill_key else None),
            created_at=_utcnow_naive(),
            payload_json=safe_payload,
            tags_json=safe_tags,
        )
        db.add(row)
        db.commit()
    except Exception:
        # Keep log concise in production; include details only if VOZLIA_DEBUG_ANALYTICS=1
        if (os.getenv("VOZLIA_DEBUG_ANALYTICS", "0") or "0").strip() == "1":
            logger.exception(
                "ANALYTICS_EVENT_WRITE_FAIL tenant_id=%s type=%s skill=%s",
                tenant_id[:8], event_type, (skill_key or ""),
            )
        else:
            logger.warning("ANALYTICS_EVENT_WRITE_FAIL tenant_id=%s type=%s", tenant_id[:8], event_type)
    finally:
        try:
            db.close()
        except Exception:
            pass

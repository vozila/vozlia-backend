# services/longterm_memory.py
from __future__ import annotations

import os
import random
from datetime import datetime, timedelta
from typing import Any, Optional, Any as _Any

from core.logging import logger
from services.memory_enricher import enrich_turn

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
        kind_val = "turn" if str(skill_key or "").lower().startswith("turn_") else "skill"
        row = CallerMemoryEvent(
            tenant_id=tenant_id,
            caller_id=str(caller_id),
            call_sid=(str(call_sid) if call_sid else None),
            kind=kind_val,
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
            caller_tail = (str(caller_id)[-4:] if caller_id else "----")
            logger.info("MEMORY_WRITE_OK tenant_id=%s caller_id=***%s skill=%s chars=%s", tenant_id[:8], caller_tail, skill_key, len(body))

        # Probabilistic purge
        if random.random() < _purge_prob():
            purge_expired_memory(db, tenant_uuid=tenant_id)

        return True
    except Exception as e:
        db.rollback()
        caller_tail = (str(caller_id)[-4:] if caller_id else "----")
        logger.exception("MEMORY_WRITE_FAIL tenant_id=%s caller_id=***%s skill=%s err=%s", tenant_id[:8], caller_tail, skill_key, e)
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
    call_sid: str | None = None,  # NEW
) -> bool:
    _ = input_text
    _ = expires_in_s
    return write_memory_event(
        db,
        tenant_uuid=str(tenant_uuid),
        caller_id=str(caller_id),
        call_sid=(str(call_sid) if call_sid else None),  # NEW
        skill_key=skill_key,
        text=memory_text,
        data_json=data_json,
        tags_json=[f"skill:{skill_key or 'skill'}"],
    )


def record_turn_event(
    db,
    *,
    tenant_uuid: str,
    caller_id: str,
    call_sid: str | None,
    role: str,
    text: str,
    session_id: str | None = None,
) -> bool:
    """Write a turn-level memory event (user or assistant). Stores cleaned text in `text`,
    raw text in data_json, and tags_json with kw:* and fact:* tags.
    """
    try:
        enriched = enrich_turn(text or "")
        skill_key = f"turn_{role}".lower().strip() or "turn_user"
        data = {
            "raw": (text or "").strip(),
            "role": role,
        }
        if session_id:
            data["session_id"] = session_id
        # include extracted facts (small / structured)
        if enriched.facts:
            data["facts"] = enriched.facts

        return write_memory_event(
            db,
            tenant_uuid=str(tenant_uuid),
            caller_id=str(caller_id),
            call_sid=call_sid,
            skill_key=skill_key,
            text=(enriched.cleaned_text or "").strip() or (text or "").strip() or "[empty]",
            data_json=data,
            tags_json=enriched.tags,
        )
    except Exception as e:
        logger.exception("MEMORY_TURN_WRITE_FAIL tenant_id=%s caller_id=%s role=%s err=%s", tenant_uuid, caller_id, role, e)
        return False

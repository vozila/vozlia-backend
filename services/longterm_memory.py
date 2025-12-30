# services/longterm_memory.py
from __future__ import annotations

import os
import random
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from core.logging import logger

# Rollback-safe import: if the model isn't present, we fail-open and disable.
try:
    from models import CallerMemoryEvent  # type: ignore
except Exception as e:
    CallerMemoryEvent = None  # type: ignore
    logger.warning("LONGTERM_MEMORY_DISABLED (missing CallerMemoryEvent): %s", e)


DEBUG_MEMORY = (os.getenv("VOZLIA_DEBUG_MEMORY", "0") or "0").strip() == "1"

# Configurable retention policy (days). Set to 0 to disable retention purge.
RETENTION_DAYS = int((os.getenv("LONGTERM_MEMORY_RETENTION_DAYS", "30") or "30").strip() or 30)

# Limit how much memory text we inject into the FSM context (prevents prompt bloat).
CONTEXT_MAX_CHARS = int((os.getenv("LONGTERM_MEMORY_CONTEXT_MAX_CHARS", "1200") or "1200").strip() or 1200)

# Keep the old allowlist/blocklist semantics.
def longterm_memory_enabled_for_tenant(tenant_uuid: str) -> bool:
    if not tenant_uuid:
        return False
    default_on = (os.getenv("LONGTERM_MEMORY_ENABLED_DEFAULT", "0") or "0").strip() == "1"
    allowlist = (os.getenv("LONGTERM_MEMORY_TENANT_ALLOWLIST", "") or "").strip()
    blocklist = (os.getenv("LONGTERM_MEMORY_TENANT_BLOCKLIST", "") or "").strip()

    if blocklist:
        blocked = {t.strip() for t in blocklist.split(",") if t.strip()}
        if tenant_uuid in blocked:
            return False

    if allowlist:
        allowed = {t.strip() for t in allowlist.split(",") if t.strip()}
        return tenant_uuid in allowed

    return default_on


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def purge_expired_memory(db: Any, *, tenant_id: str, caller_id: Optional[str] = None) -> int:
    """Best-effort retention purge. Never raises."""
    if CallerMemoryEvent is None:
        return 0
    if not tenant_id:
        return 0

    try:
        q = db.query(CallerMemoryEvent).filter(CallerMemoryEvent.tenant_id == tenant_id)

        if caller_id:
            q = q.filter(CallerMemoryEvent.caller_id == caller_id)

        now = _utcnow()

        # Expiration-based purge
        q_exp = q.filter(CallerMemoryEvent.expires_at.isnot(None)).filter(CallerMemoryEvent.expires_at < now)
        deleted_exp = q_exp.delete(synchronize_session=False)

        deleted_ret = 0
        if RETENTION_DAYS and RETENTION_DAYS > 0:
            cutoff = now - timedelta(days=RETENTION_DAYS)
            q_ret = q.filter(CallerMemoryEvent.created_at < cutoff)
            deleted_ret = q_ret.delete(synchronize_session=False)

        total = int((deleted_exp or 0) + (deleted_ret or 0))
        if total:
            db.commit()
            logger.info(
                "MEMORY_PURGE_OK tenant_id=%s caller_id=%s deleted=%s retention_days=%s",
                tenant_id,
                caller_id or None,
                total,
                RETENTION_DAYS,
            )
        return total
    except Exception as e:
        # Fail-open. Never block calls.
        try:
            db.rollback()
        except Exception:
            pass
        logger.warning("MEMORY_PURGE_FAIL tenant_id=%s caller_id=%s err=%s", tenant_id, caller_id or None, e)
        return 0


def fetch_recent_memory_text(
    db: Any,
    *,
    tenant_uuid: str,
    caller_id: str,
    limit: int = 8,
) -> str:
    """Returns a compact text block for FSM context injection."""
    if CallerMemoryEvent is None:
        return ""
    if not longterm_memory_enabled_for_tenant(tenant_uuid):
        return ""
    if not caller_id:
        return ""

    tenant_id = str(tenant_uuid)

    try:
        now = _utcnow()
        q = (
            db.query(CallerMemoryEvent)
            .filter(CallerMemoryEvent.tenant_id == tenant_id)
            .filter(CallerMemoryEvent.caller_id == caller_id)
            .filter((CallerMemoryEvent.expires_at.is_(None)) | (CallerMemoryEvent.expires_at >= now))
            .order_by(CallerMemoryEvent.created_at.desc())
            .limit(int(limit or 8))
        )
        rows = q.all() or []
        if not rows:
            return ""

        # Oldest-first for readability in prompt context
        rows = list(reversed(rows))

        parts: list[str] = []
        for r in rows:
            ts = (r.created_at or now).astimezone(timezone.utc).strftime("%Y-%m-%d %H:%MZ")
            sk = (r.skill_key or "memory").strip()
            txt = (r.text or "").strip()
            if not txt:
                continue
            line = f"[{ts}] ({sk}) {txt}"
            parts.append(line)

        out = "\n".join(parts).strip()
        if not out:
            return ""

        if len(out) > CONTEXT_MAX_CHARS:
            out = out[-CONTEXT_MAX_CHARS:]
            # Trim to a clean line boundary
            nl = out.find("\n")
            if nl != -1 and nl < 200:
                out = out[nl + 1 :]

        if DEBUG_MEMORY:
            logger.info(
                "MEMORY_CONTEXT_READY tenant_id=%s caller_id=%s rows=%s chars=%s",
                tenant_id,
                caller_id,
                len(rows),
                len(out),
            )
        return out

    except Exception as e:
        logger.warning("MEMORY_CONTEXT_FAIL tenant_id=%s caller_id=%s err=%s", tenant_id, caller_id, e)
        return ""


def write_memory_event(
    db: Any,
    *,
    tenant_uuid: str,
    caller_id: str,
    call_sid: Optional[str],
    skill_key: str,
    text: str,
    data_json: Optional[dict[str, Any]] = None,
    tags: Optional[list[str]] = None,
    expires_in_s: Optional[int] = None,
) -> bool:
    """Best-effort durable memory write. Never raises."""
    if CallerMemoryEvent is None:
        return False
    if not longterm_memory_enabled_for_tenant(str(tenant_uuid)):
        return False

    tenant_id = str(tenant_uuid)
    if not (tenant_id and caller_id and skill_key and (text or "").strip()):
        return False

    try:
        now = _utcnow()
        expires_at = None
        if expires_in_s is not None and int(expires_in_s) > 0:
            expires_at = now + timedelta(seconds=int(expires_in_s))

        row = CallerMemoryEvent(
            tenant_id=tenant_id,
            caller_id=caller_id,
            call_sid=call_sid,
            skill_key=skill_key,
            text=(text or "").strip(),
            data_json=data_json,
            tags_json=tags or None,
            created_at=now,
            expires_at=expires_at,
        )
        db.add(row)
        db.commit()

        if DEBUG_MEMORY:
            logger.info(
                "MEMORY_WRITE_OK tenant_id=%s caller_id=%s call_sid=%s skill_key=%s chars=%s tags=%s",
                tenant_id,
                caller_id,
                call_sid or None,
                skill_key,
                len((text or "").strip()),
                len(tags or []),
            )

        # Opportunistic retention purge (best-effort, low frequency).
        # Tune with LONGTERM_MEMORY_PURGE_PROB (default 0.03 = ~3% of writes).
        p = float((os.getenv("LONGTERM_MEMORY_PURGE_PROB", "0.03") or "0.03").strip() or 0.03)
        if p > 0 and random.random() < p:
            purge_expired_memory(db, tenant_id=tenant_id, caller_id=caller_id)

        return True
    except Exception as e:
        try:
            db.rollback()
        except Exception:
            pass
        logger.warning(
            "MEMORY_WRITE_FAIL tenant_id=%s caller_id=%s call_sid=%s skill_key=%s err=%s",
            str(tenant_uuid),
            caller_id,
            call_sid or None,
            skill_key,
            e,
        )
        return False

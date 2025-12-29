# services/longterm_memory.py
from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from core.logging import logger
from models import CallerMemoryEvent


def _truthy(v: str | None) -> bool:
    return (v or "").strip().lower() in ("1", "true", "yes", "on")


def longterm_memory_enabled_for_tenant(tenant_id: str) -> bool:
    """Per-tenant enable gate using env vars.

    Defaults:
      - LONGTERM_MEMORY_ENABLED_DEFAULT=0 (off)
      - LONGTERM_MEMORY_TENANT_ALLOWLIST=  (empty means: apply default to all)
      - LONGTERM_MEMORY_TENANT_BLOCKLIST=  (always off for listed tenants)

    Allowlist/blocklist values are comma-separated. Tenant IDs are compared as strings.
    """
    default_on = _truthy(os.getenv("LONGTERM_MEMORY_ENABLED_DEFAULT", "0"))

    allow = (os.getenv("LONGTERM_MEMORY_TENANT_ALLOWLIST") or "").strip()
    block = (os.getenv("LONGTERM_MEMORY_TENANT_BLOCKLIST") or "").strip()

    tenant_id_s = str(tenant_id or "").strip()
    if not tenant_id_s:
        return False

    if block:
        blocked = {t.strip() for t in block.split(",") if t.strip()}
        if tenant_id_s in blocked:
            return False

    if allow:
        allowed = {t.strip() for t in allow.split(",") if t.strip()}
        return tenant_id_s in allowed

    return default_on


def record_skill_result(
    db,
    *,
    tenant_uuid,
    caller_id: str,
    skill_key: str,
    input_text: str | None,
    memory_text: str | None,
    data_json: Dict[str, Any] | None = None,
    expires_in_s: int | None = None,
) -> None:
    """Write a durable memory event. Fail-open: never break the call."""
    try:
        if not (tenant_uuid and caller_id and skill_key):
            return

        expires_at = None
        if expires_in_s and int(expires_in_s) > 0:
            expires_at = datetime.utcnow() + timedelta(seconds=int(expires_in_s))

        ev = CallerMemoryEvent(
            tenant_id=tenant_uuid,
            caller_id=caller_id,
            kind="skill_result",
            skill_key=skill_key,
            input_text=(input_text or None),
            memory_text=(memory_text or None),
            data_json=(data_json or None),
            expires_at=expires_at,
        )
        db.add(ev)
        db.commit()
    except Exception:
        try:
            db.rollback()
        except Exception:
            pass
        logger.exception("LONGTERM_MEM_WRITE_FAILED skill=%s caller_id=%s", skill_key, caller_id)


def fetch_recent_memory_text(
    db,
    *,
    tenant_uuid,
    caller_id: str,
    limit: int = 8,
) -> str:
    """Return a compact text block suitable for prompt injection.

    Fail-open: returns empty string on error.
    """
    try:
        if not (tenant_uuid and caller_id):
            return ""
        now = datetime.utcnow()

        q = (
            db.query(CallerMemoryEvent)
            .filter(CallerMemoryEvent.tenant_id == tenant_uuid)
            .filter(CallerMemoryEvent.caller_id == caller_id)
            .filter((CallerMemoryEvent.expires_at == None) | (CallerMemoryEvent.expires_at > now))  # noqa: E711
            .order_by(CallerMemoryEvent.created_at.desc())
            .limit(int(limit))
        )
        rows: List[CallerMemoryEvent] = list(q.all())
        if not rows:
            return ""

        lines: List[str] = []
        for r in rows:
            ts = r.created_at.strftime("%Y-%m-%d")
            sk = r.skill_key or "memory"
            mt = (r.memory_text or "").strip()
            if not mt:
                continue
            lines.append(f"- [{ts}] {sk}: {mt}")

        return "\n".join(lines).strip()
    except Exception:
        logger.exception("LONGTERM_MEM_FETCH_FAILED caller_id=%s", caller_id)
        return ""

# services/longterm_memory.py
from __future__ import annotations

import os
from typing import Any, Optional

from core.logging import logger

# IMPORTANT:
# We are in rollback / short-term-only mode. Some deployments may still import this module.
# If the CallerMemoryEvent model is not present, we fail-open and disable longterm memory.
try:
    from models import CallerMemoryEvent  # type: ignore
except Exception as e:
    CallerMemoryEvent = None  # type: ignore
    logger.warning("LONGTERM_MEMORY_DISABLED (missing CallerMemoryEvent): %s", e)


def longterm_memory_enabled_for_tenant(tenant_uuid: str) -> bool:
    """
    Tenant-gated longterm memory flag.
    In rollback mode (no model), this returns False.
    """
    if CallerMemoryEvent is None:
        return False

    # Keep existing env-gating semantics (safe defaults).
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


def fetch_recent_memory_text(
    db: Any,
    *,
    tenant_uuid: str,
    caller_id: str,
    limit: int = 8,
) -> str:
    """
    Rollback-safe: returns "" if longterm memory isn't available.
    """
    if CallerMemoryEvent is None:
        return ""
    if not longterm_memory_enabled_for_tenant(tenant_uuid):
        return ""

    # If you re-enable longterm later, re-add the real query implementation here.
    # For now (rollback), we intentionally do not query.
    return ""


def record_skill_result(
    db: Any,
    *,
    tenant_uuid: str,
    caller_id: str,
    skill_key: str,
    input_text: Optional[str],
    memory_text: str,
    data_json: Optional[dict[str, Any]] = None,
    expires_in_s: Optional[int] = None,
) -> bool:
    """
    Rollback-safe: no-op if longterm memory isn't available.
    """
    if CallerMemoryEvent is None:
        return False
    if not longterm_memory_enabled_for_tenant(tenant_uuid):
        return False

    # If you re-enable longterm later, re-add the real write implementation here.
    return False

"""
Observability helpers (Flow B MVP).

Design goals:
- Safe-by-default: importing this module must never crash the app.
- No-op unless ENABLE_OBS=true (or you explicitly pass enabled=True).
- Keep logging / event capture out of latency-sensitive paths.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

# Default gate. Keep behavior stable and safe.
ENABLE_OBS = os.getenv("ENABLE_OBS", "false").lower() == "true"


def maybe_record_event(
    event_name: str,
    payload: Optional[Dict[str, Any]] = None,
    *,
    enabled: Optional[bool] = None,
) -> None:
    """
    Best-effort event capture.

    - If OBS is disabled, returns immediately.
    - If the underlying obs implementation isn't present, it silently no-ops.
    - Never raises.
    """
    try:
        is_enabled = ENABLE_OBS if enabled is None else enabled
        if not is_enabled:
            return

        # Try to call your existing obs implementation if it exists.
        # Supports either vozlia_obs.py or vozlia_obs (1).py-style copies that were renamed.
        try:
            from vozlia_obs import maybe_record_event as impl  # type: ignore
            impl(event_name, payload or {})
            return
        except Exception:
            pass

        # Fallback: do nothing (safe default).
        return

    except Exception:
        # Observability must never break production.
        return

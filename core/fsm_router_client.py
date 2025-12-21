# core/fsm_router_client.py
from __future__ import annotations

from typing import Optional
import os

import httpx

from core.logging import logger
from core import config as cfg


async def call_fsm_router(
    text: str,
    context: Optional[dict] = None,
    account_id: Optional[str] = None,
) -> dict:
    """
    Calls the unified /assistant/route endpoint.

    Kept behavior-aligned with the version that previously lived in main.py.
    """

    # --- Compatibility shim ---
    # Some callers accidentally pass {"text": "...", "context": {...}} as the `text` arg.
    # Unwrap it so /assistant/route receives the shape it expects.
    if isinstance(text, dict):
        logger.error("FSM_ROUTER: received dict for `text` (legacy shape). Unwrapping. text=%r", text)
        if context is None:
            context = text.get("context")
        if account_id is None:
            account_id = text.get("account_id")
        text = text.get("text") or ""

    if not isinstance(text, str):
        logger.error("FSM_ROUTER: `text` is not a string after unwrap: %r", text)
        text = str(text or "")

    text = text.strip()
    if not text:
        return {"spoken_reply": "", "fsm": {}, "gmail": None}


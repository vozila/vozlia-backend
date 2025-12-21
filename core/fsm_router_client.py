# core/fsm_router_client.py
from __future__ import annotations

from typing import Optional
import os

import httpx

from core.logging import logger
from core import config as cfg

# core/fsm_router_client.py
from __future__ import annotations

from typing import Optional, Any
import os
import httpx

from core.logging import logger
from core import config as cfg


async def call_fsm_router(
    text: str | dict,
    context: Optional[dict] = None,
    account_id: Optional[str] = None,
) -> dict:
    """
    Calls the unified /assistant/route endpoint.
    Always returns a dict with keys: spoken_reply, fsm, gmail.
    """

    # --- Legacy shape support (defensive) ---
    if isinstance(text, dict):
        logger.error("FSM_ROUTER: received dict for `text` (legacy shape). Unwrapping. text=%r", text)
        # Unwrap: {"text": "...", "context": {...}, "account_id": "..."} if present
        legacy = text
        text = legacy.get("text") or ""
        if context is None and isinstance(legacy.get("context"), dict):
            context = legacy["context"]
        if account_id is None and legacy.get("account_id") is not None:
            account_id = legacy.get("account_id")

    if not isinstance(text, str):
        logger.error("FSM_ROUTER: invalid `text` type after unwrap: %r", type(text))
        text = str(text)

    text = text.strip()
    if not text:
        return {"spoken_reply": "", "fsm": {}, "gmail": None}

    base = (cfg.VOZLIA_BACKEND_BASE_URL or "").rstrip("/")
    url = base + "/assistant/route"
    payload: dict[str, Any] = {"text": text}
    if context is not None:
        payload["context"] = context
    if account_id is not None:
        payload["account_id"] = account_id

    timeout_s = float(os.getenv("FSM_ROUTER_TIMEOUT_S", "10.0"))

    logger.info("FSM_ROUTER POST url=%s payload=%s", url, payload)

    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client_http:
            resp = await client_http.post(url, json=payload)

            if resp.status_code >= 400:
                logger.error(
                    "FSM_ROUTER failed: status=%s body=%s",
                    resp.status_code,
                    resp.text[:2000],
                )

            resp.raise_for_status()

            data = resp.json()
            if not isinstance(data, dict):
                logger.error("FSM_ROUTER: non-dict JSON returned: %r", data)
                return {
                    "spoken_reply": "Something went wrong checking that. Please try again in a moment.",
                    "fsm": {"error": "non-dict response"},
                    "gmail": None,
                }

            return data

    except Exception as e:
        logger.exception("FSM_ROUTER exception calling %s: %s", url, e)
        return {
            "spoken_reply": (
                "I tried to check that information in the backend, "
                "but something went wrong. Please try again in a moment."
            ),
            "fsm": {"error": str(e)},
            "gmail": None,
        }

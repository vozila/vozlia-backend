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
    if not text:
        return {"spoken_reply": "", "fsm": {}, "gmail": None}

    base = (cfg.VOZLIA_BACKEND_BASE_URL or "").rstrip("/")
    url = base + "/assistant/route"

    payload: dict = {"text": text}
    if context is not None:
        payload["context"] = context
    if account_id is not None:
        payload["account_id"] = account_id

    timeout_s = float(os.getenv("FSM_ROUTER_TIMEOUT_S", "10.0"))

    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client_http:
            resp = await client_http.post(url, json=payload)
            if resp.status_code >= 400:
                logger.error("Router call failed: status=%s body=%s", resp.status_code, resp.text)
            resp.raise_for_status()
            return resp.json()
    except Exception as e:
        logger.exception("Error calling /assistant/route at %s: %s", url, e)
        return {
            "spoken_reply": (
                "I tried to check that information in the backend, "
                "but something went wrong. Please try again in a moment."
            ),
            "fsm": {"error": str(e)},
            "gmail": None,
        }

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
    Calls POST /assistant/route on the Vozlia backend.

    Expected request body:
      {"text": "<string>", "context": {...}?, "account_id": "...?"}
    """
    text = (text or "").strip()
    if not text:
        return {"spoken_reply": "", "fsm": {}, "gmail": None}

    base = (cfg.VOZLIA_BACKEND_BASE_URL or "").rstrip("/")
    url = f"{base}/assistant/route"

    payload: dict = {"text": text}
    if context is not None:
        payload["context"] = context
    if account_id is not None:
        payload["account_id"] = account_id

    timeout_s = float(os.getenv("FSM_ROUTER_TIMEOUT_S", "15.0"))

    # Optional debug logging (safe)
    if os.getenv("FSM_ROUTER_LOG_PAYLOAD", "0") == "1":
        logger.info("FSM_ROUTER POST url=%s payload=%s", url, payload)

    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            resp = await client.post(url, json=payload)

            if resp.status_code >= 400:
                logger.error(
                    "FSM_ROUTER failed: status=%s body=%s",
                    resp.status_code,
                    resp.text[:2000],
                )

            resp.raise_for_status()
            return resp.json()

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

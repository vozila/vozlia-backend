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

    log_details = os.getenv("FSM_ROUTER_LOG_DETAILS", "0") == "1"
    t0 = None
    if log_details:
        # Avoid logging full content; keep it small and safe.
        text_snip = (text[:160] + "…") if len(text) > 160 else text
        logger.info("FSM_ROUTER_START url=%s account_id=%s text=%r timeout_s=%s", url, account_id, text_snip, timeout_s)
        t0 = __import__("time").perf_counter()

    # Optional debug logging (safe)
    if os.getenv("FSM_ROUTER_LOG_PAYLOAD", "0") == "1":
        logger.info("FSM_ROUTER POST url=%s payload=%s", url, payload)

    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            resp = await client.post(url, json=payload)

            if log_details and t0 is not None:
                dt_ms = int((__import__('time').perf_counter() - t0) * 1000)
                logger.info("FSM_ROUTER_HTTP status=%s dt_ms=%s bytes=%s", resp.status_code, dt_ms, len(resp.content or b""))

            if resp.status_code >= 400:
                logger.error(
                    "FSM_ROUTER failed: status=%s body=%s",
                    resp.status_code,
                    resp.text[:2000],
                )

            resp.raise_for_status()
            data = resp.json()
            if log_details and isinstance(data, dict):
                try:
                    fsm = data.get('fsm') if isinstance(data.get('fsm'), dict) else {}
                    backend_call = fsm.get('backend_call') if isinstance(fsm, dict) else None
                    bc_type = backend_call.get('type') if isinstance(backend_call, dict) else None
                    gmail = data.get('gmail') if isinstance(data.get('gmail'), dict) else None
                    used_acct = gmail.get('used_account_id') if isinstance(gmail, dict) else None
                    logger.info(
                        "FSM_ROUTER_END keys=%s spoken_len=%s backend_call=%s used_account_id=%s",
                        sorted(list(data.keys())),
                        len((data.get('spoken_reply') or '')),
                        bc_type,
                        used_acct,
                    )
                except Exception:
                    logger.exception("FSM_ROUTER_END logging failed")
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

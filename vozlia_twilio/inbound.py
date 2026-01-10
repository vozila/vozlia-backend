from __future__ import annotations

import os
from typing import Optional

from fastapi import APIRouter, Request
from fastapi.responses import Response
from twilio.twiml.voice_response import VoiceResponse, Connect

from core.logging import logger
from db import SessionLocal
from services.user_service import get_or_create_primary_user

router = APIRouter()


def _resolve_tenant_id(to_number: Optional[str]) -> Optional[str]:
    """Resolve tenant_id for this inbound call.

    Current behavior (safe default):
      1) If VOZLIA_TENANT_ID or TENANT_ID is set, use it.
      2) Else fall back to the deterministic primary user (ADMIN_EMAIL) ID.

    Future (multi-tenant):
      - Replace this with a lookup by `to_number` (dialed DID → tenant).
      - Keep the Twilio <Stream><Parameter> approach so /twilio/stream always receives it.
    """
    env_tenant = (os.getenv("VOZLIA_TENANT_ID") or os.getenv("TENANT_ID") or "").strip()
    if env_tenant:
        return env_tenant

    db = SessionLocal()
    try:
        user = get_or_create_primary_user(db)
        return str(user.id)
    finally:
        db.close()


@router.post("/twilio/inbound")
async def twilio_inbound(request: Request) -> Response:
    """Twilio inbound → TwiML (Media Streams).

    IMPORTANT:
    - We pass (tenant_id, caller_id) into Media Streams customParameters so /twilio/stream
      always knows identity at finalize time (stop/disconnect), without relying on
      best-effort inference.
    - This is NOT in the audio hot path; it runs once at call start.
    """
    form = await request.form()
    from_number = form.get("From")
    to_number = form.get("To")
    call_sid = form.get("CallSid")

    tenant_id = _resolve_tenant_id(str(to_number) if to_number else None)
    caller_id = str(from_number) if from_number else None

    logger.info(
        "Incoming call: tenant_id=%s From=%s To=%s CallSid=%s",
        tenant_id,
        from_number,
        to_number,
        call_sid,
    )

    resp = VoiceResponse()
    connect = Connect()

    stream_url = os.getenv(
        "TWILIO_STREAM_URL",
        "wss://vozlia-backend.onrender.com/twilio/stream",
    )

    stream = connect.stream(url=stream_url)

    # These become start.customParameters in the Twilio Media Stream "start" event
    # (and are available in the WS handler immediately).
    try:
        if tenant_id:
            stream.parameter(name="tenant_id", value=str(tenant_id))
        if caller_id:
            stream.parameter(name="caller_id", value=str(caller_id))

        # Back-compat / helpful debugging
        if from_number:
            stream.parameter(name="from", value=str(from_number))
        if to_number:
            stream.parameter(name="to", value=str(to_number))
        if call_sid:
            stream.parameter(name="call_sid", value=str(call_sid))
    except Exception:
        # Fail-open: the call should still work even if parameters fail
        logger.exception(
            "Failed to attach customParameters to Twilio <Stream>; identity propagation may be degraded."
        )

    resp.append(connect)
    return Response(content=str(resp), media_type="application/xml")

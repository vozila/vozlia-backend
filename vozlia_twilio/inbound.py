from __future__ import annotations

import os

from fastapi import APIRouter, Request
from fastapi.responses import Response
from twilio.twiml.voice_response import VoiceResponse, Connect

from core.logging import logger

router = APIRouter()


@router.post("/twilio/inbound")
async def twilio_inbound(request: Request) -> Response:
    """
    Twilio inbound → TwiML.

    IMPORTANT:
    - We pass caller identity into Media Streams customParameters so /twilio/stream
      can forward it to /assistant/route and enable caller-scoped Postgres TTL cache.
    - This is NOT in the audio hot path; it runs once at call start.
    """
    form = await request.form()
    from_number = form.get("From")
    to_number = form.get("To")
    call_sid = form.get("CallSid")

    logger.info("Incoming call: From=%s To=%s CallSid=%s", from_number, to_number, call_sid)

    resp = VoiceResponse()
    connect = Connect()

    stream_url = os.getenv(
        "TWILIO_STREAM_URL",
        "wss://vozlia-backend.onrender.com/twilio/stream",
    )

    stream = connect.stream(url=stream_url)

    # These become start.customParameters in the Twilio Media Stream "start" event
    try:
        if from_number:
            stream.parameter(name="from", value=str(from_number))
        if to_number:
            stream.parameter(name="to", value=str(to_number))
        if call_sid:
            stream.parameter(name="call_sid", value=str(call_sid))
    except Exception:
        # Fail-open: the call should still work even if parameters fail
        logger.exception("Failed to attach customParameters to Twilio <Stream>; caller cache may be disabled.")

    resp.append(connect)
    return Response(content=str(resp), media_type="application/xml")

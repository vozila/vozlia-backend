
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
    Behavior should remain identical to the previous inline handler in main.py.
    """
    form = await request.form()
    from_number = form.get("From")
    to_number = form.get("To")
    call_sid = form.get("CallSid")

    logger.info(f"Incoming call: From={from_number}, To={to_number}, CallSid={call_sid}")

    resp = VoiceResponse()
    connect = Connect()

    stream_url = os.getenv(
        "TWILIO_STREAM_URL",
        "wss://vozlia-backend.onrender.com/twilio/stream",
    )
    connect.stream(url=stream_url)
    resp.append(connect)

    xml = str(resp)
    return Response(content=xml, media_type="application/xml")

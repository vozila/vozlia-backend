from fastapi import FastAPI, Request
from fastapi.responses import Response

from twilio.twiml.voice_response import VoiceResponse

app = FastAPI()


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/twilio/inbound")
async def twilio_inbound(request: Request):
    """
    This endpoint will be called by Twilio whenever someone calls your number.
    For now it just plays a simple message using TwiML.
    """
    form = await request.form()
    from_number = form.get("From")
    to_number = form.get("To")

    # Build TwiML response
    vr = VoiceResponse()
    vr.say(
        "Hi, you’ve reached Vozlia, your A.I. voice assistant. "
        "This is a test call to confirm the system is online.",
        voice="alice",
        language="en-US"
    )
    vr.pause(length=1)
    vr.say("Goodbye for now.")
    
    twiml_str = str(vr)

    # Twilio expects XML with the proper content type
    return Response(content=twiml_str, media_type="application/xml")


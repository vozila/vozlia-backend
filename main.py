import os
import json
import asyncio
import logging

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import PlainTextResponse, Response, JSONResponse

from twilio.twiml.voice_response import VoiceResponse, Connect, Stream as TwilioStream

from openai import OpenAI
import websockets

# -------------------------------------------------------------------
# Logging
# -------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vozlia")
logger.setLevel(logging.INFO)

# -------------------------------------------------------------------
# OpenAI configuration
# -------------------------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY is not set. GPT / Realtime calls will fail.")

client = OpenAI(api_key=OPENAI_API_KEY)

# Realtime model & WS URL
OPENAI_REALTIME_MODEL = os.getenv(
    "OPENAI_REALTIME_MODEL",
    "gpt-4o-mini-realtime-preview-2024-12-17",  # adjust if needed
)
OPENAI_REALTIME_URL = (
    f"wss://api.openai.com/v1/realtime?model={OPENAI_REALTIME_MODEL}"
)

OPENAI_REALTIME_HEADERS = {
    "Authorization": f"Bearer {OPENAI_API_KEY}" if OPENAI_API_KEY else "",
    "OpenAI-Beta": "realtime=v1",
}

VOICE_NAME = os.getenv("OPENAI_REALTIME_VOICE", "alloy")

SYSTEM_PROMPT = (
    "You are Vozlia, a friendly, efficient AI phone assistant. "
    "You are talking to callers over a standard phone line. "
    "Keep your answers concise, clear, and conversational. "
    "You can ask clarifying questions, but avoid long monologues."
)

# -------------------------------------------------------------------
# FastAPI app
# -------------------------------------------------------------------
app = FastAPI()


# -------------------------------------------------------------------
# Simple health / root endpoints
# -------------------------------------------------------------------
@app.get("/")
async def root():
    return PlainTextResponse("OK")


@app.get("/health")
async def health():
    return {"status": "ok"}


# -------------------------------------------------------------------
# Debug GPT endpoint (text → GPT → text)
# -------------------------------------------------------------------
async def generate_gpt_reply(text: str) -> str:
    """
    Simple helper that uses chat.completions (non-realtime) to test GPT.
    """
    logger.info(f"/debug/gpt called with text: {text!r}")

    if not OPENAI_API_KEY:
        return "OpenAI API key is not configured on the server."

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are Vozlia, a helpful AI assistant."},
                {"role": "user", "content": text},
            ],
        )
        return resp.choices[0].message.content
    except Exception as e:
        logger.error(f"Error calling OpenAI chat.completions: {e}")
        return f"Error talking to GPT: {e}"


@app.get("/debug/gpt")
async def debug_gpt(text: str = "Hello Vozlia"):
    reply = await generate_gpt_reply(text)
    return JSONResponse({"reply": reply})


# -------------------------------------------------------------------
# Twilio inbound voice webhook → returns TwiML
# -------------------------------------------------------------------
@app.post("/twilio/inbound")
async def twilio_inbound(request: Request):
    """
    Called by Twilio when someone dials your Vozlia number.
    We:
      1. Play a short greeting.
      2. Connect the call to a Twilio Media Stream pointing
         at our WebSocket: /twilio/stream
    """
    form = await request.form()
    from_number = form.get("From")
    to_number = form.get("To")
    call_sid = form.get("CallSid")

    logger.info(
        f"Incoming call: From={from_number}, To={to_number}, CallSid={call_sid}"
    )

    resp = VoiceResponse()

    # Short greeting before we switch to streaming mode
    resp.say(
        "Hi, this is your Vozlia A.I. assistant. "
        "Please hold for a moment while I connect you.",
        voice="alice",
        language="en-US",
    )

    # Media Stream → our WebSocket endpoint
    connect = Connect()
    # If you ever change your backend domain, update TWILIO_STREAM_URL in env
    stream_url = os.getenv(
        "TWILIO_STREAM_URL",
        "wss://vozlia-backend.onrender.com/twilio/stream",
    )
    connect.stream(url=stream_url)
    resp.append(connect)

    xml = str(resp)
    logger.debug(f"Generated TwiML:\n{xml}")

    return Response(content=xml, media_type="application/xml")


# -------------------------------------------------------------------
# Helper: create & configure an OpenAI Realtime session
# -------------------------------------------------------------------
async def create_realtime_session():
    """
    Opens a WebSocket to the OpenAI Realtime API and sends a session.update
    so it understands it's talking over a phone line (g711_ulaw).
    """
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set; cannot use Realtime API.")

    logger.info("Connecting to OpenAI Realtime WebSocket...")
    openai_ws = await websockets.connect(
        OPENAI_REALTIME_URL,
        extra_headers=OPENAI_REALTIME_HEADERS,
    )

    # Configure the session: audio in/out over g711_ulaw, server-side VAD, etc.
    session_update = {
        "type": "session.update",
        "session": {
            "instructions": SYSTEM_PROMPT,
            "voice": VOICE_NAME,
            "modalities": ["text", "audio"],
            "input_audio_format": "g711_ulaw",
            "output_audio_format": "g711_ulaw",
            "turn_detection": {"type": "server_vad"},
            # Optional: ask it to also transcribe the audio
            "input_audio_transcription": {
                "model": "gpt-4o-mini-transcribe"
            },
            "temperature": 0.5,
        },
    }

    await openai_ws.send(json.dumps(session_update))
    logger.info("Sent session.update to OpenAI Realtime")

    return openai_ws


# -------------------------------------------------------------------
# Twilio Media Stream WebSocket ↔ OpenAI Realtime WebSocket
# -------------------------------------------------------------------
@app.websocket("/twilio/stream")
async def twilio_stream(websocket: WebSocket):
    """
    Twilio connects here via Media Streams (WebSocket).

    We:
      * Accept the WebSocket from Twilio.
      * Open a second WebSocket to OpenAI Realtime.
      * Forward Twilio audio (g711_ulaw in base64) → OpenAI input_audio_buffer.append
      * Forward OpenAI audio → Twilio media events (so caller hears GPT live).
    """
    await websocket.accept()
    logger.info("Twilio media stream connected")

    openai_ws = None

    try:
        # Open connection to OpenAI Realtime API
        openai_ws = await create_realtime_session()

        async def twilio_to_openai():
            """
            Read events from Twilio → push audio into OpenAI's input buffer.
            """
            while True:
                try:
                    msg_text = await websocket.receive_text()
                except WebSocketDisconnect:
                    logger.info("Twilio WebSocket disconnected")
                    break

                try:
                    data = json.loads(msg_text)
                except json.JSONDecodeError:
                    logger.warning(f"Non-JSON message from Twilio: {msg_text!r}")
                    continue

                event_type = data.get("event")
                logger.info(f"Twilio stream event: {event_type}")

                if event_type == "start":
                    # You can store streamSid if needed
                    stream_sid = data.get("start", {}).get("streamSid")
                    logger.info(f"Stream started: {stream_sid}")
                elif event_type == "media":
                    # Twilio gives base64-encoded g711_ulaw audio
                    payload_b64 = data.get("media", {}).get("payload")
                    if not payload_b64:
                        continue

                    # Send into the Realtime input audio buffer
                    audio_event = {
                        "type": "input_audio_buffer.append",
                        "audio": payload_b64,
                    }
                    await openai_ws.send(json.dumps(audio_event))

                elif event_type == "stop":
                    logger.info("Twilio sent stop; closing call.")
                    # Let OpenAI know we're done with this turn; VAD may also handle it
                    try:
                        await openai_ws.send(
                            json.dumps({"type": "input_audio_buffer.commit"})
                        )
                    except Exception as e:
                        logger.warning(
                            f"Error sending commit to OpenAI on stop: {e}"
                        )
                    break

        async def openai_to_twilio():
            """
            Read events from OpenAI → send audio back to Twilio as media events.
            """
            async for msg in openai_ws:
                try:
                    event = json.loads(msg)
                except json.JSONDecodeError:
                    logger.warning(f"Non-JSON message from OpenAI: {msg!r}")
                    continue

                etype = event.get("type")

                # When audio is being streamed back, you'll see audio delta chunks
                if etype == "response.audio.delta":
                    audio_chunk_b64 = event.get("delta")
                    if not audio_chunk_b64:
                        continue

                    twilio_msg = {
                        "event": "media",
                        "media": {
                            "payload": audio_chunk_b64,
                        },
                    }
                    await websocket.send_text(json.dumps(twilio_msg))

                elif etype == "response.audio.done":
                    # One spoken response finished
                    logger.info("OpenAI finished an audio response")

                elif etype == "response.text.delta":
                    # Optional: log partial text
                    delta = event.get("delta", "")
                    if delta:
                        logger.info(f"AI (text delta): {delta}")

                elif etype == "response.text.done":
                    text = event.get("text", "")
                    if text:
                        logger.info(f"AI full text response: {text}")

                elif etype == "error":
                    logger.error(f"OpenAI error event: {event}")
                    break

        # Run both directions concurrently
        await asyncio.gather(
            twilio_to_openai(),
            openai_to_twilio(),
        )

    except Exception as e:
        logger.exception(f"Error in /twilio/stream: {e}")
    finally:
        # Clean up OpenAI WebSocket
        if openai_ws is not None:
            try:
                await openai_ws.close()
            except Exception:
                pass

        # Close Twilio WebSocket
        try:
            await websocket.close()
        except Exception:
            pass

        logger.info("Twilio stream closed")

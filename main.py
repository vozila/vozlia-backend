import os
import json
import base64
import asyncio
import logging
from dataclasses import dataclass
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
import websockets

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s:%(name)s:%(message)s",
)
logger = logging.getLogger("vozlia")

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_REALTIME_URL = os.getenv(
    "OPENAI_REALTIME_URL",
    "wss://api.openai.com/v1/realtime?model=gpt-4.1-realtime-preview",
)

INITIAL_SYSTEM_MESSAGE = (
    "You are Vozlia, a friendly voice AI receptionist for a small business. "
    "Keep answers concise, speak clearly, and pause often so callers can interrupt. "
    "If the caller interrupts you, immediately stop speaking and listen."
)

# -----------------------------------------------------------------------------
# Call state
# -----------------------------------------------------------------------------
@dataclass
class CallState:
    stream_sid: Optional[str] = None
    barge_in_enabled: bool = False
    suppress_assistant_audio: bool = False
    cancel_in_progress: bool = False
    # True between 'we requested a new response' and first audio chunk of that response
    awaiting_new_response_audio: bool = False


# -----------------------------------------------------------------------------
# FastAPI app
# -----------------------------------------------------------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def health():
    return {"status": "ok", "service": "vozlia-backend"}


# -----------------------------------------------------------------------------
# Twilio inbound webhook → start media stream
# -----------------------------------------------------------------------------
@app.post("/twilio/inbound")
async def twilio_inbound(request: Request):
    body = await request.body()
    logger.info("Incoming Twilio webhook: %s", body.decode(errors="ignore"))

    # Twilio will open a Media Stream WebSocket to /twilio/stream
    # The URL below must match your deployed backend URL.
    ws_url = os.getenv(
        "TWILIO_STREAM_URL",
        "wss://vozlia-backend.onrender.com/twilio/stream",
    )

    twiml = f"""
<Response>
  <Connect>
    <Stream url="{ws_url}" />
  </Connect>
</Response>
""".strip()

    return PlainTextResponse(content=twiml, media_type="text/xml")


# -----------------------------------------------------------------------------
# Core bridge: Twilio media stream ↔ OpenAI Realtime
# -----------------------------------------------------------------------------
async def connect_to_openai():
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set")

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "OpenAI-Beta": "realtime=v1",
    }

    return await websockets.connect(OPENAI_REALTIME_URL, extra_headers=headers)


async def handle_twilio_stream(websocket: WebSocket):
    await websocket.accept()
    logger.info("Twilio media stream connected")

    state = CallState()

    # Connect to OpenAI Realtime
    openai_ws = await connect_to_openai()
    logger.info("OpenAI Realtime WebSocket connection open")

    async def send_to_openai(event: dict):
        await openai_ws.send(json.dumps(event))

    async def send_to_twilio_audio(audio_bytes: bytes):
        # Twilio expects base64-encoded audio payload
        if not state.stream_sid:
            return
        payload = base64.b64encode(audio_bytes).decode("ascii")
        msg = {
            "event": "media",
            "streamSid": state.stream_sid,
            "media": {"payload": payload},
        }
        await websocket.send_text(json.dumps(msg))

    # ------------------------ OpenAI session setup ---------------------------
    session_update = {
        "type": "session.update",
        "session": {
            "instructions": INITIAL_SYSTEM_MESSAGE,
            "modalities": ["audio"],
            "input_audio_format": "mulaw",
            "output_audio_format": "mulaw",
            "voice": "coral",
            "turn_detection": {
                "type": "server_vad",
                "threshold": 0.45,
                "prefix_padding_ms": 300,
                "silence_duration_ms": 600,
            },
        },
    }
    await send_to_openai(session_update)

    # Initial greeting (no barge-in on this first response)
    await send_to_openai({"type": "response.create"})
    logger.info("Sent initial greeting request to OpenAI Realtime")

    # -------------------------- Bridge tasks --------------------------------
    async def twilio_reader():
        try:
            while True:
                message = await websocket.receive_text()
                data = json.loads(message)
                event_type = data.get("event")

                if event_type == "start":
                    state.stream_sid = data.get("streamSid")
                    logger.info("Twilio stream event: start (sid=%s)", state.stream_sid)

                elif event_type == "media":
                    media = data.get("media", {})
                    payload = media.get("payload")
                    if not payload:
                        continue
                    # Audio from Twilio is base64-encoded mulaw at 8kHz; forward to OpenAI
                    audio_bytes = base64.b64decode(payload)
                    await send_to_openai(
                        {
                            "type": "input_audio_buffer.append",
                            "audio": base64.b64encode(audio_bytes).decode("ascii"),
                        }
                    )

                elif event_type == "stop":
                    logger.info("Twilio stream event: stop")
                    break

        except WebSocketDisconnect:
            logger.info("Twilio WebSocket disconnected (reader)")
        except Exception as e:
            logger.exception("Error in twilio_reader: %s", e)

        # Tell OpenAI the call is done (safe to ignore errors here)
        try:
            await send_to_openai({"type": "input_audio_buffer.commit"})
        except Exception:
            pass

    async def openai_reader():
        try:
            async for raw in openai_ws:
                try:
                    event = json.loads(raw)
                except json.JSONDecodeError:
                    logger.warning("Non-JSON event from OpenAI, ignoring")
                    continue

                etype = event.get("type")

                # ----------------- Audio out to Twilio -----------------
                if etype == "response.audio.delta":
                    delta = event.get("delta", {})
                    audio_b64 = delta.get("audio")
                    if not audio_b64:
                        continue

                    # Barge-in suppression: drop chunks from an old, canceled response
                    if state.suppress_assistant_audio:
                        if state.awaiting_new_response_audio:
                            # First chunk of *new* response after barge-in; allow audio again
                            state.suppress_assistant_audio = False
                            state.awaiting_new_response_audio = False
                            logger.info("Resuming assistant audio after barge-in.")
                        else:
                            # Still draining old response audio, drop it
                            continue

                    audio_bytes = base64.b64decode(audio_b64)
                    await send_to_twilio_audio(audio_bytes)

                # --------------- Response lifecycle events ---------------
                elif etype == "response.created":
                    logger.info("OpenAI created a response.")

                elif etype == "response.completed":
                    logger.info("OpenAI finished an audio response")
                    if not state.barge_in_enabled:
                        state.barge_in_enabled = True
                        logger.info("Barge-in is now ENABLED for subsequent responses.")
                    state.cancel_in_progress = False

                elif etype == "response.stopped":
                    logger.info("OpenAI response stopped.")
                    state.cancel_in_progress = False

                # ----------------- VAD events (barge-in) ----------------
                elif etype == "input_audio_buffer.speech_started":
                    logger.info("OpenAI VAD: user speech START")
                    if state.barge_in_enabled:
                        # User is speaking while AI may be talking → barge-in
                        state.suppress_assistant_audio = True
                        state.awaiting_new_response_audio = True
                        if not state.cancel_in_progress:
                            try:
                                await send_to_openai({"type": "response.cancel"})
                                state.cancel_in_progress = True
                                logger.info(
                                    "BARGE-IN: user speech started while AI speaking; "
                                    "sent response.cancel and suppressing assistant audio."
                                )
                            except Exception as e:
                                # This will log the 'response_cancel_not_active' errors you saw
                                logger.warning(
                                    "response.cancel failed (likely no active response): %s",
                                    e,
                                )

                elif etype == "input_audio_buffer.speech_stopped":
                    logger.info("OpenAI VAD: user speech STOP")
                    # Commit user's utterance and trigger a new response
                    try:
                        await send_to_openai({"type": "input_audio_buffer.commit"})
                        await send_to_openai({"type": "response.create"})
                        state.awaiting_new_response_audio = True
                        state.cancel_in_progress = False
                        logger.info("Committed user audio and requested new response.")
                    except Exception as e:
                        logger.warning(
                            "Failed to commit buffer / create response after VAD stop: %s",
                            e,
                        )

                # --------------- Error events from OpenAI ---------------
                elif etype == "error":
                    err = event.get("error", {})
                    logger.error("OpenAI error event: %s", err)

        except Exception as e:
            logger.exception("Error in openai_reader: %s", e)

    async def cleanup():
        try:
            await openai_ws.close()
        except Exception:
            pass
        try:
            await websocket.close()
        except Exception:
            pass

    # Run both readers until one finishes
    await asyncio.gather(twilio_reader(), openai_reader(), return_exceptions=True)
    await cleanup()


@app.websocket("/twilio/stream")
async def twilio_stream(websocket: WebSocket):
    await handle_twilio_stream(websocket)

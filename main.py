import os
import json
import base64
import logging
import asyncio
from dataclasses import dataclass
from typing import Optional

import websockets
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import JSONResponse, PlainTextResponse, Response
from fastapi.middleware.cors import CORSMiddleware

# ------------------------------------------------------------------------------
# Config & logging
# ------------------------------------------------------------------------------

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY environment variable is not set")

# Use the correct realtime-capable model
OPENAI_REALTIME_URL = (
    "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview"
)

TWILIO_STREAM_URL = os.getenv(
    "TWILIO_STREAM_URL",
    "wss://vozlia-backend.onrender.com/twilio/stream",
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s:%(name)s:%(message)s",
)
logger = logging.getLogger("vozlia")

# ------------------------------------------------------------------------------
# State
# ------------------------------------------------------------------------------

@dataclass
class CallState:
    stream_sid: Optional[str] = None
    is_ai_speaking: bool = False
    last_response_id: Optional[str] = None
    openai_ready: bool = False


# ------------------------------------------------------------------------------
# FastAPI app
# ------------------------------------------------------------------------------

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)


@app.get("/")
async def root():
    return {"status": "ok", "service": "vozlia-backend"}


@app.get("/health")
async def health():
    return JSONResponse({"status": "healthy"})


# ------------------------------------------------------------------------------
# Twilio inbound: return TwiML pointing to our /twilio/stream WebSocket
# ------------------------------------------------------------------------------

@app.post("/twilio/inbound")
async def twilio_inbound(request: Request) -> Response:
    # We log the raw params just to help debugging
    form = await request.body()
    logger.info("Incoming Twilio webhook: %s", form.decode("utf-8", errors="ignore"))

    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Say voice="alice">Connecting you to your AI assistant.</Say>
  <Connect>
    <Stream url="{TWILIO_STREAM_URL}" />
  </Connect>
</Response>"""

    return Response(content=twiml, media_type="text/xml")


# ------------------------------------------------------------------------------
# Helper to send events to OpenAI
# ------------------------------------------------------------------------------

async def send_to_openai(openai_ws, event: dict):
    try:
        await openai_ws.send(json.dumps(event))
    except Exception as e:
        logger.error("Failed to send to OpenAI: %s | event=%s", e, event)


# ------------------------------------------------------------------------------
# Twilio reader: forward caller audio to OpenAI; do amplitude-based barge-in
# ------------------------------------------------------------------------------

async def twilio_reader(
    twilio_ws: WebSocket,
    openai_ws,
    state: CallState,
):
    """
    Read media events from Twilio and forward them to OpenAI as input_audio_buffer.append.
    When AI is speaking, use a simple amplitude heuristic to trigger barge-in.
    """
    try:
        while True:
            msg_text = await twilio_ws.receive_text()
            data = json.loads(msg_text)
            event_type = data.get("event")

            if event_type == "start":
                start_info = data.get("start", {})
                state.stream_sid = start_info.get("streamSid")
                logger.info("Twilio stream start: sid=%s", state.stream_sid)

            elif event_type == "media":
                media = data.get("media", {})
                payload_b64 = media.get("payload")
                if not payload_b64:
                    continue

                # Simple amplitude check on μ-law bytes
                try:
                    decoded = base64.b64decode(payload_b64)
                except Exception:
                    decoded = b""

                if state.is_ai_speaking and decoded:
                    # Treat bytes as centered around 128; this is crude but works
                    max_dev = max(abs(b - 128) for b in decoded)
                    # Threshold tuned to avoid noise but catch real speech
                    if max_dev > 18:
                        logger.info(
                            "BARGE-IN (Twilio amplitude): user speech detected while "
                            "AI speaking; sent response.cancel and suppressed assistant audio"
                        )
                        # Cancel any active response (if none, OpenAI returns a benign error)
                        await send_to_openai(
                            openai_ws,
                            {"type": "response.cancel"},
                        )
                        state.is_ai_speaking = False

                # Forward audio to OpenAI
                await send_to_openai(
                    openai_ws,
                    {
                        "type": "input_audio_buffer.append",
                        "audio": payload_b64,
                    },
                )

            elif event_type == "stop":
                logger.info("Twilio stream event: stop")
                break

            else:
                # Other events (mark, etc.) can be ignored or logged
                logger.debug("Twilio event (ignored): %s", data)

    except WebSocketDisconnect:
        logger.info("Twilio WebSocket disconnected")
    except Exception as e:
        logger.error("Error in twilio_reader: %s", e)


# ------------------------------------------------------------------------------
# OpenAI reader: stream assistant audio to Twilio; handle VAD-based barge-in
# ------------------------------------------------------------------------------

async def openai_reader(
    twilio_ws: WebSocket,
    openai_ws,
    state: CallState,
):
    """
    Read events from OpenAI Realtime and forward audio deltas to Twilio.
    Also handle server-side VAD events for barge-in.
    """
    try:
        async for raw in openai_ws:
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                logger.error("Failed to decode OpenAI event: %s", raw)
                continue

            etype = event.get("type")

            # --- Errors from OpenAI ---
            if etype == "error":
                logger.error("OpenAI error event: %s", event)
                continue

            # --- Response lifecycle ---
            if etype == "response.created":
                resp = event.get("response", {})
                state.last_response_id = resp.get("id")
                logger.info("OpenAI response.created: id=%s", state.last_response_id)

            elif etype in ("response.output_audio.delta", "response.audio.delta"):
                # Mark that AI is currently speaking
                state.is_ai_speaking = True

                # Realtime audio delta: use "delta" if present, else "audio"
                audio_b64 = event.get("delta") or event.get("audio")
                if not audio_b64:
                    continue

                # Send audio to Twilio as media
                try:
                    await twilio_ws.send_text(
                        json.dumps(
                            {
                                "event": "media",
                                "media": {
                                    "payload": audio_b64,
                                },
                            }
                        )
                    )
                except Exception as e:
                    logger.error("Failed to send audio to Twilio: %s", e)

            elif etype in ("response.completed", "response.stopped", "response.canceled"):
                # AI is done speaking
                logger.info("OpenAI %s", etype)
                state.is_ai_speaking = False

            # --- Server-side VAD: speech detection on input buffer ---
            elif etype == "input_audio_buffer.speech_started":
                logger.info("OpenAI VAD: user speech START")
                # Barge-in based only on VAD (no amplitude check)
                if state.is_ai_speaking:
                    logger.info(
                        "BARGE-IN (VAD only): user speech started while AI speaking; "
                        "sent response.cancel and suppressed audio"
                    )
                    await send_to_openai(openai_ws, {"type": "response.cancel"})
                    state.is_ai_speaking = False

            elif etype == "input_audio_buffer.speech_stopped":
                logger.info("OpenAI VAD: user speech STOP")

            else:
                # We log other event types at debug level
                logger.debug("OpenAI event: %s", event)

    except websockets.exceptions.ConnectionClosedError as e:
        logger.error("OpenAI connection closed: %s", e)
    except Exception as e:
        logger.error("Error in openai_reader: %s", e)


# ------------------------------------------------------------------------------
# WebSocket endpoint: /twilio/stream
# ------------------------------------------------------------------------------

@app.websocket("/twilio/stream")
async def twilio_stream(ws: WebSocket):
    """
    This endpoint is used by Twilio Media Streams. We:
      - Accept Twilio WebSocket
      - Connect to OpenAI Realtime
      - Wire up two tasks: twilio_reader <-> openai_reader
      - Configure the Realtime session with audio + VAD + voice
      - Send an initial greeting response
    """
    await ws.accept()
    logger.info("Twilio media stream connected")

    state = CallState()

    # Connect to OpenAI Realtime
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "OpenAI-Beta": "realtime=v1",
    }

    try:
        async with websockets.connect(
            OPENAI_REALTIME_URL,
            extra_headers=headers,
        ) as openai_ws:
            logger.info("OpenAI Realtime WebSocket connection open")

            # Configure the Realtime session
            session_update = {
                "type": "session.update",
                "session": {
                    # Model comes from URL, but it's safe to repeat:
                    "model": "gpt-4o-realtime-preview",
                    "modalities": ["audio", "text"],
                    "voice": "coral",
                    "input_audio_format": "g711_ulaw",
                    "output_audio_format": "g711_ulaw",
                    "turn_detection": {
                        "type": "server_vad",
                        # Play with these if barge-in feels off:
                        "threshold": 0.6,
                        "silence_duration_ms": 700,
                    },
                    "instructions": (
                        "You are Vozlia, an AI receptionist and assistant. "
                        "Keep your responses short and conversational. "
                        "Pause frequently to allow the caller to interrupt you. "
                        "If the caller starts speaking, stop talking immediately and listen. "
                        "Never lecture; use one or two sentences at a time."
                    ),
                },
            }
            await send_to_openai(openai_ws, session_update)

            # Initial greeting
            greeting = {
                "type": "response.create",
                "response": {
                    "instructions": (
                        "Greet the caller briefly as Vozlia, the AI assistant, "
                        "and ask how you can help. Use one or two short sentences."
                    )
                },
            }
            await send_to_openai(openai_ws, greeting)

            # Start bidirectional piping
            twilio_task = asyncio.create_task(twilio_reader(ws, openai_ws, state))
            openai_task = asyncio.create_task(openai_reader(ws, openai_ws, state))

            done, pending = await asyncio.wait(
                {twilio_task, openai_task},
                return_when=asyncio.FIRST_EXCEPTION,
            )

            # If either side errors, cancel the other
            for task in pending:
                task.cancel()

    except Exception as e:
        logger.error("Fatal error in /twilio/stream: %s", e)
    finally:
        try:
            await ws.close()
        except Exception:
            pass
        logger.info("Twilio /twilio/stream connection closed")

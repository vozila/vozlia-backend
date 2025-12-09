import os
import json
import asyncio
import logging
import base64
import time

from fastapi import (
    FastAPI,
    Request,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse

import websockets
from openai import OpenAI

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vozlia")
logger.setLevel(logging.INFO)

# ---------- FastAPI app ----------
app = FastAPI()

# CORS (adjust allow_origins for prod as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Health check ----------
@app.get("/health", response_class=PlainTextResponse)
async def health():
    return "OK"

# ---------- OpenAI Realtime config ----------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY is not set. GPT / Realtime calls will fail.")

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

OPENAI_REALTIME_MODEL = os.getenv(
    "OPENAI_REALTIME_MODEL",
    "gpt-4o-mini-realtime-preview-2024-12-17",
)
OPENAI_REALTIME_URL = f"wss://api.openai.com/v1/realtime?model={OPENAI_REALTIME_MODEL}"

OPENAI_REALTIME_HEADERS = {
    "Authorization": f"Bearer {OPENAI_API_KEY}" if OPENAI_API_KEY else "",
    "OpenAI-Beta": "realtime=v1",
}

SUPPORTED_VOICES = {
    "coral": "coral",
    "alloy": "alloy",
}

VOICE_NAME = os.getenv("VOZLIA_VOICE", "coral")
if VOICE_NAME not in SUPPORTED_VOICES:
    VOICE_NAME = "coral"

SYSTEM_PROMPT = os.getenv(
    "VOZLIA_SYSTEM_PROMPT",
    (
        "You are Vozlia, a calm, competent, AI-powered assistant on a phone call. "
        "You speak clearly, concisely, and naturally. "
        "You are allowed to ask clarifying questions when needed, but avoid rambling. "
        "Do not guess about facts you are uncertain about; say what you *can* do. "
        "You are talking to a real human on a call, so be polite and easy to interrupt."
    ),
)

async def create_realtime_session():
    """
    Create an OpenAI Realtime WebSocket session and send a session.update
    configuring audio + transcription.
    """
    logger.info("Connecting to OpenAI Realtime WebSocket via websockets...")

    openai_ws = await websockets.connect(
        OPENAI_REALTIME_URL,
        extra_headers=OPENAI_REALTIME_HEADERS,
    )

    session_update = {
        "type": "session.update",
        "session": {
            "instructions": SYSTEM_PROMPT,
            "voice": VOICE_NAME,
            "modalities": ["audio", "text"],
            "input_audio_format": "g711_ulaw",
            "output_audio_format": "g711_ulaw",
            "turn_detection": {
                "type": "server_vad",
            },
            "input_audio_transcription": {
                "model": "gpt-4o-mini-transcribe",
            },
        },
    }

    await openai_ws.send(json.dumps(session_update))
    logger.info("Sent session.update to OpenAI Realtime (with transcription enabled)")

    return openai_ws

# ---------- Twilio inbound → TwiML ----------
@app.post("/twilio/inbound")
async def twilio_inbound(request: Request):
    """
    Twilio hits this first. We respond with TwiML instructing it to open a
    Media Stream WebSocket to /twilio/stream.
    """
    form = await request.form()
    from_number = form.get("From")
    to_number = form.get("To")

    logger.info(f"Incoming call from {from_number} to {to_number}")

    # IMPORTANT: update the URL below to match your Render hostname if needed.
    host = request.url.hostname or "localhost"

    twiml = f"""
<Response>
    <Start>
        <Stream url="wss://{host}/twilio/stream" />
    </Start>
    <Say>Connecting you to Vozlia, your AI assistant.</Say>
    <Pause length="60" />
</Response>
    """.strip()

    return PlainTextResponse(twiml, media_type="text/xml")

# ---------- Twilio Media Stream WebSocket ----------
FRAME_MS = 20
SAMPLE_RATE = 8000
BYTES_PER_FRAME = int(SAMPLE_RATE * (FRAME_MS / 1000.0))

PREBUFFER_FRAMES = 5
PREBUFFER_BYTES = PREBUFFER_FRAMES * BYTES_PER_FRAME

MAX_TWILIO_BACKLOG_SECONDS = 1.0

@app.websocket("/twilio/stream")
async def twilio_stream(websocket: WebSocket):
    await websocket.accept()
    logger.info("Twilio media stream connected")

    openai_ws = None
    stream_sid = None

    barge_in_enabled = False
    cancel_in_progress = False

    user_speaking_vad = False

    audio_buffer = bytearray()
    assistant_last_audio_time = 0.0

    prebuffer_active = False

    call_transcripts: list[str] = []

    def assistant_actively_speaking() -> bool:
        if audio_buffer:
            return True
        return (time.monotonic() - assistant_last_audio_time) < 0.6

    try:
        openai_ws = await create_realtime_session()

        async def twilio_to_openai():
            nonlocal stream_sid, openai_ws
            while True:
                try:
                    msg_text = await websocket.receive_text()
                except WebSocketDisconnect:
                    logger.info("Twilio WebSocket disconnected")
                    break
                except Exception as e:
                    logger.error(f"Error receiving from Twilio WS: {e}")
                    break

                try:
                    data = json.loads(msg_text)
                except json.JSONDecodeError:
                    logger.warning(f"Non-JSON payload from Twilio: {msg_text!r}")
                    continue

                event = data.get("event")

                if event == "start":
                    stream_sid = data.get("streamSid")
                    logger.info(f"Twilio stream started (streamSid={stream_sid})")
                elif event == "media":
                    if not openai_ws:
                        continue

                    chunk = data["media"]["payload"]
                    audio_bytes = base64.b64decode(chunk)

                    base64_ulaw = base64.b64encode(audio_bytes).decode("ascii")
                    realtime_event = {
                        "type": "input_audio_buffer.append",
                        "audio": base64_ulaw,
                    }
                    await openai_ws.send(json.dumps(realtime_event))
                elif event == "stop":
                    logger.info("Twilio stream stopped by Twilio")
                    break

        async def openai_to_twilio():
            nonlocal stream_sid, barge_in_enabled, cancel_in_progress
            nonlocal user_speaking_vad, assistant_last_audio_time, audio_buffer, prebuffer_active, call_transcripts

            async for msg in openai_ws:
                try:
                    event = json.loads(msg)
                except json.JSONDecodeError:
                    logger.warning(f"Non-JSON message from OpenAI: {msg!r}")
                    continue

                etype = event.get("type")

                if etype == "response.audio.delta":
                    audio_chunk_b64 = event.get("delta")
                    if not audio_chunk_b64:
                        continue

                    try:
                        raw_bytes = base64.b64decode(audio_chunk_b64)
                    except Exception as e:
                        logger.error(f"Error decoding audio delta: {e}")
                        continue

                    if len(audio_buffer) == 0:
                        prebuffer_active = True

                    audio_buffer.extend(raw_bytes)

                elif etype == "response.audio.done":
                    logger.info("OpenAI finished an audio response.")
                    assistant_last_audio_time = time.monotonic()
                    barge_in_enabled = True
                    prebuffer_active = False

                elif etype == "response.text.delta":
                    text_delta = event.get("delta", "")
                    if text_delta:
                        logger.info(f"AI text delta: {text_delta}")

                elif etype == "response.text.done":
                    text = event.get("text", "")
                    if text:
                        logger.info(f"AI full text response: {text}")

                elif etype == "conversation.item.input_audio_transcription.completed":
                    transcript = event.get("transcript")
                    item_id = event.get("item_id")
                    if transcript:
                        logger.info(f"[ASR] User said (item_id={item_id}): {transcript!r}")
                        call_transcripts.append(transcript)

                elif etype == "input_audio_buffer.speech_started":
                    user_speaking_vad = True
                    logger.info("OpenAI VAD: user speech START")

                    if (
                        barge_in_enabled
                        and assistant_actively_speaking()
                        and not cancel_in_progress
                    ):
                        try:
                            await openai_ws.send(
                                json.dumps({"type": "response.cancel"})
                            )
                            cancel_in_progress = True

                            audio_buffer.clear()
                            logger.info(
                                "BARGE-IN: user speech started while AI speaking; "
                                "sent response.cancel and cleared audio buffer."
                            )
                        except Exception as e:
                            logger.error(f"Error sending response.cancel: {e}")

                elif etype == "input_audio_buffer.speech_stopped":
                    user_speaking_vad = False
                    logger.info("OpenAI VAD: user speech STOP")

                    try:
                        await openai_ws.send(json.dumps({"type": "response.create"}))
                        logger.info("Sent response.create after VAD speech stop")
                    except Exception as e:
                        logger.error(f"Error sending response.create: {e}")

                elif etype == "error":
                    logger.error(f"OpenAI error event: {event}")
                    err = event.get("error") or {}
                    code = err.get("code")

                    if code == "response_cancel_not_active":
                        cancel_in_progress = False

        async def twilio_audio_sender():
            nonlocal audio_buffer, assistant_last_audio_time, prebuffer_active

            frame_interval = FRAME_MS / 1000.0
            next_send_time = time.monotonic()
            frames_sent = 0

            while True:
                now = time.monotonic()

                if not prebuffer_active and audio_buffer:
                    backlog_seconds = frames_sent * frame_interval - (
                        now - assistant_last_audio_time
                    )
                    if backlog_seconds > MAX_TWILIO_BACKLOG_SECONDS:
                        drop_frames = int(
                            (backlog_seconds - MAX_TWILIO_BACKLOG_SECONDS)
                            / frame_interval
                        )
                        drop_bytes = drop_frames * BYTES_PER_FRAME
                        audio_buffer = audio_buffer[drop_bytes:]
                        logger.info(
                            f"Dropping {drop_frames} frames from audio buffer to "
                            f"keep backlog around {MAX_TWILIO_BACKLOG_SECONDS}s"
                        )

                if not audio_buffer:
                    await asyncio.sleep(0.005)
                    continue

                if prebuffer_active and len(audio_buffer) < PREBUFFER_BYTES:
                    await asyncio.sleep(0.005)
                    continue

                if now < next_send_time:
                    await asyncio.sleep(next_send_time - now)
                    continue

                frame_bytes = audio_buffer[:BYTES_PER_FRAME]
                audio_buffer = audio_buffer[BYTES_PER_FRAME:]

                try:
                    frame_b64 = base64.b64encode(frame_bytes).decode("ascii")
                    twilio_msg = json.dumps(
                        {
                            "event": "media",
                            "streamSid": stream_sid,
                            "media": {"payload": frame_b64},
                        }
                    )
                    await websocket.send_text(twilio_msg)
                    frames_sent += 1
                except WebSocketDisconnect:
                    logger.info("WebSocket disconnected while sending audio")
                    return
                except Exception as e:
                    logger.error(f"Error sending audio frame to Twilio: {e}")

                next_send_time += frame_interval

        await asyncio.gather(
            twilio_to_openai(),
            openai_to_twilio(),
            twilio_audio_sender(),
        )

    except WebSocketDisconnect:
        logger.info("Twilio WebSocket disconnected (outer)")
    except Exception as e:
        logger.error(f"Unhandled error in twilio_stream: {e}")
    finally:
        if openai_ws:
            try:
                await openai_ws.close()
            except Exception:
                pass

        try:
            await websocket.close()
        except Exception:
            pass

        logger.info("Twilio stream closed")

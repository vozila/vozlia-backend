import os
import json
import asyncio
import logging
import base64
import time

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import PlainTextResponse, Response, JSONResponse

from twilio.twiml.voice_response import VoiceResponse, Connect

from openai import OpenAI
import websockets

from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from uuid import UUID
from db import Base, engine
from models import User, EmailAccount
from schemas import EmailAccountCreate, EmailAccountRead
from deps import get_db

# Create tables if not using Alembic yet
Base.metadata.create_all(bind=engine)

app = FastAPI()


# TODO: replace with real auth
def get_current_user(db: Session) -> User:
    # For now, just fetch first user or create a dummy
    user = db.query(User).first()
    if not user:
        user = User(email="demo@vozlia.com")
        db.add(user)
        db.commit()
        db.refresh(user)
    return user


@app.get("/email/accounts", response_model=List[EmailAccountRead])
def list_email_accounts(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    accounts = (
        db.query(EmailAccount)
        .filter(EmailAccount.user_id == current_user.id)
        .all()
    )
    return accounts


@app.post("/email/accounts", response_model=EmailAccountRead)
def create_email_account(
    payload: EmailAccountCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    # Example: if provider_type == "imap_custom" and payload.password is set,
    # encrypt the password before saving:
    password_enc = None
    if payload.provider_type == "imap_custom" and payload.password:
        from cryptography.fernet import Fernet

        key = os.getenv("ENCRYPTION_KEY")
        if not key:
            raise HTTPException(status_code=500, detail="ENCRYPTION_KEY not configured")

        f = Fernet(key.encode() if not key.startswith("gAAAA") else key)
        password_enc = f.encrypt(payload.password.encode()).decode()

    account = EmailAccount(
        user_id=current_user.id,
        provider_type=payload.provider_type,
        email_address=str(payload.email_address),
        display_name=payload.display_name,
        is_primary=payload.is_primary,
        is_active=payload.is_active,
        imap_host=payload.imap_host,
        imap_port=payload.imap_port,
        imap_ssl=payload.imap_ssl,
        smtp_host=payload.smtp_host,
        smtp_port=payload.smtp_port,
        smtp_ssl=payload.smtp_ssl,
        username=payload.username,
        password_enc=password_enc,
    )

    db.add(account)
    db.commit()
    db.refresh(account)

    return account


# ---------- Logging ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vozlia")
logger.setLevel(logging.INFO)

# ---------- OpenAI config ----------
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
    "alloy",
    "ash",
    "ballad",
    "coral",
    "echo",
    "sage",
    "shimmer",
    "verse",
    "marin",
    "cedar",
}

voice_env = os.getenv("OPENAI_REALTIME_VOICE", "coral")
if voice_env not in SUPPORTED_VOICES:
    logger.warning(
        f"Unsupported voice '{voice_env}' for Realtime. "
        "Falling back to 'coral'. Supported voices: "
        + ", ".join(sorted(SUPPORTED_VOICES))
    )
    voice_env = "coral"

VOICE_NAME = voice_env

SYSTEM_PROMPT = (
 "You are Vozlia, a warm, friendly, highly capable AI phone assistant using the "
    "Coral voice persona. Speak naturally, confidently, and in a professional tone, "
    "similar to a helpful customer support agent. "
    "You greet callers immediately at the start of the call, introduce yourself as "
    "Vozlia, and invite them to describe what they need help with. "
    "Keep your responses very concise: usually no more than 2–3 short sentences "
    "(around 5–7 seconds of speech) before pausing. "
    "If the caller asks for a long explanation or story, summarize the key points "
    "briefly and offer to go deeper only if they ask. "
    "Be attentive to interruptions: if the caller starts speaking while you are "
    "talking, immediately stop and listen. "
    "Your goal is to make callers feel welcome, understood, and supported."
)

# ---------- Audio framing for G.711 μ-law ----------
SAMPLE_RATE = 8000        # Hz
FRAME_MS = 20             # 20 ms per frame
BYTES_PER_FRAME = int(SAMPLE_RATE * FRAME_MS / 1000)  # 160 bytes

# Real-time pacing: one 20ms frame every 20ms
FRAME_INTERVAL = FRAME_MS / 1000.0  # 0.020 seconds

# Prebuffer at start of each utterance to smooth jitter
# 4 frames = 80 ms
PREBUFFER_FRAMES = 4
PREBUFFER_BYTES = PREBUFFER_FRAMES * BYTES_PER_FRAME

# Limit how far ahead we've sent audio to Twilio (in seconds)
# This directly bounds the maximum "tail" after barge-in.
MAX_TWILIO_BACKLOG_SECONDS = 1.0


# ---------- FastAPI app ----------
app = FastAPI()


@app.get("/")
async def root():
    return PlainTextResponse("OK")


@app.get("/health")
async def health():
    return {"status": "ok"}


# ---------- Debug GPT (text only) ----------
async def generate_gpt_reply(text: str) -> str:
    logger.info(f"/debug/gpt called with text: {text!r}")

    if not OPENAI_API_KEY or client is None:
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


# ---------- Twilio inbound → TwiML ----------
@app.post("/twilio/inbound")
async def twilio_inbound(request: Request):
    form = await request.form()
    from_number = form.get("From")
    to_number = form.get("To")
    call_sid = form.get("CallSid")

    logger.info(
        f"Incoming call: From={from_number}, To={to_number}, CallSid={call_sid}"
    )

    resp = VoiceResponse()
    connect = Connect()

    stream_url = os.getenv(
        "TWILIO_STREAM_URL",
        "wss://vozlia-backend.onrender.com/twilio/stream",
    )
    connect.stream(url=stream_url)
    resp.append(connect)

    xml = str(resp)
    logger.debug(f"Generated TwiML:\n{xml}")

    return Response(content=xml, media_type="application/xml")


# ---------- Helper: initial greeting via Realtime ----------
async def send_initial_greeting(openai_ws):
    try:
        convo_item = {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "Greet the caller as Vozlia, a friendly AI phone assistant using "
                            "the Coral voice. Briefly introduce yourself and invite them to "
                            "say what they need help with."
                        ),
                    }
                ],
            },
        }
        await openai_ws.send(json.dumps(convo_item))
        await openai_ws.send(json.dumps({"type": "response.create"}))
        logger.info("Sent initial greeting request to OpenAI Realtime")
    except Exception as e:
        logger.error(f"Error sending initial greeting: {e}")


# ---------- Helper: OpenAI Realtime session via websockets ----------
async def create_realtime_session():
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set; cannot use Realtime API.")

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
            "modalities": ["text", "audio"],
            "input_audio_format": "g711_ulaw",
            "output_audio_format": "g711_ulaw",
            "turn_detection": {
                "type": "server_vad",
            },
        },
    }

    await openai_ws.send(json.dumps(session_update))
    logger.info("Sent session.update to OpenAI Realtime")

    await send_initial_greeting(openai_ws)

    return openai_ws


# ---------- Twilio media stream ↔ OpenAI Realtime ----------
@app.websocket("/twilio/stream")
async def twilio_stream(websocket: WebSocket):
    await websocket.accept()
    logger.info("Twilio media stream connected")

    openai_ws = None
    stream_sid = None

    # AI / call state
    barge_in_enabled = False  # after first full response
    cancel_in_progress = False

    # Server VAD-based user speech flag
    user_speaking_vad = False

    # Outgoing assistant audio buffer (raw μ-law bytes)
    audio_buffer = bytearray()
    assistant_last_audio_time = 0.0

    # Prebuffer state: we hold back sending until we have PREBUFFER_BYTES
    prebuffer_active = False

    def assistant_actively_speaking() -> bool:
        # If there's buffered audio, or we've sent audio very recently,
        # treat the assistant as actively speaking.
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

                try:
                    data = json.loads(msg_text)
                except json.JSONDecodeError:
                    logger.warning(f"Non-JSON message from Twilio: {msg_text!r}")
                    continue

                event_type = data.get("event")

                if event_type != "media":
                    logger.info(f"Twilio stream event: {event_type}")

                if event_type == "start":
                    stream_sid = data.get("start", {}).get("streamSid")
                    logger.info(f"Stream started: {stream_sid}")

                elif event_type == "media":
                    payload_b64 = data.get("media", {}).get("payload")
                    if not payload_b64:
                        continue

                    # Always forward μ-law audio to OpenAI
                    audio_event = {
                        "type": "input_audio_buffer.append",
                        "audio": payload_b64,
                    }
                    await openai_ws.send(json.dumps(audio_event))

                elif event_type == "stop":
                    logger.info("Twilio sent stop; closing call.")
                    break

                elif event_type == "connected":
                    logger.info("Twilio reports call connected")

        async def openai_to_twilio():
            nonlocal stream_sid, barge_in_enabled, cancel_in_progress
            nonlocal user_speaking_vad, assistant_last_audio_time, audio_buffer, prebuffer_active

            async for msg in openai_ws:
                try:
                    event = json.loads(msg)
                except json.JSONDecodeError:
                    logger.warning(f"Non-JSON message from OpenAI: {msg!r}")
                    continue

                etype = event.get("type")

                # ----- AUDIO FROM OPENAI → BUFFER FOR TWILIO -----
                if etype == "response.audio.delta":
                    audio_chunk_b64 = event.get("delta")
                    if not audio_chunk_b64:
                        continue

                    try:
                        raw_bytes = base64.b64decode(audio_chunk_b64)
                    except Exception as e:
                        logger.error(f"Error decoding audio delta: {e}")
                        continue

                    # If we were idle (no buffered audio), this is a new utterance:
                    if len(audio_buffer) == 0:
                        prebuffer_active = True

                    audio_buffer.extend(raw_bytes)

                elif etype == "response.audio.done":
                    logger.info("OpenAI finished an audio response")
                    cancel_in_progress = False
                    # After the first full response, allow barge-in.
                    if not barge_in_enabled:
                        barge_in_enabled = True
                        logger.info("Barge-in is now ENABLED for subsequent responses.")

                # ----- TEXT LOGGING (optional) -----
                elif etype == "response.text.delta":
                    delta = event.get("delta", "")
                    if delta:
                        logger.info(f"AI (text delta): {delta}")

                elif etype == "response.text.done":
                    text = event.get("text", "")
                    if text:
                        logger.info(f"AI full text response: {text}")

                # ----- VAD EVENTS: drive user_speaking_vad & barge-in -----
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

                            # Drop any queued assistant audio so Twilio tail is minimal
                            audio_buffer.clear()
                            prebuffer_active = False

                            logger.info(
                                "BARGE-IN: user speech started while AI speaking; "
                                "sent response.cancel and cleared audio buffer."
                            )
                        except Exception as e:
                            logger.error(f"Error sending response.cancel: {e}")

                elif etype == "input_audio_buffer.speech_stopped":
                    user_speaking_vad = False
                    logger.info("OpenAI VAD: user speech STOP")

                # ----- ERROR EVENTS -----
                elif etype == "error":
                    logger.error(f"OpenAI error event: {event}")
                    err = event.get("error") or {}
                    code = err.get("code")

                    if code == "response_cancel_not_active":
                        # harmless: we tried to cancel after it already finished
                        cancel_in_progress = False

        async def twilio_audio_sender():
            """
            Convert the continuous μ-law audio buffer into fixed 20 ms frames
            and send them to Twilio at a steady cadence, with:
              - a small prebuffer per utterance to avoid choppiness
              - an explicit cap on how far ahead we've sent audio (backlog)
            """
            nonlocal assistant_last_audio_time, stream_sid, audio_buffer, prebuffer_active

            # Real-time pacing variables
            next_send_time = time.monotonic()
            call_start_time = time.monotonic()
            frames_sent = 0

            try:
                while True:
                    now = time.monotonic()

                    # Compute how far ahead Twilio is scheduled to play vs. wall clock.
                    audio_sent_duration = frames_sent * (FRAME_MS / 1000.0)
                    call_elapsed = max(0.0, now - call_start_time)
                    backlog_seconds = audio_sent_duration - call_elapsed

                    if backlog_seconds > MAX_TWILIO_BACKLOG_SECONDS:
                        # We've sent too far ahead. Pause sending and let real time catch up.
                        await asyncio.sleep(0.01)
                        # Reset next_send_time to "now" so we don't try to rush later.
                        next_send_time = time.monotonic()
                        continue

                    if stream_sid and len(audio_buffer) >= BYTES_PER_FRAME:
                        # If we're still prebuffering this utterance, wait until we have enough
                        if prebuffer_active and len(audio_buffer) < PREBUFFER_BYTES:
                            await asyncio.sleep(0.005)
                            continue
                        else:
                            prebuffer_active = False

                        # Take the next 20ms frame
                        frame_bytes = audio_buffer[:BYTES_PER_FRAME]
                        del audio_buffer[:BYTES_PER_FRAME]

                        frame_b64 = base64.b64encode(frame_bytes).decode("utf-8")

                        twilio_msg = {
                            "event": "media",
                            "streamSid": stream_sid,
                            "media": {"payload": frame_b64},
                        }

                        try:
                            await websocket.send_text(json.dumps(twilio_msg))
                            assistant_last_audio_time = time.monotonic()
                            frames_sent += 1
                        except WebSocketDisconnect:
                            logger.info("WebSocket disconnected while sending audio")
                            return
                        except Exception as e:
                            logger.error(f"Error sending audio frame to Twilio: {e}")

                        # Schedule next frame at real-time
                        next_send_time += FRAME_INTERVAL
                        sleep_for = max(0.0, next_send_time - time.monotonic())
                        await asyncio.sleep(sleep_for)
                    else:
                        # Nothing (or not enough) to send, avoid busy loop and
                        # keep next_send_time aligned with "now"
                        next_send_time = time.monotonic()
                        await asyncio.sleep(0.005)
            except asyncio.CancelledError:
                logger.info("twilio_audio_sender cancelled")
            except Exception as e:
                logger.exception(f"Error in twilio_audio_sender: {e}")

        await asyncio.gather(
            twilio_to_openai(),
            openai_to_twilio(),
            twilio_audio_sender(),
        )

    except Exception as e:
        logger.exception(f"Error in /twilio/stream: {e}")
    finally:
        if openai_ws is not None:
            try:
                await openai_ws.close()
            except Exception:
                pass

        try:
            await websocket.close()
        except Exception:
            pass

        logger.info("Twilio stream closed")

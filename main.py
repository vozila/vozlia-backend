import os
import json
import asyncio
import logging
import base64
import time
from typing import List
from datetime import datetime, timedelta

import httpx  # <-- for Google OAuth + Gmail API

from fastapi import (
    FastAPI,
    Request,
    WebSocket,
    WebSocketDisconnect,
    Depends,
    HTTPException,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, RedirectResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

import websockets

from openai import OpenAI

from db import SessionLocal
from deps import get_db
from models import GmailAccount
from schemas import GmailAccountCreate, GmailAccountRead

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vozlia")
logger.setLevel(logging.INFO)

# ---------- FastAPI app ----------
app = FastAPI()

# CORS (if you’re calling from a browser-based landing page or admin UI)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten for prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- Internal brain wiring (call /debug/brain from inside the app) ----------

VOICE_USER_ID_CACHE: str | None = None


def _get_internal_base_url() -> str:
    """
    Internal base URL to call this same service from inside the container.
    On Render, uvicorn listens on PORT (e.g. 10000) on 0.0.0.0.
    """
    port = os.getenv("PORT", "8000")
    return f"http://127.0.0.1:{port}"


async def get_voice_user_id() -> str:
    """
    Resolve which user_id to use for phone calls.

    Strategy:
      1) If VOZLIA_VOICE_USER_ID env var is set, use that.
      2) Otherwise, call /debug/me once and cache the resulting user id.
    """
    global VOICE_USER_ID_CACHE

    if VOICE_USER_ID_CACHE:
        return VOICE_USER_ID_CACHE

    env_user_id = os.getenv("VOZLIA_VOICE_USER_ID")
    if env_user_id:
        VOICE_USER_ID_CACHE = env_user_id
        return env_user_id

    internal_base = _get_internal_base_url()
    url = f"{internal_base}/debug/me"

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(url)
        resp.raise_for_status()
        data = resp.json()
        user_id = str(data.get("id") or data.get("user_id") or "voice-demo-user")
        VOICE_USER_ID_CACHE = user_id
        logger.info(f"Resolved voice user_id={user_id!r} via /debug/me")
        return user_id
    except Exception as e:
        logger.error(f"Error resolving voice user id via /debug/me: {e}")
        fallback_id = "voice-demo-user"
        VOICE_USER_ID_CACHE = fallback_id
        return fallback_id


async def run_brain_on_transcript(transcript: str) -> str | None:
    """
    Call the existing /debug/brain endpoint with the transcript text and
    return the 'speech' string if any.

    NOTE: In the current PROD version, we’re no longer using this inside the
    Twilio <-> OpenAI loop. It’s kept here only for future LAB experiments.
    """
    internal_base = _get_internal_base_url()
    url = f"{internal_base}/debug/brain"

    try:
        user_id = await get_voice_user_id()
    except Exception as e:
        logger.error(f"Error obtaining voice user id: {e}")
        return None

    payload = {
        "user_id": user_id,
        "transcript": transcript,
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(url, json=payload)
        if resp.status_code != 200:
            logger.error(f"/debug/brain HTTP {resp.status_code}: {resp.text}")
            return None

        data = resp.json()
        speech = data.get("speech")
        if not isinstance(speech, str) or not speech.strip():
            logger.warning(f"/debug/brain response missing 'speech': {data}")
            return None

        logger.info(f"Brain returned speech: {speech!r}")
        return speech.strip()
    except Exception as e:
        logger.error(f"Error calling /debug/brain: {e}")
        return None


# ---------- Health check ----------
@app.get("/health", response_class=PlainTextResponse)
async def health():
    return "OK"


# ---------- Database dependency ----------
def get_db_sync():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ---------- Gmail account models / schemas ----------

class GmailAuthInitResponse(BaseModel):
    auth_url: str


class GmailAuthCallbackRequest(BaseModel):
    code: str
    state: str


# ---------- Google OAuth config ----------

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
GOOGLE_REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI")

GOOGLE_AUTH_BASE_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_GMAIL_SCOPE = "https://www.googleapis.com/auth/gmail.readonly"

if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET or not GOOGLE_REDIRECT_URI:
    logger.warning(
        "Google OAuth env vars are not fully set. "
        "Gmail account linking will not work."
    )


def build_google_auth_url(state: str) -> str:
    from urllib.parse import urlencode

    params = {
        "client_id": GOOGLE_CLIENT_ID,
        "redirect_uri": GOOGLE_REDIRECT_URI,
        "response_type": "code",
        "scope": GOOGLE_GMAIL_SCOPE,
        "access_type": "offline",
        "prompt": "consent",
        "state": state,
    }
    return f"{GOOGLE_AUTH_BASE_URL}?{urlencode(params)}"


async def exchange_code_for_tokens(code: str) -> dict:
    data = {
        "code": code,
        "client_id": GOOGLE_CLIENT_ID,
        "client_secret": GOOGLE_CLIENT_SECRET,
        "redirect_uri": GOOGLE_REDIRECT_URI,
        "grant_type": "authorization_code",
    }
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(GOOGLE_TOKEN_URL, data=data)
    resp.raise_for_status()
    return resp.json()


async def refresh_google_tokens(refresh_token: str) -> dict:
    data = {
        "refresh_token": refresh_token,
        "client_id": GOOGLE_CLIENT_ID,
        "client_secret": GOOGLE_CLIENT_SECRET,
        "grant_type": "refresh_token",
    }
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(GOOGLE_TOKEN_URL, data=data)
    resp.raise_for_status()
    return resp.json()


# ---------- Gmail account CRUD ----------

@app.post("/email/accounts", response_model=GmailAccountRead)
async def create_gmail_account(
    payload: GmailAccountCreate,
    db: Session = Depends(get_db_sync),
):
    """
    Create a Gmail account record with encrypted refresh token and SMTP settings.
    """
    if not payload.refresh_token:
        raise HTTPException(status_code=400, detail="Missing refresh_token")

    from cryptography.fernet import Fernet

    key = os.getenv("ENCRYPTION_KEY")
    if not key:
        raise HTTPException(
            status_code=500,
            detail="ENCRYPTION_KEY not set in environment.",
        )

    f = Fernet(key.encode("utf-8"))
    password_enc = f.encrypt(payload.password.encode("utf-8"))

    account = GmailAccount(
        email_address=payload.email_address,
        refresh_token=payload.refresh_token,
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


# ---------- Google OAuth: start flow ----------
@app.get("/auth/google/start", response_model=GmailAuthInitResponse)
async def google_auth_start():
    """
    Return URL that the user should visit to authorize Gmail access.
    """
    if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET or not GOOGLE_REDIRECT_URI:
        raise HTTPException(
            status_code=500,
            detail="Google OAuth is not configured.",
        )

    state = "dummy-state"  # TODO: generate and persist a CSRF-safe state
    url = build_google_auth_url(state)
    return GmailAuthInitResponse(auth_url=url)


# ---------- Google OAuth: callback ----------
@app.get("/auth/google/callback")
async def google_auth_callback(request: Request):
    """
    Handle the Google OAuth callback. In a real app, you'd:
      - Verify the 'state'
      - Exchange 'code' for tokens
      - Store tokens mapped to the correct user
    """
    code = request.query_params.get("code")
    state = request.query_params.get("state")

    if not code:
        raise HTTPException(status_code=400, detail="Missing 'code' in query params")

    try:
        token_data = await exchange_code_for_tokens(code)
    except httpx.HTTPError as e:
        logger.error(f"Error exchanging code for tokens: {e}")
        raise HTTPException(status_code=500, detail="Token exchange failed")

    logger.info(f"Google token data: {token_data}")

    return RedirectResponse(url="/auth/success")  # or wherever your UI lives


# ---------- Gmail simple queries ----------

async def get_gmail_headers(access_token: str, max_results: int = 10) -> List[dict]:
    url = "https://www.googleapis.com/gmail/v1/users/me/messages"
    params = {"maxResults": max_results}
    headers = {"Authorization": f"Bearer {access_token}"}

    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(url, params=params, headers=headers)
    resp.raise_for_status()
    data = resp.json()
    return data.get("messages", [])


@app.get("/email/accounts/{account_id}/summary")
async def gmail_summary(
    account_id: int,
    max_results: int = 10,
    query: str | None = None,
    db: Session = Depends(get_db_sync),
):
    """
    Very simple Gmail summary endpoint:
      - Looks up GmailAccount by ID.
      - Refreshes tokens if needed.
      - Calls Gmail API to fetch recent messages.
    """
    account: GmailAccount | None = db.query(GmailAccount).filter(
        GmailAccount.id == account_id
    ).first()

    if not account:
        raise HTTPException(status_code=404, detail="Account not found")

    from cryptography.fernet import Fernet

    key = os.getenv("ENCRYPTION_KEY")
    if not key:
        raise HTTPException(
            status_code=500,
            detail="ENCRYPTION_KEY not set in environment.",
        )

    f = Fernet(key.encode("utf-8"))
    # Here we only need the refresh_token; password is used for SMTP elsewhere
    refresh_token = account.refresh_token

    try:
        token_data = await refresh_google_tokens(refresh_token)
    except httpx.HTTPError as e:
        logger.error(f"Error refreshing Google tokens: {e}")
        raise HTTPException(status_code=500, detail="Failed to refresh token")

    access_token = token_data.get("access_token")
    if not access_token:
        raise HTTPException(status_code=500, detail="No access_token in token_data")

    headers_list = await get_gmail_headers(access_token, max_results=max_results)

    return {"account_id": account_id, "email_address": account.email_address, "messages": headers_list}


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

    # IMPORTANT:
    #  - modalities MUST be ["audio", "text"] if you want audio.
    #  - input/output formats stay g711_ulaw for Twilio.
    #  - input_audio_transcription only takes a 'model' field (no 'enabled').
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

    twiml = f"""
<Response>
    <Start>
        <Stream url="wss://{request.url.hostname}/twilio/stream" />
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

# Prebuffer ~100ms (5 frames) of audio before sending to Twilio
PREBUFFER_FRAMES = 5
PREBUFFER_BYTES = PREBUFFER_FRAMES * BYTES_PER_FRAME

# Limit how far ahead we've sent audio to Twilio (in seconds)
# This directly bounds the maximum "tail" after barge-in.
MAX_TWILIO_BACKLOG_SECONDS = 1.0


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

    # Per-call transcript history (last few utterances)
    call_transcripts: list[str] = []

    def assistant_actively_speaking() -> bool:
        # If there's buffered audio, or
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

                # ----- USER TRANSCRIPTS (ASR from caller audio) -----
                elif etype == "conversation.item.input_audio_transcription.completed":
                    transcript = event.get("transcript")
                    item_id = event.get("item_id")
                    if transcript:
                        logger.info(
                            f"[ASR] User said (item_id={item_id}): {transcript!r}"
                        )
                        # Keep a short rolling history; useful for logging or future NLU
                        call_transcripts.append(transcript)

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
                            logger.info(
                                "BARGE-IN: user speech started while AI speaking; "
                                "sent response.cancel and cleared audio buffer."
                            )
                        except Exception as e:
                            logger.error(
                                f"Error sending response.cancel: {e}"
                            )

                elif etype == "input_audio_buffer.speech_stopped":
                    user_speaking_vad = False
                    logger.info("OpenAI VAD: user speech STOP")

                    # Trigger a model response when the user finishes speaking
                    try:
                        await openai_ws.send(json.dumps({"type": "response.create"}))
                        logger.info("Sent response.create after VAD speech stop")
                    except Exception as e:
                        logger.error(f"Error sending response.create: {e}")

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
            Convert the continuous μ-law audio_buffer into 20ms frames and
            send them to Twilio, respecting a small prebuffer.
            """
            nonlocal audio_buffer, assistant_last_audio_time, prebuffer_active

            frame_interval = FRAME_MS / 1000.0
            next_send_time = time.monotonic()

            frames_sent = 0

            while True:
                now = time.monotonic()

                if not prebuffer_active and audio_buffer:
                    # If Twilio is too far ahead, drop frames.
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

                # Take the next 20ms frame
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

                # Schedule next frame at real-time
                next_send_time += frame_interval

        # Run them concurrently
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

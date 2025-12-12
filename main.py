import os
import json
import time
import base64
import asyncio
import logging
from typing import Optional, Dict, Any

import httpx
import websockets
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import PlainTextResponse

# -------------------------
# Logging
# -------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vozlia")

# -------------------------
# Env / Config
# -------------------------
PORT = int(os.getenv("PORT", "10000"))

# IMPORTANT: must be your public https base (Render URL), no trailing slash
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "https://vozlia-backend.onrender.com").rstrip("/")

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "")
ELEVENLABS_MODEL_ID = os.getenv("ELEVENLABS_MODEL_ID", "eleven_multilingual_v2")

FSM_ROUTER_URL = os.getenv("FSM_ROUTER_URL", f"{PUBLIC_BASE_URL}/assistant/route").rstrip("/")

TWILIO_SAMPLE_RATE = 8000
TWILIO_CODEC = "mulaw"

DEEPGRAM_MODEL = os.getenv("DEEPGRAM_MODEL", "nova-2-phonecall")
DEEPGRAM_LANGUAGE = os.getenv("DEEPGRAM_LANGUAGE", "en-US")

# Transcript handling
FINAL_UTTERANCE_COOLDOWN_MS = int(os.getenv("FINAL_UTTERANCE_COOLDOWN_MS", "450"))
MIN_FINAL_CHARS = int(os.getenv("MIN_FINAL_CHARS", "2"))

# -------------------------
# App
# -------------------------
app = FastAPI()


@app.get("/")
def root():
    return {"ok": True, "service": "vozlia-backend", "pipeline": "twilio->deepgram->fsm->elevenlabs"}


@app.get("/healthz")
def healthz():
    return {"ok": True}


# -------------------------
# Twilio inbound (TwiML)
# -------------------------
@app.post("/twilio/inbound")
async def twilio_inbound(request: Request):
    """
    Guaranteed-audio greeting FIRST (Twilio Say), then start Media Stream.
    This prevents dead-silence even if ElevenLabs fails.
    """
    form = await request.form()
    from_num = form.get("From")
    to_num = form.get("To")
    call_sid = form.get("CallSid")
    logger.info(f"Incoming call: From={from_num}, To={to_num}, CallSid={call_sid}")

    # Twilio requires wss. Render is https, so we convert:
    stream_url = f"{PUBLIC_BASE_URL.replace('https://', 'wss://')}/twilio/stream"

    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Say voice="alice">Hi, you’ve reached Vozlia. One moment while I connect.</Say>
  <Connect>
    <Stream url="{stream_url}" />
  </Connect>
</Response>
"""
    return PlainTextResponse(twiml, media_type="application/xml")


# -------------------------
# Helpers
# -------------------------
def _b64_to_bytes(b64: str) -> bytes:
    return base64.b64decode(b64.encode("utf-8"))


def _bytes_to_b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode("utf-8")


async def call_fsm_router(text: str, meta: Optional[Dict[str, Any]] = None) -> str:
    """
    Calls your unified router (FSM/skills). Must return a string to speak.
    If router is unavailable, fall back gracefully.
    """
    payload = {"text": text, "meta": meta or {}}

    try:
        async with httpx.AsyncClient(timeout=25.0) as client:
            r = await client.post(FSM_ROUTER_URL, json=payload)
            r.raise_for_status()
            data = r.json()

        # Accept common fields
        for key in ("spoken_reply", "reply", "text", "message"):
            if isinstance(data, dict) and isinstance(data.get(key), str) and data[key].strip():
                return data[key].strip()

        # If it's just a string JSON
        if isinstance(data, str) and data.strip():
            return data.strip()

        return "Okay. What would you like to do next?"

    except Exception as e:
        logger.exception(f"FSM router call failed: {e}")
        return "I’m having trouble reaching my skills service right now. Please try again."


async def elevenlabs_tts_ulaw_bytes(text: str) -> bytes:
    """
    Returns μ-law 8k audio bytes suitable for Twilio Media Streams.
    """
    url = (
        f"https://api.elevenlabs.io/v1/text-to-speech/"
        f"{ELEVENLABS_VOICE_ID}?output_format=ulaw_8000"
    )

    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json",
        "Accept": "audio/ulaw",
    }

    body = {
        "text": text,
        "model_id": ELEVENLABS_MODEL_ID,
        "voice_settings": {
            "stability": 0.4,
            "similarity_boost": 0.75,
        },
    }

    async with httpx.AsyncClient(timeout=45.0) as client:
        r = await client.post(url, headers=headers, json=body)
        r.raise_for_status()

        logger.info(
            f"ElevenLabs OK: content-type={r.headers.get('content-type')} "
            f"bytes={len(r.content)} first10={r.content[:10]}"
        )

        return r.content



async def stream_ulaw_audio_to_twilio(
    twilio_ws: WebSocket,
    stream_sid: str,
    ulaw_audio: bytes,
    cancel_event: asyncio.Event,
):
    """
    Sends ulaw bytes to Twilio in small frames.
    Can be cancelled via cancel_event.
    """
    # Twilio expects small chunks; 20ms @ 8k = 160 bytes for ulaw
    frame_size = 160
    idx = 0
    while idx < len(ulaw_audio):
        if cancel_event.is_set():
            return
        chunk = ulaw_audio[idx : idx + frame_size]
        idx += frame_size

        msg = {
            "event": "media",
            "streamSid": stream_sid,
            "media": {"payload": _bytes_to_b64(chunk)},
        }
        await twilio_ws.send_text(json.dumps(msg))
        #await asyncio.sleep(0.02)  # pace ~20ms


async def twilio_clear(twilio_ws: WebSocket, stream_sid: str):
    """
    Clears queued audio on Twilio side (barge-in).
    """
    try:
        await twilio_ws.send_text(json.dumps({"event": "clear", "streamSid": stream_sid}))
    except Exception:
        logger.exception("Failed to send Twilio clear")


def deepgram_ws_url() -> str:
    # Deepgram realtime endpoint
    # encoding=mulaw&sample_rate=8000 matches Twilio MediaStreams payload
    params = (
        f"model={DEEPGRAM_MODEL}"
        f"&language={DEEPGRAM_LANGUAGE}"
        f"&encoding=mulaw"
        f"&sample_rate={TWILIO_SAMPLE_RATE}"
        f"&punctuate=true"
        f"&interim_results=true"
        f"&endpointing=300"
        f"&smart_format=true"
    )
    return f"wss://api.deepgram.com/v1/listen?{params}"


# -------------------------
# Twilio stream WebSocket
# -------------------------
@app.websocket("/twilio/stream")
async def twilio_stream(websocket: WebSocket):
    await websocket.accept()
    logger.info("Twilio media stream connected")

    stream_sid: Optional[str] = None

    dg_ws = None
    dg_task: Optional[asyncio.Task] = None
    speak_task: Optional[asyncio.Task] = None
    speak_cancel = asyncio.Event()

    # transcript debouncing
    last_final_ts = 0.0
    last_final_text = ""

    # barge-in state
    assistant_speaking = False

    async def cancel_speaking():
        nonlocal speak_task, assistant_speaking
        if speak_task and not speak_task.done():
            speak_cancel.set()
            # Clear Twilio queue immediately
            if stream_sid:
                await twilio_clear(websocket, stream_sid)
            try:
                speak_task.cancel()
            except Exception:
                pass
        assistant_speaking = False
        speak_cancel.clear()

    async def speak(text: str):
        nonlocal speak_task, assistant_speaking
        if not stream_sid:
            logger.warning("speak() called before stream_sid; skipping")
            return

        await cancel_speaking()

        assistant_speaking = True
        speak_cancel.clear()

        async def _run():
            nonlocal assistant_speaking
            try:
                ulaw = await elevenlabs_tts_ulaw_bytes(text)
                await stream_ulaw_audio_to_twilio(websocket, stream_sid, ulaw, speak_cancel)
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.exception(f"ElevenLabs/Twilio speak failed: {e}")
            finally:
                assistant_speaking = False

        speak_task = asyncio.create_task(_run())

    async def deepgram_reader():
        """
        Receives Deepgram transcripts and triggers FSM routing on finals.
        Also triggers barge-in cancellation on user speech (interim).
        """
        nonlocal last_final_ts, last_final_text, assistant_speaking

        try:
            async for msg in dg_ws:
                try:
                    data = json.loads(msg)
                except Exception:
                    continue

                # Deepgram transcript shape: channel.alternatives[0].transcript
                ch = data.get("channel") or {}
                alts = ch.get("alternatives") or []
                if not alts:
                    continue

                transcript = (alts[0].get("transcript") or "").strip()
                if not transcript:
                    continue

                is_final = bool(data.get("is_final"))
                speech_final = bool(data.get("speech_final"))

                # If user starts talking and assistant is speaking -> barge-in cancel
                if assistant_speaking and not is_final:
                    await cancel_speaking()

                if not is_final:
                    continue

                # Final debouncing
                now = time.time()
                if len(transcript) < MIN_FINAL_CHARS:
                    continue

                if transcript == last_final_text and (now - last_final_ts) < 1.0:
                    continue

                # cool-down between finals (to avoid double triggers)
                if (now - last_final_ts) * 1000 < FINAL_UTTERANCE_COOLDOWN_MS:
                    # allow if it's meaningfully different
                    if transcript.lower() in last_final_text.lower():
                        continue

                last_final_text = transcript
                last_final_ts = now

                logger.info(f"Deepgram FINAL: {transcript}")

                # Route to skills/router, then speak reply
                reply = await call_fsm_router(transcript, meta={"stream_sid": stream_sid})
                logger.info(f"Router reply: {reply}")
                await speak(reply)

        except asyncio.CancelledError:
            return
        except Exception as e:
            logger.exception(f"Deepgram reader error: {e}")

    try:
        # Connect to Deepgram immediately
        if not DEEPGRAM_API_KEY:
            logger.error("Missing DEEPGRAM_API_KEY; cannot transcribe")
        else:
            logger.info("Connecting to Deepgram realtime WebSocket...")
            dg_ws = await websockets.connect(
                deepgram_ws_url(),
                extra_headers={"Authorization": f"Token {DEEPGRAM_API_KEY}"},
                ping_interval=10,
                ping_timeout=20,
                close_timeout=5,
                max_size=2**23,
            )
            logger.info("Connected to Deepgram realtime.")
            dg_task = asyncio.create_task(deepgram_reader())

        # Main Twilio loop
        while True:
            raw = await websocket.receive_text()
            evt = json.loads(raw)
            etype = evt.get("event")

            if etype == "connected":
                logger.info("Twilio stream event: connected")
                continue

            if etype == "start":
                stream_sid = evt.get("start", {}).get("streamSid")
                logger.info(f"Twilio stream event: start (streamSid={stream_sid})")

                # IMPORTANT: Say something via ElevenLabs too (optional),
                # so you can confirm TTS streaming works end-to-end.
                # If it fails, logs will show the exact error.
                try:
                    await speak("I’m connected. How can I help you today?")
                except Exception:
                    logger.exception("Initial ElevenLabs greeting failed")

                continue

            if etype == "media":
                # forward Twilio audio -> Deepgram
                if dg_ws is not None:
                    payload_b64 = evt.get("media", {}).get("payload")
                    if payload_b64:
                        try:
                            audio_bytes = _b64_to_bytes(payload_b64)
                            await dg_ws.send(audio_bytes)
                        except Exception:
                            logger.exception("Failed sending audio to Deepgram")
                continue

            if etype == "stop":
                logger.info("Twilio stream event: stop")
                break

    except Exception as e:
        logger.exception(f"Twilio stream handler error: {e}")

    finally:
        # Cleanup
        try:
            await cancel_speaking()
        except Exception:
            pass

        if dg_task:
            dg_task.cancel()
            try:
                await dg_task
            except Exception:
                pass

        if dg_ws:
            try:
                await dg_ws.close()
            except Exception:
                pass

        try:
            await websocket.close()
        except Exception:
            pass

        logger.info("Twilio media stream handler completed")


# -------------------------
# Placeholder router endpoint (optional)
# If you already have /assistant/route elsewhere, you can remove this.
# -------------------------
@app.post("/assistant/route")
async def assistant_route(request: Request):
    body = await request.json()
    text = (body.get("text") or "").strip()
    if not text:
        return {"spoken_reply": "I didn’t catch that. Please say it again."}

    # Minimal fallback behavior (replace with your real FSM)
    if "email" in text.lower():
        return {"spoken_reply": "Email access is connected through the skills router. Tell me which inbox to read."}

    return {"spoken_reply": f"You said: {text}. What would you like to do next?"}

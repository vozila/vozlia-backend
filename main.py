import os
import json
import time
import base64
import asyncio
import logging
import signal
import re
import random
from typing import Optional, Dict, Any, List, Tuple

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
# App
# -------------------------
app = FastAPI()

# -------------------------
# Shared HTTP clients (performance)
# NOTE: Do NOT use http2=True unless you install httpx[http2] (h2).
# -------------------------
openai_client: httpx.AsyncClient | None = None
eleven_client: httpx.AsyncClient | None = None
router_client: httpx.AsyncClient | None = None

# -------------------------
# Per-call state (session memory)
# streamSid -> {"topic": str|None, "last_user": str, "last_bot": str}
# -------------------------
CALL_STATE: dict[str, dict] = {}

# -------------------------
# Env / Config
# -------------------------
PORT = int(os.getenv("PORT", "10000"))
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

FINAL_UTTERANCE_COOLDOWN_MS = int(os.getenv("FINAL_UTTERANCE_COOLDOWN_MS", "450"))
MIN_FINAL_CHARS = int(os.getenv("MIN_FINAL_CHARS", "2"))

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")

# Fallback model when OpenAI returns “reasoning-only” output or intermittent 500s
OPENAI_FALLBACK_MODEL = os.getenv("OPENAI_FALLBACK_MODEL", "gpt-4o-mini")

# Retries/backoff
OPENAI_MAX_RETRIES = int(os.getenv("OPENAI_MAX_RETRIES", "2"))
OPENAI_TIMEOUT_S = float(os.getenv("OPENAI_TIMEOUT_S", "25"))

# Voice agent behavior
VOICE_MAX_TOKENS = int(os.getenv("VOICE_MAX_TOKENS", "240"))
VOICE_ACK_ENABLED = os.getenv("VOICE_ACK_ENABLED", "0").strip() == "1"
VOICE_ACK_TEXT = os.getenv("VOICE_ACK_TEXT", "Okay.")

# -------------------------
# Lifecycle + SIGTERM
# -------------------------
@app.on_event("startup")
async def startup():
    global openai_client, eleven_client, router_client
    logger.info("APP STARTUP")
    openai_client = httpx.AsyncClient(timeout=OPENAI_TIMEOUT_S)
    eleven_client = httpx.AsyncClient(timeout=45.0)
    router_client = httpx.AsyncClient(timeout=25.0)

@app.on_event("shutdown")
async def shutdown():
    global openai_client, eleven_client, router_client
    logger.warning("APP SHUTDOWN (process terminated)")
    if openai_client:
        await openai_client.aclose()
    if eleven_client:
        await eleven_client.aclose()
    if router_client:
        await router_client.aclose()

def _handle_sigterm(*_):
    logger.warning("SIGTERM received")

signal.signal(signal.SIGTERM, _handle_sigterm)

# -------------------------
# Basic endpoints
# -------------------------
@app.get("/")
def root():
    return {"ok": True, "service": "vozlia-backend", "pipeline": "twilio->deepgram->openai->elevenlabs"}

@app.get("/healthz")
def healthz():
    return {"ok": True}

# -------------------------
# Twilio inbound (TwiML)
# -------------------------
@app.post("/twilio/inbound")
async def twilio_inbound(request: Request):
    form = await request.form()
    from_num = form.get("From")
    to_num = form.get("To")
    call_sid = form.get("CallSid")
    logger.info(f"Incoming call: From={from_num}, To={to_num}, CallSid={call_sid}")

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

def deepgram_ws_url() -> str:
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

async def twilio_clear(twilio_ws: WebSocket, stream_sid: str):
    try:
        await twilio_ws.send_text(json.dumps({"event": "clear", "streamSid": stream_sid}))
    except Exception:
        logger.exception("Failed to send Twilio clear")

async def stream_ulaw_audio_to_twilio(
    twilio_ws: WebSocket,
    stream_sid: str,
    ulaw_audio: bytes,
    cancel_event: asyncio.Event,
):
    # 20ms @ 8kHz μ-law = 160 bytes
    frame_size = 160
    idx = 0
    while idx < len(ulaw_audio):
        if cancel_event.is_set():
            return
        chunk = ulaw_audio[idx: idx + frame_size]
        idx += frame_size
        msg = {"event": "media", "streamSid": stream_sid, "media": {"payload": _bytes_to_b64(chunk)}}
        await twilio_ws.send_text(json.dumps(msg))

async def elevenlabs_tts_ulaw_bytes(text: str) -> bytes:
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}?output_format=ulaw_8000"
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json",
        "Accept": "audio/ulaw",
    }
    body = {
        "text": text,
        "model_id": ELEVENLABS_MODEL_ID,
        "voice_settings": {"stability": 0.4, "similarity_boost": 0.75},
    }

    if eleven_client is None:
        async with httpx.AsyncClient(timeout=45.0) as client:
            r = await client.post(url, headers=headers, json=body)
    else:
        r = await eleven_client.post(url, headers=headers, json=body)

    r.raise_for_status()
    logger.info(
        f"ElevenLabs OK: content-type={r.headers.get('content-type')} bytes={len(r.content)} first10={r.content[:10]}"
    )
    return r.content

async def call_fsm_router(text: str, meta: Optional[Dict[str, Any]] = None) -> str:
    """
    By default FSM_ROUTER_URL points to our own /assistant/route,
    but you can set it to a separate skills router later.
    """
    payload = {"text": text, "meta": meta or {}}
    try:
        if router_client is None:
            async with httpx.AsyncClient(timeout=25.0) as client:
                r = await client.post(FSM_ROUTER_URL, json=payload)
        else:
            r = await router_client.post(FSM_ROUTER_URL, json=payload)

        r.raise_for_status()
        data = r.json()

        for key in ("spoken_reply", "reply", "text", "message"):
            if isinstance(data, dict) and isinstance(data.get(key), str) and data[key].strip():
                return data[key].strip()

        if isinstance(data, str) and data.strip():
            return data.strip()

        return "Okay."

    except Exception as e:
        logger.exception(f"FSM router call failed: {e}")
        return "I’m having trouble reaching my brain service right now. Please try again."

# -------------------------
# Topic inference (generic, light)
# -------------------------
_TOPIC_PREFIXES = [
    "tell me about ",
    "learn about ",
    "what is ",
    "what are ",
    "explain ",
    "help me understand ",
]

def _infer_topic(text: str) -> Optional[str]:
    if not text:
        return None
    low = text.lower().strip()
    for p in _TOPIC_PREFIXES:
        if low.startswith(p) and len(text) > len(p) + 2:
            return text[len(p):].strip(" .?!")
    return None

def _is_short_followup(text: str) -> bool:
    # One to three words like: "breeds", "history", "price", "steps"
    t = (text or "").strip()
    t = re.sub(r"[^\w\s']", "", t.lower()).strip()
    if not t:
        return False
    return len(t.split()) <= 3

# -------------------------
# OpenAI helpers
# -------------------------
def _extract_openai_text(data: dict) -> str:
    """
    Responses API returns an 'output' array which can include messages, tool calls,
    and reasoning items. We must aggregate all output_text blocks from message items.
    """
    out = data.get("output")
    parts: List[str] = []

    if isinstance(out, list):
        for item in out:
            if not isinstance(item, dict):
                continue
            if item.get("type") != "message":
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if isinstance(part, dict) and part.get("type") == "output_text":
                    t = part.get("text")
                    if isinstance(t, str) and t.strip():
                        parts.append(t.strip())

    joined = "\n".join(parts).strip()
    if joined:
        return joined

    # Some SDKs expose response.output_text, but the raw API may not.
    ot = data.get("output_text")
    if isinstance(ot, str) and ot.strip():
        return ot.strip()

    t = data.get("text")
    if isinstance(t, str) and t.strip():
        return t.strip()

    return ""

def _summarize_output_types(data: dict) -> List[dict]:
    """
    For logging only: show item types and any content types.
    """
    out = data.get("output")
    if not isinstance(out, list):
        return []
    sample = []
    for item in out[:3]:
        if not isinstance(item, dict):
            continue
        itype = item.get("type")
        ctypes = []
        c = item.get("content")
        if isinstance(c, list):
            for p in c[:3]:
                if isinstance(p, dict) and p.get("type"):
                    ctypes.append(p.get("type"))
        sample.append({"item_type": itype, "content_types": ctypes})
    return sample

async def _openai_post_with_retries(payload: dict) -> dict:
    if openai_client is None:
        client = httpx.AsyncClient(timeout=OPENAI_TIMEOUT_S)
        close_client = True
    else:
        client = openai_client
        close_client = False

    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    last_err: Exception | None = None

    try:
        for attempt in range(OPENAI_MAX_RETRIES + 1):
            try:
                r = await client.post("https://api.openai.com/v1/responses", headers=headers, json=payload)
                r.raise_for_status()
                return r.json()
            except httpx.HTTPStatusError as e:
                last_err = e
                status = e.response.status_code if e.response is not None else None
                # Retry on transient 5xx
                if status is not None and 500 <= status <= 599 and attempt < OPENAI_MAX_RETRIES:
                    backoff = (0.6 * (2 ** attempt)) + random.random() * 0.2
                    logger.warning(f"OpenAI 5xx (status={status}); retrying in {backoff:.2f}s")
                    await asyncio.sleep(backoff)
                    continue
                raise
            except (httpx.TimeoutException, httpx.NetworkError) as e:
                last_err = e
                if attempt < OPENAI_MAX_RETRIES:
                    backoff = (0.6 * (2 ** attempt)) + random.random() * 0.2
                    logger.warning(f"OpenAI network/timeout; retrying in {backoff:.2f}s")
                    await asyncio.sleep(backoff)
                    continue
                raise

        raise last_err or RuntimeError("OpenAI call failed")
    finally:
        if close_client:
            await client.aclose()

async def openai_reply(user_text: str, state: Optional[Dict[str, Any]] = None) -> str:
    if not OPENAI_API_KEY:
        return "OpenAI isn’t configured yet. Set OPENAI_API_KEY in Render."

    st = state or {}
    topic = (st.get("topic") or "").strip() or "unknown"
    last_user = (st.get("last_user") or "").strip()
    last_bot = (st.get("last_bot") or "").strip()

    # Use high-authority `instructions` per OpenAI docs. :contentReference[oaicite:3]{index=3}
    instructions = (
        "You are Vozlia, a calm, confident AI voice assistant.\n"
        "PHONE STYLE:\n"
        "- Answer immediately with a helpful response (1–3 sentences).\n"
        "- If the user asks something broad, give a quick overview + offer up to 3 options.\n"
        "- If the user gives a short follow-up like 'breeds', 'history', 'price', 'steps', 'care tips', etc.,\n"
        "  interpret it in the context of the CURRENT TOPIC and continue.\n"
        "- Do NOT say: 'Tell me what you want to know' or 'What would you like to do next?'\n"
        "- Keep it natural, like a great phone assistant.\n"
        "- Return plain text only.\n"
    )

    # Light context (helps follow-ups)
    context = (
        f"CURRENT_TOPIC: {topic}\n"
        f"LAST_USER: {last_user}\n"
        f"LAST_ASSISTANT: {last_bot}\n"
    )

    # If the user says a very short follow-up, nudge the model to treat it as continuation.
    effective_user_text = user_text
    if _is_short_followup(user_text) and topic != "unknown":
        effective_user_text = f"Follow-up on topic '{topic}': {user_text}"

    async def _run(model_name: str) -> Tuple[str, dict]:
        payload = {
            "model": model_name,
            "instructions": instructions,
            # Lower reasoning for voice latency; still smart. :contentReference[oaicite:4]{index=4}
            "reasoning": {"effort": "low"},
            "input": [
                {"role": "system", "content": context},
                {"role": "user", "content": effective_user_text},
            ],
            "max_output_tokens": VOICE_MAX_TOKENS,
        }
        data = await _openai_post_with_retries(payload)
        text_out = _extract_openai_text(data)
        return text_out, data

    try:
        text1, data1 = await _run(OPENAI_MODEL)
        if text1:
            return text1

        logger.warning(f"OpenAI parse failed. output_sample_types={_summarize_output_types(data1)}")

        # Fallback model (often fixes “reasoning-only” outputs + transient weirdness)
        if OPENAI_FALLBACK_MODEL and OPENAI_FALLBACK_MODEL != OPENAI_MODEL:
            text2, data2 = await _run(OPENAI_FALLBACK_MODEL)
            if text2:
                logger.warning(f"Recovered via fallback model={OPENAI_FALLBACK_MODEL}")
                return text2
            logger.warning(f"Fallback parse failed. output_sample_types={_summarize_output_types(data2)}")

        # Last resort (still phone-friendly)
        return "I’m having trouble formatting my reply, but I did hear you. Please repeat that once."

    except Exception as e:
        logger.exception(f"OpenAI reply failed: {e}")
        return "I had trouble reaching my brain service. Please try again."

# -------------------------
# Router endpoint (topic-agnostic)
# -------------------------
@app.post("/assistant/route")
async def assistant_route(request: Request):
    body = await request.json()
    text = (body.get("text") or "").strip()
    meta = body.get("meta") or {}

    state = (meta.get("state") if isinstance(meta, dict) else None) or {"topic": None, "last_user": "", "last_bot": ""}

    if not text:
        return {"spoken_reply": "I didn’t catch that. Can you say it again?", "intent": "general", "actions": [], "state": state}

    # Infer/update topic from “tell me about X”, etc.
    inferred = _infer_topic(text)
    if inferred:
        state["topic"] = inferred

    reply = await openai_reply(text, state=state)

    state["last_user"] = text
    state["last_bot"] = reply

    return {"spoken_reply": reply, "intent": "general", "actions": [], "state": state}

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

    last_final_ts = 0.0
    last_final_text = ""

    assistant_speaking = False

    async def cancel_speaking():
        nonlocal speak_task, assistant_speaking
        if speak_task and not speak_task.done():
            speak_cancel.set()
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
        nonlocal last_final_ts, last_final_text, assistant_speaking

        try:
            async for msg in dg_ws:
                try:
                    data = json.loads(msg)
                except Exception:
                    continue

                ch = data.get("channel") or {}
                alts = ch.get("alternatives") or []
                if not alts:
                    continue

                transcript = (alts[0].get("transcript") or "").strip()
                if not transcript:
                    continue

                is_final = bool(data.get("is_final"))

                # barge-in: if user speaks while assistant talking
                if assistant_speaking and not is_final:
                    await cancel_speaking()

                if not is_final:
                    continue

                now = time.time()
                if len(transcript) < MIN_FINAL_CHARS:
                    continue

                if transcript == last_final_text and (now - last_final_ts) < 1.0:
                    continue

                if (now - last_final_ts) * 1000 < FINAL_UTTERANCE_COOLDOWN_MS:
                    if transcript.lower() in last_final_text.lower():
                        continue

                last_final_text = transcript
                last_final_ts = now

                logger.info(f"Deepgram FINAL: {transcript}")

                if not stream_sid:
                    continue

                # Pull state for this call
                state = CALL_STATE.get(stream_sid) or {"topic": None, "last_user": "", "last_bot": ""}
                state["last_user"] = transcript
                CALL_STATE[stream_sid] = state

                # Optional “ack” (can be annoying; default OFF)
                if VOICE_ACK_ENABLED:
                    await speak(VOICE_ACK_TEXT)

                # Route (calls our local /assistant/route by default)
                reply = await call_fsm_router(transcript, meta={"stream_sid": stream_sid, "state": state})
                logger.info(f"Router reply: {reply}")

                state["last_bot"] = reply
                CALL_STATE[stream_sid] = state

                await speak(reply)

        except asyncio.CancelledError:
            return
        except Exception as e:
            logger.exception(f"Deepgram reader error: {e}")

    try:
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
                if stream_sid:
                    CALL_STATE[stream_sid] = {"topic": None, "last_user": "", "last_bot": ""}

                try:
                    await speak("I’m connected. How can I help you today?")
                except Exception:
                    logger.exception("Initial ElevenLabs greeting failed")
                continue

            if etype == "media":
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

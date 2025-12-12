import os
import json
import time
import base64
import asyncio
import logging
import signal
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
# -------------------------
CALL_STATE: dict[str, dict] = {}  # streamSid -> {"topic": str|None, "last_user": str, "last_bot": str}

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

# -------------------------
# Lifecycle + SIGTERM
# -------------------------
@app.on_event("startup")
async def startup():
    global openai_client, eleven_client, router_client
    logger.info("APP STARTUP")
    # http2=True requires `h2` dependency. Keep it off for Render simplicity.
    openai_client = httpx.AsyncClient(timeout=25.0)
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
    frame_size = 160  # 20ms @ 8kHz μ-law
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
    logger.info(f"ElevenLabs OK: content-type={r.headers.get('content-type')} bytes={len(r.content)} first10={r.content[:10]}")
    return r.content

async def call_fsm_router(text: str, meta: Optional[Dict[str, Any]] = None) -> str:
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

        return "Got it."

    except Exception as e:
        logger.exception(f"FSM router call failed: {e}")
        return "I’m having trouble reaching my skills service right now. Please try again."

# -------------------------
# OpenAI helpers
# -------------------------
def _extract_openai_text(data: dict) -> str:
    # 1) output -> content -> output_text
    out = data.get("output")
    if isinstance(out, list):
        parts = []
        for item in out:
            if not isinstance(item, dict):
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

    # 2) data["text"] can be str, list, or dict depending on server tier/format
    t = data.get("text")
    if isinstance(t, str) and t.strip():
        return t.strip()
    if isinstance(t, dict):
        # common shapes: {"value": "..."} or {"text": "..."}
        for k in ("value", "text", "content"):
            v = t.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
    if isinstance(t, list):
        parts = []
        for blk in t:
            if isinstance(blk, str) and blk.strip():
                parts.append(blk.strip())
            elif isinstance(blk, dict):
                v = blk.get("text") or blk.get("value") or blk.get("content")
                if isinstance(v, str) and v.strip():
                    parts.append(v.strip())
        joined = "\n".join([p for p in parts if p]).strip()
        if joined:
            return joined

    # 3) fallback
    ot = data.get("output_text")
    if isinstance(ot, str) and ot.strip():
        return ot.strip()

    return ""

async def openai_reply(user_text: str, state: Optional[Dict[str, Any]] = None) -> str:
    if not OPENAI_API_KEY:
        return "OpenAI isn’t configured yet. Set OPENAI_API_KEY in Render."

    topic = (state or {}).get("topic") or "none"

    system = (
        "You are Vozlia, a calm, confident AI voice assistant. "
        "PHONE RULES: answer first in 1–2 sentences. "
        "Do NOT ask clarifying questions unless absolutely required. "
        "Do NOT end with 'what would you like to do next'. "
        "If user asks something broad, give a short overview + 2 options. "
        f"Current topic: {topic}."
    )

    payload = {
        "model": OPENAI_MODEL,
        "input": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_text},
        ],
        "max_output_tokens": 180,
    }

    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}

    try:
        if openai_client is None:
            async with httpx.AsyncClient(timeout=25.0) as client:
                r = await client.post("https://api.openai.com/v1/responses", headers=headers, json=payload)
        else:
            r = await openai_client.post("https://api.openai.com/v1/responses", headers=headers, json=payload)

        r.raise_for_status()
        data = r.json()

        text_out = _extract_openai_text(data)
        if text_out:
            return text_out

        logger.warning(f"OpenAI response had no parsable text. Keys={list(data.keys())}")
        return "Got it. Tell me what you want to know."

    except Exception as e:
        logger.exception(f"OpenAI reply failed: {e}")
        return "I had trouble reaching my brain service. Please try again."

# -------------------------
# Router endpoint (THIS is where the behavior is enforced)
# -------------------------
@app.post("/assistant/route")
async def assistant_route(request: Request):
    body = await request.json()
    text = (body.get("text") or "").strip()
    meta = body.get("meta") or {}
    state = (meta.get("state") if isinstance(meta, dict) else None) or {"topic": None, "last_user": "", "last_bot": ""}

    if not text:
        return {"spoken_reply": "I didn’t catch that. Can you say it again?", "intent": "general", "actions": [], "state": state}

    low = text.lower()

    # ---- Topic lock: cats (more robust) ----
    if "cat" in low or "cats" in low or "cat breeds" in low:
        state["topic"] = "cats"
    topic = state.get("topic")

    # ---- Hard guardrail: cat breeds request should NOT go to OpenAI clarifiers ----
    # Catch many phrasings: "learn about cat breeds", "tell me about cat breeds", "summarize cat breeds"
    if ("breed" in low or "breeds" in low) and (("cat" in low) or topic == "cats"):
        # If they asked to summarize, do a concise summary rather than asking registry/species
        if any(k in low for k in ["summarize", "summary", "different breeds", "all cat breeds", "all breeds"]):
            reply = (
                "There are dozens of recognized cat breeds, but here’s a clean overview by type: "
                "1) Longhair: Maine Coon, Persian, Ragdoll. "
                "2) Shorthair: British Shorthair, American Shorthair, Russian Blue. "
                "3) Vocal/social: Siamese, Oriental Shorthair. "
                "4) Active/athletic: Bengal, Abyssinian. "
                "5) Unique coats: Sphynx (hairless), Devon Rex (curly). "
                "If you tell me your vibe—calm, playful, low-shedding—I’ll recommend 2–3 perfect matches."
            )
            return {"spoken_reply": reply, "intent": "general", "actions": [], "state": state}

        reply = (
            "Sure. Popular cat breeds include Maine Coon, Ragdoll, British Shorthair, Siamese, Persian, Bengal, and Sphynx. "
            "Do you want a calm family cat, a playful cat, or low-shedding?"
        )
        return {"spoken_reply": reply, "intent": "general", "actions": [], "state": state}

    # ---- Answer-first: open-ended cats requests (broadened) ----
    if topic == "cats" and (
        "learn about" in low
        or "tell me about" in low
        or "about cats" in low
        or low.strip() in ["cats", "cats.", "cat", "cat."]
        or "just tell" in low
        or "i don't know" in low
    ):
        reply = (
            "Cats are intelligent, curious animals that often bond strongly with people. "
            "Day to day, the big keys are good food, daily play, clean litter, and regular vet care. "
            "Want care tips, behavior, or breeds?"
        )
        return {"spoken_reply": reply, "intent": "general", "actions": [], "state": state}

    # ---- Default: OpenAI ----
    reply = await openai_reply(text, state=state)
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

                # Lively: instant ack
                await speak("Okay.")

                state = CALL_STATE.get(stream_sid) or {"topic": None, "last_user": "", "last_bot": ""}
                state["last_user"] = transcript
                CALL_STATE[stream_sid] = state

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

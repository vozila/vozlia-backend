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

# ----- System behavior prompt -----
SYSTEM_PROMPT = (
    "You are Vozlia, a warm, friendly, highly capable AI phone assistant using the "
    "Coral voice persona. Speak naturally, confidently, and in a professional tone, "
    "similar to a helpful customer support agent. "
    "You greet callers immediately at the start of the call, introduce yourself as "
    "Vozlia, and invite them to describe what they need help with. "
    "Keep your responses VERY short and concise: usually 1–2 short sentences and "
    "no more than about 5 seconds of speech before pausing. "
    "If the caller asks for a long explanation or story, summarize only the key "
    "points and ask if they want more details. "
    "Be highly attentive to interruptions: if the caller starts speaking while you "
    "are talking, you must immediately stop talking and listen. "
    "If the caller says the word 'stop' while you are speaking, treat that as an "
    "immediate instruction to stop talking. "
    "Your goal is to make callers feel welcome, understood, and supported."
)

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
                            "say what they need help with. Keep it to 1–2 short sentences."
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
    ai_speaking = False
    barge_in_enabled = False  # after first full response
    cancel_in_progress = False
    suppress_assistant_audio = False

    # Server VAD-based user speech flag
    user_speaking_vad = False

    # Track last STOP-trigger time to avoid spamming cancels
    last_stop_barge_in_ts = 0.0
    STOP_COOLDOWN_SECONDS = 1.0

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
            nonlocal stream_sid, ai_speaking, barge_in_enabled, cancel_in_progress
            nonlocal suppress_assistant_audio, openai_ws, user_speaking_vad
            nonlocal last_stop_barge_in_ts

            async for msg in openai_ws:
                try:
                    event = json.loads(msg)
                except json.JSONDecodeError:
                    logger.warning(f"Non-JSON message from OpenAI: {msg!r}")
                    continue

                etype = event.get("type")

                # ----- AUDIO FROM OPENAI → TWILIO -----
                if etype == "response.audio.delta":
                    audio_chunk_b64 = event.get("delta")
                    if not audio_chunk_b64:
                        continue

                    ai_speaking = True

                    if suppress_assistant_audio:
                        logger.debug(
                            "Suppressing assistant audio delta due to active barge-in"
                        )
                        continue

                    if not stream_sid:
                        logger.warning(
                            "Got audio from OpenAI but stream_sid is not set yet."
                        )
                        continue

                    try:
                        raw_bytes = base64.b64decode(audio_chunk_b64)
                        audio_payload = base64.b64encode(raw_bytes).decode("utf-8")
                    except Exception:
                        audio_payload = audio_chunk_b64

                    twilio_msg = {
                        "event": "media",
                        "streamSid": stream_sid,
                        "media": {"payload": audio_payload},
                    }
                    await websocket.send_text(json.dumps(twilio_msg))

                elif etype == "response.audio.done":
                    logger.info("OpenAI finished an audio response")
                    ai_speaking = False
                    cancel_in_progress = False
                    suppress_assistant_audio = False

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

                # ----- USER TRANSCRIPT (for STOP keyword) -----
                elif etype.startswith("input_text."):
                    # Realtime may send input_text.delta / input_text.completed etc.
                    text = event.get("delta") or event.get("text") or ""
                    if text:
                        logger.info(f"User transcript: {text!r}")
                        lower = text.lower()

                        # Check for explicit STOP keyword
                        if "stop" in lower:
                            now_ts = time.time()
                            if (
                                now_ts - last_stop_barge_in_ts
                                > STOP_COOLDOWN_SECONDS
                            ):
                                last_stop_barge_in_ts = now_ts
                                if (
                                    barge_in_enabled
                                    and ai_speaking
                                    and not cancel_in_progress
                                ):
                                    try:
                                        await openai_ws.send(
                                            json.dumps({"type": "response.cancel"})
                                        )
                                        cancel_in_progress = True
                                        ai_speaking = False
                                        suppress_assistant_audio = True
                                        logger.info(
                                            "BARGE-IN (STOP keyword): user said 'stop' "
                                            "while AI speaking; sent response.cancel "
                                            "and suppressed audio"
                                        )
                                    except Exception as e:
                                        logger.error(
                                            f"Error sending response.cancel (STOP): {e}"
                                        )

                # ----- VAD EVENTS: drive user_speaking_vad & barge-in -----
                elif etype == "input_audio_buffer.speech_started":
                    user_speaking_vad = True
                    logger.info("OpenAI VAD: user speech START")

                    # Pure VAD-based barge-in:
                    if (
                        barge_in_enabled
                        and ai_speaking
                        and not cancel_in_progress
                    ):
                        try:
                            await openai_ws.send(
                                json.dumps({"type": "response.cancel"})
                            )
                            cancel_in_progress = True
                            ai_speaking = False
                            suppress_assistant_audio = True

                            logger.info(
                                "BARGE-IN (VAD only): user speech started while AI "
                                "speaking; sent response.cancel and suppressed audio"
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
                        # This just means we tried to cancel when there was
                        # no active response; we can safely clear flags.
                        cancel_in_progress = False
                        suppress_assistant_audio = False

        await asyncio.gather(
            twilio_to_openai(),
            openai_to_twilio(),
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

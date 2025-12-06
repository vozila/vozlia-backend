# main.py

import os
import json
import asyncio
import logging
import base64

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, Query
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
    "gpt-4o-mini-realtime-preview-2024-12-17",  # default; override via env if needed
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
    "Keep your responses concise but conversational. Avoid long monologues. "
    "Pause briefly to let callers speak, and be attentive to interruptions. "
    "Your goal is to make callers feel welcome, understood, and supported."
)

# ---------- FastAPI app ----------
app = FastAPI()


# ---------- Basic endpoints ----------
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
async def debug_gpt(text: str = Query(default="Hello Vozlia")):
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

    # Directly connect to media stream; let Coral/Vozlia greet the caller.
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
    """
    Ask the model to greet the caller once the audio bridge is ready.
    """
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
    """
    Opens a WebSocket to the OpenAI Realtime API using websockets.connect,
    configures the session for μ-law audio, server-side VAD, and Coral voice,
    then sends an initial greeting.
    """
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set; cannot use Realtime API.")

    logger.info("Connecting to OpenAI Realtime WebSocket via websockets...")

    openai_ws = await websockets.connect(
        OPENAI_REALTIME_URL,
        extra_headers=OPENAI_REALTIME_HEADERS,
    )

    # Session config
    session_update = {
        "type": "session.update",
        "session": {
            "instructions": SYSTEM_PROMPT,
            "voice": VOICE_NAME,
            "modalities": ["text", "audio"],
            "input_audio_format": "g711_ulaw",
            "output_audio_format": "g711_ulaw",
            # Use server-side VAD for speech detection
            "turn_detection": {"type": "server_vad"},
        },
    }

    await openai_ws.send(json.dumps(session_update))
    logger.info("Sent session.update to OpenAI Realtime")

    # Proactive greeting so the AI talks first
    await send_initial_greeting(openai_ws)

    return openai_ws


# ---------- Twilio media stream ↔ OpenAI Realtime ----------
@app.websocket("/twilio/stream")
async def twilio_stream(websocket: WebSocket):
    """
    Twilio connects here via Media Streams (WebSocket).

    We:
      * Accept the WebSocket from Twilio.
      * Open a second WebSocket to OpenAI Realtime.
      * Forward Twilio audio → OpenAI.
      * Forward OpenAI audio → Twilio.
      * Use OpenAI server VAD events (speech_started / speech_stopped)
        to decide when a user turn is complete, then:
           - input_audio_buffer.commit
           - response.create
      * Implement barge-in:
           - If the user speaks while AI is talking (after the first full response),
             send response.cancel and treat it as a new user turn.
    """
    await websocket.accept()
    logger.info("Twilio media stream connected")

    openai_ws = None
    stream_sid = None  # Twilio stream ID for outbound audio

    # AI / call state
    ai_speaking = False
    cancel_in_progress = False
    barge_in_enabled = False  # Only after first full response

    # Turn-taking state
    awaiting_user = False  # True when we expect the caller to speak next

    # VAD state (from OpenAI)
    speech_active = False
    speech_frame_count = 0
    MIN_SPEECH_FRAMES = 40  # Rough threshold to treat as real speech turn

    try:
        openai_ws = await create_realtime_session()

        async def twilio_to_openai():
            nonlocal (
                stream_sid,
                ai_speaking,
                cancel_in_progress,
                barge_in_enabled,
                awaiting_user,
                speech_active,
                speech_frame_count,
            )

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
                    stream_sid = data.get("start", {}).get("streamSid")
                    logger.info(f"Stream started: {stream_sid}")

                elif event_type == "media":
                    payload_b64 = data.get("media", {}).get("payload")
                    if not payload_b64:
                        continue

                    # ----- BARGE-IN -----
                    # After first AI response, allow user speech to cancel ongoing audio.
                    if barge_in_enabled and ai_speaking and not cancel_in_progress:
                        try:
                            await openai_ws.send(
                                json.dumps({"type": "response.cancel"})
                            )
                            cancel_in_progress = True
                            ai_speaking = False

                            # After barge-in, we expect the user to speak a new turn.
                            awaiting_user = True
                            speech_active = False
                            speech_frame_count = 0

                            logger.info(
                                "BARGE-IN: user spoke, sent response.cancel to OpenAI"
                            )
                        except Exception as e:
                            logger.error(f"Error sending response.cancel: {e}")

                    # Always forward raw μ-law audio to OpenAI
                    audio_event = {
                        "type": "input_audio_buffer.append",
                        "audio": payload_b64,
                    }
                    await openai_ws.send(json.dumps(audio_event))

                    # Count frames only when VAD says speech is active
                    if speech_active:
                        speech_frame_count += 1

                elif event_type == "stop":
                    logger.info("Twilio sent stop; closing call.")
                    break

        async def openai_to_twilio():
            nonlocal (
                stream_sid,
                ai_speaking,
                cancel_in_progress,
                barge_in_enabled,
                awaiting_user,
                speech_active,
                speech_frame_count,
            )

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
                    cancel_in_progress = False  # new audio turn

                    if not stream_sid:
                        logger.warning(
                            "Got audio from OpenAI but stream_sid is not set yet."
                        )
                        continue

                    # Round-trip base64 (as in Twilio examples)
                    try:
                        raw_bytes = base64.b64decode(audio_chunk_b64)
                        audio_payload = base64.b64encode(raw_bytes).decode("utf-8")
                    except Exception:
                        audio_payload = audio_chunk_b64

                    twilio_msg = {
                        "event": "media",
                        "streamSid": stream_sid,
                        "media": {
                            "payload": audio_payload,
                        },
                    }
                    await websocket.send_text(json.dumps(twilio_msg))

                elif etype == "response.audio.done":
                    logger.info("OpenAI finished an audio response")
                    ai_speaking = False
                    cancel_in_progress = False

                    # After the first full response is done, enable barge-in
                    if not barge_in_enabled:
                        barge_in_enabled = True
                        logger.info("Barge-in is now ENABLED for subsequent responses.")

                    # Now we expect the caller to speak
                    awaiting_user = True
                    speech_active = False
                    speech_frame_count = 0

                # ----- TEXT LOGGING (optional) -----
                elif etype == "response.text.delta":
                    delta = event.get("delta", "")
                    if delta:
                        logger.info(f"AI (text delta): {delta}")

                elif etype == "response.text.done":
                    text = event.get("text", "")
                    if text:
                        logger.info(f"AI full text response: {text}")

                # ----- VAD EVENTS (server-side speech detection) -----
                elif etype == "input_audio_buffer.speech_started":
                    logger.info("OpenAI detected speech START in input buffer")
                    speech_active = True
                    speech_frame_count = 0

                elif etype == "input_audio_buffer.speech_stopped":
                    logger.info("OpenAI detected speech STOP in input buffer")

                    # Only treat this as a user turn if:
                    #  - we were actually expecting user speech (awaiting_user)
                    #  - speech was active
                    #  - we saw enough frames to consider it real speech
                    if (
                        awaiting_user
                        and speech_active
                        and speech_frame_count >= MIN_SPEECH_FRAMES
                    ):
                        try:
                            await openai_ws.send(
                                json.dumps({"type": "input_audio_buffer.commit"})
                            )
                            await openai_ws.send(
                                json.dumps({"type": "response.create"})
                            )
                            logger.info(
                                "User turn (via VAD): committed buffer & requested new response."
                            )
                        except Exception as e:
                            logger.error(
                                f"Error committing user audio (VAD) / creating response: {e}"
                            )
                        finally:
                            awaiting_user = False

                    # Reset VAD state
                    speech_active = False
                    speech_frame_count = 0

                # ----- ERROR EVENTS -----
                elif etype == "error":
                    logger.error(f"OpenAI error event: {event}")
                    # response_cancel_not_active or similar are usually harmless
                    # We'll just log them and keep the stream running.
                    continue

                # Optional: log other event types for debugging
                else:
                    # Uncomment for very verbose logging:
                    # logger.debug(f"Unhandled OpenAI event type: {etype}")
                    pass

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

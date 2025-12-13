import asyncio
import base64
import json
import time
import re
from typing import Optional, Callable, Any, Dict

import websockets
from fastapi import WebSocket, WebSocketDisconnect


EMAIL_KEYWORDS_LOCAL = [
    "email","emails","e-mail","e-mails","inbox","gmail","g mail","mailbox","my mail",
    "my messages","unread","new mail","new emails","read my email","read my emails",
    "check my email","check my emails","how many emails","how many messages","email today",
    "emails today","read the email","read that email","read it","just read it",
    "read the first message","read the first email","read the message",
]

# Phrases that usually refer to the *current* email context (after a list/summary):
FOLLOWUP_EMAIL_PHRASES = [
    "read that",
    "read it",
    "read this",
    "open that",
    "open it",
    "show that",
    "show it",
    "what does it say",
    "what does that say",
    "the one from",
    "from klarna",
    "from alibaba",
]


FOLLOWUP_EMAIL_PHRASES = {
    "read it","just read it","read that","read that email","read the email",
    "open it","open that","what does it say","read the first one","read the first message",
    "read the first email","read the one from",
}

FILLER_ONLY = {"um", "uh", "er", "hmm"}
SMALL_TOSS = {"awesome", "great", "okay", "ok", "hello", "hi", "thanks", "thank you", "k"}


def _normalize_text(text: str) -> str:
    t = (text or "").lower()
    out = []
    for ch in t:
        out.append(ch if (ch.isalnum() or ch.isspace()) else " ")
    return " ".join("".join(out).split())


def _looks_like_email_intent(text: str, email_context_active: bool) -> bool:
    if not text:
        return False
    n = _normalize_text(text)

    # Strong signals (works even without prior email context)
    if re.search(r"\b(email|emails|inbox|gmail|messages|message)\b", n) and re.search(r"\b(read|check|list|open|show|summari[sz]e|search|find)\b", n):
        return True

    # Common keyword substrings
    for kw in EMAIL_KEYWORDS_LOCAL:
        if kw in n:
            return True

    # Requests like: 'read the email from Klarna', 'first message from Alibaba'
    if re.search(r"\b(read|open|show)\b.*\b(email|message)\b.*\bfrom\b", n):
        return True
    if re.search(r"\b(first|latest|recent)\b.*\b(email|message)\b", n) and re.search(r"\b(read|open|show)\b", n):
        return True

    # Follow-ups after an email summary/list was just spoken
    if email_context_active:
        for p in FOLLOWUP_EMAIL_PHRASES:
            if n.startswith(p) or (p in n):
                return True
        if re.search(r"\b(that|this|it)\b", n) and re.search(r"\b(read|open|show)\b", n):
            return True

    if "how many" in n and ("mail" in n or "message" in n or "inbox" in n):
        return True

    return False


def _should_reply(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return False
    words = t.split()
    if len(words) == 1 and (words[0] in FILLER_ONLY or words[0] in SMALL_TOSS):
        return False
    return True


def register_flow_a(
    app,
    logger,
    call_fsm_router: Callable[..., Any],
    settings: Dict[str, Any],
):
    """Register the /twilio/stream websocket route on an existing FastAPI app.

    This keeps your monolith slim while preserving runtime behavior.
    """

    OPENAI_REALTIME_URL = settings["OPENAI_REALTIME_URL"]
    OPENAI_REALTIME_HEADERS = settings["OPENAI_REALTIME_HEADERS"]
    REALTIME_SYSTEM_PROMPT = settings["REALTIME_SYSTEM_PROMPT"]
    VOICE_NAME = settings["VOICE_NAME"]
    BYTES_PER_FRAME = settings["BYTES_PER_FRAME"]
    FRAME_INTERVAL = settings["FRAME_INTERVAL"]
    PREBUFFER_BYTES = settings["PREBUFFER_BYTES"]
    MAX_TWILIO_BACKLOG_SECONDS = settings["MAX_TWILIO_BACKLOG_SECONDS"]

    async def create_realtime_session():
        openai_ws = await websockets.connect(
            OPENAI_REALTIME_URL,
            extra_headers=OPENAI_REALTIME_HEADERS,
            ping_interval=None,
            ping_timeout=None,
            max_size=None,
        )

        session_update = {
            "type": "session.update",
            "session": {
                "instructions": REALTIME_SYSTEM_PROMPT,
                "voice": VOICE_NAME,
                "modalities": ["text", "audio"],
                "input_audio_format": "g711_ulaw",
                "output_audio_format": "g711_ulaw",
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                    "silence_duration_ms": 500,
                    "create_response": False,
                    "interrupt_response": True,
                },
                "input_audio_transcription": {"model": "gpt-4o-mini-transcribe"},
            },
        }
        await openai_ws.send(json.dumps(session_update))
        logger.info("Sent session.update to OpenAI Realtime")

        await openai_ws.send(json.dumps({"type": "response.create"}))
        logger.info("Sent initial greeting request to OpenAI Realtime")
        return openai_ws

    @app.websocket("/twilio/stream")
    async def twilio_stream(websocket: WebSocket):
        await websocket.accept()
        logger.info("Twilio media stream connected")

        openai_ws: Optional[websockets.WebSocketClientProtocol] = None
        stream_sid: Optional[str] = None

        audio_buffer = bytearray()
        stop_event = asyncio.Event()  # set when Twilio sends 'stop' or websocket closes
        prebuffer_active = True
        barge_in_enabled = False
        assistant_last_audio_time = 0.0

        active_response_id: Optional[str] = None

        email_context_until = 0.0
        EMAIL_CONTEXT_TTL = 120.0

        def email_context_active() -> bool:
            return time.monotonic() < email_context_until

        def assistant_actively_speaking() -> bool:
            if audio_buffer:
                return True
            if assistant_last_audio_time and (time.monotonic() - assistant_last_audio_time) < 0.5:
                return True
            return False

        async def twilio_clear_buffer():
            if stream_sid is None:
                return
            try:
                await websocket.send_text(json.dumps({"event": "clear", "streamSid": stream_sid}))
            except Exception:
                pass

        async def send_audio_frame():
            nonlocal assistant_last_audio_time, prebuffer_active, barge_in_enabled
            if stream_sid is None or len(audio_buffer) < BYTES_PER_FRAME:
                return

            if prebuffer_active and len(audio_buffer) < PREBUFFER_BYTES:
                return
            if prebuffer_active and len(audio_buffer) >= PREBUFFER_BYTES:
                prebuffer_active = False
                logger.info("Prebuffer complete; starting to send audio to Twilio")
                if not barge_in_enabled:
                    barge_in_enabled = True
                    logger.info("Barge-in is now ENABLED (audio streaming started).")

            chunk = bytes(audio_buffer[:BYTES_PER_FRAME])
            del audio_buffer[:BYTES_PER_FRAME]

            payload = base64.b64encode(chunk).decode("ascii")
            msg = {"event": "media", "streamSid": stream_sid, "media": {"payload": payload}}
            await websocket.send_text(json.dumps(msg))
            assistant_last_audio_time = time.monotonic()

        async def twilio_audio_sender():
            next_send = time.monotonic()
            frames_sent = 0
            play_start_ts = None
            underruns = 0
            late_ms_max = 0.0
            last_stats = time.monotonic()

            while not stop_event.is_set():
                if stream_sid is None:
                    await asyncio.sleep(0.01)
                    continue

                now = time.monotonic()
                if now < next_send:
                    await asyncio.sleep(min(0.01, next_send - now))
                    continue

                if play_start_ts is None:
                    play_start_ts = now
                    frames_sent = 0
                    underruns = 0
                    late_ms_max = 0.0

                call_elapsed = now - play_start_ts
                sent_dur = frames_sent * FRAME_INTERVAL
                backlog = sent_dur - call_elapsed

                if backlog > MAX_TWILIO_BACKLOG_SECONDS:
                    await asyncio.sleep(0.01)
                    next_send = time.monotonic()
                    continue

                if len(audio_buffer) >= BYTES_PER_FRAME:
                    try:
                        await send_audio_frame()
                    except WebSocketDisconnect:
                        logger.info("twilio_audio_sender: websocket closed while sending; stopping sender")
                        return
                    except Exception:
                        logger.info("twilio_audio_sender: send failed; stopping sender")
                        return
                    frames_sent += 1
                    next_send = time.monotonic() + FRAME_INTERVAL
                else:
                    underruns += 1
                    await asyncio.sleep(0.005)
                    next_send = time.monotonic() + FRAME_INTERVAL

                if now - last_stats > 1.0:
                    q_bytes = len(audio_buffer)
                    logger.info(
                        "twilio_send stats: q_bytes=%s frames_sent=%s underruns=%s late_ms_max=%.1f prebuf=%s",
                        q_bytes, frames_sent, underruns, late_ms_max, prebuffer_active
                    )
                    last_stats = now

        sender_task = asyncio.create_task(twilio_audio_sender())

        async def _cancel_active_and_clear_buffer(reason: str):
            nonlocal active_response_id, prebuffer_active
            audio_buffer.clear()
            prebuffer_active = True

            if not openai_ws or not active_response_id:
                logger.info("_cancel_active_and_clear_buffer: no active response (reason=%s)", reason)
                active_response_id = None
                return

            rid = active_response_id
            active_response_id = None
            try:
                await openai_ws.send(json.dumps({"type": "response.cancel", "response_id": rid}))
                logger.info("Sent response.cancel for %s due to %s", rid, reason)
            except Exception:
                logger.info("OpenAI cancel race (expected): rid=%s reason=%s", rid, reason)

        async def create_processing_ack():
            await _cancel_active_and_clear_buffer("processing_ack")
            instructions = (
                "Say one short sentence to the caller acknowledging you're checking email, "
                "for example: 'Okay—one moment while I check your inbox.' Then stop."
            )
            await openai_ws.send(json.dumps({"type": "response.create", "response": {"instructions": instructions}}))

        async def create_generic_response():
            await _cancel_active_and_clear_buffer("create_generic_response")
            await openai_ws.send(json.dumps({"type": "response.create"}))
            logger.info("Sent generic response.create for chit-chat turn")

        async def create_fsm_spoken_reply(spoken_reply: str):
            await _cancel_active_and_clear_buffer("create_fsm_spoken_reply")
            instructions = (
                "You are on a live phone call as Vozlia. "
                "The secure backend already checked the caller's email and produced this output. "
                "Speak it naturally, light rephrasing is allowed but keep key facts. "
                "Never say you can't access email. "
                f'Output: "{spoken_reply}"'
            )
            await openai_ws.send(json.dumps({"type": "response.create", "response": {"instructions": instructions}}))
            logger.info("Sent FSM-driven spoken reply into Realtime session")

        async def handle_transcript(transcript: str):
            nonlocal email_context_until
            if not transcript or not _should_reply(transcript):
                return

            logger.info("USER Transcript completed: %r", transcript)
            is_email = _looks_like_email_intent(transcript, email_context_active())

            if is_email:
                email_context_until = time.monotonic() + EMAIL_CONTEXT_TTL
                logger.info("Debounce: transcript looks like an email/skill request; routing to FSM + backend.")

                await create_processing_ack()

                data = await call_fsm_router(text=transcript, context={"channel": "phone"})
                spoken = (data or {}).get("spoken_reply")
                logger.info("FSM spoken_reply to send: %r", spoken)
                if spoken:
                    await create_fsm_spoken_reply(spoken)
                else:
                    await create_generic_response()
            else:
                logger.info("Debounce: transcript does NOT look like an email/skill intent; using generic response.create.")
                await create_generic_response()

        async def handle_barge_in():
            nonlocal active_response_id
            if not barge_in_enabled:
                return
            if not assistant_actively_speaking():
                return
            logger.info("BARGE-IN: user speech started while AI speaking; locally muting and clearing.")
            active_response_id = None
            audio_buffer.clear()
            await twilio_clear_buffer()

        async def openai_loop():
            nonlocal active_response_id, barge_in_enabled
            try:
                async for raw in openai_ws:
                    event = json.loads(raw)
                    etype = event.get("type")

                    if etype == "response.created":
                        rid = (event.get("response") or {}).get("id")
                        if rid:
                            active_response_id = rid
                            logger.info("Tracking allowed MANUAL response_id: %s", rid)

                    elif etype in ("response.completed", "response.failed", "response.canceled"):
                        rid = (event.get("response") or {}).get("id")
                        if active_response_id and rid == active_response_id:
                            active_response_id = None
                        if not barge_in_enabled:
                            barge_in_enabled = True

                    elif etype == "response.audio.delta":
                        rid = event.get("response_id")
                        if rid != active_response_id:
                            continue
                        delta_b64 = event.get("delta")
                        if not delta_b64:
                            continue
                        try:
                            audio_buffer.extend(base64.b64decode(delta_b64))
                        except Exception:
                            continue

                    elif etype == "input_audio_buffer.speech_started":
                        logger.info("OpenAI VAD: user speech START")
                        if assistant_actively_speaking():
                            await handle_barge_in()

                    elif etype == "input_audio_buffer.speech_stopped":
                        logger.info("OpenAI VAD: user speech STOP")

                    elif etype == "conversation.item.input_audio_transcription.completed":
                        transcript = (event.get("transcript") or "").strip()
                        await handle_transcript(transcript)

                    elif etype == "error":
                        err = event.get("error", {})
                        if err.get("code") == "response_cancel_not_active":
                            logger.info("OpenAI cancel race (expected): %s", event)
                        else:
                            logger.error("OpenAI error event: %s", event)

            except websockets.ConnectionClosed:
                logger.info("OpenAI Realtime WebSocket closed")
            except Exception:
                logger.exception("Error in OpenAI event loop")

        async def twilio_loop():
            nonlocal stream_sid, prebuffer_active
            try:
                async for msg in websocket.iter_text():
                    data = json.loads(msg)
                    event_type = data.get("event")

                    if event_type == "start":
                        stream_sid = (data.get("start") or {}).get("streamSid")
                        prebuffer_active = True
                        logger.info("Stream started: %s", stream_sid)

                    elif event_type == "media":
                        if not openai_ws:
                            continue
                        payload = (data.get("media") or {}).get("payload")
                        if payload:
                            await openai_ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": payload}))

                    elif event_type == "stop":
                        logger.info("Twilio sent stop; closing call.")
                        stop_event.set()
                        break
            except WebSocketDisconnect:
                logger.info("Twilio WebSocket disconnected")
            except Exception:
                logger.exception("Error in Twilio event loop")

        try:
            openai_ws = await create_realtime_session()
            await asyncio.gather(openai_loop(), twilio_loop())
        finally:
            stop_event.set()
            try:
                await openai_ws.close()
            except Exception:
                pass
            try:
                sender_task.cancel()
            except Exception:
                pass
            try:
                if openai_ws is not None:
                    await openai_ws.close()
            except Exception:
                pass
            try:
                await websocket.close()
            except Exception:
                pass
            logger.info("Call websocket cleanup complete")

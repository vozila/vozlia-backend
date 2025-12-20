from __future__ import annotations

import asyncio
import base64
import json
import time
from typing import Optional

import websockets
from fastapi import WebSocket, WebSocketDisconnect

from core.logging import logger


def _main():
    """
    Lazy access to main.py module-level globals (constants, prompts, helpers).
    This avoids circular imports at import-time while keeping behavior identical.
    """
    import main  # uvicorn main:app => module name is 'main'
    return main


async def create_realtime_session():
    """
    Connect to OpenAI Realtime WS and send session.update + an initial greeting.
    (Copied from main.py, behavior unchanged.)
    """
    m = _main()

    logger.info(f"Connecting to OpenAI Realtime at {m.OPENAI_REALTIME_URL}")

    ws = await websockets.connect(
        m.OPENAI_REALTIME_URL,
        extra_headers=m.OPENAI_REALTIME_HEADERS,
        max_size=16 * 1024 * 1024,
    )

    session_update = {
        "type": "session.update",
        "session": {
            "turn_detection": {
                "type": "server_vad",
                "threshold": float(os.getenv("REALTIME_VAD_THRESHOLD", "0.4")),
                "silence_duration_ms": int(os.getenv("REALTIME_VAD_SILENCE_MS", "1200")),
                "prefix_padding_ms": int(os.getenv("REALTIME_VAD_PREFIX_MS", "200")),
                "create_response": False,
                "interrupt_response": True,
            },
            "input_audio_format": "g711_ulaw",
            "output_audio_format": "g711_ulaw",
            "voice": m.VOICE_NAME,
            "instructions": m.REALTIME_SYSTEM_PROMPT,
            "input_audio_transcription": {
                "model": "whisper-1",
            },
        },
    }

    await ws.send(json.dumps(session_update))
    logger.info("Sent session.update to OpenAI Realtime")

    await ws.send(json.dumps({"type": "response.create"}))
    logger.info("Sent initial greeting request to OpenAI Realtime")

    return ws


async def twilio_stream(websocket: WebSocket):
    """
    Handles Twilio <-> OpenAI Realtime audio for a single phone call.

    Changes vs previous version:
    - Fixes email intent detection for normalized "e-mails" -> "e mails"
    - Adds a pending-response latch so we don't drop early audio deltas
      when response.audio.delta arrives before response.created.
    - Uses existing create_email_processing_ack() before slow email backend calls.
    """
    m = _main()

    await websocket.accept()

    sender_task = None  # will be started after helper defs

    logger.info("Twilio media stream connected")

    # --- Call + AI state -----------------------------------------------------
    openai_ws: Optional[websockets.WebSocketClientProtocol] = None
    stream_sid: Optional[str] = None

    # After we first start sending assistant audio, we allow barge-in
    barge_in_enabled: bool = False

    # Lifecycle flag: becomes True when Twilio sends stop or disconnects.
    twilio_ws_closed: bool = False

    # Only run one transcript action at a time (avoid overlapping Gmail fetches).
    transcript_action_task: Optional[asyncio.Task] = None

    user_speaking_vad: bool = False

    audio_buffer = bytearray()
    assistant_last_audio_time: float = 0.0

    prebuffer_active: bool = True

    active_response_id: Optional[str] = None
    allowed_response_ids: set[str] = set()

    # ✅ Latch: a response.create was sent, but we may not have received response.created yet.
    pending_response_create: bool = False

    def assistant_actively_speaking() -> bool:
        if audio_buffer:
            return True
        if assistant_last_audio_time:
            if (time.monotonic() - assistant_last_audio_time) < 0.5:
                return True
        return False

    async def send_audio_to_twilio():
        nonlocal audio_buffer, assistant_last_audio_time

        if stream_sid is None:
            return
        if len(audio_buffer) < m.BYTES_PER_FRAME:
            return

        frame = bytes(audio_buffer[:m.BYTES_PER_FRAME])
        del audio_buffer[:m.BYTES_PER_FRAME]

        payload = base64.b64encode(frame).decode("ascii")
        msg = {
            "event": "media",
            "streamSid": stream_sid,
            "media": {"payload": payload},
        }
        await websocket.send_text(json.dumps(msg))
        assistant_last_audio_time = time.monotonic()

    async def twilio_audio_sender():
        nonlocal audio_buffer, prebuffer_active, assistant_last_audio_time, barge_in_enabled

        send_start_ts: Optional[float] = None
        frame_idx: int = 0

        last_stat_ts: float = time.monotonic()
        frames_sent_interval: int = 0
        underruns: int = 0
        late_ms_max: float = 0.0

        try:
            while True:
                if twilio_ws_closed:
                    return
                if stream_sid is None:
                    await asyncio.sleep(0.01)
                    continue

                now = time.monotonic()

                if now - last_stat_ts >= 1.0:
                    logger.info(
                        "twilio_send stats: q_bytes=%d frames_sent=%d underruns=%d late_ms_max=%.1f prebuf=%s",
                        len(audio_buffer),
                        frames_sent_interval,
                        underruns,
                        late_ms_max,
                        prebuffer_active,
                    )
                    last_stat_ts = now
                    frames_sent_interval = 0
                    underruns = 0
                    late_ms_max = 0.0

                if len(audio_buffer) == 0:
                    if assistant_last_audio_time and (time.monotonic() - assistant_last_audio_time) > 1.0:
                        send_start_ts = None
                        frame_idx = 0
                    await asyncio.sleep(0.005)
                    continue

                if prebuffer_active:
                    if len(audio_buffer) < m.PREBUFFER_BYTES:
                        await asyncio.sleep(0.005)
                        continue

                    prebuffer_active = False
                    logger.info("Prebuffer complete; starting to send audio to Twilio")
                    if not barge_in_enabled:
                        barge_in_enabled = True
                        logger.info("Barge-in is now ENABLED (audio streaming started).")

                    send_start_ts = time.monotonic()
                    frame_idx = 0

                if send_start_ts is None:
                    send_start_ts = time.monotonic()
                    frame_idx = 0

                call_elapsed = now - send_start_ts
                audio_sent_duration = frame_idx * m.FRAME_INTERVAL
                backlog_seconds = audio_sent_duration - call_elapsed
                if backlog_seconds > m.MAX_TWILIO_BACKLOG_SECONDS:
                    await asyncio.sleep(0.005)
                    continue

                target = send_start_ts + frame_idx * m.FRAME_INTERVAL
                now = time.monotonic()
                if now < target:
                    await asyncio.sleep(target - now)
                    continue

                now = time.monotonic()
                late_ms = (now - target) * 1000.0
                if late_ms > late_ms_max:
                    late_ms_max = late_ms

                if len(audio_buffer) >= m.BYTES_PER_FRAME:
                    try:
                        await send_audio_to_twilio()
                    except WebSocketDisconnect:
                        logger.info("Twilio WebSocket closed; stopping audio sender task")
                        return
                    except Exception:
                        logger.exception("Error sending audio to Twilio; stopping sender")
                        return
                    frame_idx += 1
                    frames_sent_interval += 1
                else:
                    underruns += 1
                    await asyncio.sleep(0.005)

        except asyncio.CancelledError:
            return
        except Exception:
            logger.exception("twilio_audio_sender crashed")
            return

    sender_task = asyncio.create_task(twilio_audio_sender())

    async def twilio_clear_buffer():
        if stream_sid is None:
            return
        try:
            await websocket.send_text(json.dumps({"event": "clear", "streamSid": stream_sid}))
        except Exception:
            logger.exception("Failed to send Twilio clear")

    async def handle_barge_in():
        nonlocal active_response_id, audio_buffer, pending_response_create, prebuffer_active

        if not barge_in_enabled:
            logger.info("BARGE-IN: ignored (not yet enabled)")
            return

        if not assistant_actively_speaking():
            logger.info("BARGE-IN: assistant not actively speaking; nothing to mute")
            return

        logger.info(
            "BARGE-IN: user speech started while AI speaking; "
            "locally muting current response and clearing audio buffer."
        )

        active_response_id = None
        pending_response_create = False
        prebuffer_active = True
        await twilio_clear_buffer()
        audio_buffer.clear()

    # ✅ Include normalized variants like "e mails"
    EMAIL_KEYWORDS_LOCAL = [
        "email", "emails", "e-mail", "e-mails",
        "e mail", "e mails",
        "inbox", "gmail", "g mail", "mailbox",
        "my mail", "my messages",
        "unread", "new mail", "new emails",
        "today's emails", "today emails",
        "read my email", "read my emails",
        "check my email", "check my emails",
        "how many emails", "how many messages",
        "email today", "emails today",
        "summary of my email", "summary of my emails",
        "summary of my e mail", "summary of my e mails",
    ]

    def looks_like_email_intent(text: str) -> bool:
        if not text:
            return False
        t = text.lower()
        normalized = []
        for ch in t:
            if ch.isalnum() or ch.isspace():
                normalized.append(ch)
            else:
                normalized.append(" ")
        normalized = " ".join("".join(normalized).split())

        for kw in EMAIL_KEYWORDS_LOCAL:
            if kw in normalized:
                return True

        if "how many" in normalized and ("mail" in normalized or "message" in normalized or "inbox" in normalized):
            return True

        if "check my" in normalized and ("inbox" in normalized or "gmail" in normalized or "g mail" in normalized):
            return True

        if "read my" in normalized and ("messages" in normalized or "mail" in normalized or "inbox" in normalized):
            return True

        return False

    FILLER_ONLY = {"um", "uh", "er", "hmm"}
    SMALL_TOSS = {"awesome", "great", "okay", "ok", "hello", "hi", "thanks", "thank you"}

    def should_reply(text: str) -> bool:
        t = text.strip().lower()
        if not t:
            return False
        words = t.split()
        if len(words) == 1 and (words[0] in FILLER_ONLY or words[0] in SMALL_TOSS):
            return False
        return True

    async def route_to_fsm_and_get_reply(transcript: str) -> Optional[str]:
        try:
            data = await m.call_fsm_router(
                text=transcript,
                context={"channel": "phone"},
            )
            spoken = data.get("spoken_reply")
            logger.info("FSM spoken_reply to send: %r", spoken)
            return spoken
        except Exception:
            logger.exception("Error calling /assistant/route")
            return None

    async def _cancel_active_and_clear_buffer(reason: str):
        nonlocal active_response_id, audio_buffer, prebuffer_active, pending_response_create

        pending_response_create = False

        if not openai_ws:
            logger.info("_cancel_active_and_clear_buffer: no openai_ws (reason=%s)", reason)
            audio_buffer.clear()
            prebuffer_active = True
            return

        if not active_response_id:
            logger.info("_cancel_active_and_clear_buffer: no active response (reason=%s)", reason)
            audio_buffer.clear()
            prebuffer_active = True
            return

        rid = active_response_id
        logger.info("Sent response.cancel for %s due to %s", rid, reason)

        try:
            await openai_ws.send(json.dumps({"type": "response.cancel", "response_id": rid}))
        except Exception:
            logger.exception("Error sending response.cancel for %s", rid)

        active_response_id = None
        audio_buffer.clear()
        prebuffer_active = True

    async def create_generic_response():
        nonlocal pending_response_create
        await _cancel_active_and_clear_buffer("create_generic_response")
        pending_response_create = True
        await openai_ws.send(json.dumps({"type": "response.create"}))
        logger.info("Sent generic response.create for chit-chat turn")

    async def create_fsm_spoken_reply(spoken_reply: str):
        nonlocal pending_response_create

        if not spoken_reply:
            logger.warning("create_fsm_spoken_reply called with empty spoken_reply")
            await create_generic_response()
            return

        await _cancel_active_and_clear_buffer("create_fsm_spoken_reply")
        pending_response_create = True

        instructions = (
            "You are on a live phone call as Vozlia.\n"
            "The secure backend has already checked the caller's email account "
            "and produced a short summary of their messages.\n\n"
            "Here is the summary you must speak to the caller:\n"
            f"\"{spoken_reply}\"\n\n"
            "Your job for THIS response only is:\n"
            "- Say this summary to the caller in a natural, friendly way.\n"
            "- You MAY lightly rephrase for flow, but you MUST keep all of the "
            "  important facts (counts, topics, deadlines) from the summary.\n"
            "- DO NOT mention security, privacy, 'I cannot access your email', "
            "  or anything about being unable to read their email.\n"
            "- DO NOT apologize or refuse.\n"
            "- Just give the summary, as if you personally checked it.\n"
        )

        await openai_ws.send(json.dumps({
            "type": "response.create",
            "response": {"instructions": instructions},
        }))
        logger.info("Sent FSM-driven spoken reply into Realtime session")

    async def create_email_processing_ack():
        nonlocal pending_response_create
        if not openai_ws:
            return

        instructions = (
            "You are Vozlia on a live phone call. "
            "The caller just asked you to check their email. "
            "Say ONE short sentence acknowledging you're checking now, like "
            "'Okay — I’m checking your email now; one moment.' "
            "Then stop speaking and wait."
        )

        try:
            pending_response_create = True
            await openai_ws.send(json.dumps({
                "type": "response.create",
                "response": {"instructions": instructions},
            }))
            logger.info("Sent email processing acknowledgement into Realtime session")
        except Exception:
            logger.exception("Failed to send email processing acknowledgement")

    async def handle_transcript_event(event: dict):
        transcript: str = event.get("transcript", "").strip()
        if not transcript:
            return

        logger.info("USER Transcript completed: %r", transcript)

        if not should_reply(transcript):
            logger.info("Ignoring filler transcript: %r", transcript)
            return

        if looks_like_email_intent(transcript):
            logger.info("Debounce: transcript looks like an email/skill request; routing to FSM + backend.")
            await create_email_processing_ack()

            spoken_reply = await route_to_fsm_and_get_reply(transcript)
            if spoken_reply:
                await create_fsm_spoken_reply(spoken_reply)
            else:
                logger.warning("FSM returned no spoken_reply; falling back to generic reply.")
                await create_generic_response()
        else:
            logger.info("Debounce: transcript does NOT look like an email/skill intent; using generic GPT response.")
            await create_generic_response()

    async def openai_loop():
        nonlocal active_response_id, barge_in_enabled, user_speaking_vad, transcript_action_task, pending_response_create

        try:
            async for raw in openai_ws:
                event = json.loads(raw)
                etype = event.get("type")

                if etype == "response.created":
                    resp = event.get("response", {}) or {}
                    rid = resp.get("id")
                    if rid:
                        active_response_id = rid
                        pending_response_create = False
                        allowed_response_ids.add(rid)
                        logger.info("Tracking allowed MANUAL response_id: %s", rid)

                elif etype in ("response.completed", "response.failed", "response.canceled"):
                    resp = event.get("response", {}) or {}
                    rid = resp.get("id")
                    if active_response_id is not None and rid == active_response_id:
                        logger.info("Response %s finished with event '%s'; clearing active_response_id", rid, etype)
                        active_response_id = None

                    if not barge_in_enabled:
                        barge_in_enabled = True
                        logger.info("First response finished (event=%s, id=%s); barge-in is now ENABLED.", etype, rid)

                elif etype == "response.audio.delta":
                    resp_id = event.get("response_id")
                    delta_b64 = event.get("delta")

                    # ✅ Adopt the response_id if audio arrives before response.created
                    if active_response_id is None and pending_response_create and resp_id:
                        logger.info(
                            "Adopting response_id=%s from audio.delta (response.created not seen yet)",
                            resp_id,
                        )
                        active_response_id = resp_id
                        pending_response_create = False

                    if resp_id != active_response_id:
                        logger.info(
                            "Dropping unsolicited audio for response_id=%s (active=%s)",
                            resp_id, active_response_id
                        )
                        continue

                    if not delta_b64:
                        continue

                    try:
                        delta_bytes = base64.b64decode(delta_b64)
                    except Exception:
                        logger.exception("Failed to decode response.audio.delta")
                        continue

                    audio_buffer.extend(delta_bytes)

                elif etype == "input_audio_buffer.speech_started":
                    user_speaking_vad = True
                    logger.info("OpenAI VAD: user speech START")
                    if assistant_actively_speaking():
                        await handle_barge_in()

                elif etype == "input_audio_buffer.speech_stopped":
                    user_speaking_vad = False
                    logger.info("OpenAI VAD: user speech STOP")

                elif etype == "conversation.item.input_audio_transcription.completed":
                    if transcript_action_task and not transcript_action_task.done():
                        transcript_action_task.cancel()
                    transcript_action_task = asyncio.create_task(handle_transcript_event(event))

                elif etype == "error":
                    err = (event.get("error") or {})
                    code = err.get("code")
                    if code == "response_cancel_not_active":
                        logger.info("OpenAI cancel race (expected): %s", event)
                    else:
                        logger.error("OpenAI error event: %s", event)

        except websockets.ConnectionClosed:
            logger.info("OpenAI Realtime WebSocket closed")
        except Exception:
            logger.exception("Error in OpenAI event loop")

    async def twilio_loop():
        nonlocal stream_sid, prebuffer_active, twilio_ws_closed

        try:
            async for msg in websocket.iter_text():
                try:
                    data = json.loads(msg)
                except json.JSONDecodeError:
                    logger.warning("Non-JSON frame from Twilio: %r", msg)
                    continue

                event_type = data.get("event")

                if event_type == "connected":
                    logger.info("Twilio stream event: connected")
                    logger.info("Twilio reports call connected")

                elif event_type == "start":
                    start = data.get("start", {})
                    stream_sid = start.get("streamSid")
                    prebuffer_active = True
                    logger.info("Twilio stream event: start")
                    logger.info("Stream started: %s", stream_sid)

                elif event_type == "media":
                    if not openai_ws:
                        continue
                    media = data.get("media", {})
                    payload = media.get("payload")
                    if not payload:
                        continue

                    try:
                        base64.b64decode(payload)
                    except Exception:
                        logger.exception("Failed to base64-decode Twilio payload")
                        continue

                    await openai_ws.send(json.dumps({
                        "type": "input_audio_buffer.append",
                        "audio": payload,
                    }))

                elif event_type == "stop":
                    logger.info("Twilio stream event: stop")
                    logger.info("Twilio sent stop; closing call.")
                    twilio_ws_closed = True
                    break

        except WebSocketDisconnect:
            logger.info("Twilio WebSocket disconnected")
            twilio_ws_closed = True
        except Exception:
            logger.exception("Error in Twilio event loop")

    try:
        openai_ws = await create_realtime_session()
        logger.info("connection open")

        await asyncio.gather(
            openai_loop(),
            twilio_loop(),
        )

    finally:
        try:
            if transcript_action_task and not transcript_action_task.done():
                transcript_action_task.cancel()
        except Exception:
            pass

        try:
            if sender_task:
                sender_task.cancel()
        except Exception:
            pass

        try:
            if openai_ws is not None:
                await openai_ws.close()
        except Exception:
            logger.exception("Error closing OpenAI WebSocket")

        try:
            await websocket.close()
        except Exception:
            logger.exception("Error closing Twilio WebSocket")

        logger.info("WebSocket disconnected while sending audio")

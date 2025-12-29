# vozlia_twilio/stream.py
from __future__ import annotations

import asyncio
import base64
import json
import os
import time
from collections import deque
from typing import Optional

import websockets
from fastapi import WebSocket, WebSocketDisconnect

from core.logging import logger
from core.config import (
    REALTIME_LOG_TEXT,
    REALTIME_LOG_ALL_EVENTS,
    SKILL_GATED_ROUTING,
    OPENAI_INTERRUPT_RESPONSE,
    BYTES_PER_FRAME,
    FRAME_INTERVAL,
    PREBUFFER_BYTES,
    MAX_TWILIO_BACKLOG_SECONDS,
    OPENAI_REALTIME_URL,
    OPENAI_REALTIME_HEADERS,
    VOICE_NAME,
    REALTIME_SYSTEM_PROMPT,
    REALTIME_INPUT_AUDIO_FORMAT,
    REALTIME_OUTPUT_AUDIO_FORMAT,
    REALTIME_VAD_THRESHOLD,
    REALTIME_VAD_SILENCE_MS,
    REALTIME_VAD_PREFIX_MS,
)

from db import SessionLocal
from services.user_service import get_or_create_primary_user
from services.settings_service import get_realtime_prompt_addendum, get_agent_greeting
from services.longterm_memory import (
    fetch_recent_memory_text,
    longterm_memory_enabled_for_tenant,
    record_skill_result,
)

from core.fsm_router_client import call_fsm_router

# Speech output controller (shadow-mode wiring)
from vozlia_twilio.speech_controller import (
    SpeechOutputController,
    TenantSpeechDefaults,
    TenantSpeechPolicyMap,
    ExecutionContext,
    SpeechRequest,
)


# ---------------------------------------------------------------------------
# Tenant speech policy provider (tenant-level settings; NOT per-skill)
# ---------------------------------------------------------------------------
def _tenant_policy_provider(tenant_id: str):
    defaults = TenantSpeechDefaults(
        priority_default=50,
        speech_mode_default="natural",
        conversation_mode_default="default",
        can_interrupt_default=True,
        barge_grace_ms_default=int(os.getenv("BARGE_IN_GRACE_MS", "250") or 250),
        barge_debounce_ms_default=int(os.getenv("BARGE_IN_DEBOUNCE_MS", "200") or 200),
        max_chars_default=int(os.getenv("SPEECH_MAX_CHARS", "900") or 900),
    )
    policy_map: TenantSpeechPolicyMap = {}
    return defaults, policy_map


def _normalize_text(s: str) -> str:
    t = (s or "").lower()
    out = []
    for ch in t:
        out.append(ch if (ch.isalnum() or ch.isspace()) else " ")
    return " ".join("".join(out).split())


def get_style_for_feature(feature: str) -> str:
    key = f"VOZLIA_STYLE_{feature.upper()}"
    s = (os.getenv(key, "") or "").strip().lower()
    if s in {"warm", "concise"}:
        return s

    default = (os.getenv("VOZLIA_DEFAULT_STYLE", "warm") or "warm").strip().lower()
    return default if default in {"warm", "concise"} else "warm"


HARD_IGNORE = {"um", "uh", "er", "hmm", "mm", "mmm", "uh huh", "mhm"}
ACKS = {"awesome", "great", "okay", "ok", "thanks", "thank you", "right", "cool"}
CONTINUE_TRIGGERS = {"continue", "go on", "keep going", "tell me more", "what else"}


def should_reply(text: str, style: str, *, is_skill_intent: bool) -> bool:
    n = _normalize_text(text)
    if not n:
        return False

    if n in HARD_IGNORE:
        return False

    if n in CONTINUE_TRIGGERS:
        return True

    if style == "concise":
        concise_acks = os.getenv("VOZLIA_CONCISE_ACKS", "0") == "1"
        if (n in ACKS) and (not concise_acks):
            return False

        if len(n.split()) <= 2 and not is_skill_intent:
            return False

        return True

    return True


def _build_realtime_instructions(base: str, prompt_addendum: Optional[str]) -> str:
    add = (prompt_addendum or "").strip()
    if not add:
        return base

    delimiter = "--- PORTAL OPENING RULE ---"
    if add.startswith(delimiter):
        add = add[len(delimiter):].lstrip("\n ").strip()

    return f"{base}\n\n{delimiter}\n{add}"


async def create_realtime_session(prompt_addendum: str, agent_greeting: str):
    logger.info(f"Connecting to OpenAI Realtime at {OPENAI_REALTIME_URL}")

    try:
        ws = await websockets.connect(
            OPENAI_REALTIME_URL,
            extra_headers=OPENAI_REALTIME_HEADERS,
            max_size=16 * 1024 * 1024,
            ping_interval=None,
            ping_timeout=None,
        )
    except TypeError:
        ws = await websockets.connect(
            OPENAI_REALTIME_URL,
            additional_headers=OPENAI_REALTIME_HEADERS,
            max_size=16 * 1024 * 1024,
            ping_interval=None,
            ping_timeout=None,
        )

    instructions = _build_realtime_instructions(REALTIME_SYSTEM_PROMPT, prompt_addendum)

    session_update = {
        "type": "session.update",
        "session": {
            "turn_detection": {
                "type": "server_vad",
                "threshold": REALTIME_VAD_THRESHOLD,
                "silence_duration_ms": REALTIME_VAD_SILENCE_MS,
                "prefix_padding_ms": REALTIME_VAD_PREFIX_MS,
                "create_response": False,
                "interrupt_response": OPENAI_INTERRUPT_RESPONSE,
            },
            "input_audio_format": REALTIME_INPUT_AUDIO_FORMAT,
            "output_audio_format": REALTIME_OUTPUT_AUDIO_FORMAT,
            "voice": VOICE_NAME,
            "instructions": instructions,
            "input_audio_transcription": {"model": "whisper-1"},
        },
    }

    await ws.send(json.dumps(session_update))
    logger.info("Sent session.update to OpenAI Realtime")

    opening = (agent_greeting or "").strip()
    if os.getenv("FORCE_REALTIME_OPENING", "1") == "1" and opening:
        evt = {
            "type": "response.create",
            "response": {
                "instructions": (
                    "CALL OPENING (FIRST UTTERANCE ONLY): "
                    "Say EXACTLY this one sentence with no extra words before or after: "
                    f"\"{opening}\""
                ),
            },
        }
    else:
        evt = {"type": "response.create"}

    await ws.send(json.dumps(evt))
    logger.info("Sent initial greeting request to OpenAI Realtime")

    return ws


async def twilio_stream(websocket: WebSocket):
    prompt_addendum = ""
    agent_greeting = ""
    db = SessionLocal()
    try:
        user = get_or_create_primary_user(db)
        prompt_addendum = get_realtime_prompt_addendum(db, user)
        agent_greeting = get_agent_greeting(db, user)

        logger.info("Realtime prompt addendum loaded (len=%d)", len(prompt_addendum or ""))
        logger.info("Agent greeting loaded (len=%d)", len(agent_greeting or ""))

    except Exception:
        logger.exception("Failed to load settings; proceeding with defaults")
        prompt_addendum = ""
        agent_greeting = ""
    finally:
        db.close()

    await websocket.accept()
    logger.info("Twilio media stream connected")

    openai_ws: Optional[websockets.WebSocketClientProtocol] = None
    speech_ctrl: Optional[SpeechOutputController] = None
    stream_sid: Optional[str] = None
    call_sid: Optional[str] = None
    from_number: Optional[str] = None

    barge_in_enabled: bool = False
    twilio_ws_closed: bool = False
    transcript_action_task: Optional[asyncio.Task] = None
    user_speaking_vad: bool = False

    audio_buffer = bytearray()
    assistant_last_audio_time: float = 0.0
    prebuffer_active: bool = True

    active_response_id: Optional[str] = None

    # --- Chitchat durable memory plumbing (maps user turn -> response_id -> assistant transcript)
    pending_response_queue = deque()  # FIFO of dicts for next response.created
    response_meta_by_id = {}  # response_id -> meta

    def assistant_actively_speaking() -> bool:
        if audio_buffer:
            return True
        window_s = float(os.getenv("ASSISTANT_SPEAKING_RECENCY_S", "1.5") or 1.5)
        if assistant_last_audio_time and (time.monotonic() - assistant_last_audio_time) < window_s:
            return True
        return False

    async def send_audio_to_twilio():
        nonlocal audio_buffer, assistant_last_audio_time

        if stream_sid is None:
            return
        if len(audio_buffer) < BYTES_PER_FRAME:
            return

        frame = bytes(audio_buffer[:BYTES_PER_FRAME])
        del audio_buffer[:BYTES_PER_FRAME]

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
                    if len(audio_buffer) < PREBUFFER_BYTES:
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
                audio_sent_duration = frame_idx * FRAME_INTERVAL
                backlog_seconds = audio_sent_duration - call_elapsed
                if backlog_seconds > MAX_TWILIO_BACKLOG_SECONDS:
                    await asyncio.sleep(0.005)
                    continue

                target = send_start_ts + frame_idx * FRAME_INTERVAL
                now = time.monotonic()
                if now < target:
                    await asyncio.sleep(target - now)
                    continue

                late_ms = (time.monotonic() - target) * 1000.0
                if late_ms > late_ms_max:
                    late_ms_max = late_ms

                if len(audio_buffer) >= BYTES_PER_FRAME:
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
        nonlocal active_response_id, prebuffer_active
        if not barge_in_enabled:
            logger.info("BARGE-IN: ignored (not yet enabled)")
            return

        if not assistant_actively_speaking():
            logger.info("BARGE-IN: assistant not actively speaking; nothing to mute")
            return

        logger.info("BARGE-IN: user speech started while AI speaking; clearing audio buffer.")

        # Cancel server-side generation if possible (best-effort)
        if openai_ws is not None and active_response_id is not None:
            rid = active_response_id
            try:
                await openai_ws.send(json.dumps({"type": "response.cancel", "response_id": rid}))
                logger.info("BARGE-IN: Sent response.cancel for %s", rid)
            except Exception:
                logger.exception("BARGE-IN: Failed sending response.cancel for %s", rid)

        await twilio_clear_buffer()
        audio_buffer.clear()
        prebuffer_active = True
        active_response_id = None

    EMAIL_KEYWORDS_LOCAL = [
        "email",
        "emails",
        "e-mail",
        "e-mails",
        "e mail",
        "e mails",
        "inbox",
        "gmail",
        "g mail",
        "mailbox",
        "my mail",
        "my messages",
        "unread",
        "new mail",
        "new emails",
        "today's emails",
        "today emails",
        "read my email",
        "read my emails",
        "check my email",
        "check my emails",
        "how many emails",
        "how many messages",
        "email today",
        "emails today",
        "summary of my email",
        "summary of my emails",
        "summary of my e mail",
        "summary of my e mails",
    ]

    def looks_like_email_intent(text: str) -> bool:
        if not text:
            return False
        normalized = _normalize_text(text)

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

    async def route_to_fsm_and_get_reply(transcript: str) -> Optional[str]:
        ctx = {"channel": "phone"}
        if stream_sid:
            ctx["stream_sid"] = stream_sid
        if call_sid:
            ctx["call_sid"] = call_sid
        if from_number:
            ctx["from_number"] = from_number
        data = await call_fsm_router(transcript, context=ctx)
        if isinstance(data, dict):
            return (data.get("spoken_reply") or data.get("reply") or data.get("text") or "").strip() or None
        if isinstance(data, str):
            return data.strip() or None
        return None

    async def _cancel_active_and_clear_buffer(reason: str):
        nonlocal active_response_id, prebuffer_active

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

    async def create_generic_response(user_text: str | None = None):
        # Queue meta so we can persist assistant transcript when response finishes.
        try:
            tenant_uuid = os.getenv("VOZLIA_TENANT_ID") or os.getenv("TENANT_ID") or "default"
            if from_number and longterm_memory_enabled_for_tenant(str(tenant_uuid)):
                pending_response_queue.append(
                    {
                        "kind": "chitchat",
                        "tenant_uuid": str(tenant_uuid),
                        "caller_id": from_number,
                        "user_text": (user_text or "").strip() or None,
                        "t0": time.monotonic(),
                    }
                )
        except Exception:
            logger.exception("CHITCHAT_MEM_QUEUE_FAILED")

        await _cancel_active_and_clear_buffer("create_generic_response")
        await openai_ws.send(json.dumps({"type": "response.create"}))
        logger.info("Sent generic response.create for chit-chat turn")

    async def create_fsm_spoken_reply(spoken_reply: str):
        if not spoken_reply:
            logger.warning("create_fsm_spoken_reply called with empty spoken_reply")
            await create_generic_response()
            return

        instructions = (
            "You are on a live phone call as Vozlia.\n"
            "The secure backend has already checked the caller's email account and produced a short summary.\n\n"
            "Here is the summary you must speak to the caller:\n"
            f"\"{spoken_reply}\"\n\n"
            "For THIS response only:\n"
            "- Say this summary naturally.\n"
            "- You MAY lightly rephrase for flow, but keep all important facts.\n"
            "- DO NOT mention tools, security, privacy, or inability to access email.\n"
            "- DO NOT apologize or refuse.\n"
        )

        tool_only = os.getenv("SPEECH_CONTROLLER_TOOL_ONLY", "0") == "1"
        use_ctrl = (speech_ctrl is not None and getattr(speech_ctrl, "enabled", False) and tool_only)

        logger.info(
            "FSM_SPEECH_SEND_PATH use_ctrl=%s tool_only=%s ctrl_enabled=%s",
            use_ctrl,
            tool_only,
            (getattr(speech_ctrl, "enabled", None) if speech_ctrl is not None else None),
        )

        if use_ctrl:
            await _cancel_active_and_clear_buffer("create_fsm_spoken_reply_ctrl")

            tenant_id = os.getenv("VOZLIA_TENANT_ID") or os.getenv("TENANT_ID") or "default"
            ctx = ExecutionContext(
                tenant_id=str(tenant_id),
                call_sid=None,
                session_id=None,
                skill_key="fsm",
            )

            req = SpeechRequest(
                text=spoken_reply,
                reason="fsm_spoken_reply",
                ctx=ctx,
                instructions_override=instructions,
                content_text_override=spoken_reply,
            )
            try:
                ok = await speech_ctrl.enqueue(req)
            except Exception:
                logger.exception("FSM_SPEECH_CONTROLLER_ENQUEUE_EXCEPTION")
                ok = False

            if ok:
                logger.info("FSM_SPEECH_CONTROLLER_ENQUEUED trace_id=%s reason=%s", req.trace_id, req.reason)
                return
            logger.warning("FSM_SPEECH_CONTROLLER_FALLBACK_LEGACY")

        await _cancel_active_and_clear_buffer("create_fsm_spoken_reply")
        await openai_ws.send(
            json.dumps(
                {
                    "type": "response.create",
                    "response": {"instructions": instructions},
                }
            )
        )
        logger.info("Sent FSM-driven spoken reply into Realtime session")

    async def handle_transcript_event(event: dict):
        transcript: str = (event.get("transcript") or "").strip()
        if not transcript:
            return

        logger.info("USER Transcript completed: %r", transcript)

        is_email = looks_like_email_intent(transcript)
        feature = "email" if is_email else "chitchat"
        style = get_style_for_feature(feature)

        if not should_reply(transcript, style, is_skill_intent=is_email):
            logger.info("Ignoring transcript (style=%s feature=%s): %r", style, feature, transcript)
            return

        if SKILL_GATED_ROUTING and not is_email:
            logger.info("Skill-gated routing: bypassing /assistant/route for non-email utterance: %r", transcript)
            await create_generic_response(transcript)
            return

        spoken_reply = await route_to_fsm_and_get_reply(transcript)

        if spoken_reply:
            await create_fsm_spoken_reply(spoken_reply)
        else:
            await create_generic_response(transcript)

    def _log_realtime_audio_transcript_delta(event: dict):
        delta = event.get("delta")
        if isinstance(delta, str) and delta.strip():
            logger.info("Realtime assistant said (delta): %r", delta)

    def _log_realtime_text_delta(event: dict):
        delta = event.get("delta")
        if isinstance(delta, dict):
            txt = delta.get("text")
            if isinstance(txt, str) and txt.strip():
                logger.info("Realtime text delta: %r", txt)
                return
        txt2 = event.get("text")
        if isinstance(txt2, str) and txt2.strip():
            logger.info("Realtime text delta: %r", txt2)
            return
        resp = event.get("response") or {}
        if isinstance(resp, dict):
            out = resp.get("output_text") or resp.get("text")
            if isinstance(out, str) and out.strip():
                logger.info("Realtime text delta: %r", out)

    async def maybe_inject_longterm_memory(tenant_uuid: str, from_number: str):
        if not (tenant_uuid and from_number):
            return
        if (os.getenv("REALTIME_LONGTERM_MEMORY", "0") or "").strip().lower() not in ("1", "true", "yes", "on"):
            return
        if not longterm_memory_enabled_for_tenant(tenant_uuid):
            return

        limit = int(os.getenv("LONGTERM_MEMORY_CONTEXT_LIMIT", "8") or 8)

        dbx = SessionLocal()
        try:
            mem_text = fetch_recent_memory_text(
                dbx,
                tenant_uuid=tenant_uuid,
                caller_id=from_number,
                limit=limit,
            )
        finally:
            dbx.close()

        if not mem_text:
            logger.info("REALTIME_MEM_INJECT_SKIP no_memory tenant=%s caller=%s", tenant_uuid, from_number)
            return

        max_chars = int(os.getenv("REALTIME_MEMORY_MAX_CHARS", "1500") or 1500)
        if len(mem_text) > max_chars:
            mem_text = mem_text[: max_chars - 3].rstrip() + "..."

        base_instr = _build_realtime_instructions(REALTIME_SYSTEM_PROMPT, prompt_addendum)
        memory_block = f"\n\n--- CALLER MEMORY (RECENT) ---\n{mem_text}\n"
        new_instr = (base_instr + memory_block).strip()

        try:
            await openai_ws.send(json.dumps({"type": "session.update", "session": {"instructions": new_instr}}))
            logger.info(
                "REALTIME_MEM_INJECT_OK tenant=%s caller=%s chars=%d lines=%d",
                tenant_uuid,
                from_number,
                len(mem_text),
                mem_text.count("\n") + 1,
            )
        except Exception:
            logger.exception("REALTIME_MEM_INJECT_FAILED tenant=%s caller=%s", tenant_uuid, from_number)

    async def openai_loop():
        nonlocal active_response_id, barge_in_enabled, user_speaking_vad, transcript_action_task, prebuffer_active

        try:
            async for raw in openai_ws:
                event = json.loads(raw)

                if speech_ctrl is not None:
                    try:
                        speech_ctrl.on_realtime_event(event)
                    except Exception:
                        logger.exception("SPEECH_CTRL_EVENT_INGEST_ERROR")

                etype = event.get("type")

                if REALTIME_LOG_ALL_EVENTS:
                    logger.info("Realtime event: type=%s keys=%s", etype, list(event.keys()))

                if etype == "response.created":
                    resp = event.get("response", {}) or {}
                    rid = resp.get("id")
                    if rid:
                        active_response_id = rid
                        logger.info("Tracking allowed response_id: %s", rid)
                        if pending_response_queue:
                            meta = pending_response_queue.popleft()
                            response_meta_by_id[rid] = meta
                            logger.info("CHITCHAT_MEM_ATTACHED response_id=%s kind=%s", rid, meta.get("kind"))

                # FIX: treat response.done as completion (your logs show response.done, not response.completed)
                elif etype in ("response.completed", "response.failed", "response.canceled", "response.done"):
                    resp = event.get("response", {}) or {}
                    rid = resp.get("id")
                    if active_response_id is not None and rid == active_response_id:
                        logger.info("Response %s finished with event '%s'; clearing active_response_id", rid, etype)
                        active_response_id = None
                        prebuffer_active = True

                    if not barge_in_enabled:
                        barge_in_enabled = True
                        logger.info("First response finished (event=%s, id=%s); barge-in is now ENABLED.", etype, rid)

                elif etype == "response.audio_transcript.delta":
                    if REALTIME_LOG_TEXT:
                        _log_realtime_audio_transcript_delta(event)

                elif etype == "response.audio_transcript.done":
                    if REALTIME_LOG_TEXT:
                        transcript = event.get("transcript")
                        if transcript:
                            logger.info("Realtime assistant said (final): %r", transcript)

                    # Persist chitchat turns (durable memory) using assistant final transcript.
                    try:
                        rid = event.get("response_id")
                        transcript_final = (event.get("transcript") or "").strip()
                        meta = response_meta_by_id.get(rid) if rid else None
                        if meta and meta.get("kind") == "chitchat" and transcript_final:
                            max_chars = int(os.getenv("LONGTERM_CHAT_TURN_MAX_CHARS", "600") or 600)
                            if len(transcript_final) > max_chars:
                                transcript_final = transcript_final[: max_chars - 3].rstrip() + "..."

                            user_text = (meta.get("user_text") or "").strip() or None
                            tenant_uuid = meta.get("tenant_uuid")
                            caller_id = meta.get("caller_id")

                            async def _persist():
                                def _sync_write():
                                    db2 = SessionLocal()
                                    try:
                                        record_skill_result(
                                            db2,
                                            tenant_uuid=tenant_uuid,
                                            caller_id=caller_id,
                                            skill_key="chitchat",
                                            input_text=user_text,
                                            memory_text=transcript_final,
                                            data_json={"channel": "realtime"},
                                            expires_in_s=None,
                                        )
                                    finally:
                                        db2.close()

                                await asyncio.to_thread(_sync_write)

                            asyncio.create_task(_persist())
                            logger.info("CHITCHAT_MEM_PERSIST_QUEUED response_id=%s", rid)
                            try:
                                response_meta_by_id.pop(rid, None)
                            except Exception:
                                pass
                    except Exception:
                        logger.exception("CHITCHAT_MEM_PERSIST_FAILED")

                elif etype in ("response.output_text.delta", "response.text.delta", "response.output_text"):
                    if REALTIME_LOG_TEXT:
                        _log_realtime_text_delta(event)

                elif etype in ("response.output_text.done", "response.text.done"):
                    if REALTIME_LOG_TEXT:
                        logger.info("Realtime text done")

                elif etype == "response.audio.delta":
                    resp_id = event.get("response_id")
                    delta_b64 = event.get("delta")

                    if resp_id != active_response_id:
                        logger.info("Dropping unsolicited audio for response_id=%s (active=%s)", resp_id, active_response_id)
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
                    else:
                        # Still clear if we spoke very recently; Twilio may still be playing buffered audio.
                        try:
                            window_s = float(os.getenv("ASSISTANT_SPEAKING_RECENCY_S", "1.5") or 1.5)
                            if assistant_last_audio_time and (time.monotonic() - assistant_last_audio_time) < window_s:
                                await handle_barge_in()
                        except Exception:
                            pass

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
                        if speech_ctrl is not None:
                            try:
                                dbg = speech_ctrl.get_last_tool_payload_debug()
                                logger.error("SPEECH_CTRL_LAST_TOOL_PAYLOAD_ON_ERROR %s", dbg)
                            except Exception:
                                logger.exception("SPEECH_CTRL_LAST_TOOL_PAYLOAD_ON_ERROR_FAILED")

        except websockets.ConnectionClosed:
            logger.info("OpenAI Realtime WebSocket closed")
        except Exception:
            logger.exception("Error in OpenAI event loop")

    async def twilio_loop():
        nonlocal stream_sid, prebuffer_active, twilio_ws_closed, call_sid, from_number

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
                    call_sid = start.get("callSid") or start.get("call_sid")
                    custom = start.get("customParameters") or {}
                    from_number = custom.get("from") or custom.get("From") or start.get("from") or start.get("From")

                    prebuffer_active = True
                    logger.info("Twilio stream event: start")
                    logger.info("Stream started: %s", stream_sid)

                    # Inject durable memory at call start (safe: one-time, not per-frame)
                    try:
                        tenant_uuid = os.getenv("VOZLIA_TENANT_ID") or os.getenv("TENANT_ID") or "default"
                        if from_number:
                            await maybe_inject_longterm_memory(str(tenant_uuid), from_number)
                    except Exception:
                        logger.exception("REALTIME_MEM_INJECT_EXCEPTION")

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

                    try:
                        await openai_ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": payload}))
                    except Exception:
                        logger.exception("OpenAI WS send failed while streaming audio; ending call loop")
                        twilio_ws_closed = True
                        break

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
        openai_ws = await create_realtime_session(prompt_addendum, agent_greeting)
        logger.info("connection open")

        async def _send_realtime_json(payload: dict):
            await openai_ws.send(json.dumps(payload))

        speech_ctrl = SpeechOutputController(
            tenant_defaults_provider=_tenant_policy_provider,
            send_realtime_json=_send_realtime_json,
            cancel_active_cb=_cancel_active_and_clear_buffer,
            clear_audio_buffer_cb=None,
            name="speech_ctrl",
        )
        speech_ctrl.start()
        logger.info(
            "SPEECH_CTRL_WIRED enabled=%s shadow=%s fail_open=%s",
            os.getenv("SPEECH_CONTROLLER_ENABLED", "0"),
            os.getenv("SPEECH_CONTROLLER_SHADOW", "1"),
            os.getenv("SPEECH_CONTROLLER_FAILOPEN", "1"),
        )

        await asyncio.gather(openai_loop(), twilio_loop())

    finally:
        try:
            if transcript_action_task and not transcript_action_task.done():
                transcript_action_task.cancel()
        except Exception:
            pass

        try:
            if speech_ctrl is not None:
                await speech_ctrl.stop()
        except Exception:
            logger.exception("SPEECH_CTRL_STOP_ERROR")

        try:
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

# vozlia_twilio/stream.py
from __future__ import annotations

import asyncio
import base64
import json
import os
import time
from contextlib import suppress
from typing import Optional
from starlette.websockets import WebSocketState
from db import SessionLocal
from services.user_service import get_or_create_primary_user
#from services.settings_service import get_realtime_prompt_addendum
from services.settings_service import get_realtime_prompt_addendum, get_agent_greeting



import websockets
from fastapi import WebSocket, WebSocketDisconnect

from core.logging import logger

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
# Step 2: placeholder implementation (env-driven defaults, no per-reason overrides).
# Later: load from control-plane settings cache.
# ---------------------------------------------------------------------------
def _tenant_policy_provider(tenant_id: str):
    defaults = TenantSpeechDefaults(
        priority_default=50,
        speech_mode_default="natural",
        conversation_mode_default="auto",
        can_interrupt_default=True,
        barge_grace_ms_default=int(os.getenv("BARGE_IN_GRACE_MS", "250") or 250),
        barge_debounce_ms_default=int(os.getenv("BARGE_IN_DEBOUNCE_MS", "200") or 200),
        max_chars_default=int(os.getenv("SPEECH_MAX_CHARS", "900") or 900),
    )
    policy_map: TenantSpeechPolicyMap = {}
    return defaults, policy_map


# Config / constants (env-driven)
from core.config import (
    # logging toggles
    REALTIME_LOG_TEXT,
    REALTIME_LOG_ALL_EVENTS,
    # feature flags
    SKILL_GATED_ROUTING,
    OPENAI_INTERRUPT_RESPONSE,
    # twilio audio constants
    BYTES_PER_FRAME,
    FRAME_INTERVAL,
    PREBUFFER_BYTES,
    MAX_TWILIO_BACKLOG_SECONDS,
    # realtime session config
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

# Router client (Flow B)
# NOTE: This must be implemented as an async function returning a dict (or adjust below accordingly).
#from services.fsm_router_client import call_fsm_router
from core.fsm_router_client import call_fsm_router



def _normalize_text(s: str) -> str:
    t = (s or "").lower()
    out = []
    for ch in t:
        out.append(ch if (ch.isalnum() or ch.isspace()) else " ")
    return " ".join("".join(out).split())


def get_style_for_feature(feature: str) -> str:
    # feature: "email", "chitchat", "calendar", etc.
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

    # Never ignore continuation commands
    if n in CONTINUE_TRIGGERS:
        return True

    if style == "concise":
        # In concise mode, ignore acknowledgements unless configured otherwise
        concise_acks = os.getenv("VOZLIA_CONCISE_ACKS", "0") == "1"
        if (n in ACKS) and (not concise_acks):
            return False

        # Also ignore super-short non-skill utterances
        if len(n.split()) <= 2 and not is_skill_intent:
            return False

        return True

    # Warm: respond to almost everything
    return True


def _build_realtime_instructions(base: str, prompt_addendum: Optional[str]) -> str:
    """
    Build the Realtime `instructions` string for session.update.

    Hardening rules:
    - Only append once (this function is only called at session start).
    - Ignore empty/whitespace addenda.
    - Strip leading/trailing whitespace on addendum.
    - Insert a clear delimiter so the "portal opening rule" stays scoped and readable.
    """
    add = (prompt_addendum or "").strip()
    if not add:
        return base

    # Prevent accidental double-delimiter if the saved addendum already contains it.
    delimiter = "--- PORTAL OPENING RULE ---"
    if add.startswith(delimiter):
        add = add[len(delimiter):].lstrip("\n ").strip()

    return f"{base}\n\n{delimiter}\n{add}"


async def create_realtime_session(prompt_addendum: str, agent_greeting: str):
    """
    Connect to OpenAI Realtime WS and send session.update + an initial greeting.
    """
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
        # Newer websockets versions renamed extra_headers -> additional_headers
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

    # Keep this behind a flag so you can disable instantly without rollback if needed
    if os.getenv("FORCE_REALTIME_OPENING", "1") == "1" and opening:
        evt = {
            "type": "response.create",
            "response": {
                # per-response instructions override session instructions for this response :contentReference[oaicite:1]{index=1}
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
    """
    Pattern 1 (no response_id adoption):
    """
    # Load portal-controlled Realtime prompt addendum ONCE per call (not in hot path)
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
        #logger.exception("Failed to load realtime prompt addendum; proceeding without it")
        logger.exception("Failed to load settings; proceeding with defaults")
        prompt_addendum = ""
        agent_greeting = ""
    finally:
        db.close()

    await websocket.accept()
    logger.info("Twilio media stream connected")


    # --- Call + AI state -----------------------------------------------------
    openai_ws: Optional[websockets.WebSocketClientProtocol] = None
    speech_ctrl: Optional[SpeechOutputController] = None
    stream_sid: Optional[str] = None
    call_sid: Optional[str] = None
    from_number: Optional[str] = None

    barge_in_enabled: bool = False
    twilio_ws_closed: bool = False

    def _ws_can_send() -> bool:
        try:
            return websocket.application_state == WebSocketState.CONNECTED
        except Exception:
            return True
    transcript_action_task: Optional[asyncio.Task] = None
    user_speaking_vad: bool = False

    audio_buffer = bytearray()
    assistant_last_audio_time: float = 0.0
    prebuffer_active: bool = True

    # Response tracking (Pattern 1)
    active_response_id: Optional[str] = None

    # --- Simple helper: is assistant currently speaking? ---------------------
    def assistant_actively_speaking() -> bool:
        if audio_buffer:
            return True
        if assistant_last_audio_time and (time.monotonic() - assistant_last_audio_time) < 0.5:
            return True
        return False

    # --- Helper: send μ-law audio TO Twilio ---------------------------------
    async def send_audio_to_twilio():
        nonlocal audio_buffer, assistant_last_audio_time, twilio_ws_closed

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
        }        if twilio_ws_closed or not _ws_can_send():
            return
        try:
            await websocket.send_text(json.dumps(msg))
        except RuntimeError as e:
            # Starlette raises this if we try to send after websocket.close / response completed.
            if "Unexpected ASGI message" in str(e) or ("websocket.send" in str(e) and "websocket.close" in str(e)):
                logger.info("Twilio WS send after close (expected during teardown); stopping sender")
            else:
                logger.exception("Twilio WS send RuntimeError")
            twilio_ws_closed = True
            return
        except Exception:
            logger.exception("Twilio WS send failed; stopping sender")
            twilio_ws_closed = True
            return
        assistant_last_audio_time = time.monotonic()

    # --- Background task: paced audio sender to Twilio ----------------------
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

                # 1Hz stats
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

                # idle reset
                if len(audio_buffer) == 0:
                    if assistant_last_audio_time and (time.monotonic() - assistant_last_audio_time) > 1.0:
                        send_start_ts = None
                        frame_idx = 0
                    await asyncio.sleep(0.005)
                    continue

                # prebuffer at utterance start
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

                # backlog cap
                call_elapsed = now - send_start_ts
                audio_sent_duration = frame_idx * FRAME_INTERVAL
                backlog_seconds = audio_sent_duration - call_elapsed
                if backlog_seconds > MAX_TWILIO_BACKLOG_SECONDS:
                    await asyncio.sleep(0.005)
                    continue

                # deadline-based pacing
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
        nonlocal twilio_ws_closed
        if stream_sid is None or twilio_ws_closed or not _ws_can_send():
            return
        try:
            await websocket.send_text(json.dumps({"event": "clear", "streamSid": stream_sid}))
        except RuntimeError as e:
            if "Unexpected ASGI message" in str(e):
                logger.info("Twilio clear after close (expected during teardown)")
            else:
                logger.exception("Twilio clear RuntimeError")
            twilio_ws_closed = True
        except Exception:
            logger.exception("Failed to send Twilio clear")
            twilio_ws_closed = True

    # --- Barge-in: local mute only ------------------------------------------
    async def handle_barge_in():
        """
        Cancel the active OpenAI response on barge-in and clear Twilio audio.
        """
        nonlocal active_response_id, prebuffer_active
        if not barge_in_enabled:
            logger.info("BARGE-IN: ignored (not yet enabled)")
            return

        if not assistant_actively_speaking():
            logger.info("BARGE-IN: assistant not actively speaking; nothing to mute")
            return

        logger.info(
            "BARGE-IN: user speech started while AI speaking; canceling active response and clearing audio buffer."
        )

        # Cancel server-side generation if possible
        if openai_ws is not None and active_response_id is not None:
            rid = active_response_id
            try:
                await openai_ws.send(json.dumps({"type": "response.cancel", "response_id": rid}))
                logger.info("BARGE-IN: Sent response.cancel for %s", rid)
            except Exception:
                logger.exception("BARGE-IN: Failed sending response.cancel for %s", rid)

        # Clear local audio immediately
        await twilio_clear_buffer()
        audio_buffer.clear()

        # Reset playback state
        prebuffer_active = True
        active_response_id = None

    # --- Intent helpers ------------------------------------------------------
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

    # --- FSM router ----------------------------------------------------------
    async def route_to_fsm_and_get_reply(transcript: str) -> Optional[str]:
        try:
            ctx = {"channel": "phone"}
            if stream_sid:
                ctx["stream_sid"] = stream_sid
            if call_sid:
                ctx["call_sid"] = call_sid
            if from_number:
                ctx["from_number"] = from_number

            data = await call_fsm_router(transcript, context=ctx)
            if isinstance(data, dict):
                # Common patterns we’ve used across codepaths
                spoken = (
                    data.get("spoken_reply")
                    or (data.get("result") or {}).get("spoken_reply")
                    or (data.get("skill_result") or {}).get("spoken_reply")
                )
                if isinstance(spoken, str) and spoken.strip():
                    return spoken.strip()
            return None
        except Exception:
            logger.exception("FSM_ROUTE_ERROR")
            return None


    # --- Cancel active response & clear audio buffer -------------------------
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

    # --- Create responses ----------------------------------------------------
    async def create_generic_response():
        await _cancel_active_and_clear_buffer("create_generic_response")
        await openai_ws.send(json.dumps({"type": "response.create"}))
        logger.info("Sent generic response.create for chit-chat turn")

    async def create_fsm_spoken_reply(spoken_reply: str):
        if not spoken_reply:
            logger.warning("create_fsm_spoken_reply called with empty spoken_reply")
            await create_generic_response()
            return

        # Cancellation handled in send-path selection (controller vs legacy)

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

        # Step 3: tool/FSM speech cutover (controller owns response.create) behind flag.
        tool_only = os.getenv("SPEECH_CONTROLLER_TOOL_ONLY", "0") == "1"
        use_ctrl = (speech_ctrl is not None and getattr(speech_ctrl, "enabled", False) and tool_only)

        logger.info(
            "FSM_SPEECH_SEND_PATH use_ctrl=%s tool_only=%s ctrl_enabled=%s",
            use_ctrl,
            tool_only,
            (getattr(speech_ctrl, "enabled", None) if speech_ctrl is not None else None),
        )

        if use_ctrl:
            # Preserve existing behavior: cancel any active response first (same as legacy path).
            await _cancel_active_and_clear_buffer("create_fsm_spoken_reply_ctrl")

            tenant_id = os.getenv("VOZLIA_TENANT_ID") or os.getenv("TENANT_ID") or "default"
            ctx = ExecutionContext(
                tenant_id=str(tenant_id),
                call_sid=call_sid,
                session_id=stream_sid,
                skill_key="fsm",
            )

            req = SpeechRequest(
                text=spoken_reply,
                reason="fsm_spoken_reply",
                ctx=ctx,
                instructions_override=instructions,  # preserve legacy scaffolding
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

        # Legacy path (unchanged)
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

    # --- Transcript handling -------------------------------------------------
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

        # Optional: skill-gated routing
        if SKILL_GATED_ROUTING and not is_email:
            logger.info(
                "Skill-gated routing: bypassing /assistant/route for non-email utterance: %r",
                transcript,
            )
            await create_generic_response()
            return

        spoken_reply = await route_to_fsm_and_get_reply(transcript)

        if spoken_reply:
            await create_fsm_spoken_reply(spoken_reply)
        else:
            await create_generic_response()

    # --- Logging helpers -----------------------------------------------------
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

    # --- OpenAI event loop ---------------------------------------------------
    async def openai_loop():
        nonlocal active_response_id, barge_in_enabled, user_speaking_vad, transcript_action_task, prebuffer_active

        try:
            async for raw in openai_ws:
                event = json.loads(raw)
                # SpeechOutputController (Step 2): observe Realtime lifecycle events (shadow mode)
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

                elif etype in ("response.completed", "response.failed", "response.canceled"):
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

                elif etype in ("response.output_text.delta", "response.text.delta", "response.output_text"):
                    if REALTIME_LOG_TEXT:
                        _log_realtime_text_delta(event)

                elif etype in ("response.output_text.done", "response.text.done"):
                    if REALTIME_LOG_TEXT:
                        logger.info("Realtime text done")

                elif etype == "response.audio.delta":
                    resp_id = event.get("response_id")
                    delta_b64 = event.get("delta")

                    # Pattern 1: ONLY accept audio for the active_response_id
                    if resp_id != active_response_id:
                        logger.info(
                            "Dropping unsolicited audio for response_id=%s (active=%s)",
                            resp_id,
                            active_response_id,
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
                        # Attach last tool payload trace (if controller is wired) for immediate diagnosis.
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

    # --- Twilio event loop ---------------------------------------------------
    async def twilio_loop():
        nonlocal stream_sid, call_sid, from_number, prebuffer_active, twilio_ws_closed

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
            

    # --- Main orchestration --------------------------------------------------
    try:
        openai_ws = await create_realtime_session(prompt_addendum, agent_greeting)
        logger.info("connection open")

        # -------------------------------------------------------------------
        # SpeechOutputController (Step 2): shadow wiring (observe-only by default)
        # No behavior changes: controller is disabled unless SPEECH_CONTROLLER_ENABLED=1
        # -------------------------------------------------------------------
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
            with suppress(asyncio.CancelledError):
                await sender_task
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

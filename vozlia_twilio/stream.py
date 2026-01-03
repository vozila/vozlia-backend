# vozlia_twilio/stream.py
from __future__ import annotations

import asyncio
import base64
import json
import os
import time
import re
from typing import Optional
from contextlib import suppress
from db import SessionLocal
from models import CallerMemoryEvent
from services.user_service import get_or_create_primary_user
#from services.settings_service import get_realtime_prompt_addendum
from services.settings_service import get_realtime_prompt_addendum, get_agent_greeting, get_skills_config
from services.gmail_service import get_default_gmail_account_id
from services.call_summary_service import ensure_call_summary_for_call
from skills.registry import skill_registry



import websockets
from fastapi import WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from core.logging import logger

# --- Discovery offer resolution (Add-to-greeting) ----------------------------
# When a skill is announced in the greeting ("Would you like me to ...?"),
# the next user turn is often just "yes/no". The FSM router can't infer context from that,
# so we resolve it deterministically here.

AFFIRMATIVE_PREFIXES = {"yes", "yeah", "yep", "yup", "sure", "ok", "okay"}
NEGATIVE_PREFIXES = {"no", "nope", "nah"}

# Full-phrase matches (after normalization)
AFFIRMATIVE_WORDS = {
    "yes please",
    "yeah please",
    "sure",
    "sure thing",
    "okay",
    "ok",
    "please",
    "do it",
    "go ahead",
    "sounds good",
}
NEGATIVE_WORDS = {
    "no",
    "no thanks",
    "no thank you",
    "nope",
    "nah",
    "not now",
    "dont",
    "don't",
    "do not",
    "not really",
}


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

_ACK_RE = re.compile(
    r"""^\s*(?:"""
    r"""sure|of\s+course|okay|ok|alright|certainly|absolutely|no\s+problem|"""
    r"""right\s+away|got\s+it|sounds\s+good|happy\s+to"""
    r""")\s*[,!.:-]*\s*""",
    re.IGNORECASE,
)

def _strip_ack_preamble(text: str) -> str:
    """Remove leading conversational acknowledgements (esp. for auto-exec)."""
    if not isinstance(text, str):
        return ""
    t = text.strip()
    for _ in range(3):
        new_t = _ACK_RE.sub("", t).strip()
        if new_t == t:
            break
        t = new_t
    return t



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
                    "Say EXACTLY this text with no extra words before or after: "
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
    skills_cfg: dict = {}
    auto_execute_skill_id: str | None = None
    auto_execute_trigger_text: str = ""
    db = SessionLocal()
    try:
        user = get_or_create_primary_user(db)
        # Resolve effective default Gmail account once per call.
        # Used for proactive flows (auto-exec + add-to-greeting) so the intended inbox is explicit.
        default_gmail_account_id: str | None = None
        try:
            default_gmail_account_id = get_default_gmail_account_id(user, db)
        except Exception:
            default_gmail_account_id = None
        logger.info("GMAIL_DEFAULT_ACCOUNT_EFFECTIVE account_id=%s", default_gmail_account_id)
        prompt_addendum = get_realtime_prompt_addendum(db, user)
        agent_greeting = get_agent_greeting(db, user)
        skills_cfg = get_skills_config(db, user) or {}

        logger.info("Realtime prompt addendum loaded (len=%d)", len(prompt_addendum or ""))
        logger.info("Agent greeting loaded (len=%d)", len(agent_greeting or ""))

        # ---- Skill announcement + auto-exec selection (computed once per call; NOT in hot path) ----
        discovery_offer_skill_id: str | None = None
        discovery_offer_trigger_text: str = ""
        try:
            announce_lines: list[str] = []
            auto_exec_candidates: list[str] = []

            if isinstance(skills_cfg, dict):
                # Deterministic order (stable across runs)
                for sid in sorted(skills_cfg.keys()):
                    cfg = skills_cfg.get(sid) or {}
                    if not isinstance(cfg, dict):
                        continue

                    enabled = bool(cfg.get("enabled", True))
                    if not enabled:
                        continue

                    # Discovery toggle (announce in greeting): legacy key name is add_to_greeting
                    if bool(cfg.get("add_to_greeting", False)):
                        sk = skill_registry.get(sid)
                        gline = (getattr(sk, "greeting", "") or "").strip() if sk else ""
                        if gline:
                            announce_lines.append(gline)

                            # Keep ONE pending discovery offer so a bare "yes/no" can be resolved.
                            if discovery_offer_skill_id is None:
                                discovery_offer_skill_id = sid

                                # Choose a trigger phrase (portal engagement_phrases > manifest trigger phrases > fallback)
                                trig = ""
                                phrases = cfg.get("engagement_phrases") or cfg.get("engagementPrompt") or []
                                if isinstance(phrases, list) and phrases:
                                    trig = str(phrases[0]).strip()

                                if not trig and sk is not None:
                                    try:
                                        tphr = getattr(getattr(sk, "trigger", None), "phrases", None)
                                        if isinstance(tphr, list) and tphr:
                                            trig = str(tphr[0]).strip()
                                    except Exception:
                                        pass

                                if not trig:
                                    trig = sid.replace("_", " ")

                                discovery_offer_trigger_text = trig

                    # Execution toggle (auto-execute after greeting): new key
                    if bool(cfg.get("auto_execute_after_greeting", False)):
                        auto_exec_candidates.append(sid)

            # Append announcement to greeting (discovery-only)
            if announce_lines:
                base = (agent_greeting or "").strip()
                suffix = " ".join([s for s in announce_lines if s])
                agent_greeting = (base + (" " if base and suffix else "") + suffix).strip()

            # Choose ONE auto-exec skill deterministically to avoid surprise cascades
            auto_execute_skill_id = auto_exec_candidates[0] if auto_exec_candidates else None


            # Pick a trigger phrase so auto-exec uses the same routing path as voice intent (FSM),
            # instead of sending an empty utterance.
            try:
                if auto_execute_skill_id and isinstance(skills_cfg, dict):
                    cfg0 = skills_cfg.get(auto_execute_skill_id) or {}
                    phrases0 = cfg0.get("engagement_phrases") or cfg0.get("engagementPrompt") or []
                    trig = ""
                    if isinstance(phrases0, list):
                        for p in phrases0:
                            if isinstance(p, str) and p.strip():
                                trig = p.strip()
                                break
                    elif isinstance(phrases0, str) and phrases0.strip():
                        # If portal/CP returned a multiline string, use the first non-empty line
                        for line in phrases0.splitlines():
                            if line.strip():
                                trig = line.strip()
                                break

                    # Fallback per-skill defaults
                    if not trig:
                        if auto_execute_skill_id == "gmail_summary":
                            trig = "email summaries"
                        elif auto_execute_skill_id == "memory":
                            trig = "memory"
                        else:
                            trig = auto_execute_skill_id.replace("_", " ")

                    auto_execute_trigger_text = trig
            except Exception:
                auto_execute_trigger_text = ""
            # Kill-switch: disable auto-exec globally without rollback (still allow announcements)
            if os.getenv("SKILLS_AUTO_EXEC_KILL_SWITCH", "0") == "1":
                auto_execute_skill_id = None

            if auto_execute_skill_id:
                logger.info("Auto-exec candidate selected: %s", auto_execute_skill_id)

        except Exception:
            logger.exception("Failed to compute skill announce/auto-exec; proceeding without")
            auto_execute_skill_id = None

    except Exception:
        #logger.exception("Failed to load realtime prompt addendum; proceeding without it")
        logger.exception("Failed to load settings; proceeding with defaults")
        prompt_addendum = ""
        agent_greeting = ""
    finally:
        db.close()

    await websocket.accept()
    logger.info("Twilio media stream connected")

    def _ws_can_send() -> bool:
        """Return True if the Twilio WS is still open for sending."""
        if twilio_ws_closed:
            return False
        try:
            if getattr(websocket, "client_state", None) != WebSocketState.CONNECTED:
                return False
            if getattr(websocket, "application_state", None) != WebSocketState.CONNECTED:
                return False
        except Exception:
            # If we can't read state, be conservative.
            return False
        return True



    # --- Call + AI state -----------------------------------------------------
    openai_ws: Optional[websockets.WebSocketClientProtocol] = None
    speech_ctrl: Optional[SpeechOutputController] = None
    stream_sid: Optional[str] = None
    call_sid: Optional[str] = None
    from_number: Optional[str] = None

    # Greeting protection: barge-in should never clip or cancel the opening greeting.
    greeting_audio_protected: bool = True
    greeting_drain_task: Optional[asyncio.Task] = None
    pending_transcript_during_greeting: Optional[str] = None

    barge_in_enabled: bool = False
    twilio_ws_closed: bool = False
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
        if not _ws_can_send():
            return
        try:
            await websocket.send_text(json.dumps(msg))
        except RuntimeError:
            # Can happen if the websocket is closing/closed (send after close).
            return
        except WebSocketDisconnect:
            return
        assistant_last_audio_time = time.monotonic()

    # --- Background task: paced audio sender to Twilio ----------------------
    async def twilio_audio_sender():
        nonlocal audio_buffer, prebuffer_active, assistant_last_audio_time, barge_in_enabled, twilio_ws_closed

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


    async def _wait_for_audio_drain(label: str, timeout_s: float = 2.5, target_bytes: int = 0, grace_ms: float = 0.0) -> bool:
        """Wait until the outbound Twilio audio buffer drains.

        This prevents cutting off the end of the greeting when we enqueue an immediate follow-on response
        (e.g., AUTO_EXECUTE_AFTER_GREETING), because create_fsm_spoken_reply cancels/clears the buffer.
        """
        start = time.monotonic()
        while True:
            if twilio_ws_closed:
                return False
            # We consider 'drained' when there's essentially nothing left queued for Twilio.
            if len(audio_buffer) <= target_bytes:
                if grace_ms and grace_ms > 0:
                    await asyncio.sleep(float(grace_ms) / 1000.0)
                return True
            if (time.monotonic() - start) >= timeout_s:
                logger.warning(
                    "AUDIO_DRAIN_TIMEOUT label=%s q_bytes=%d prebuf=%s",
                    label,
                    len(audio_buffer),
                    prebuffer_active,
                )
                return False
            await asyncio.sleep(0.02)



    async def _after_greeting_drain():
        """Release greeting protection once all greeting audio has been sent to Twilio.

        Also processes any transcript that arrived during the greeting, so the caller doesn't lose input.
        """
        nonlocal greeting_audio_protected, pending_transcript_during_greeting

        timeout_s = float(os.getenv("GREETING_DRAIN_TIMEOUT_S", "4.0") or 4.0)
        await _wait_for_audio_drain("greeting_protect", timeout_s=timeout_s, target_bytes=0)

        greeting_audio_protected = False
        logger.info("GREETING_PROTECT: released (audio drained or timeout)")

        # Process the latest transcript captured during the greeting (if any).
        if pending_transcript_during_greeting:
            t = pending_transcript_during_greeting
            pending_transcript_during_greeting = None
            logger.info("GREETING_PROTECT: processing deferred transcript: %r", t)
            try:
                await handle_transcript_event({"transcript": t})
            except Exception:
                logger.exception("DEFERRED_TRANSCRIPT_PROCESS_ERROR")

    async def twilio_clear_buffer():
        if stream_sid is None:
            return
        try:
            if not _ws_can_send():
                return
            try:
                await websocket.send_text(json.dumps({"event": "clear", "streamSid": stream_sid}))
            except RuntimeError:
                return
            except WebSocketDisconnect:
                return
        except Exception:
            logger.exception("Failed to send Twilio clear")

    # --- Barge-in: local mute only ------------------------------------------
    async def handle_barge_in():
        """
        Cancel the active OpenAI response on barge-in and clear Twilio audio.
        """
        nonlocal active_response_id, prebuffer_active
        nonlocal greeting_audio_protected
        if greeting_audio_protected:
            logger.info("BARGE-IN: ignored (greeting protected)")
            return

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

        # Optionally inform the Realtime conversation that the assistant output was interrupted.
        # This helps the model avoid "reset" replies when the caller says "continue/go on".
        if openai_ws is not None and os.getenv("BARGE_IN_CONTEXT_NOTE", "1") == "1":
            try:
                await openai_ws.send(json.dumps({
                    "type": "conversation.item.create",
                    "item": {
                        "type": "message",
                        "role": "system",
                        "content": [
                            {"type": "input_text", "text": "Caller barged in. Your previous response audio was interrupted and may not have been heard. Do NOT restart the conversation. If the caller says continue/go on, continue the interrupted thought rather than greeting again."}
                        ],
                    },
                }))
            except Exception:
                logger.exception("BARGE_IN_CONTEXT_NOTE send failed (non-fatal)")


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
    async def route_to_fsm_and_get_reply(transcript: str, account_id: str | None = None) -> Optional[str]:
        try:
            ctx = {"channel": "phone"}
            if stream_sid:
                ctx["stream_sid"] = stream_sid
            if call_sid:
                ctx["call_sid"] = call_sid
            if from_number:
                ctx["from_number"] = from_number

            data = await call_fsm_router(transcript, context=ctx, account_id=account_id)
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
    
    async def _cancel_active_and_clear_buffer(
        reason: str,
        *,
        clear_twilio_playback: bool = False,
        clear_local_audio_buffer: bool = True,
    ):
        """Cancel the active OpenAI response.

        Critical rule:
        - If there is NO active_response_id, do NOT clear local buffers or Twilio playback.
          Clearing local audio_buffer at that point can clip the tail end of the greeting (or any finished response)
          because Twilio may not have received/sent all frames yet.
        """
        nonlocal active_response_id, prebuffer_active

        if not openai_ws:
            logger.info("_cancel_active_and_clear_buffer: no openai_ws (reason=%s)", reason)
            return

        if not active_response_id:
            logger.info("_cancel_active_and_clear_buffer: no active response (reason=%s)", reason)
            return

        rid = active_response_id
        logger.info("Sent response.cancel for %s due to %s", rid, reason)

        try:
            await openai_ws.send(json.dumps({"type": "response.cancel", "response_id": rid}))
        except Exception:
            logger.exception("Error sending response.cancel for %s", rid)

        active_response_id = None

        if clear_twilio_playback:
            await twilio_clear_buffer()

        if clear_local_audio_buffer:
            audio_buffer.clear()

        prebuffer_active = True


# --- Create responses ----------------------------------------------------
    async def create_generic_response():
        await _cancel_active_and_clear_buffer("create_generic_response")
        await openai_ws.send(json.dumps({"type": "response.create"}))
        logger.info("Sent generic response.create for chit-chat turn")

    async def create_fsm_spoken_reply(spoken_reply: str, *, clear_playback: bool = True, reason_tag: str = "fsm_spoken_reply", verbatim: bool = False):
        if not spoken_reply:
            logger.warning("create_fsm_spoken_reply called with empty spoken_reply")
            await create_generic_response()
            return

        # Cancellation handled in send-path selection (controller vs legacy)

        instructions = ""
        if verbatim:
            instructions = (
                "You are Vozlia on a live phone call. This response was auto-executed after the greeting.\n"
                "Speak the following text exactly as written.\n\n"
                f"\"{spoken_reply}\"\n\n"
                "Rules (STRICT):\n"
                "- Start immediately with the text; do NOT add any preface like 'Sure', 'Okay', 'Of course', 'No problem', etc.\n"
                "- Do NOT add any extra words before or after the text.\n"
                "- Do NOT rephrase or paraphrase.\n"
                "- Do NOT mention tools, security, privacy, or inability to access email.\n"
            )
        else:
            instructions = (
                "You are on a live phone call as Vozlia.\n"
                "The secure backend has already checked the caller's email account and produced a short summary.\n\n"
                "Here is the summary you must speak to the caller:\n"
                f"\"{spoken_reply}\"\n\n"
                "For THIS response only:\n"
                "- Say this summary naturally.\n"
                "- Do NOT begin with acknowledgements like 'Sure', 'Okay', or 'Of course'.\n"
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
            if clear_playback:
                await _cancel_active_and_clear_buffer(
                    "create_fsm_spoken_reply_ctrl", clear_twilio_playback=False, clear_local_audio_buffer=True
                )

            tenant_id = os.getenv("VOZLIA_TENANT_ID") or os.getenv("TENANT_ID") or "default"
            ctx = ExecutionContext(
                tenant_id=str(tenant_id),
                call_sid=call_sid,
                session_id=stream_sid,
                skill_key="fsm",
            )

            req = SpeechRequest(
                text=spoken_reply,
                reason=reason_tag,
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
        if clear_playback:
            await _cancel_active_and_clear_buffer(
                "create_fsm_spoken_reply", clear_twilio_playback=False, clear_local_audio_buffer=True
            )
        else:
            logger.info("SPEAK_NO_CLEAR len=%d", len(spoken_reply))

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

        # Greeting protection: never let early caller speech cancel/clip the greeting.
        nonlocal pending_transcript_during_greeting
        if greeting_audio_protected:
            pending_transcript_during_greeting = transcript
            logger.info("DEFER_TRANSCRIPT_DURING_GREETING: %r", transcript)
            return

        # Resolve a pending discovery offer (Add-to-greeting) deterministically.
        nonlocal pending_discovery_skill_id, pending_discovery_trigger_text
        if pending_discovery_skill_id:
            norm = _normalize_text(transcript)
            first = norm.split(" ", 1)[0] if norm else ""

            is_yes = (norm in AFFIRMATIVE_WORDS) or (first in AFFIRMATIVE_PREFIXES)
            is_no = (norm in NEGATIVE_WORDS) or (first in NEGATIVE_PREFIXES)

            if is_yes:
                sid = pending_discovery_skill_id
                trig = pending_discovery_trigger_text or sid.replace("_", " ")
                pending_discovery_skill_id = None
                pending_discovery_trigger_text = ""
                logger.info("DISCOVERY_OFFER_ACCEPTED skill_id=%s trigger=%r", sid, trig)

                spoken_reply = await route_to_fsm_and_get_reply(trig, account_id=default_gmail_account_id if sid == "gmail_summary" else None)
                if spoken_reply:
                    await create_fsm_spoken_reply(spoken_reply, clear_playback=False)
                else:
                    await create_generic_response()
                return

            if is_no:
                sid = pending_discovery_skill_id
                pending_discovery_skill_id = None
                pending_discovery_trigger_text = ""
                logger.info("DISCOVERY_OFFER_DECLINED skill_id=%s", sid)
                await create_fsm_spoken_reply("No problem. What can I help you with today?", clear_playback=False)
                return

            logger.info("DISCOVERY_OFFER_UNCLEAR clearing pending offer; user said: %r", transcript)
            pending_discovery_skill_id = None
            pending_discovery_trigger_text = ""


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
        nonlocal initial_greeting_pending, greeting_response_id, auto_exec_fired, user_spoke_once, greeting_audio_protected, greeting_drain_task

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
                        # Capture the initial greeting response id (first response created after connect)
                        if initial_greeting_pending and greeting_response_id is None:
                            greeting_response_id = rid
                            initial_greeting_pending = False
                            logger.info("Greeting response_id captured: %s", greeting_response_id)

                elif etype in ("response.done", "response.completed", "response.failed", "response.canceled"):
                    resp = event.get("response", {}) or {}
                    rid = resp.get("id")
                    if active_response_id is not None and rid == active_response_id:
                        logger.info("Response %s finished with event '%s'; clearing active_response_id", rid, etype)
                        active_response_id = None
                        prebuffer_active = True


                        # Greeting protection: don't allow barge-in or user transcripts to cancel/clip
                        # the opening greeting. We keep a lock until the outbound Twilio audio buffer drains.
                        if greeting_response_id and rid == greeting_response_id and greeting_audio_protected:
                            if greeting_drain_task is None or greeting_drain_task.done():
                                greeting_drain_task = asyncio.create_task(_after_greeting_drain())
                                logger.info("GREETING_PROTECT: drain task scheduled")

                        # Auto-execute a configured skill after the greeting finishes (once per call).
                        if (
                            (not auto_exec_fired)
                            and auto_execute_skill_id
                            and greeting_response_id
                            and rid == greeting_response_id
                            and (not user_spoke_once)
                        ):
                            auto_exec_fired = True
                            logger.info("AUTO_EXECUTE_AFTER_GREETING skill_id=%s", auto_execute_skill_id)

                            async def _auto_exec():
                                try:
                                    ctx = {"channel": "phone", "auto_execute": True, "forced_skill_id": auto_execute_skill_id}
                                    if stream_sid:
                                        ctx["stream_sid"] = stream_sid
                                    if call_sid:
                                        ctx["call_sid"] = call_sid
                                    if from_number:
                                        ctx["from_number"] = from_number

                                    data = await call_fsm_router((auto_execute_trigger_text or ""), context=ctx, account_id=(default_gmail_account_id if auto_execute_skill_id == "gmail_summary" else None))
                                    spoken = None
                                    if isinstance(data, dict):
                                        spoken = (
                                            data.get("spoken_reply")
                                            or (data.get("result") or {}).get("spoken_reply")
                                            or (data.get("skill_result") or {}).get("spoken_reply")
                                        )
                                    if isinstance(spoken, str) and spoken.strip():
                                        # Use the existing "backend-produced speech" path
                                        # Wait for any remaining greeting audio to finish sending before
                                        # enqueuing the auto-exec reply (create_fsm_spoken_reply clears the buffer).
                                        timeout_s = float(os.getenv("AUTO_EXEC_POST_GREETING_DRAIN_TIMEOUT_S", "2.5"))
                                        grace_ms = float(os.getenv("POST_GREETING_GRACE_MS", "150") or 150)
                                        await _wait_for_audio_drain("auto_exec_after_greeting", timeout_s=timeout_s, target_bytes=0, grace_ms=grace_ms)
                                        clean = _strip_ack_preamble(spoken.strip())
                                        await create_fsm_spoken_reply(clean, clear_playback=False, reason_tag="auto_exec_after_greeting", verbatim=True)
                                    else:
                                        logger.warning("AUTO_EXECUTE_AFTER_GREETING produced no spoken_reply")
                                except Exception:
                                    logger.exception("AUTO_EXECUTE_AFTER_GREETING failed")

                            asyncio.create_task(_auto_exec())

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
                    if not user_spoke_once:
                        user_spoke_once = True
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
                    elif code == "session_expired":
                        # Realtime sessions have a max lifetime; treat as recoverable.
                        logger.error("speech_ctrl_REALTIME_ERROR code=session_expired message=%s", err.get("message"))
                        # Break out so the outer logic can close/reconnect cleanly.
                        try:
                            await openai_ws.close()
                        except Exception:
                            pass
                        break
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
                    logger.info("Stream start call_sid=%s from_number=%s", call_sid, from_number)

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
        initial_greeting_pending = True
        greeting_response_id: str | None = None
        auto_exec_fired = False
        user_spoke_once = False
        pending_discovery_skill_id: str | None = (
            discovery_offer_skill_id if (discovery_offer_skill_id and not auto_execute_skill_id) else None
        )
        pending_discovery_trigger_text: str = discovery_offer_trigger_text
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

        # --- Call summary (end-of-call) ----------------------------------------
        try:
            if os.getenv("CALL_SUMMARY_ENABLED", "0").strip() == "1" and call_sid:
                db2 = SessionLocal()
                try:
                    caller_id_for_summary = from_number
                    # Twilio 'start' event often omits the caller number unless you pass it via customParameters.
                    # Fallback: infer caller_id from the first stored memory event for this call_sid.
                    if not caller_id_for_summary:
                        row = (
                            db2.query(CallerMemoryEvent.caller_id)
                            .filter(CallerMemoryEvent.call_sid == str(call_sid))
                            .filter(CallerMemoryEvent.caller_id.isnot(None))
                            .order_by(CallerMemoryEvent.created_at.asc())
                            .first()
                        )
                        if row and row[0]:
                            caller_id_for_summary = row[0]

                    if not caller_id_for_summary:
                        logger.warning("CALL_SUMMARY_SKIP_MISSING_CALLER_ID call_sid=%s from_number=%s", call_sid, from_number)
                    else:
                        ensure_call_summary_for_call(db2, call_sid=str(call_sid), caller_id=str(caller_id_for_summary))
                finally:
                    db2.close()
        except Exception:
            logger.exception("CALL_SUMMARY_FINALIZE_ERROR call_sid=%s from=%s", call_sid, from_number)

        logger.info("WebSocket disconnected while sending audio")

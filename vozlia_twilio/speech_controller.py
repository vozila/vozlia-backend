# vozlia_twilio/speech_controller.py
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, Optional, Tuple

from core.logging import logger

SPEECH_CTRL_VERSION = "2025-12-28.payload-trace.v1"

# -----------------------------
# Tenant speech policy (tenant-level, NOT per skill)
# -----------------------------

@dataclass
class TenantSpeechDefaults:
    """
    Tenant-wide defaults. These are NOT stored per skill.

    These should eventually be loaded from the control-plane settings cache.
    """
    priority_default: int = 50
    speech_mode_default: str = "natural"          # natural | verbatim
    conversation_mode_default: str = "default"    # default | none  (mapped before sending to Realtime)
    can_interrupt_default: bool = True
    barge_grace_ms_default: int = 250
    barge_debounce_ms_default: int = 200
    max_chars_default: int = 900


@dataclass
class TenantSpeechPolicy:
    """
    Per-reason overrides (still tenant-owned).
    Any field may be None to mean "inherit defaults".
    """
    priority: Optional[int] = None
    speech_mode: Optional[str] = None            # natural | verbatim
    conversation_mode: Optional[str] = None      # default | none
    can_interrupt: Optional[bool] = None
    barge_grace_ms: Optional[int] = None
    barge_debounce_ms: Optional[int] = None
    max_chars: Optional[int] = None


TenantSpeechPolicyMap = Dict[str, TenantSpeechPolicy]


# -----------------------------
# Execution context (correlation only)
# -----------------------------

@dataclass
class ExecutionContext:
    """
    Correlates speech to skill/playbook/template runs.
    """
    tenant_id: str
    execution_id: str = field(default_factory=lambda: uuid.uuid4().hex)

    template_id: Optional[str] = None
    playbook_id: Optional[str] = None
    skill_key: Optional[str] = None

    call_sid: Optional[str] = None
    session_id: Optional[str] = None


# -----------------------------
# Speech request (what controller consumes)
# -----------------------------

@dataclass
class SpeechRequest:
    """
    What skills/playbooks emit: text + reason + context.

    NOTE: skills do not set tenant policy knobs; those come from TenantSpeechDefaults/PolicyMap.
    """
    text: str
    reason: str
    ctx: ExecutionContext

    # Optional debugging / scaffolding overrides used during cutover
    instructions_override: Optional[str] = None
    content_text_override: Optional[str] = None

    # Correlation
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex)


# -----------------------------
# Small transcript deduper (for future)
# -----------------------------

class TranscriptDeduper:
    def __init__(self, window_s: float = 1.5) -> None:
        self.window_s = window_s
        self._last_sha1: Optional[str] = None
        self._last_t: float = 0.0

    def seen_recently(self, text: str) -> bool:
        norm = (text or "").strip().lower()
        sha1 = hashlib.sha1(norm.encode("utf-8")).hexdigest()
        now = time.time()
        if self._last_sha1 == sha1 and (now - self._last_t) <= self.window_s:
            return True
        self._last_sha1 = sha1
        self._last_t = now
        return False


# -----------------------------
# SpeechOutputController
# -----------------------------

class SpeechOutputController:
    """
    Single owner for tool/skill speech output into OpenAI Realtime.

    Long-term responsibilities:
    - arbitration: never overlap response.create
    - cancellation coordination
    - barge-in policy
    - payload schema validation
    - deep observability for debugging
    """

    def __init__(
        self,
        *,
        tenant_defaults_provider: Callable[[str], Tuple[TenantSpeechDefaults, TenantSpeechPolicyMap]],
        send_realtime_json: Callable[[Dict[str, Any]], Awaitable[None]],
        cancel_active_cb: Optional[Callable[[str], Awaitable[None]]] = None,
        clear_audio_buffer_cb: Optional[Callable[[str], None]] = None,
        name: str = "speech_ctrl",
    ) -> None:
        self.name = name

        self.tenant_defaults_provider = tenant_defaults_provider
        self.send_realtime_json = send_realtime_json
        self.cancel_active_cb = cancel_active_cb
        self.clear_audio_buffer_cb = clear_audio_buffer_cb

        # Feature flags (env today; portal toggles later)
        self.enabled = os.getenv("SPEECH_CONTROLLER_ENABLED", "0") == "1"
        self.shadow = os.getenv("SPEECH_CONTROLLER_SHADOW", "1") == "1"      # observe-only by default
        self.fail_open = os.getenv("SPEECH_CONTROLLER_FAILOPEN", "1") == "1"

        # Logging flags
        self.payload_trace = os.getenv("REALTIME_TOOL_PAYLOAD_TRACE", "0") == "1"
        self.payload_trace_max = int(os.getenv("REALTIME_TOOL_PAYLOAD_TRACE_MAX", "900") or 900)
        self.metadata_trace = os.getenv("SPEECH_METADATA_TRACE", "1") == "1"  # default ON since you asked
        self.heartbeat_s = float(os.getenv("SPEECH_CTRL_HEARTBEAT_S", "15.0") or 15.0)

        # Timing
        self.wait_timeout_s = float(os.getenv("SPEECH_WAIT_TIMEOUT_S", "12.0") or 12.0)

        # Serialization + queue
        self._lock = asyncio.Lock()
        self._queue: asyncio.Queue[SpeechRequest] = asyncio.Queue()
        self._worker: Optional[asyncio.Task] = None
        self._hb_task: Optional[asyncio.Task] = None

        # Active response tracking (driven by Realtime events)
        self.active_response_id: Optional[str] = None
        self.active_started_at: float = 0.0
        self._active_done = asyncio.Event()
        self._active_done.set()

        # Payload debug (for stream.py to print on errors)
        self._last_tool_trace_id: Optional[str] = None
        self._last_tool_payload_trunc: Optional[str] = None
        self._last_tool_payload_keys: Optional[list[str]] = None

        self.deduper = TranscriptDeduper(window_s=float(os.getenv("TRANSCRIPT_DEDUPE_WINDOW_S", "1.5") or 1.5))

        logger.info(
            "%s_INIT version=%s enabled=%s shadow=%s fail_open=%s payload_trace=%s metadata_trace=%s auto_exec_force_verbatim=%s file=%s",
            self.name,
            SPEECH_CTRL_VERSION,
            self.enabled,
            self.shadow,
            self.fail_open,
            self.payload_trace,
            self.metadata_trace,
            os.getenv("AUTO_EXEC_FORCE_VERBATIM", "0"),
            __file__,
        )

        self.start()

    def start(self) -> None:
        if self._worker is None:
            self._worker = asyncio.create_task(self._run(), name=f"{self.name}_worker")
            logger.info("%s_WORKER_STARTED", self.name)
        if self._hb_task is None:
            self._hb_task = asyncio.create_task(self._heartbeat(), name=f"{self.name}_heartbeat")

    async def stop(self) -> None:
        if self._hb_task is not None:
            self._hb_task.cancel()
            try:
                await self._hb_task
            except asyncio.CancelledError:
                pass
            self._hb_task = None

        if self._worker is not None:
            self._worker.cancel()
            try:
                await self._worker
            except asyncio.CancelledError:
                pass
            self._worker = None
            logger.info("%s_WORKER_STOPPED", self.name)

    async def _heartbeat(self) -> None:
        while True:
            try:
                await asyncio.sleep(self.heartbeat_s)
                logger.info(
                    "%s_HEARTBEAT enabled=%s shadow=%s qsize=%s active_response_id=%s",
                    self.name,
                    self.enabled,
                    self.shadow,
                    self._queue.qsize(),
                    self.active_response_id,
                )
            except asyncio.CancelledError:
                return
            except Exception:
                logger.exception("%s_HEARTBEAT_ERROR", self.name)

    async def enqueue(self, req: SpeechRequest) -> bool:
        """
        Returns True if enqueued, False if controller is disabled.
        Does NOT raise (fail-open contract).
        """
        try:
            meta = self._log_meta(req)
            if not self.enabled:
                logger.info("%s_DISABLED_DROP %s", self.name, meta)
                return False
            logger.info("%s_ENQUEUE %s", self.name, meta)
            await self._queue.put(req)
            return True
        except Exception:
            # Never let tool speech go silent because controller threw.
            logger.exception("%s_ENQUEUE_EXCEPTION trace_id=%s", self.name, getattr(req, "trace_id", None))
            return False

    # -----------------------------
    # Realtime event ingestion
    # -----------------------------

    def on_realtime_event(self, event: Dict[str, Any]) -> None:
        et = event.get("type")
        if not et:
            return

        if et == "response.created":
            resp = event.get("response") or {}
            rid = resp.get("id")
            if rid:
                self.active_response_id = rid
                self.active_started_at = time.time()
                self._active_done.clear()
                logger.info("%s_ACTIVE_CREATED response_id=%s", self.name, rid)
            return

        # We wait for these as true completion signals to avoid overlapping response.create calls.
        if et in ("response.done", "response.completed", "response.failed", "response.canceled"):
            resp = event.get("response") or {}
            rid = resp.get("id")
            # If rid is missing, clear defensively.
            if self.active_response_id is None or rid is None or rid == self.active_response_id:
                dt_ms = int((time.time() - self.active_started_at) * 1000) if self.active_started_at else None
                logger.info("%s_ACTIVE_DONE type=%s response_id=%s dt_ms=%s", self.name, et, (rid or self.active_response_id), dt_ms)
                self.active_response_id = None
                self.active_started_at = 0.0
                self._active_done.set()
            return

        if et == "error":
            # Attach last tool payload trace for immediate diagnosis.
            err = event.get("error") or {}
            code = err.get("code")
            message = err.get("message")

            # Expected cancel race: we may send response.cancel after a response already ended.
            # Don't page on-call / don't pollute error logs for this.
            if code == "response_cancel_not_active":
                logger.info("%s_REALTIME_CANCEL_RACE code=%s message=%s last_tool_trace_id=%s",
                            self.name, code, message, self._last_tool_trace_id)
                return

            logger.error("%s_REALTIME_ERROR code=%s message=%s last_tool_trace_id=%s",
                         self.name, code, message, self._last_tool_trace_id)
            if self._last_tool_payload_trunc:
                logger.error("%s_LAST_TOOL_PAYLOAD trace_id=%s payload_trunc=%s",
                             self.name, self._last_tool_trace_id, self._last_tool_payload_trunc)
            return


    def get_last_tool_payload_debug(self) -> Dict[str, Any]:
        return {
            "trace_id": self._last_tool_trace_id,
            "payload_keys": self._last_tool_payload_keys,
            "payload_trunc": self._last_tool_payload_trunc,
        }

    # -----------------------------
    # Worker
    # -----------------------------

    async def _run(self) -> None:
        while True:
            req = await self._queue.get()
            try:
                await self._process(req)
            except Exception:
                logger.exception("%s_PROCESS_ERROR trace_id=%s", self.name, req.trace_id)
            finally:
                self._queue.task_done()

    async def _process(self, req: SpeechRequest) -> None:
        meta = self._log_meta(req)

        # Shadow mode: observe-only (controller disabled)
        if self.shadow and not self.enabled:
            logger.info("%s_SHADOW_WOULD_PROCESS %s", self.name, meta)
            return

        # Wait until no active response (event is set initially; cleared on response.created)
        if self.active_response_id is not None:
            logger.info(
                "%s_WAIT_ACTIVE response_id=%s next_trace_id=%s reason=%s",
                self.name, self.active_response_id, req.trace_id, req.reason
            )

        try:
            await asyncio.wait_for(self._active_done.wait(), timeout=self.wait_timeout_s)
        except asyncio.TimeoutError:
            logger.warning(
                "%s_WAIT_TIMEOUT active_response_id=%s timeout_s=%s %s",
                self.name, self.active_response_id, self.wait_timeout_s, meta
            )

            # Recovery: cancel + clear buffer if configured
            if self.cancel_active_cb:
                try:
                    await self.cancel_active_cb("speech_wait_timeout")
                except Exception:
                    logger.exception("%s_CANCEL_ON_TIMEOUT_FAILED response_id=%s", self.name, self.active_response_id)

            if self.clear_audio_buffer_cb:
                try:
                    self.clear_audio_buffer_cb("speech_wait_timeout")
                except Exception:
                    logger.exception("%s_CLEAR_BUFFER_ON_TIMEOUT_FAILED", self.name)

            self.active_response_id = None
            self.active_started_at = 0.0
            self._active_done.set()

        async with self._lock:
            payload = self._build_payload(req)
            ok, err = self._validate_payload(payload)
            if not ok:
                logger.error("%s_PAYLOAD_INVALID err=%s %s", self.name, err, meta)
                if self.fail_open:
                    logger.error("%s_FAILOPEN_REQUIRED %s", self.name, meta)
                return

            resp_obj = payload.get("response") if isinstance(payload, dict) else None
            conv_val = (resp_obj.get("conversation") if isinstance(resp_obj, dict) else None)
            has_input = (isinstance(resp_obj, dict) and "input" in resp_obj)
            has_metadata = (isinstance(resp_obj, dict) and "metadata" in resp_obj)
            instr = resp_obj.get("instructions") if isinstance(resp_obj, dict) else None
            instr_len = len(instr) if isinstance(instr, str) else None

            # Structured payload trace (redacted/truncated)
            payload_json = None
            if self.payload_trace:
                try:
                    payload_json = json.dumps(payload, ensure_ascii=False)
                except Exception:
                    payload_json = None

            payload_trunc = None
            if payload_json:
                payload_trunc = payload_json[: self.payload_trace_max]

            keys = list(resp_obj.keys()) if isinstance(resp_obj, dict) else []
            self._last_tool_trace_id = req.trace_id
            self._last_tool_payload_keys = keys
            self._last_tool_payload_trunc = payload_trunc

            logger.info(
                "%s_PAYLOAD_SHAPE trace_id=%s conversation=%s speech_mode=%s has_input=%s has_metadata=%s instr_len=%s keys=%s",
                self.name, req.trace_id, conv_val, has_input, has_metadata, instr_len, keys
            )
            if payload_trunc:
                logger.info("%s_PAYLOAD_JSON_TRUNC trace_id=%s payload=%s", self.name, req.trace_id, payload_trunc)

            logger.info("%s_SEND_RESPONSE_CREATE %s", self.name, meta)
            await self.send_realtime_json(payload)

    # -----------------------------
    # Payload construction
    # -----------------------------

    def _resolve_policy(self, req: SpeechRequest) -> Tuple[TenantSpeechDefaults, TenantSpeechPolicy]:
        defaults, policy_map = self.tenant_defaults_provider(req.ctx.tenant_id)
        per_reason = policy_map.get(req.reason) if policy_map else None

        merged = TenantSpeechPolicy(
            priority=(per_reason.priority if per_reason and per_reason.priority is not None else None),
            speech_mode=(per_reason.speech_mode if per_reason and per_reason.speech_mode else None),
            conversation_mode=(per_reason.conversation_mode if per_reason and per_reason.conversation_mode else None),
            can_interrupt=(per_reason.can_interrupt if per_reason and per_reason.can_interrupt is not None else None),
            barge_grace_ms=(per_reason.barge_grace_ms if per_reason and per_reason.barge_grace_ms is not None else None),
            barge_debounce_ms=(per_reason.barge_debounce_ms if per_reason and per_reason.barge_debounce_ms is not None else None),
            max_chars=(per_reason.max_chars if per_reason and per_reason.max_chars is not None else None),
        )

        return defaults, merged

    def _build_payload(self, req: SpeechRequest) -> Dict[str, Any]:
        defaults, policy = self._resolve_policy(req)

        # Determine policy knobs
        speech_mode = policy.speech_mode or defaults.speech_mode_default
        # Feature-flag: make auto-exec deterministic by forcing "verbatim" wrapper.
        # This reduces cases where Realtime ignores raw instructions and replies generically.
        if req.reason == "auto_exec_after_greeting":
            if os.getenv("AUTO_EXEC_FORCE_VERBATIM", "0").lower() in ("1", "true", "yes", "on"):
                speech_mode = "verbatim"
        conversation_mode = policy.conversation_mode or defaults.conversation_mode_default
        max_chars = policy.max_chars if policy.max_chars is not None else defaults.max_chars_default

        # Content (truncate for safety)
        content_text = req.content_text_override or req.text or ""
        content_text = content_text.strip()
        if max_chars and len(content_text) > max_chars:
            content_text = content_text[:max_chars].rstrip() + "…"

        # Instructions: preserve legacy scaffolding if provided
        if req.instructions_override:
            instructions = req.instructions_override
        else:
            if speech_mode == "verbatim":
                instructions = (
                    "Speak the following text exactly as written. "
                    "Do not summarize, do not paraphrase, do not add extra words:\n\n"
                    f"{content_text}"
                )
            else:
                instructions = content_text

        # Prepare metadata for logging/correlation, but DO NOT send it to Realtime.
        metadata = {
            "trace_id": req.trace_id,
            "reason": req.reason,
            "tenant_id": req.ctx.tenant_id,
            "execution_id": req.ctx.execution_id,
            "template_id": req.ctx.template_id,
            "playbook_id": req.ctx.playbook_id,
            "skill_key": req.ctx.skill_key,
            "call_sid": req.ctx.call_sid,
            "session_id": req.ctx.session_id,
        }

        if self.metadata_trace:
            none_count = sum(1 for _k, v in metadata.items() if v is None)
            logger.info("%s_METADATA_PREPARED trace_id=%s keys=%s none_count=%s", self.name, req.trace_id, list(metadata.keys()), none_count)
            logger.info("%s_METADATA_STRIPPED trace_id=%s stripped=True", self.name, req.trace_id)

        # conversation_mode mapping for Realtime:
        # - tenant 'default' means omit conversation entirely (let server decide auto behavior)
        # - tenant 'none' maps to conversation='none'
        resp: Dict[str, Any] = {"instructions": instructions}
        if conversation_mode == "none":
            resp["conversation"] = "none"
        # If conversation_mode == "default" we omit the field (Realtime supports only auto|none)

        return {"type": "response.create", "response": resp}

    def _validate_payload(self, payload: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate for known Realtime schema requirements.
        """
        try:
            if not isinstance(payload, dict):
                return False, "payload must be dict"

            if payload.get("type") != "response.create":
                return False, "payload.type must be response.create"

            resp = payload.get("response")
            if not isinstance(resp, dict):
                return False, "payload.response missing or not a dict"

            conv = resp.get("conversation")
            if conv is not None and conv not in ("auto", "none"):
                return False, "response.conversation must be 'auto' or 'none' when provided"

            instr = resp.get("instructions")
            if not isinstance(instr, str) or not instr.strip():
                return False, "response.instructions must be a non-empty string"

            # Ensure we are not accidentally sending metadata/input
            if "metadata" in resp:
                return False, "response.metadata must not be sent to Realtime"
            if "input" in resp:
                return False, "response.input must not be sent from controller path"

            return True, ""
        except Exception as e:
            return False, f"exception validating payload: {e}"

    def _log_meta(self, req: SpeechRequest) -> Dict[str, Any]:
        defaults, policy = self._resolve_policy(req)

        # Merge resolved policy into meta for logs
        speech_mode = policy.speech_mode or defaults.speech_mode_default
        conversation_mode = policy.conversation_mode or defaults.conversation_mode_default
        can_interrupt = policy.can_interrupt if policy.can_interrupt is not None else defaults.can_interrupt_default
        max_chars = policy.max_chars if policy.max_chars is not None else defaults.max_chars_default
        barge_grace_ms = policy.barge_grace_ms if policy.barge_grace_ms is not None else defaults.barge_grace_ms_default
        barge_debounce_ms = policy.barge_debounce_ms if policy.barge_debounce_ms is not None else defaults.barge_debounce_ms_default
        priority = policy.priority if policy.priority is not None else defaults.priority_default

        txt = req.content_text_override or req.text or ""
        sha1 = hashlib.sha1(txt.encode("utf-8")).hexdigest()[:10]

        return {
            "trace_id": req.trace_id,
            "reason": req.reason,
            "tenant_id": req.ctx.tenant_id,
            "execution_id": req.ctx.execution_id,
            "template_id": req.ctx.template_id,
            "playbook_id": req.ctx.playbook_id,
            "skill_key": req.ctx.skill_key,
            "call_sid": req.ctx.call_sid,
            "session_id": req.ctx.session_id,
            "len": len(txt),
            "sha1": sha1,
            "has_at": ("@" in txt),
            "policy_speech_mode": speech_mode,
            "policy_conversation_mode": conversation_mode,
            "policy_can_interrupt": can_interrupt,
            "policy_max_chars": max_chars,
            "policy_barge_grace_ms": barge_grace_ms,
            "policy_barge_debounce_ms": barge_debounce_ms,
            "policy_priority": priority,
        }

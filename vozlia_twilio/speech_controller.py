# vozlia_twilio/speech_controller.py
from __future__ import annotations

import asyncio
import hashlib
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, Optional, Tuple

from core.logging import logger


# -----------------------------
# Tenant speech policy (NOT skill config)
# -----------------------------

@dataclass
class TenantSpeechDefaults:
    """
    Tenant-wide defaults. These are NOT stored per skill.
    You will later load this from your control-plane settings cache.
    """
    # The user said "priority" is tenant-based, not skill-based.
    # We support per-reason overrides via TenantSpeechPolicyMap below.
    priority_default: int = 50

    speech_mode_default: str = "natural"          # "natural" | "verbatim"
    conversation_mode_default: str = "default"    # "default" | "none"
    can_interrupt_default: bool = True

    barge_grace_ms_default: int = 250
    barge_debounce_ms_default: int = 200

    max_chars_default: int = 900


@dataclass
class TenantSpeechPolicy:
    """
    Optional per-reason overrides (still tenant-level).
    Example: inbox_menu might be verbatim + conversation=none.
    """
    priority: Optional[int] = None
    speech_mode: Optional[str] = None           # "natural" | "verbatim"
    conversation_mode: Optional[str] = None     # "default" | "none"
    can_interrupt: Optional[bool] = None
    barge_grace_ms: Optional[int] = None
    barge_debounce_ms: Optional[int] = None
    max_chars: Optional[int] = None


TenantSpeechPolicyMap = Dict[str, TenantSpeechPolicy]


# -----------------------------
# Execution context (skills/playbooks/templates)
# -----------------------------

@dataclass
class ExecutionContext:
    """
    Correlates speech to skill/playbook/template runs.
    """
    tenant_id: str
    execution_id: str = field(default_factory=lambda: uuid.uuid4().hex)

    # Optional higher-level identifiers
    template_id: Optional[str] = None
    playbook_id: Optional[str] = None
    skill_key: Optional[str] = None

    # Request/call correlation
    call_sid: Optional[str] = None
    session_id: Optional[str] = None


# -----------------------------
# Speech request (what controller consumes)
# -----------------------------

@dataclass
class SpeechRequest:
    """
    What skills/playbooks emit: text + reason + context.
    NO speech knobs here (those come from tenant policy),
    but we allow optional overrides for debugging / admin test.
    """
    text: str
    reason: str
    ctx: ExecutionContext

    # Optional overrides (should be used rarely; e.g. admin test)
    override_policy: Optional[TenantSpeechPolicy] = None

    # Optional overrides to preserve legacy instruction scaffolding (e.g. FSM tool speech)
    # If provided, the controller will use these directly instead of generating instructions/content.
    instructions_override: Optional[str] = None
    content_text_override: Optional[str] = None

    created_at: float = field(default_factory=lambda: time.time())
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def text_meta(self) -> Dict[str, Any]:
        t = (self.content_text_override or self.text or "")
        sha1 = hashlib.sha1(t.encode("utf-8", errors="ignore")).hexdigest()[:10]
        return {
            "len": len(t),
            "sha1": sha1,
            "has_at": ("@" in t),
        }


# -----------------------------
# Transcript de-dupe
# -----------------------------

class TranscriptDeduper:
    def __init__(self, window_s: float = 1.5) -> None:
        self.window_s = window_s
        self._last_norm: Optional[str] = None
        self._last_ts: float = 0.0

    @staticmethod
    def normalize(text: str) -> str:
        t = (text or "").strip().lower()
        t = " ".join(t.split())
        return t

    def seen_recently(self, text: str) -> bool:
        now = time.time()
        norm = self.normalize(text)
        if not norm:
            return False
        if self._last_norm == norm and (now - self._last_ts) <= self.window_s:
            return True
        self._last_norm = norm
        self._last_ts = now
        return False


# -----------------------------
# Controller
# -----------------------------

class SpeechOutputController:
    """
    Single owner for response.create to OpenAI Realtime.

    Long-term responsibilities:
    - arbitration: never overlap response.create
    - queueing: (future) priority scheduling based on tenant speech policy
    - cancellation + buffer clear coordination
    - barge-in policy per tenant/per reason
    - schema validation of payloads (prevent silence)
    - logs with trace_id + execution/template/playbook/skill correlation
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

        # External dependencies
        self.tenant_defaults_provider = tenant_defaults_provider
        self.send_realtime_json = send_realtime_json
        self.cancel_active_cb = cancel_active_cb
        self.clear_audio_buffer_cb = clear_audio_buffer_cb

        # Feature flags
        self.enabled = os.getenv("SPEECH_CONTROLLER_ENABLED", "0") == "1"
        self.shadow = os.getenv("SPEECH_CONTROLLER_SHADOW", "1") == "1"  # default observe-only
        self.fail_open = os.getenv("SPEECH_CONTROLLER_FAILOPEN", "1") == "1"

        # Timing safety
        self.wait_timeout_s = float(os.getenv("SPEECH_WAIT_TIMEOUT_S", "12.0") or 12.0)

        # Serialization
        self._lock = asyncio.Lock()
        self._queue: asyncio.Queue[SpeechRequest] = asyncio.Queue()
        self._worker: Optional[asyncio.Task] = None

        # Active response tracking (driven by Realtime events)
        self.active_response_id: Optional[str] = None
        self.active_started_at: float = 0.0
        self._active_done = asyncio.Event()
        self._active_done.set()

        # Optional utilities
        self.deduper = TranscriptDeduper(window_s=float(os.getenv("TRANSCRIPT_DEDUPE_WINDOW_S", "1.5") or 1.5))

        logger.info(
            "%s_INIT enabled=%s shadow=%s fail_open=%s wait_timeout_s=%s",
            self.name, self.enabled, self.shadow, self.fail_open, self.wait_timeout_s
        )

    def start(self) -> None:
        if self._worker is None:
            self._worker = asyncio.create_task(self._run(), name=f"{self.name}_worker")
            logger.info("%s_WORKER_STARTED", self.name)

    async def stop(self) -> None:
        if self._worker is None:
            return
        self._worker.cancel()
        try:
            await self._worker
        except asyncio.CancelledError:
            pass
        self._worker = None
        logger.info("%s_WORKER_STOPPED", self.name)

    async def enqueue(self, req: SpeechRequest) -> bool:
        meta = self._log_meta(req)
        if not self.enabled:
            logger.info("%s_DISABLED_DROP %s", self.name, meta)
            return False
        logger.info("%s_ENQUEUE %s", self.name, meta)
        await self._queue.put(req)
        return True

    # -----------------------------
    # Realtime event ingestion (shadow first)
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

        if et in ("response.completed", "response.failed", "response.canceled"):
            resp = event.get("response") or {}
            rid = resp.get("id")
            if self.active_response_id is None or rid == self.active_response_id:
                dt_ms = int((time.time() - self.active_started_at) * 1000) if self.active_started_at else None
                logger.info("%s_ACTIVE_DONE type=%s response_id=%s dt_ms=%s", self.name, et, rid, dt_ms)
                self.active_response_id = None
                self.active_started_at = 0.0
                self._active_done.set()
            return

    # -----------------------------
    # Worker loop
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

        # Shadow mode: observe-only
        if self.shadow and not self.enabled:
            logger.info("%s_SHADOW_WOULD_PROCESS %s", self.name, meta)
            return

        # Wait until no active response
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
            if self.cancel_active_cb and self.active_response_id:
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
                    # caller can fallback to legacy path during cutover
                    logger.error("%s_FAILOPEN_REQUIRED %s", self.name, meta)
                return

            resp_obj = payload.get("response") if isinstance(payload, dict) else None
            conv_val = (resp_obj.get("conversation") if isinstance(resp_obj, dict) else None)
            has_input = (isinstance(resp_obj, dict) and "input" in resp_obj)
            instr_len = (len(resp_obj.get("instructions")) if isinstance(resp_obj, dict) and isinstance(resp_obj.get("instructions"), str) else None)
            logger.info("%s_PAYLOAD_SHAPE conversation=%s has_input=%s instr_len=%s", self.name, conv_val, has_input, instr_len)
            logger.info("%s_SEND_RESPONSE_CREATE %s", self.name, meta)
            await self.send_realtime_json(payload)

    # -----------------------------
    # Payload construction based on tenant policy
    # -----------------------------

    def _resolve_policy(self, req: SpeechRequest) -> Tuple[TenantSpeechDefaults, TenantSpeechPolicy]:
        defaults, policy_map = self.tenant_defaults_provider(req.ctx.tenant_id)
        per_reason = policy_map.get(req.reason) if policy_map else None

        # merge: defaults -> per_reason -> overrides
        merged = TenantSpeechPolicy(
            priority=(per_reason.priority if per_reason and per_reason.priority is not None else None),
            speech_mode=(per_reason.speech_mode if per_reason and per_reason.speech_mode else None),
            conversation_mode=(per_reason.conversation_mode if per_reason and per_reason.conversation_mode else None),
            can_interrupt=(per_reason.can_interrupt if per_reason and per_reason.can_interrupt is not None else None),
            barge_grace_ms=(per_reason.barge_grace_ms if per_reason and per_reason.barge_grace_ms is not None else None),
            barge_debounce_ms=(per_reason.barge_debounce_ms if per_reason and per_reason.barge_debounce_ms is not None else None),
            max_chars=(per_reason.max_chars if per_reason and per_reason.max_chars is not None else None),
        )

        if req.override_policy:
            o = req.override_policy
            if o.priority is not None: merged.priority = o.priority
            if o.speech_mode: merged.speech_mode = o.speech_mode
            if o.conversation_mode: merged.conversation_mode = o.conversation_mode
            if o.can_interrupt is not None: merged.can_interrupt = o.can_interrupt
            if o.barge_grace_ms is not None: merged.barge_grace_ms = o.barge_grace_ms
            if o.barge_debounce_ms is not None: merged.barge_debounce_ms = o.barge_debounce_ms
            if o.max_chars is not None: merged.max_chars = o.max_chars

        return defaults, merged

    def _build_payload(self, req: SpeechRequest) -> Dict[str, Any]:
        defaults, policy = self._resolve_policy(req)

        speech_mode = policy.speech_mode or defaults.speech_mode_default
        conversation_mode = policy.conversation_mode or defaults.conversation_mode_default
        max_chars = policy.max_chars or defaults.max_chars_default

        content_text = (req.content_text_override or req.text or "").strip()
        if max_chars and len(content_text) > max_chars:
            content_text = content_text[: max_chars].rstrip() + "…"

        # Instructions: allow override to preserve legacy scaffolding for certain reasons
        if req.instructions_override:
            instructions = req.instructions_override
        else:
            # For "verbatim", force exact reading
            if speech_mode == "verbatim":
                instructions = (
                    "Speak the following text exactly as written. "
                    "Do not summarize, do not paraphrase, do not add extra words:\n\n"
                    f"{content_text}"
                )
            else:
                instructions = content_text

        # IMPORTANT: content[].type must be "text" (schema strict)
        payload: Dict[str, Any] = {
            "type": "response.create",
            "response": {
                "instructions": instructions,
                "metadata": {
                    "trace_id": req.trace_id,
                    "reason": req.reason,
                    "tenant_id": req.ctx.tenant_id,
                    "execution_id": req.ctx.execution_id,
                    "template_id": req.ctx.template_id,
                    "playbook_id": req.ctx.playbook_id,
                    "skill_key": req.ctx.skill_key,
                    "call_sid": req.ctx.call_sid,
                    "session_id": req.ctx.session_id,
                },
            },
        }
        # Only set conversation when explicitly isolating; otherwise omit to use default server behavior.
        if conversation_mode == "none":
            payload["response"]["conversation"] = "none"
        return payload

    
    def _validate_payload(self, payload: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Keep validation lightweight and aligned with the Realtime schema.
        For tool speech we intentionally use the legacy-stable shape:
          {"type":"response.create","response":{"instructions": "...", "conversation":"none"? , "metadata": {...}}}

        If an "input" field is present (future use), validate its first content part is {"type":"text","text":...}.
        """
        try:
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

            # Optional input validation (not used in legacy-stable tool speech)
            if "input" in resp:
                inp = resp.get("input")
                if not isinstance(inp, list) or not inp:
                    return False, "response.input must be a non-empty list when provided"
                msg = inp[0]
                if not isinstance(msg, dict) or msg.get("type") != "message":
                    return False, "response.input[0].type must be message"
                content = msg.get("content")
                if not isinstance(content, list) or not content:
                    return False, "response.input[0].content must be non-empty list"
                c0 = content[0]
                if not isinstance(c0, dict) or c0.get("type") != "text":
                    return False, "response.input[0].content[0].type must be text"
                if not isinstance(c0.get("text"), str):
                    return False, "response.input[0].content[0].text must be str"

            return True, ""
        except Exception as e:
            return False, f"exception validating payload: {e}"


    def _log_meta(self, req: SpeechRequest) -> Dict[str, Any]:
        defaults, policy = self._resolve_policy(req)
        meta = {
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
        meta.update(req.text_meta())
        meta.update({
            "policy_speech_mode": (policy.speech_mode or defaults.speech_mode_default),
            "policy_conversation_mode": (policy.conversation_mode or defaults.conversation_mode_default),
            "policy_can_interrupt": (policy.can_interrupt if policy.can_interrupt is not None else defaults.can_interrupt_default),
            "policy_max_chars": (policy.max_chars or defaults.max_chars_default),
            "policy_barge_grace_ms": (policy.barge_grace_ms or defaults.barge_grace_ms_default),
            "policy_barge_debounce_ms": (policy.barge_debounce_ms or defaults.barge_debounce_ms_default),
            "policy_priority": (policy.priority if policy.priority is not None else defaults.priority_default),
        })
        return meta
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

        # ✅ Treat transcript/text done events as completion too (Realtime often ends here)
        # NOTE: We do NOT treat transcript/text completion as full response completion.
        # We wait for response.done / response.completed / response.canceled to avoid sending the next response too early.
        if et in ("response.audio_transcript.done", "response.output_text.done", "response.text.done"):
            return

        if et in ("response.completed", "response.failed", "response.canceled"):
            resp = event.get("response") or {}
            rid = resp.get("id")
            if self.active_response_id is None or rid == self.active_response_id:
                dt_ms = int((time.time() - self.active_started_at) * 1000) if self.active_started_at else None
                logger.info("%s_ACTIVE_DONE type=%s response_id=%s dt_ms=%s", self.name, et, rid, dt_ms)
                self.active_response_id = None
                self.active_started_at = 0.0
                self._active_done.set()
            return

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

        # ✅ ALSO treat these as completion signals
        if et in ("response.done",):
            rid = (event.get("response") or {}).get("id") or event.get("response_id")
            if self.active_response_id is not None and (rid is None or rid == self.active_response_id):
                dt_ms = int((time.time() - self.active_started_at) * 1000) if self.active_started_at else None
                logger.info("%s_ACTIVE_DONE type=%s response_id=%s dt_ms=%s", self.name, et, self.active_response_id, dt_ms)
                self.active_response_id = None
                self.active_started_at = 0.0
                self._active_done.set()
            return

        # keep your existing done handlers too:
        if et in ("response.audio_transcript.done", "response.output_text.done", "response.text.done"):
            rid = event.get("response_id") or (event.get("response") or {}).get("id")
            if self.active_response_id is not None and (rid is None or rid == self.active_response_id):
                dt_ms = int((time.time() - self.active_started_at) * 1000) if self.active_started_at else None
                logger.info("%s_ACTIVE_DONE type=%s response_id=%s dt_ms=%s", self.name, et, self.active_response_id, dt_ms)
                self.active_response_id = None
                self.active_started_at = 0.0
                self._active_done.set()
            return

        if et in ("response.completed", "response.failed", "response.canceled"):
            resp = event.get("response") or {}
            rid = resp.get("id")
            if self.active_response_id is None or rid == self.active_response_id:
                dt_ms = int((time.time() - self.active_started_at) * 1000) if self.active_started_at else None
                logger.info("%s_ACTIVE_DONE type=%s response_id=%s dt_ms=%s", self.name, et, rid, dt_ms)
                self.active_response_id = None
                self.active_started_at = 0.0
                self._active_done.set()
            return

        # Optional (helps when your state is stale after barge-in cancel races):
        if et == "error":
            err = event.get("error") or {}
            if err.get("code") == "response_cancel_not_active" and self.active_response_id is not None:
                logger.warning("%s_ACTIVE_CLEAR_STALE reason=response_cancel_not_active response_id=%s", self.name, self.active_response_id)
                self.active_response_id = None
                self.active_started_at = 0.0
                self._active_done.set()
            return
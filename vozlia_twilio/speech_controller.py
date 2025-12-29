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

SPEECH_CTRL_VERSION = "2025-12-28_step3_fixmeta_strip"


# -----------------------------
# Tenant speech policy (tenant-level settings; NOT per-skill)
# -----------------------------

@dataclass
class TenantSpeechDefaults:
    priority_default: int = 50
    speech_mode_default: str = "natural"          # "natural" | "verbatim"
    conversation_mode_default: str = "default"    # "default" | "none" (mapped internally)
    can_interrupt_default: bool = True
    barge_grace_ms_default: int = 250
    barge_debounce_ms_default: int = 200
    max_chars_default: int = 900


@dataclass
class TenantSpeechPolicy:
    priority: Optional[int] = None
    speech_mode: Optional[str] = None
    conversation_mode: Optional[str] = None
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
    text: str
    reason: str
    ctx: ExecutionContext

    # Optional overrides (admin test / legacy scaffolding)
    override_policy: Optional[TenantSpeechPolicy] = None
    instructions_override: Optional[str] = None
    content_text_override: Optional[str] = None

    created_at: float = field(default_factory=lambda: time.time())
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def text_meta(self) -> Dict[str, Any]:
        t = (self.content_text_override or self.text or "").strip()
        sha1 = hashlib.sha1(t.encode("utf-8", errors="ignore")).hexdigest()[:10]
        return {"len": len(t), "sha1": sha1, "has_at": ("@" in t)}


# -----------------------------
# Controller
# -----------------------------

class SpeechOutputController:
    """
    Single owner for response.create to OpenAI Realtime (tool speech).

    Step 3 contract:
    - Enqueue speech requests and serialize response.create
    - Track "active response" via Realtime events to avoid overlap
    - Keep payload shape close to legacy-known-good:
        { "type":"response.create", "response": { "instructions": "...", "conversation":"none"? } }
      (No response.input; no response.metadata)
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

        self.enabled = os.getenv("SPEECH_CONTROLLER_ENABLED", "0") == "1"
        self.shadow = os.getenv("SPEECH_CONTROLLER_SHADOW", "0") == "1"
        self.fail_open = os.getenv("SPEECH_CONTROLLER_FAILOPEN", "1") == "1"
        self.wait_timeout_s = float(os.getenv("SPEECH_WAIT_TIMEOUT_S", "12.0") or 12.0)

        self._lock = asyncio.Lock()
        self._queue: asyncio.Queue[SpeechRequest] = asyncio.Queue()
        self._worker: Optional[asyncio.Task] = None

        # Active response tracking
        self.active_response_id: Optional[str] = None
        self.active_started_at: float = 0.0
        self._active_done = asyncio.Event()
        self._active_done.set()

        logger.info(
            "%s_INIT version=%s file=%s enabled=%s shadow=%s fail_open=%s wait_timeout_s=%s has__log_meta=%s",
            self.name,
            SPEECH_CTRL_VERSION,
            __file__,
            self.enabled,
            self.shadow,
            self.fail_open,
            self.wait_timeout_s,
            hasattr(self, "_log_meta"),
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
        """
        Returns False if controller is disabled or if enqueue fails.
        Never raises (fail-open friendly).
        """
        if not self.enabled:
            return False

        # If worker died, don't accept requests.
        if self._worker is not None and self._worker.done():
            exc = self._worker.exception()
            logger.error("%s_WORKER_DEAD exc=%r trace_id=%s", self.name, exc, req.trace_id)
            return False

        meta = None
        try:
            meta = self._log_meta(req)
            logger.info("%s_ENQUEUE %s", self.name, meta)
        except Exception:
            # Don't crash the call-flow; return False so caller can fall back.
            logger.exception("%s_ENQUEUE_META_ERROR trace_id=%s reason=%s", self.name, req.trace_id, req.reason)
            return False

        await self._queue.put(req)
        return True

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

        # Prefer definitive completion signals.
        if et in ("response.done", "response.completed", "response.failed", "response.canceled"):
            rid = (event.get("response") or {}).get("id") or event.get("response_id")
            if self.active_response_id is None or rid is None or rid == self.active_response_id:
                dt_ms = int((time.time() - self.active_started_at) * 1000) if self.active_started_at else None
                logger.info("%s_ACTIVE_DONE type=%s response_id=%s dt_ms=%s", self.name, et, self.active_response_id or rid, dt_ms)
                self.active_response_id = None
                self.active_started_at = 0.0
                self._active_done.set()
            return

        # Helpful fallbacks (some sessions only emit transcript/text done).
        if et in ("response.audio_transcript.done", "response.output_text.done", "response.text.done"):
            rid = event.get("response_id") or (event.get("response") or {}).get("id")
            if self.active_response_id is not None and (rid is None or rid == self.active_response_id):
                dt_ms = int((time.time() - self.active_started_at) * 1000) if self.active_started_at else None
                logger.info("%s_ACTIVE_DONE type=%s response_id=%s dt_ms=%s", self.name, et, self.active_response_id, dt_ms)
                self.active_response_id = None
                self.active_started_at = 0.0
                self._active_done.set()
            return

        if et == "error":
            err = event.get("error") or {}
            # If we get a cancel-not-active race, clear stale state so we don't deadlock.
            if err.get("code") == "response_cancel_not_active" and self.active_response_id is not None:
                logger.warning("%s_ACTIVE_CLEAR_STALE reason=response_cancel_not_active response_id=%s", self.name, self.active_response_id)
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

        # Wait until no active response (serialize response.create)
        if self.active_response_id is not None:
            logger.info("%s_WAIT_ACTIVE response_id=%s next_trace_id=%s reason=%s", self.name, self.active_response_id, req.trace_id, req.reason)

        try:
            await asyncio.wait_for(self._active_done.wait(), timeout=self.wait_timeout_s)
        except asyncio.TimeoutError:
            logger.warning("%s_WAIT_TIMEOUT active_response_id=%s timeout_s=%s %s", self.name, self.active_response_id, self.wait_timeout_s, meta)

            # Best-effort recovery
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
            payload, meta_info = self._build_payload(req)

            ok, err = self._validate_payload(payload)
            if not ok:
                logger.error("%s_PAYLOAD_INVALID err=%s %s", self.name, err, meta)
                return

            logger.info(
                "%s_PAYLOAD_SHAPE trace_id=%s conversation=%s has_input=%s has_metadata=%s instr_len=%s none_meta=%s",
                self.name,
                req.trace_id,
                meta_info.get("conversation"),
                meta_info.get("has_input"),
                meta_info.get("has_metadata"),
                meta_info.get("instr_len"),
                meta_info.get("meta_none_count"),
            )

            logger.info("%s_SEND_RESPONSE_CREATE %s", self.name, meta)
            await self.send_realtime_json(payload)

    # -----------------------------
    # Payload building
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

    def _build_payload(self, req: SpeechRequest) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        defaults, policy = self._resolve_policy(req)

        speech_mode = policy.speech_mode or defaults.speech_mode_default
        conversation_mode = policy.conversation_mode or defaults.conversation_mode_default
        max_chars = policy.max_chars or defaults.max_chars_default

        content_text = (req.content_text_override or req.text or "").strip()
        if max_chars and len(content_text) > max_chars:
            content_text = content_text[: max_chars].rstrip() + "…"

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
        none_count = sum(1 for v in metadata.values() if v is None)

        logger.info(
            "%s_METADATA_PREPARED trace_id=%s keys=%s none_count=%s",
            self.name,
            req.trace_id,
            sorted(metadata.keys()),
            none_count,
        )
        logger.info("%s_METADATA_STRIPPED trace_id=%s stripped=True", self.name, req.trace_id)

        payload: Dict[str, Any] = {
            "type": "response.create",
            "response": {
                "instructions": instructions,
            },
        }

        # Only set conversation when explicitly isolating; otherwise omit (use server default).
        if conversation_mode == "none":
            payload["response"]["conversation"] = "none"

        meta_info = {
            "conversation": payload["response"].get("conversation"),
            "has_input": ("input" in payload["response"]),
            "has_metadata": ("metadata" in payload["response"]),
            "instr_len": len(instructions or ""),
            "meta_none_count": none_count,
        }
        return payload, meta_info

    def _validate_payload(self, payload: Dict[str, Any]) -> Tuple[bool, str]:
        try:
            if payload.get("type") != "response.create":
                return False, "payload.type must be response.create"
            resp = payload.get("response")
            if not isinstance(resp, dict):
                return False, "payload.response missing"
            conv = resp.get("conversation")
            if conv is not None and conv not in ("auto", "none"):
                # We omit by default; if present must be allowed values.
                return False, f"invalid response.conversation={conv}"
            if "metadata" in resp:
                return False, "response.metadata must not be sent"
            if "instructions" not in resp or not isinstance(resp.get("instructions"), str):
                return False, "response.instructions must be str"
            return True, ""
        except Exception as e:
            return False, f"exception validating payload: {e}"

    # -----------------------------
    # Logging helper (safe)
    # -----------------------------

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

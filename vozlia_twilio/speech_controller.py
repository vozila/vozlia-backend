# vozlia_twilio/speech_controller.py
from __future__ import annotations

import asyncio
import hashlib
import os
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Optional, Tuple

from core.logging import logger

SPEECH_CTRL_VERSION = "2025-12-28.step3.failopen.v2"


# -----------------------------
# Tenant policy primitives
# -----------------------------
@dataclass(frozen=True)
class TenantSpeechDefaults:
    priority_default: int = 50
    speech_mode_default: str = "natural"  # natural | verbatim
    conversation_mode_default: str = "default"  # default | none (tenant-facing)
    can_interrupt_default: bool = True
    barge_grace_ms_default: int = 250
    barge_debounce_ms_default: int = 200
    max_chars_default: int = 900


@dataclass(frozen=True)
class TenantSpeechPolicy:
    priority: Optional[int] = None
    speech_mode: Optional[str] = None
    conversation_mode: Optional[str] = None
    can_interrupt: Optional[bool] = None
    barge_grace_ms: Optional[int] = None
    barge_debounce_ms: Optional[int] = None
    max_chars: Optional[int] = None


TenantSpeechPolicyMap = Dict[str, TenantSpeechPolicy]


@dataclass(frozen=True)
class ExecutionContext:
    tenant_id: str
    execution_id: str
    template_id: Optional[str] = None
    playbook_id: Optional[str] = None
    skill_key: Optional[str] = None
    call_sid: Optional[str] = None
    session_id: Optional[str] = None

    @staticmethod
    def new(
        tenant_id: str,
        *,
        template_id: Optional[str] = None,
        playbook_id: Optional[str] = None,
        skill_key: Optional[str] = None,
        call_sid: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> "ExecutionContext":
        # cheap, collision-resistant enough for logs
        eid = hashlib.sha1(f"{tenant_id}:{time.time_ns()}".encode("utf-8")).hexdigest()[:32]
        return ExecutionContext(
            tenant_id=str(tenant_id),
            execution_id=eid,
            template_id=template_id,
            playbook_id=playbook_id,
            skill_key=skill_key,
            call_sid=call_sid,
            session_id=session_id,
        )


@dataclass
class SpeechRequest:
    text: str
    reason: str
    ctx: ExecutionContext
    priority: Optional[int] = None

    # Overrides to keep legacy scaffolding stable for FSM/tool speech.
    instructions_override: Optional[str] = None
    content_text_override: Optional[str] = None

    # Internal fields
    trace_id: str = ""

    def __post_init__(self) -> None:
        if not self.trace_id:
            # stable trace for correlation
            self.trace_id = hashlib.md5(f"{self.reason}:{time.time_ns()}".encode("utf-8")).hexdigest()

    def text_meta(self) -> Dict[str, Any]:
        t = (self.content_text_override or self.text or "").strip()
        return {
            "len": len(t),
            "sha1": hashlib.sha1(t.encode("utf-8")).hexdigest()[:10],
            "has_at": ("@" in t),
        }


# -----------------------------
# Speech output controller
# -----------------------------
class SpeechOutputController:
    def __init__(
        self,
        *,
        tenant_defaults_provider: Callable[[str], Tuple[TenantSpeechDefaults, TenantSpeechPolicyMap]],
        send_realtime_json: Callable[[Dict[str, Any]], Awaitable[None]],
        cancel_active_cb: Optional[Callable[[], Awaitable[None]]] = None,
        clear_audio_buffer_cb: Optional[Callable[[], None]] = None,
        name: str = "speech_ctrl",
    ) -> None:
        self.tenant_defaults_provider = tenant_defaults_provider
        self.send_realtime_json = send_realtime_json
        self.cancel_active_cb = cancel_active_cb
        self.clear_audio_buffer_cb = clear_audio_buffer_cb
        self.name = name

        self.enabled = os.getenv("SPEECH_CONTROLLER_ENABLED", "0") == "1"
        self.shadow = os.getenv("SPEECH_CONTROLLER_SHADOW", "0") == "1"
        self.fail_open = os.getenv("SPEECH_CONTROLLER_FAILOPEN", "1") == "1"

        self._q: "asyncio.Queue[SpeechRequest]" = asyncio.Queue()
        self._worker_task: Optional[asyncio.Task] = None

        self.active_response_id: Optional[str] = None
        self.active_started_at: float = 0.0
        self._active_done = asyncio.Event()
        self._active_done.set()

        logger.info(
            "%s_INIT version=%s file=%s enabled=%s shadow=%s fail_open=%s has__log_meta=%s",
            self.name,
            SPEECH_CTRL_VERSION,
            __file__,
            self.enabled,
            self.shadow,
            self.fail_open,
            hasattr(self, "_log_meta"),
        )

    def start(self) -> None:
        if self._worker_task is None:
            self._worker_task = asyncio.create_task(self._worker(), name=f"{self.name}_worker")
            logger.info("%s_WORKER_STARTED", self.name)

    async def stop(self) -> None:
        if self._worker_task is None:
            return
        try:
            self._worker_task.cancel()
            await asyncio.gather(self._worker_task, return_exceptions=True)
        finally:
            self._worker_task = None
            logger.info("%s_WORKER_STOPPED", self.name)

    # ---- realtime event ingestion (shadow + active response tracking) ----
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

        # Hard completion signals (most reliable across sessions)
        if et in ("response.done", "response.completed", "response.failed", "response.canceled"):
            rid = (event.get("response") or {}).get("id") or event.get("response_id")
            if self.active_response_id is not None and (rid is None or rid == self.active_response_id):
                dt_ms = int((time.time() - self.active_started_at) * 1000) if self.active_started_at else None
                logger.info("%s_ACTIVE_DONE type=%s response_id=%s dt_ms=%s", self.name, et, self.active_response_id, dt_ms)
                self.active_response_id = None
                self.active_started_at = 0.0
                self._active_done.set()
            return

        # Transcript done is also a good completion indicator, but we keep it as a helper.
        if et in ("response.audio_transcript.done", "response.text.done", "response.output_text.done"):
            rid = event.get("response_id") or (event.get("response") or {}).get("id")
            if self.active_response_id is not None and (rid is None or rid == self.active_response_id):
                dt_ms = int((time.time() - self.active_started_at) * 1000) if self.active_started_at else None
                logger.info("%s_ACTIVE_DONE type=%s response_id=%s dt_ms=%s", self.name, et, self.active_response_id, dt_ms)
                self.active_response_id = None
                self.active_started_at = 0.0
                self._active_done.set()
            return

        # Keep errors visible; also clear stale state for cancel races.
        if et == "error":
            err = event.get("error") or {}
            logger.error("%s_OAI_ERROR %s", self.name, err)
            if err.get("code") == "response_cancel_not_active" and self.active_response_id is not None:
                logger.warning("%s_ACTIVE_CLEAR_STALE reason=response_cancel_not_active response_id=%s", self.name, self.active_response_id)
                self.active_response_id = None
                self.active_started_at = 0.0
                self._active_done.set()
            return

    # ---- enqueue API (fail-open safe) ----
    async def enqueue(self, req: SpeechRequest) -> bool:
        if not self.enabled:
            return False
        try:
            meta = self._log_meta(req)  # will also log metadata prepared/stripped proof
            logger.info("%s_ENQUEUE %s", self.name, meta)
            await self._q.put(req)
            return True
        except Exception:
            # Never let tool speech go silent because the controller threw.
            logger.exception("%s_ENQUEUE_EXCEPTION reason=%s trace_id=%s", self.name, req.reason, req.trace_id)
            return False

    # ---- internal worker ----
    async def _worker(self) -> None:
        while True:
            req = await self._q.get()
            try:
                await self._send(req)
            except Exception:
                logger.exception("%s_WORKER_SEND_EXCEPTION trace_id=%s reason=%s", self.name, req.trace_id, req.reason)
            finally:
                self._q.task_done()

    async def _wait_for_active_clear(self, timeout_s: float = 8.0) -> None:
        if self._active_done.is_set():
            return
        rid = self.active_response_id
        logger.info("%s_WAIT_ACTIVE response_id=%s", self.name, rid)
        try:
            await asyncio.wait_for(self._active_done.wait(), timeout=timeout_s)
        except asyncio.TimeoutError:
            logger.warning("%s_WAIT_TIMEOUT active_response_id=%s", self.name, rid)

    async def _send(self, req: SpeechRequest) -> None:
        # Shadow mode: do not send, but still drain queue and log
        meta = self._log_meta(req)
        if self.shadow:
            logger.info("%s_SHADOW_DROP %s", self.name, meta)
            return

        # Serialize tool speech: wait for any active response to complete.
        await self._wait_for_active_clear(timeout_s=8.0)

        payload = self._build_payload(req)

        ok, why = self._validate_payload(payload)
        if not ok:
            logger.error("%s_SCHEMA_GUARD_TRIP trace_id=%s reason=%s why=%s", self.name, req.trace_id, req.reason, why)
            return

        # Payload shape log (explicit proof metadata removed)
        resp_obj = payload.get("response") if isinstance(payload, dict) else None
        conv_val = (resp_obj.get("conversation") if isinstance(resp_obj, dict) else None)
        has_input = (isinstance(resp_obj, dict) and "input" in resp_obj)
        has_metadata = (isinstance(resp_obj, dict) and "metadata" in resp_obj)
        instr_len = (len(resp_obj.get("instructions")) if isinstance(resp_obj, dict) and isinstance(resp_obj.get("instructions"), str) else None)
        logger.info(
            "%s_PAYLOAD_SHAPE trace_id=%s conversation=%s has_input=%s has_metadata=%s instr_len=%s",
            self.name, req.trace_id, conv_val, has_input, has_metadata, instr_len
        )

        logger.info("%s_SEND_RESPONSE_CREATE %s", self.name, meta)
        await self.send_realtime_json(payload)

    # -----------------------------
    # Policy & payload building
    # -----------------------------
    def _resolve_policy(self, req: SpeechRequest) -> Tuple[TenantSpeechDefaults, TenantSpeechPolicy]:
        defaults, policy_map = self.tenant_defaults_provider(req.ctx.tenant_id)
        per_reason = policy_map.get(req.reason) if policy_map else None

        # Merge: defaults -> per_reason -> req.priority override (others can be added later)
        merged = TenantSpeechPolicy(
            priority=(per_reason.priority if per_reason and per_reason.priority is not None else None),
            speech_mode=(per_reason.speech_mode if per_reason and per_reason.speech_mode is not None else None),
            conversation_mode=(per_reason.conversation_mode if per_reason and per_reason.conversation_mode is not None else None),
            can_interrupt=(per_reason.can_interrupt if per_reason and per_reason.can_interrupt is not None else None),
            barge_grace_ms=(per_reason.barge_grace_ms if per_reason and per_reason.barge_grace_ms is not None else None),
            barge_debounce_ms=(per_reason.barge_debounce_ms if per_reason and per_reason.barge_debounce_ms is not None else None),
            max_chars=(per_reason.max_chars if per_reason and per_reason.max_chars is not None else None),
        )
        if req.priority is not None:
            merged = TenantSpeechPolicy(**{**merged.__dict__, "priority": req.priority})
        return defaults, merged

    def _build_payload(self, req: SpeechRequest) -> Dict[str, Any]:
        defaults, policy = self._resolve_policy(req)

        conversation_mode = policy.conversation_mode or defaults.conversation_mode_default
        max_chars = policy.max_chars if policy.max_chars is not None else defaults.max_chars_default

        content_text = (req.content_text_override or req.text or "").strip()
        if max_chars and len(content_text) > max_chars:
            content_text = content_text[: max_chars].rstrip() + "…"

        # Instructions: allow override to preserve legacy scaffolding for certain reasons
        if req.instructions_override:
            instructions = req.instructions_override
        else:
            # Minimal safe instruction; matches legacy "generic response" behavior.
            instructions = "Speak the following message to the caller in a natural tone:\n" + content_text

        # IMPORTANT: Realtime supports conversation: auto|none. Tenant-facing 'default' is mapped to omission.
        response_obj: Dict[str, Any] = {"instructions": instructions}
        if str(conversation_mode).lower() == "none":
            response_obj["conversation"] = "none"
        # else: omit field (server decides default behavior)

        return {"type": "response.create", "response": response_obj}

    def _validate_payload(self, payload: Dict[str, Any]) -> Tuple[bool, str]:
        try:
            if payload.get("type") != "response.create":
                return False, "payload.type must be response.create"
            resp = payload.get("response")
            if not isinstance(resp, dict):
                return False, "payload.response must be dict"
            if not isinstance(resp.get("instructions"), str) or not resp.get("instructions"):
                return False, "response.instructions must be non-empty str"
            conv = resp.get("conversation")
            if conv is not None and conv not in ("auto", "none"):
                # we intentionally omit "auto" by default; if present it must be valid
                return False, f"invalid response.conversation={conv!r}"
            if "metadata" in resp:
                return False, "metadata must not be present (intentionally stripped)"
            if "input" in resp:
                # We are not using input for tool speech in MVP; if it appears, validate shape.
                inp = resp.get("input")
                if not isinstance(inp, list) or not inp:
                    return False, "response.input must be non-empty list"
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

    # -----------------------------
    # Metadata logging (log prepared, then stripped)
    # -----------------------------
    def _log_meta(self, req: SpeechRequest) -> Dict[str, Any]:
        defaults, policy = self._resolve_policy(req)

        # Build the metadata we *would* attach (for debugging), then strip it.
        prepared = {
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
        prepared.update(req.text_meta())

        # Count None values as a sanity check
        none_count = sum(1 for v in prepared.values() if v is None)
        logger.info("%s_METADATA_PREPARED trace_id=%s keys=%s none_count=%s", self.name, req.trace_id, list(prepared.keys()), none_count)
        logger.info("%s_METADATA_STRIPPED trace_id=%s stripped=True", self.name, req.trace_id)

        # What we actually log as the compact meta blob (no PII).
        effective_conv = (policy.conversation_mode or defaults.conversation_mode_default)
        effective_max = (policy.max_chars if policy.max_chars is not None else defaults.max_chars_default)
        return {
            "trace_id": req.trace_id,
            "reason": req.reason,
            "tenant_id": req.ctx.tenant_id,
            "execution_id": req.ctx.execution_id,
            **req.text_meta(),
            "policy_conversation_mode": effective_conv,
            "policy_max_chars": effective_max,
        }

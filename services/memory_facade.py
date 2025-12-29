# services/memory_facade.py
"""Session + durable memory facade (MVP: session cache only).

Design goals:
- Keep memory work OUT of the audio hot path (no per-frame reads/writes).
- Fast per-turn recall inside /assistant/route.
- Feature-flagged and safe to disable instantly.

Current scope (Step 1):
- In-memory TTL session cache for skill outputs + reference handles.
- Durable memory is a stub (to be implemented with Postgres + tenancy).

NOTE: In-memory session cache is best-effort. If you run multiple Render instances,
a future step should swap SessionMemoryBackend to Redis for cross-worker safety.
"""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass
from threading import Lock
from typing import Any, Dict, Optional, Tuple

from core.logging import logger


SESSION_MEMORY_ENABLED = (os.getenv("SESSION_MEMORY_ENABLED") or "1").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
SESSION_MEMORY_TTL_S = int((os.getenv("SESSION_MEMORY_TTL_S") or "1800").strip() or "1800")  # 30 min
SESSION_MEMORY_DEBUG = (os.getenv("SESSION_MEMORY_DEBUG") or "0").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


def make_skill_cache_key_hash(*parts: Any) -> str:
    # Stable hash for (skill_name + params). Keep it short in logs.
    joined = "|".join([str(p) for p in parts])
    return _sha1(joined)


class SessionMemoryBackend:
    """Tiny TTL key/value store keyed by call_id.

    Stored as:
      store[call_id][key] = (expires_at_epoch, value)
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._store: Dict[str, Dict[str, Tuple[float, Any]]] = {}

    def _purge_expired_locked(self, call_id: str) -> None:
        bucket = self._store.get(call_id)
        if not bucket:
            return
        now = time.time()
        expired = [k for k, (exp, _v) in bucket.items() if exp <= now]
        for k in expired:
            bucket.pop(k, None)
        if not bucket:
            self._store.pop(call_id, None)

    def get(self, call_id: str, key: str, default: Any = None) -> Any:
        if not (SESSION_MEMORY_ENABLED and call_id and key):
            return default
        with self._lock:
            self._purge_expired_locked(call_id)
            bucket = self._store.get(call_id) or {}
            item = bucket.get(key)
            if not item:
                return default
            exp, value = item
            if exp <= time.time():
                bucket.pop(key, None)
                return default
            return value

    def set(self, call_id: str, key: str, value: Any, ttl_s: int) -> None:
        if not (SESSION_MEMORY_ENABLED and call_id and key):
            return
        ttl = max(1, int(ttl_s))
        exp = time.time() + ttl
        with self._lock:
            self._purge_expired_locked(call_id)
            if call_id not in self._store:
                self._store[call_id] = {}
            self._store[call_id][key] = (exp, value)

    def clear(self, call_id: str) -> None:
        if not call_id:
            return
        with self._lock:
            self._store.pop(call_id, None)


@dataclass(frozen=True)
class CachedSkillResult:
    skill_key: str
    cache_key_hash: str
    created_at_epoch: float
    result: Dict[str, Any]


class MemoryFacade:
    """High-level API for skills + router."""

    def __init__(self, backend: SessionMemoryBackend) -> None:
        self._backend = backend

    # ---------- Skill result cache ----------
    def get_cached_skill_result(
        self,
        *,
        tenant_id: str,
        call_id: str,
        skill_key: str,
        cache_key_hash: str,
    ) -> Optional[CachedSkillResult]:
        if not (SESSION_MEMORY_ENABLED and tenant_id and call_id and skill_key and cache_key_hash):
            return None

        key = f"skill_cache:{tenant_id}:{skill_key}:{cache_key_hash}"
        obj = self._backend.get(call_id, key, default=None)
        if not obj:
            if SESSION_MEMORY_DEBUG:
                logger.info("SESSION_MEM_MISS skill=%s hash=%s call_id=%s", skill_key, cache_key_hash[:8], call_id)
            return None

        if SESSION_MEMORY_DEBUG:
            logger.info("SESSION_MEM_HIT skill=%s hash=%s call_id=%s", skill_key, cache_key_hash[:8], call_id)

        try:
            return CachedSkillResult(
                skill_key=skill_key,
                cache_key_hash=cache_key_hash,
                created_at_epoch=float(obj.get("created_at_epoch") or 0),
                result=dict(obj.get("result") or {}),
            )
        except Exception:
            logger.exception("SESSION_MEM_CORRUPT skill=%s hash=%s", skill_key, cache_key_hash[:8])
            return None

    def put_cached_skill_result(
        self,
        *,
        tenant_id: str,
        call_id: str,
        skill_key: str,
        cache_key_hash: str,
        result: Dict[str, Any],
        ttl_s: int = SESSION_MEMORY_TTL_S,
    ) -> None:
        if not (SESSION_MEMORY_ENABLED and tenant_id and call_id and skill_key and cache_key_hash):
            return
        key = f"skill_cache:{tenant_id}:{skill_key}:{cache_key_hash}"
        payload = {"created_at_epoch": time.time(), "result": result}
        self._backend.set(call_id, key, payload, ttl_s=ttl_s)

    # ---------- Reference handles (follow-ups) ----------
    def set_handle(
        self, *, tenant_id: str, call_id: str, name: str, value: Any, ttl_s: int = SESSION_MEMORY_TTL_S
    ) -> None:
        if not (SESSION_MEMORY_ENABLED and tenant_id and call_id and name):
            return
        key = f"handle:{tenant_id}:{name}"
        self._backend.set(call_id, key, value, ttl_s=ttl_s)

    def get_handle(self, *, tenant_id: str, call_id: str, name: str, default: Any = None) -> Any:
        if not (SESSION_MEMORY_ENABLED and tenant_id and call_id and name):
            return default
        key = f"handle:{tenant_id}:{name}"
        return self._backend.get(call_id, key, default=default)


session_memory_backend = SessionMemoryBackend()
memory = MemoryFacade(session_memory_backend)

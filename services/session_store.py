# services/session_store.py
"""Tiny in-memory store for per-call, multi-turn state.

Purpose:
- Remember caller choices (e.g., which Gmail inbox to use) across turns in the SAME call.

Notes:
- This is intentionally in-memory and best-effort.
- Keyed by streamSid/callSid passed in the assistant route context.
- Safe for the call path: O(1) lookups, no DB writes.
"""

from __future__ import annotations

from threading import Lock
from typing import Any, Dict


class SessionStore:
    def __init__(self) -> None:
        self._lock = Lock()
        self._store: Dict[str, Dict[str, Any]] = {}

    def get(self, call_id: str) -> Dict[str, Any]:
        if not call_id:
            return {}
        with self._lock:
            return dict(self._store.get(call_id, {}))

    def set(self, call_id: str, key: str, value: Any) -> None:
        if not call_id:
            return
        with self._lock:
            if call_id not in self._store:
                self._store[call_id] = {}
            self._store[call_id][key] = value

    def pop(self, call_id: str, key: str, default: Any = None) -> Any:
        if not call_id:
            return default
        with self._lock:
            bucket = self._store.get(call_id)
            if not bucket:
                return default
            return bucket.pop(key, default)

    def clear(self, call_id: str) -> None:
        if not call_id:
            return
        with self._lock:
            self._store.pop(call_id, None)


session_store = SessionStore()

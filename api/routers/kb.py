# api/routers/kb.py
from __future__ import annotations

from fastapi import APIRouter

# Placeholder router to unblock deploy.
# Real KB endpoints can be implemented later without changing main.py imports.
router = APIRouter(prefix="/kb", tags=["kb"])


@router.get("/health")
async def kb_health():
    return {
        "ok": True,
        "kb": "placeholder",
        "note": "KB router exists to satisfy import; implement endpoints later.",
    }

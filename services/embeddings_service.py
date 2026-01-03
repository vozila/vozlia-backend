# services/embeddings_service.py
from __future__ import annotations

import os
from typing import List
import httpx

from core.logging import logger

DEFAULT_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small").strip() or "text-embedding-3-small"

def _api_key() -> str:
    return (os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY") or "").strip()

def _base_url() -> str:
    return (os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")

def embed_texts(texts: List[str], *, model: str | None = None, timeout_s: float | None = None) -> List[List[float]]:
    """Return embeddings for texts. Raises on failure."""
    key = _api_key()
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set")
    m = (model or DEFAULT_EMBED_MODEL).strip() or DEFAULT_EMBED_MODEL
    t = timeout_s or float(os.getenv("OPENAI_EMBED_TIMEOUT_S", "8.0") or 8.0)

    url = f"{_base_url()}/embeddings"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    payload = {"model": m, "input": texts}

    with httpx.Client(timeout=t) as client:
        r = client.post(url, headers=headers, json=payload)
        r.raise_for_status()
        data = r.json()
        out: List[List[float]] = []
        for item in data.get("data", []):
            out.append(item.get("embedding") or [])
        if len(out) != len(texts):
            raise RuntimeError(f"Embedding count mismatch: expected {len(texts)} got {len(out)}")
        return out

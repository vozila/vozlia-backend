# services/kb_voice.py
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import text
from sqlalchemy.orm import Session

from core.logging import logger, env_flag
from core import config as cfg

try:
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore


@dataclass
class KBResult:
    ok: bool
    retrieval_strategy: str
    answer: Optional[str]
    sources: List[Dict[str, Any]]
    policy_chars: int
    context_chars: int
    latency_ms: float
    model: Optional[str]


def _truthy_env(name: str, default: str = "0") -> bool:
    v = (os.getenv(name, default) or default).strip().lower()
    return v in ("1", "true", "yes", "on")


def _clamp_int(v: Any, default: int, lo: int, hi: int) -> int:
    try:
        x = int(v)
    except Exception:
        x = default
    return max(lo, min(hi, x))


def _sanitize_query(q: str) -> str:
    q = (q or "").strip()
    # Keep queries small to protect DB / logs
    q = re.sub(r"\s+", " ", q)
    if len(q) > 400:
        q = q[:400]
    return q


# Lazily constructed OpenAI client (reuse across calls)
_OAI: Optional[Any] = None


def _oai() -> Optional[Any]:
    global _OAI
    if _OAI is not None:
        return _OAI
    if not cfg.OPENAI_API_KEY or not OpenAI:
        return None
    _OAI = OpenAI(api_key=cfg.OPENAI_API_KEY)
    return _OAI


def retrieve_kb_sources(
    db: Session,
    *,
    tenant_uuid: str,
    query: str,
    limit: int = 8,
    include_policy: bool = True,
    snippet_chars: int = 1200,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Retrieve relevant KB chunks for a tenant.

    Returns:
      (strategy, sources)
    """
    q = _sanitize_query(query)
    if not q:
        return ("empty_query", [])

    limit = _clamp_int(limit, 8, 1, 50)
    snippet_chars = _clamp_int(snippet_chars, 1200, 200, 4000)
    kinds = ("knowledge", "policy") if include_policy else ("knowledge",)

    # NOTE: kb_files.tenant_id is varchar, kb_chunks.tenant_id is uuid.
    # We join using kb_files.tenant_id = kb_chunks.tenant_id::text.
    # This matches the control-plane schema currently in use.
    base_params = {
        "tenant_uuid": str(tenant_uuid),
        "q": q,
        "limit": limit,
        "snippet_chars": snippet_chars,
        "kinds": list(kinds),
    }

    # 1) Full-text search (best signal). Uses runtime tsvector, so no schema dependency.
    try:
        fts_sql = text(
            """
            SELECT
              c.file_id AS file_id,
              f.filename AS filename,
              f.content_type AS content_type,
              c.kind AS kind,
              c.chunk_index AS chunk_index,
              ts_rank_cd(to_tsvector('english', c.text), plainto_tsquery('english', :q)) AS score,
              substring(c.text from 1 for :snippet_chars) AS snippet
            FROM kb_chunks c
            JOIN kb_files f
              ON f.id = c.file_id
             AND f.tenant_id = (c.tenant_id::text)
            WHERE c.tenant_id = (:tenant_uuid)::uuid
              AND c.kind = ANY(:kinds)
              AND to_tsvector('english', c.text) @@ plainto_tsquery('english', :q)
            ORDER BY score DESC, c.created_at DESC
            LIMIT :limit
            """
        )
        rows = db.execute(fts_sql, base_params).mappings().all()
        sources = [dict(r) for r in rows] if rows else []
        if sources:
            return ("fts", sources)
    except Exception:
        # Don't fail the call if FTS is unavailable (e.g., permissions / dialect differences).
        logger.exception("KB_RETRIEVE_FTS_FAIL tenant_uuid=%s q=%r", tenant_uuid, q)

    # 2) ILIKE fallback (works everywhere, but weaker)
    try:
        ilike_sql = text(
            """
            SELECT
              c.file_id AS file_id,
              f.filename AS filename,
              f.content_type AS content_type,
              c.kind AS kind,
              c.chunk_index AS chunk_index,
              NULL::float AS score,
              substring(c.text from 1 for :snippet_chars) AS snippet
            FROM kb_chunks c
            JOIN kb_files f
              ON f.id = c.file_id
             AND f.tenant_id = (c.tenant_id::text)
            WHERE c.tenant_id = (:tenant_uuid)::uuid
              AND c.kind = ANY(:kinds)
              AND c.text ILIKE ('%' || :q || '%')
            ORDER BY c.created_at DESC
            LIMIT :limit
            """
        )
        rows = db.execute(ilike_sql, base_params).mappings().all()
        sources = [dict(r) for r in rows] if rows else []
        if sources:
            return ("ilike", sources)
    except Exception:
        logger.exception("KB_RETRIEVE_ILIKE_FAIL tenant_uuid=%s q=%r", tenant_uuid, q)

    # 3) Recent fallback (useful for very generic questions like "what is this document about?")
    try:
        recent_sql = text(
            """
            SELECT
              c.file_id AS file_id,
              f.filename AS filename,
              f.content_type AS content_type,
              c.kind AS kind,
              c.chunk_index AS chunk_index,
              NULL::float AS score,
              substring(c.text from 1 for :snippet_chars) AS snippet
            FROM kb_chunks c
            JOIN kb_files f
              ON f.id = c.file_id
             AND f.tenant_id = (c.tenant_id::text)
            WHERE c.tenant_id = (:tenant_uuid)::uuid
              AND c.kind = ANY(:kinds)
            ORDER BY c.created_at DESC
            LIMIT :limit
            """
        )
        rows = db.execute(recent_sql, base_params).mappings().all()
        sources = [dict(r) for r in rows] if rows else []
        if sources:
            return ("recent_fallback", sources)
    except Exception:
        logger.exception("KB_RETRIEVE_RECENT_FAIL tenant_uuid=%s", tenant_uuid)

    return ("none", [])


def _load_policy_text(
    db: Session,
    *,
    tenant_uuid: str,
    max_chars: int = 2400,
) -> str:
    """Load policy chunks (kind='policy') for the tenant, truncated."""
    max_chars = _clamp_int(max_chars, 2400, 0, 20000)
    if max_chars <= 0:
        return ""

    try:
        sql = text(
            """
            SELECT substring(c.text from 1 for 4000) AS snippet
            FROM kb_chunks c
            WHERE c.tenant_id = (:tenant_uuid)::uuid
              AND c.kind = 'policy'
            ORDER BY c.created_at DESC
            LIMIT 8
            """
        )
        rows = db.execute(sql, {"tenant_uuid": str(tenant_uuid)}).mappings().all()
        parts: List[str] = []
        total = 0
        for r in rows:
            s = (r.get("snippet") or "").strip()
            if not s:
                continue
            # Keep simple separators for prompt readability
            if parts:
                parts.append("\n\n---\n\n")
                total += 7
            if total + len(s) > max_chars:
                s = s[: max(0, max_chars - total)]
            parts.append(s)
            total += len(s)
            if total >= max_chars:
                break
        return "".join(parts).strip()
    except Exception:
        logger.exception("KB_POLICY_LOAD_FAIL tenant_uuid=%s", tenant_uuid)
        return ""


def _build_context_text(
    sources: List[Dict[str, Any]],
    *,
    max_chars: int = 6500,
) -> str:
    max_chars = _clamp_int(max_chars, 6500, 0, 40000)
    if max_chars <= 0:
        return ""
    parts: List[str] = []
    total = 0
    for s in sources:
        filename = str(s.get("filename") or "unknown")
        kind = str(s.get("kind") or "knowledge")
        idx = s.get("chunk_index")
        snippet = (s.get("snippet") or "").strip()
        header = f"[{kind}] {filename} (chunk {idx})\n"
        block = header + snippet
        if parts:
            sep = "\n\n---\n\n"
            if total + len(sep) < max_chars:
                parts.append(sep)
                total += len(sep)
        if total + len(block) > max_chars:
            block = block[: max(0, max_chars - total)]
        parts.append(block)
        total += len(block)
        if total >= max_chars:
            break
    return "".join(parts).strip()


def answer_from_kb(
    db: Session,
    *,
    tenant_uuid: str,
    query: str,
    limit: int = 8,
    include_policy: bool = True,
) -> KBResult:
    """Generate a voice-friendly answer grounded in KB chunks.

    This is intentionally conservative:
    - If no sources are found, we return ok=True with answer=None (caller can fall back to FSM).
    - If OpenAI is not configured, we return ok=False with a helpful error in logs (caller can fall back).
    """
    t0 = time.perf_counter()
    q = _sanitize_query(query)

    # Tunables (safe defaults)
    snippet_chars = _clamp_int(os.getenv("VOICE_KB_SNIPPET_CHARS", "1200"), 1200, 200, 4000)
    max_context_chars = _clamp_int(os.getenv("VOICE_KB_MAX_CONTEXT_CHARS", "6500"), 6500, 0, 40000)
    max_policy_chars = _clamp_int(os.getenv("VOICE_KB_POLICY_MAX_CHARS", "2400"), 2400, 0, 20000)
    model = (os.getenv("VOICE_KB_MODEL", "") or "gpt-4o-mini").strip()

    strategy, sources = retrieve_kb_sources(
        db,
        tenant_uuid=str(tenant_uuid),
        query=q,
        limit=limit,
        include_policy=include_policy,
        snippet_chars=snippet_chars,
    )

    # If we didn't find anything, let caller decide next step
    if not sources:
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return KBResult(
            ok=True,
            retrieval_strategy=strategy,
            answer=None,
            sources=[],
            policy_chars=0,
            context_chars=0,
            latency_ms=latency_ms,
            model=None,
        )

    policy_text = _load_policy_text(db, tenant_uuid=str(tenant_uuid), max_chars=max_policy_chars) if include_policy else ""
    context_text = _build_context_text(sources, max_chars=max_context_chars)

    # If OpenAI isn't available, return retrieval-only
    client = _oai()
    if not client:
        latency_ms = (time.perf_counter() - t0) * 1000.0
        logger.warning("KB_ANSWER_NO_OPENAI tenant_uuid=%s", tenant_uuid)
        return KBResult(
            ok=False,
            retrieval_strategy=strategy,
            answer=None,
            sources=sources,
            policy_chars=len(policy_text),
            context_chars=len(context_text),
            latency_ms=latency_ms,
            model=None,
        )

    system = (
        "You are Vozlia, a real-time phone assistant for a small business.\n"
        "Answer the caller's question using ONLY the provided KB CONTEXT and BUSINESS POLICY.\n"
        "If the answer is not supported by the context, say you don't know and ask a brief clarifying question.\n"
        "Keep responses voice-friendly: short, clear, no bullet lists unless requested.\n"
        "Do NOT read citations aloud.\n"
    )

    if policy_text:
        system += "\nBUSINESS POLICY (follow these rules):\n" + policy_text.strip() + "\n"

    user = (
        f"CALLER QUESTION:\n{q}\n\n"
        "KB CONTEXT:\n"
        f"{context_text}\n"
    )

    answer = None
    try:
        # Keep latency/cost bounded
        resp = client.chat.completions.create(
            model=model,
            temperature=0.2,
            max_tokens=_clamp_int(os.getenv("VOICE_KB_MAX_TOKENS", "220"), 220, 60, 600),
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        answer = (resp.choices[0].message.content or "").strip()
        if answer:
            # Safety: avoid overly long rambling replies in voice
            max_answer_chars = _clamp_int(os.getenv("VOICE_KB_MAX_ANSWER_CHARS", "900"), 900, 200, 4000)
            if len(answer) > max_answer_chars:
                answer = answer[:max_answer_chars].rstrip() + "…"
    except Exception:
        logger.exception("KB_ANSWER_OPENAI_FAIL tenant_uuid=%s model=%s", tenant_uuid, model)
        answer = None

    latency_ms = (time.perf_counter() - t0) * 1000.0
    return KBResult(
        ok=bool(answer),
        retrieval_strategy=strategy,
        answer=answer,
        sources=sources,
        policy_chars=len(policy_text),
        context_chars=len(context_text),
        latency_ms=latency_ms,
        model=model if answer else None,
    )

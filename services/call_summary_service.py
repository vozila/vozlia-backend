# services/call_summary_service.py
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import httpx

from core.logging import logger
from models import CallerMemoryEvent
from sqlalchemy import text

from services.embeddings_service import embed_texts

def _api_key() -> str:
    return (os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY") or "").strip()

def _base_url() -> str:
    return (os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")

def _summary_model() -> str:
    return (os.getenv("CALL_SUMMARY_MODEL") or os.getenv("OPENAI_SUMMARY_MODEL") or "gpt-4o-mini").strip()

def _max_transcript_chars() -> int:
    return int(os.getenv("CALL_SUMMARY_MAX_TRANSCRIPT_CHARS", "12000") or 12000)

def _response_timeout_s() -> float:
    return float(os.getenv("CALL_SUMMARY_TIMEOUT_S", "12.0") or 12.0)

def _embedding_enabled() -> bool:
    return (os.getenv("VECTOR_MEMORY_ENABLED", "0").strip() == "1")

def _vector_str(v: List[float]) -> str:
    # pgvector accepts: '[0.1,0.2,...]'
    return "[" + ",".join(f"{x:.8f}" for x in v) + "]"

def _chat_json(messages: List[Dict[str, str]]) -> Dict[str, Any]:
    key = _api_key()
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set")
    url = f"{_base_url()}/chat/completions"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    payload = {
        "model": _summary_model(),
        "temperature": 0.2,
        "max_tokens": int(os.getenv("CALL_SUMMARY_MAX_TOKENS", "420") or 420),
        "response_format": {"type": "json_object"},
        "messages": messages,
    }
    with httpx.Client(timeout=_response_timeout_s()) as client:
        r = client.post(url, headers=headers, json=payload)
        r.raise_for_status()
        data = r.json()
        content = data["choices"][0]["message"]["content"]
        try:
            return json.loads(content)
        except Exception:
            logger.warning("CALL_SUMMARY_JSON_PARSE_FAIL content_prefix=%r", (content or "")[:200])
            return {"summary": (content or "").strip()}

def build_transcript(events: List[CallerMemoryEvent]) -> str:
    lines: List[str] = []
    for e in events:
        txt = (e.text or "").strip()
        if not txt:
            continue
        role = "Note"
        if e.kind == "turn":
            sk = (e.skill_key or "").lower()
            if "user" in sk:
                role = "Caller"
            elif "assistant" in sk:
                role = "Assistant"
            else:
                role = "Turn"
        elif e.skill_key:
            role = f"Assistant({e.skill_key})"
        ts = ""
        if getattr(e, "created_at", None):
            try:
                ts = e.created_at.strftime("%H:%M:%S")
            except Exception:
                ts = ""
        prefix = f"[{ts}] " if ts else ""
        lines.append(f"{prefix}{role}: {txt}")
    transcript = "\n".join(lines)
    mx = _max_transcript_chars()
    if len(transcript) > mx:
        transcript = transcript[-mx:]
        transcript = "(truncated)\n" + transcript
    return transcript

def generate_call_summary(transcript: str) -> Dict[str, Any]:
    sys = (
        "You summarize phone calls for long-term memory. "
        "Output JSON with keys: summary, memory_bullets, preferences, facts, todos. "
        "- summary: 4-8 bullet sentences, <= 900 chars.\n"
        "- memory_bullets: up to 8 short bullets of durable facts the assistant should remember.\n"
        "- preferences: dict (e.g., likes/dislikes).\n"
        "- facts: dict of important entities or details.\n"
        "- todos: list of follow-ups.\n"
        "Never say you lack memory; these notes ARE the memory."
    )
    user = f"TRANSCRIPT:\n{transcript}"
    return _chat_json([
        {"role": "system", "content": sys},
        {"role": "user", "content": user},
    ])

def write_call_summary_event(
    db: Any,
    *,
    tenant_id: str,
    caller_id: str,
    call_sid: str,
    summary_obj: Dict[str, Any],
) -> bool:
    summary_text = (summary_obj.get("summary") or "").strip()
    bullets = summary_obj.get("memory_bullets") or []
    if isinstance(bullets, list) and bullets:
        bullets_text = "\n".join(f"- {str(b).strip()}" for b in bullets[:12] if str(b).strip())
        if bullets_text:
            summary_text = (summary_text + "\n\nMemory bullets:\n" + bullets_text).strip()

    data_json = {
        "memory_bullets": summary_obj.get("memory_bullets") or [],
        "preferences": summary_obj.get("preferences") or {},
        "facts": summary_obj.get("facts") or {},
        "todos": summary_obj.get("todos") or [],
        "schema": "call_summary.v1",
    }

    ev = CallerMemoryEvent(
        tenant_id=tenant_id,
        caller_id=caller_id,
        call_sid=call_sid,
        kind="skill",
        skill_key="call_summary",
        text=summary_text[:4000],
        data_json=data_json,
        tags_json=["skill:call_summary"],
    )

    vec_lit: str | None = None
    if _embedding_enabled():
        try:
            emb = embed_texts([summary_text])[0]
            vec_lit = _vector_str(emb)
            # If model has embedding attr (pgvector column), store it.
            if hasattr(ev, "embedding"):
                setattr(ev, "embedding", emb)
            else:
                # fallback: store in json (keeps data even if pgvector isn't wired yet)
                data_json["embedding"] = emb
            data_json["embedding_vector"] = vec_lit
        except Exception:
            logger.exception("CALL_SUMMARY_EMBED_FAIL tenant_id=%s caller_id=%s call_sid=%s", tenant_id, caller_id, call_sid)

    db.add(ev)
    db.commit()

    # If the ORM model doesn't have an embedding column but the DB does (pgvector),
    # write it via raw SQL using the vector literal.
    if vec_lit:
        try:
            db.execute(
                text("UPDATE caller_memory_events SET embedding = (:v)::vector WHERE id = :id"),
                {"v": vec_lit, "id": ev.id},
            )
            db.commit()
        except Exception:
            logger.exception("CALL_SUMMARY_EMBED_SQL_WRITE_FAIL tenant_id=%s caller_id=%s call_sid=%s", tenant_id, caller_id, call_sid)

    return True


def ensure_call_summary_for_call(db: Any, *, call_sid: str, caller_id: str) -> None:
    """Idempotent: creates call_summary for call_sid if enabled and not present."""
    if os.getenv("CALL_SUMMARY_ENABLED", "0").strip() != "1":
        return
    if not call_sid or not caller_id:
        return

    # Find tenant_id from existing events for this call
    tenant_row = (
        db.query(CallerMemoryEvent.tenant_id)
        .filter(CallerMemoryEvent.call_sid == call_sid)
        .order_by(CallerMemoryEvent.created_at.desc())
        .first()
    )
    tenant_id = str(tenant_row[0]) if tenant_row and tenant_row[0] else None
    if not tenant_id:
        logger.warning("CALL_SUMMARY_SKIP no tenant_id for call_sid=%s", call_sid)
        return

    exists = (
        db.query(CallerMemoryEvent.id)
        .filter(CallerMemoryEvent.tenant_id == tenant_id)
        .filter(CallerMemoryEvent.caller_id == caller_id)
        .filter(CallerMemoryEvent.call_sid == call_sid)
        .filter(CallerMemoryEvent.skill_key == "call_summary")
        .first()
    )
    if exists:
        return

    events = (
        db.query(CallerMemoryEvent)
        .filter(CallerMemoryEvent.tenant_id == tenant_id)
        .filter(CallerMemoryEvent.caller_id == caller_id)
        .filter(CallerMemoryEvent.call_sid == call_sid)
        .order_by(CallerMemoryEvent.created_at.asc())
        .all()
    )
    if not events:
        logger.warning("CALL_SUMMARY_SKIP no events for call_sid=%s", call_sid)
        return

    transcript = build_transcript(events)
    summary_obj = generate_call_summary(transcript)
    write_call_summary_event(db, tenant_id=tenant_id, caller_id=caller_id, call_sid=call_sid, summary_obj=summary_obj)
    logger.info("CALL_SUMMARY_WRITE_OK tenant_id=%s caller_id=%s call_sid=%s", tenant_id, caller_id, call_sid)

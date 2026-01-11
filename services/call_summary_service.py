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


def _env_int(name: str, default: int) -> int:
    try:
        return int((os.getenv(name) or str(default)).strip())
    except Exception:
        return default


def _env_bool(name: str, default: bool) -> bool:
    v = (os.getenv(name) or "").strip().lower()
    if not v:
        return bool(default)
    return v in ("1", "true", "yes", "on")


# Guardrails: if transcript is empty / too short, DO NOT generate a call summary.
# This prevents the model from fabricating a plausible-sounding "summary" when we have no text evidence.
CALL_SUMMARY_MIN_TRANSCRIPT_CHARS = _env_int("CALL_SUMMARY_MIN_TRANSCRIPT_CHARS", 20)
CALL_SUMMARY_REQUIRE_CALLER_TURN = _env_bool("CALL_SUMMARY_REQUIRE_CALLER_TURN", True)


def _has_caller_turn(events: List[CallerMemoryEvent]) -> bool:
    """
    Require at least one caller/user turn event (prevents call summaries on memory-recall-only calls
    where turns are intentionally not persisted).
    """
    for e in events or []:
        try:
            if getattr(e, "kind", None) == "turn":
                sk = (getattr(e, "skill_key", "") or "").strip().lower()
                if sk == "turn_user" or sk.endswith("_user") or "user" in sk:
                    txt = (getattr(e, "text", "") or "").strip()
                    if txt:
                        return True
        except Exception:
            continue
    return False


def transcript_is_actionable(transcript: str, events: List[CallerMemoryEvent]) -> bool:
    t = (transcript or "").strip()
    if not t:
        return False
    if len(t) < CALL_SUMMARY_MIN_TRANSCRIPT_CHARS:
        return False
    if CALL_SUMMARY_REQUIRE_CALLER_TURN and (not _has_caller_turn(events)):
        return False
    return True


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
        # Skip internal trace/audit rows so they don't pollute call summaries
        sk_norm = (getattr(e, "skill_key", "") or "").strip().lower()
        if sk_norm in {"memory_recall_audit", "memory_context_audit"}:
            continue

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
        "You summarize phone calls for long-term memory and vector recall. "
        "Output JSON ONLY (no markdown) with keys: summary, memory_bullets, preferences, facts, todos.\n\n"
        "Rules (important):\n"
        "1) Capture the caller's questions/requests explicitly (what they asked for).\n"
        "2) Extract casual facts too: if the caller casually mentions a preference or personal detail, include it.\n"
        "   Example: 'I like cats' -> preferences.likes includes 'cats' AND memory_bullets includes 'Caller likes cats.'\n"
        "3) Put named entities into facts.entities when possible (tickers, names, animals, products, places).\n"
        "4) Make memory_bullets durable and searchable: include key nouns (e.g., 'cats', 'MSFT') verbatim.\n"
        "5) Do NOT invent details. Use only what appears in the transcript.\n\n"
        "Field specs:\n"
        "- summary: 4-8 bullet sentences, <= 900 chars total.\n"
        "- memory_bullets: up to 10 short bullets of durable facts to remember.\n"
        "- preferences: dict (likes/dislikes/favorites). Use simple strings or lists.\n"
        "- facts: dict. Prefer {entities:{...}} plus any other important details.\n"
        "- todos: list of follow-ups.\n"
        "Never say you lack memory; these notes ARE the memory."
    )
    user = f"TRANSCRIPT:\n{transcript}"
    return _chat_json(
        [
            {"role": "system", "content": sys},
            {"role": "user", "content": user},
        ]
    )


def write_call_summary_event(
    db: Any,
    *,
    tenant_id: str,
    caller_id: str,
    call_sid: str,
    summary_obj: Dict[str, Any],
    transcript: str,
) -> bool:
    if CallerMemoryEvent is None:
        return False

    summary_text = (summary_obj.get("summary") or "").strip()
    memory_bullets = summary_obj.get("memory_bullets") or []
    prefs = summary_obj.get("preferences") or {}
    facts = summary_obj.get("facts") or {}
    todos = summary_obj.get("todos") or {}

    data_json = {
        "summary": summary_text,
        "memory_bullets": memory_bullets,
        "preferences": prefs,
        "facts": facts,
        "todos": todos,
        "transcript": transcript,
        "model": _summary_model(),
        "created_at": datetime.utcnow().isoformat(timespec="seconds"),
    }

    # Create a single embedding string for vector search
    embedding_text = summary_text
    try:
        if isinstance(memory_bullets, list) and memory_bullets:
            embedding_text += "\n\nBullets:\n- " + "\n- ".join([str(b) for b in memory_bullets[:12]])
    except Exception:
        pass
    try:
        if prefs:
            embedding_text += "\n\nPreferences: " + json.dumps(prefs, ensure_ascii=False)
        if facts:
            embedding_text += "\n\nFacts: " + json.dumps(facts, ensure_ascii=False)
        if todos:
            embedding_text += "\n\nTodos: " + json.dumps(todos, ensure_ascii=False)
    except Exception:
        pass

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
            emb = embed_texts([embedding_text])[0]
            vec_lit = _vector_str(emb)
            # If model has embedding attr (pgvector column), store it.
            if hasattr(ev, "embedding"):
                setattr(ev, "embedding", emb)
            else:
                data_json["embedding"] = emb
        except Exception:
            logger.exception("CALL_SUMMARY_EMBED_FAIL tenant_id=%s caller_id=%s call_sid=%s", tenant_id, caller_id, call_sid)

    try:
        db.add(ev)
        db.commit()
    except Exception:
        db.rollback()
        logger.exception("CALL_SUMMARY_WRITE_FAIL tenant_id=%s caller_id=%s call_sid=%s", tenant_id, caller_id, call_sid)
        return False

    # If we computed a vector literal and the table supports a pgvector column, attempt an UPDATE.
    # This is defensive for deployments where SQLAlchemy model isn't synced with the DB column.
    if vec_lit and not hasattr(ev, "embedding"):
        try:
            db.execute(
                text("UPDATE caller_memory_events SET embedding = (:v)::vector WHERE id = :id"),
                {"v": vec_lit, "id": ev.id},
            )
            db.commit()
        except Exception:
            logger.exception("CALL_SUMMARY_EMBED_SQL_WRITE_FAIL tenant_id=%s caller_id=%s call_sid=%s", tenant_id, caller_id, call_sid)

    return True


def ensure_call_summary_for_call(
    db: Any,
    *,
    call_sid: str,
    tenant_id: str | None = None,
    caller_id: str | None = None,
) -> None:
    """Idempotent: creates call_summary for call_sid if enabled and not present.

    Identity strategy:
    - Preferred: (tenant_id, caller_id) passed in from Twilio Media Stream customParameters
      so finalize-time does not depend on best-effort inference.
    - Fallback: infer from existing memory events for this call_sid.
    """
    enabled = (os.getenv("CALL_SUMMARY_ENABLED", "0").strip() == "1")
    if not enabled:
        return
    if not call_sid:
        return

    tenant_id_resolved = (tenant_id or "").strip()
    caller_id_resolved = (caller_id or "").strip()

    if tenant_id_resolved and caller_id_resolved:
        logger.info(
            "CALL_SUMMARY_IDENTITY_FROM_STREAM tenant_id=%s caller_id=%s call_sid=%s",
            tenant_id_resolved,
            caller_id_resolved,
            call_sid,
        )
    else:
        # Find tenant_id + caller_id from existing events for this call
        row = (
            db.query(CallerMemoryEvent.tenant_id, CallerMemoryEvent.caller_id)
            .filter(CallerMemoryEvent.call_sid == call_sid)
            .order_by(CallerMemoryEvent.created_at.desc())
            .first()
        )
        if not row or not row[0] or not row[1]:
            logger.warning("CALL_SUMMARY_SKIP no tenant/caller for call_sid=%s", call_sid)
            return
        tenant_id_resolved = str(row[0])
        caller_id_resolved = str(row[1])

    tenant_id = tenant_id_resolved
    caller_id = caller_id_resolved

    # Already summarized?
    existing = (
        db.query(CallerMemoryEvent.id)
        .filter(CallerMemoryEvent.tenant_id == tenant_id)
        .filter(CallerMemoryEvent.caller_id == caller_id)
        .filter(CallerMemoryEvent.call_sid == call_sid)
        .filter(CallerMemoryEvent.skill_key == "call_summary")
        .first()
    )
    if existing:
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
    if not transcript_is_actionable(transcript, events):
        logger.warning(
            "CALL_SUMMARY_SKIP transcript_empty_or_no_caller_turn tenant_id=%s caller_id=%s call_sid=%s transcript_chars=%s",
            tenant_id,
            caller_id,
            call_sid,
            len((transcript or "").strip()),
        )
        return

    summary_obj = generate_call_summary(transcript)
    write_call_summary_event(
        db,
        tenant_id=tenant_id,
        caller_id=caller_id,
        call_sid=call_sid,
        summary_obj=summary_obj,
        transcript=transcript,
    )
    logger.info("CALL_SUMMARY_WRITE_OK tenant_id=%s caller_id=%s call_sid=%s", tenant_id, caller_id, call_sid)

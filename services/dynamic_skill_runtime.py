# services/dynamic_skill_runtime.py
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

from sqlalchemy.orm import Session

from core.logging import logger
from services.settings_service import get_skills_config
from services.web_search_service import run_web_search
from services.db_query_service import run_db_query
from services.longterm_memory import record_skill_result
from services.analytics_events import emit_analytics_event


# Cache skills_config by tenant for a short TTL to avoid per-turn DB hits in the voice hot path.
_CFG_CACHE: dict[str, tuple[float, dict[str, dict]]] = {}
_CFG_TTL_S = float((os.getenv("DYNAMIC_SKILLS_CACHE_TTL_S") or "15").strip() or "15")


_STOPWORDS = {
    "a", "an", "the", "my", "me", "please", "give", "show", "run", "do", "get", "tell", "today", "todays",
    "this", "that", "for", "to", "of", "in", "on", "at", "and", "or", "is", "are", "was", "were",
}


def _norm(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _tokens(s: str) -> list[str]:
    t = _norm(s)
    toks = [w for w in t.split(" ") if w and w not in _STOPWORDS]
    return toks


@dataclass(frozen=True)
class SkillMatch:
    skill_id: str
    skill_cfg: dict[str, Any]
    score: float
    reason: str


def _get_skills_config_cached(db: Session, user: Any) -> dict[str, dict]:
    tenant_id = str(getattr(user, "id", "") or "")
    if not tenant_id:
        return {}
    now = time.time()
    hit = _CFG_CACHE.get(tenant_id)
    if hit and (now - hit[0]) < _CFG_TTL_S:
        return hit[1]
    cfg = get_skills_config(db, user) or {}
    if not isinstance(cfg, dict):
        cfg = {}
    _CFG_CACHE[tenant_id] = (now, cfg)
    return cfg


def _eligible_dynamic_skill(skill_id: str, cfg: dict[str, Any]) -> bool:
    # Only route to dynamically created skills; avoid interfering with built-ins.
    # WebSearchSkill store uses keys websearch_<uuid>, type=web_search
    # DBQuerySkill store uses keys dbquery_<uuid>, type=db_query
    t = (cfg or {}).get("type")
    if skill_id.startswith("websearch_") and t == "web_search":
        return bool((cfg or {}).get("enabled", True))
    if skill_id.startswith("dbquery_") and t == "db_query":
        return bool((cfg or {}).get("enabled", True))
    return False


def match_dynamic_skill(db: Session, user: Any, text: str) -> SkillMatch | None:
    """Deterministically match an utterance to a dynamically-created skill (websearch_*/dbquery_*)."""
    if not (text or "").strip():
        return None

    enabled = (os.getenv("DYNAMIC_SKILLS_ENABLED", "1") or "1").strip().lower() in ("1", "true", "yes", "on")
    if not enabled:
        return None

    cfg_all = _get_skills_config_cached(db, user)
    if not cfg_all:
        return None

    u = _norm(text)
    u_tokens = set(_tokens(text))
    if not u_tokens and not u:
        return None

    best: SkillMatch | None = None

    for skill_id, scfg in cfg_all.items():
        if not isinstance(skill_id, str) or not isinstance(scfg, dict):
            continue
        if not _eligible_dynamic_skill(skill_id, scfg):
            continue

        phrases: list[str] = []
        ep = scfg.get("engagement_phrases")
        if isinstance(ep, list):
            phrases.extend([p for p in ep if isinstance(p, str) and p.strip()])

        # Also match on label/name (helps "give me my sports digest" for "Today's Sports Digest")
        label = scfg.get("label")
        if isinstance(label, str) and label.strip():
            phrases.append(label.strip())

        # Score each phrase
        for ph in phrases:
            pnorm = _norm(ph)
            if not pnorm:
                continue

            # Substring match (strong)
            if pnorm in u:
                score = 100.0 + min(25.0, float(len(pnorm)) / 4.0)
                reason = f"substring:{pnorm}"
            else:
                # Token overlap (moderate)
                pt = [t for t in _tokens(ph)]
                if not pt:
                    continue
                inter = len(set(pt).intersection(u_tokens))
                need = max(2, int((len(pt) * 0.6) + 0.999)) if len(pt) >= 3 else len(pt)
                if inter < need:
                    continue
                score = 50.0 + (inter / max(1, len(pt))) * 20.0
                reason = f"token_overlap:{inter}/{len(pt)}"

            if best is None or score > best.score:
                best = SkillMatch(skill_id=skill_id, skill_cfg=scfg, score=score, reason=reason)

    return best


def _format_for_voice(s: str, *, env_key: str, default_limit: int = 900) -> str:
    s2 = (s or "").strip()
    if not s2:
        return ""
    try:
        max_chars = int((os.getenv(env_key) or os.getenv("MAX_SPOKEN_CHARS") or str(default_limit)).strip() or default_limit)
    except Exception:
        max_chars = default_limit
    if len(s2) > max_chars:
        s2 = s2[: max_chars - 1].rstrip() + "…"
    return s2


def execute_dynamic_skill(
    db: Session,
    user: Any,
    *,
    match: SkillMatch,
    tenant_uuid: str,
    caller_id: str,
    call_sid: str | None,
    input_text: str,
) -> dict | None:
    """Execute a matched dynamic skill and return an assistant /assistant/route payload."""
    skill_id = match.skill_id
    cfg = match.skill_cfg or {}
    stype = cfg.get("type")

    # Defensive: refuse if disabled
    if not bool(cfg.get("enabled", True)):
        return {
            "spoken_reply": "That saved skill is currently disabled.",
            "fsm": {"mode": "dynamic_skill", "skill_id": skill_id, "disabled": True},
            "gmail": None,
        }

    if skill_id.startswith("websearch_") and stype == "web_search":
        # Prefer DB truth for enabled/name if present
        try:
            from models import WebSearchSkill
            sid = str(cfg.get("web_search_skill_id") or "").strip() or skill_id.replace("websearch_", "", 1)
            sid_uuid = None
            try:
                from uuid import UUID
                sid_uuid = UUID(str(sid))
            except Exception:
                sid_uuid = None
            q = db.query(WebSearchSkill).filter(WebSearchSkill.tenant_id == getattr(user, "id", None))
            if sid_uuid:
                q = q.filter(WebSearchSkill.id == sid_uuid)
            else:
                q = q.filter(WebSearchSkill.id == sid)
            row = q.first()
            if row and not bool(row.enabled):
                return {
                    "spoken_reply": "That saved web search skill is currently disabled.",
                    "fsm": {"mode": "dynamic_skill", "skill_id": skill_id, "type": "web_search", "disabled": True},
                    "gmail": None,
                }
            name = (getattr(row, "name", None) or cfg.get("label") or "Saved Web Search").strip()
            query = (getattr(row, "query", None) or cfg.get("query") or "").strip()
        except Exception:
            name = (cfg.get("label") or "Saved Web Search").strip()
            query = (cfg.get("query") or "").strip()

        if not query:
            return {
                "spoken_reply": "I couldn't run that web search skill because it has no query.",
                "fsm": {"mode": "dynamic_skill", "skill_id": skill_id, "type": "web_search", "error": "missing_query"},
                "gmail": None,
            }

        res = run_web_search(query)
        answer = _format_for_voice(res.answer, env_key="WEB_SEARCH_MAX_SPOKEN_CHARS", default_limit=900)
        spoken = f"{name}. {answer}".strip()

        # Record into long-term memory (best-effort)
        try:
            record_skill_result(
                db,
                tenant_uuid=str(tenant_uuid),
                caller_id=str(caller_id),
                call_sid=str(call_sid) if call_sid else None,
                skill_key=str(skill_id),
                input_text=str(input_text or ""),
                memory_text=str(spoken),
                data_json={
                    "type": "web_search",
                    "query": query,
                    "latency_ms": res.latency_ms,
                    "model": res.model,
                },
            )
        except Exception:
            pass

        # Analytics: record skill execution (best-effort, fail-open).
        try:
            emit_analytics_event(
                tenant_id=str(tenant_uuid),
                event_type="skill_executed",
                caller_id=str(caller_id) if caller_id else None,
                call_sid=(str(call_sid) if call_sid else None),
                skill_key=str(skill_id),
                payload={"source": "voice", "type": "web_search"},
                tags=["origin:voice", "kind:skill_executed", f"skill:{str(skill_id)}"],
            )
        except Exception:
            pass

        logger.info("DYNAMIC_SKILL_RUN type=web_search skill_id=%s score=%.1f reason=%s", skill_id, match.score, match.reason)
        return {"spoken_reply": spoken, "fsm": {"mode": "dynamic_skill", "type": "web_search", "skill_id": skill_id}, "gmail": None}

    if skill_id.startswith("dbquery_") and stype == "db_query":
        entity = (cfg.get("entity") or "caller_memory_events").strip()
        spec = cfg.get("spec") if isinstance(cfg.get("spec"), dict) else {}
        # Ensure entity aligns to cfg
        if isinstance(spec, dict):
            spec = dict(spec)
            spec.setdefault("entity", entity)

        qres = run_db_query(db, tenant_uuid=str(tenant_uuid), spec=spec)
        spoken = _format_for_voice(qres.spoken_summary or "", env_key="DB_QUERY_MAX_SPOKEN_CHARS", default_limit=900)
        if not spoken:
            spoken = "I ran that query, but there was nothing to report."

        # Record into long-term memory (best-effort)
        try:
            record_skill_result(
                db,
                tenant_uuid=str(tenant_uuid),
                caller_id=str(caller_id),
                call_sid=str(call_sid) if call_sid else None,
                skill_key=str(skill_id),
                input_text=str(input_text or ""),
                memory_text=str(spoken),
                data_json={
                    "type": "db_query",
                    "entity": entity,
                    "ok": bool(getattr(qres, "ok", True)),
                    "count": getattr(qres, "count", None),
                },
            )
        except Exception:
            pass

        # Analytics: record skill execution (best-effort, fail-open).
        try:
            emit_analytics_event(
                tenant_id=str(tenant_uuid),
                event_type="skill_executed",
                caller_id=str(caller_id) if caller_id else None,
                call_sid=(str(call_sid) if call_sid else None),
                skill_key=str(skill_id),
                payload={"source": "voice", "type": "db_query"},
                tags=["origin:voice", "kind:skill_executed", f"skill:{str(skill_id)}"],
            )
        except Exception:
            pass

        logger.info("DYNAMIC_SKILL_RUN type=db_query skill_id=%s score=%.1f reason=%s", skill_id, match.score, match.reason)
        return {"spoken_reply": spoken, "fsm": {"mode": "dynamic_skill", "type": "db_query", "skill_id": skill_id}, "gmail": None}

    return None

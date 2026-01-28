# services/intent_router_v2.py
"""Intent Router V2 (LLM-assisted, schema-validated).

Purpose
-------
Provide a flexible, natural-language first-stage router so callers do NOT need
standardized phrases. The LLM interprets user intent and returns a STRICT JSON plan.
Python then executes deterministically (no hallucinated actions).

Key design constraints (per Vozlia reliability philosophy)
----------------------------------------------------------
- LLM plans; Python executes deterministically.
- Any LLM output must be schema-validated before we act on it.
- Keep prompts small to protect voice latency: use candidate generation first.
- Feature-flagged so we can cut over safely and roll back instantly.

Cutover / rollback
------------------
Env: INTENT_V2_MODE = off|shadow|assist
- off    : router disabled (legacy routing path only)
- shadow : compute plan + log it, but DO NOT change behavior
- assist : execute valid plans; fall back to legacy on any failure

This module is intentionally additive: it can run alongside the legacy FSM +
dynamic skill matcher until we're confident enough to cut over.

(See CODE_DRIFT_CONTROL.md for how to maintain file intent notes.)
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, ValidationError

from core.logging import logger
from services.settings_service import get_skills_config

try:
    # openai>=1.x (used elsewhere in this repo)
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore


# ----------------------------
# Feature flags / config
# ----------------------------

def intent_v2_mode() -> str:
    return (os.getenv("INTENT_V2_MODE", "off") or "off").strip().lower()

def intent_v2_enabled() -> bool:
    return intent_v2_mode() in ("shadow", "assist")

def intent_v2_debug() -> bool:
    return (os.getenv("INTENT_V2_DEBUG", "0") or "0").strip().lower() in ("1","true","yes","on")

def _max_candidates() -> int:
    try:
        return max(3, min(20, int((os.getenv("INTENT_V2_MAX_CANDIDATES", "10") or "10").strip())))
    except Exception:
        return 10


# ----------------------------
# Candidate skill snapshot
# ----------------------------

_STOPWORDS = {
    "a","an","the","my","me","please","give","show","run","do","get","tell","today","todays",
    "this","that","for","to","of","in","on","at","and","or","is","are","was","were","it","its","about",
}

def _norm(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _tokens(s: str) -> List[str]:
    t = _norm(s)
    toks = [w for w in t.split(" ") if w and w not in _STOPWORDS]
    return toks

# ----------------------------
# Dynamic-skill activation gate
# ----------------------------
#
# Problem:
# - In natural conversation, people may mention a category word (e.g., "sports") in passing.
# - If we always allow dynamic skill disambiguation on category words, we can "punt" into saved
#   skills when the user did not intend to run a saved Skill.
#
# Solution:
# - Optionally require an activation keyword/phrase (e.g., "skill", "report") to engage
#   *category-style* dynamic routing.
# - Explicit mentions (skill name / trigger substring) still work even without activation words.
#
# Env:
#   INTENT_V2_DYNAMIC_ACTIVATION_KEYWORDS="skill,report"
#   - Comma-separated list (recommended).
#   - If you provide a single space-separated string with no commas ("skill report"),
#     we treat it as a list of single-word keywords.
#   - Empty/unset disables the gate (legacy behavior).
#
# NOTE: This gate applies ONLY to dynamic skills (websearch_* / dbquery_*). It does not affect:
# - legacy manifest skills (gmail_summary, investment_reporting, ...)
# - memory / KB / chitchat handling
#
def dynamic_skill_activation_keywords() -> List[str]:
    raw = (os.getenv("INTENT_V2_DYNAMIC_ACTIVATION_KEYWORDS") or "").strip()
    if not raw:
        return []

    parts: List[str] = []
    chunks = [c.strip() for c in re.split(r"[,\n]+", raw) if c and c.strip()]

    # If the user provided one chunk with spaces and no commas, treat it as "keywords" list.
    if len(chunks) == 1 and ("," not in raw) and (" " in chunks[0]):
        parts = [p.strip() for p in re.split(r"\s+", chunks[0]) if p and p.strip()]
    else:
        parts = chunks

    # Deduplicate by normalized form while preserving order.
    seen: set[str] = set()
    out: List[str] = []
    for p in parts:
        pn = _norm(p)
        if not pn:
            continue
        if pn not in seen:
            out.append(p)
            seen.add(pn)
    return out


def utterance_has_activation_keyword(utterance: str) -> bool:
    kws = dynamic_skill_activation_keywords()
    if not kws:
        return True  # gate disabled

    u = _norm(utterance)
    if not u:
        return True  # don't block non-utterance flows (e.g., disambiguation cache)

    padded = f" {u} "
    for kw in kws:
        kn = _norm(kw)
        if not kn:
            continue
        # "word-ish" / phrase match on normalized strings
        if f" {kn} " in padded:
            return True
    return False


def allow_dynamic_skill_candidate(utterance: str, *, score: float, reason: str) -> bool:
    """Return True if a dynamic skill candidate should be considered for this utterance."""
    kws = dynamic_skill_activation_keywords()
    if not kws:
        return True  # gate disabled
    if not (utterance or "").strip():
        return True  # e.g., disambiguation selection build
    if utterance_has_activation_keyword(utterance):
        return True

    # Without activation keywords, allow only explicit mentions.
    # - trigger_substring/label_substring are explicit
    # - score>=90 typically implies an explicit substring match (see _candidate_score)
    if score >= 90.0:
        return True
    r = (reason or "")
    if r.startswith("label_substring") or r.startswith("trigger_substring"):
        return True
    return False



@dataclass(frozen=True)
class SkillCandidate:
    skill_key: str
    label: str
    stype: str  # web_search | db_query | internal
    triggers: List[str]
    enabled: bool = True
    score: float = 0.0
    reason: str = ""


def _candidate_score(utterance: str, label: str, triggers: List[str]) -> Tuple[float, str]:
    """Return (score, reason). Higher is better. Deterministic, no network."""
    u_norm = _norm(utterance)
    u_tokens = set(_tokens(utterance))
    best = (0.0, "")

    # Label exact-ish substring
    ln = _norm(label)
    if ln and ln in u_norm:
        return (100.0 + min(25.0, float(len(ln)) / 4.0), f"label_substring:{ln}")

    # Trigger substring beats token overlap
    for tr in triggers[:10]:
        tn = _norm(tr)
        if tn and tn in u_norm:
            score = 90.0 + min(20.0, float(len(tn)) / 4.0)
            if score > best[0]:
                best = (score, f"trigger_substring:{tn}")

    # Token overlap as fallback
    for tr in triggers[:10]:
        tt = _tokens(tr)
        if not tt:
            continue
        inter = len(set(tt).intersection(u_tokens))
        need = max(1, int((len(tt) * 0.6) + 0.999))
        if inter < need:
            continue
        score = 50.0 + (inter / max(1, len(tt))) * 25.0
        if score > best[0]:
            best = (score, f"trigger_token_overlap:{inter}/{len(tt)}")

    return best


def build_skill_candidates(db, user, utterance: str) -> List[SkillCandidate]:
    """Collect + score candidates from skills_config (dynamic skills) and YAML registry (legacy)."""
    cfg_all = get_skills_config(db, user) or {}
    if not isinstance(cfg_all, dict):
        cfg_all = {}

    out: List[SkillCandidate] = []

    # Dynamic skills from skills_config: websearch_* / dbquery_*
    for skill_key, scfg in cfg_all.items():
        if not isinstance(skill_key, str) or not isinstance(scfg, dict):
            continue

        stype = str(scfg.get("type") or "").strip()
        if skill_key.startswith("websearch_") and stype != "web_search":
            continue
        if skill_key.startswith("dbquery_") and stype != "db_query":
            continue
        if not (skill_key.startswith("websearch_") or skill_key.startswith("dbquery_")):
            continue

        enabled = bool(scfg.get("enabled", True))
        label = (scfg.get("label") or "").strip() or skill_key
        triggers = []
        ep = scfg.get("engagement_phrases")
        if isinstance(ep, list):
            triggers.extend([t for t in ep if isinstance(t, str) and t.strip()])

        score, reason = _candidate_score(utterance, label, triggers + [label])
        if not allow_dynamic_skill_candidate(utterance, score=score, reason=reason):
            continue


        out.append(
            SkillCandidate(
                skill_key=skill_key,
                label=label,
                stype=stype,
                triggers=triggers[:5],
                enabled=enabled,
                score=score,
                reason=reason,
            )
        )

    # Legacy manifest skills (gmail_summary, investment_reporting, etc.)
    try:
        from skills.registry import skill_registry
        for sk in skill_registry.all():
            sid = str(getattr(sk, "id", "") or "").strip()
            if not sid:
                continue
            label = str(getattr(sk, "name", "") or sid).strip()
            triggers = []
            try:
                phrases = getattr(getattr(sk, "trigger", None), "phrases", None)
                if isinstance(phrases, list):
                    triggers = [p for p in phrases if isinstance(p, str) and p.strip()]
            except Exception:
                triggers = []

            score, reason = _candidate_score(utterance, label, triggers + [label])
            out.append(
                SkillCandidate(
                    skill_key=sid,
                    label=label,
                    stype="internal",
                    triggers=triggers[:5],
                    enabled=True,
                    score=score,
                    reason=reason,
                )
            )
    except Exception:
        pass

    # Rank and cap
    out_sorted = sorted(out, key=lambda c: c.score, reverse=True)
    return out_sorted[: _max_candidates()]


# ----------------------------
# LLM plan schema
# ----------------------------

class IntentPlanV2(BaseModel):
    """Strict plan returned by the LLM."""

    route: str = Field(..., description="run_skill|disambiguate|chitchat")
    skill_key: Optional[str] = Field(None, description="Chosen skill_key if route=run_skill")
    choices: Optional[List[str]] = Field(None, description="List of skill_keys if route=disambiguate")
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    rationale: Optional[str] = Field(None, description="Short reason; not shown to end user")


_CLIENT: OpenAI | None = None

def _get_client() -> OpenAI | None:
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT
    if OpenAI is None:
        return None
    key = (os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY") or "").strip()
    if not key:
        return None
    try:
        _CLIENT = OpenAI(api_key=key)
        return _CLIENT
    except Exception:
        return None


def _extract_json_object(text: str) -> str | None:
    """Best-effort extraction of a single JSON object from a model reply."""
    if not text:
        return None
    s = text.strip()
    if s.startswith("{") and s.endswith("}"):
        return s
    # If the model wrapped JSON with text, find outermost braces.
    i = s.find("{")
    j = s.rfind("}")
    if i >= 0 and j > i:
        return s[i : j + 1].strip()
    return None


def llm_plan_intent(utterance: str, *, candidates: List[SkillCandidate]) -> IntentPlanV2 | None:
    """Use the LLM to pick an intent plan from a small candidate list."""
    client = _get_client()
    if client is None:
        return None

    model = (os.getenv("OPENAI_INTENT_MODEL") or "").strip() or "gpt-4o-mini"
    timeout_s = float((os.getenv("OPENAI_INTENT_TIMEOUT_S", "4.0") or "4.0").strip())
    max_tokens = int((os.getenv("OPENAI_INTENT_MAX_TOKENS", "220") or "220").strip())

    # Keep prompt short for voice latency.
    system = (
        "You are Vozlia's intent router. "
        "The user speaks in natural language and does NOT use standardized phrases. "
        "Choose the BEST route using ONLY the provided candidates list. "
        "Return STRICT JSON ONLY (no markdown, no extra text). "
        "Rules: "
        "1) If the user is asking to run a specific saved skill, route='run_skill' and set skill_key to an EXACT candidate skill_key. "
        "2) If the user is asking about a topic but multiple candidates match, route='disambiguate' and provide choices (skill_key list) in best-first order. "
        "3) If nothing matches, route='chitchat'. "
        "Never invent skill keys."
    )

    # Compress candidates to keep token footprint small.
    cand_payload = [
        {
            "skill_key": c.skill_key,
            "label": c.label,
            "type": c.stype,
            "triggers": c.triggers[:3],
        }
        for c in candidates
    ]

    user_obj = {
        "utterance": (utterance or "")[:800],
        "candidates": cand_payload,
        "output_schema": {
            "route": "run_skill|disambiguate|chitchat",
            "skill_key": "string (only if route=run_skill)",
            "choices": "array of skill_key strings (only if route=disambiguate)",
            "confidence": "0.0-1.0",
            "rationale": "short string (optional)",
        },
    }

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user_obj)},
            ],
            max_tokens=max_tokens,
            temperature=0.0,
            timeout=timeout_s,
        )
        raw = (resp.choices[0].message.content or "").strip()
        js = _extract_json_object(raw)
        if not js:
            return None
        data = json.loads(js)
        plan = IntentPlanV2.model_validate(data)
        plan.route = (plan.route or "").strip().lower()
        if plan.route not in ("run_skill", "disambiguate", "chitchat"):
            return None
        return plan
    except Exception:
        return None


# ----------------------------
# Execution glue
# ----------------------------

def _format_disambiguation_prompt(options: List[SkillCandidate]) -> str:
    lines = ["Which one did you mean?"]
    for i, c in enumerate(options[:6], start=1):
        lines.append(f"{i}. {c.label}")
    lines.append("Reply with the number or the skill name.")
    return "\n".join(lines)


def maybe_route_and_execute(
    *,
    utterance: str,
    db,
    user,
    tenant_uuid: str | None,
    caller_id: str | None,
    call_id: str | None,
    account_id: str | None = None,
    context: dict | None = None,
) -> dict | None:
    """If INTENT_V2_MODE is enabled, attempt skill routing and execution.

    Returns an /assistant/route payload dict if handled, else None (caller should fall back to legacy).
    """
    if not intent_v2_enabled():
        return None

    mode = intent_v2_mode()
    if not (utterance or "").strip():
        return None

    # Candidate generation is deterministic + cheap.
    candidates = build_skill_candidates(db, user, utterance)
    if not candidates:
        return None

    if intent_v2_debug():
        kws = dynamic_skill_activation_keywords()
        if kws:
            logger.info(
                "INTENT_V2_DYNAMIC_ACTIVATION has_kw=%s keywords=%s",
                utterance_has_activation_keyword(utterance),
                [_norm(k) for k in kws][:8],
            )

    if intent_v2_debug():
        logger.info(
            "INTENT_V2_CANDIDATES mode=%s n=%s top=%s",
            mode,
            len(candidates),
            [(c.skill_key, round(c.score, 1), c.reason) for c in candidates[:5]],
        )

    # If best deterministic match is extremely strong, we can avoid LLM for speed.
    best = candidates[0]
    fast_path = best.score >= 120.0 and best.enabled
    if fast_path:
        logger.info("INTENT_V2_FASTPATH skill_key=%s score=%.1f reason=%s", best.skill_key, best.score, best.reason)
        return _execute_candidate(
            best,
            utterance=utterance,
            db=db,
            user=user,
            tenant_uuid=tenant_uuid,
            caller_id=caller_id,
            call_id=call_id,
            account_id=account_id,
            context=context,
            reason="fastpath",
        )

    plan = llm_plan_intent(utterance, candidates=candidates)
    if plan is None:
        if intent_v2_debug():
            logger.info("INTENT_V2_PLAN_NONE mode=%s", mode)
        return None if mode != "shadow" else None

    # Always log the plan (auditable, helps diagnose drift)
    logger.info(
        "INTENT_V2_PLAN mode=%s route=%s skill_key=%s conf=%.2f",
        mode,
        plan.route,
        plan.skill_key,
        float(plan.confidence or 0.0),
    )

    if mode == "shadow":
        # Shadow: do NOT change behavior.
        return None

    # Assist mode: execute valid plans
    if plan.route == "run_skill" and plan.skill_key:
        sk = plan.skill_key.strip()
        chosen = next((c for c in candidates if c.skill_key == sk), None)
        if chosen and chosen.enabled:
            return _execute_candidate(
                chosen,
                utterance=utterance,
                db=db,
                user=user,
                tenant_uuid=tenant_uuid,
                caller_id=caller_id,
                call_id=call_id,
                account_id=account_id,
                context=context,
                reason="llm",
            )
        return None

    if plan.route == "disambiguate":
        # Offer top-N as choices (either from plan.choices or from scored list)
        keys = [k for k in (plan.choices or []) if isinstance(k, str)]
        opts = [c for c in candidates if c.skill_key in keys] if keys else candidates[:6]
        if not opts:
            opts = candidates[:6]

        # Store into session for next turn selection (best-effort)
        if call_id:
            try:
                from services.session_store import session_store
                session_store.set(call_id, "intent_v2_choices", [c.skill_key for c in opts[:6]])
            except Exception:
                pass

        return {"spoken_reply": _format_disambiguation_prompt(opts), "fsm": {"mode": "intent_v2", "intent": "disambiguate"}, "gmail": None}

    # chitchat -> let legacy router handle
    return None


def maybe_consume_disambiguation_choice(
    *,
    utterance: str,
    db,
    user,
    tenant_uuid: str | None,
    caller_id: str | None,
    call_id: str | None,
    account_id: str | None = None,
    context: dict | None = None,
) -> dict | None:
    """If the previous turn asked for disambiguation, try to consume the user's selection."""
    if not (call_id and utterance and utterance.strip()):
        return None
    try:
        from services.session_store import session_store
        keys = session_store.pop(call_id, "intent_v2_choices", None)
    except Exception:
        keys = None
    if not keys or not isinstance(keys, list):
        return None

    # Numeric selection: "1", "option 2", etc.
    m = re.search(r"\b(\d{1,2})\b", utterance.strip())
    if m:
        try:
            idx = int(m.group(1)) - 1
        except Exception:
            idx = -1
        if 0 <= idx < len(keys):
            chosen_key = str(keys[idx])
            # Build a fresh candidate snapshot so we have labels/config
            candidates = build_skill_candidates(db, user, utterance="")
            chosen = next((c for c in candidates if c.skill_key == chosen_key), None)
            if chosen:
                return _execute_candidate(
                    chosen,
                    utterance=utterance,
                    db=db,
                    user=user,
                    tenant_uuid=tenant_uuid,
                    caller_id=caller_id,
                    call_id=call_id,
                    account_id=account_id,
                    context=context,
                    reason="disambiguation_number",
                )

    # Name-based fallback: run LLM with only those choices
    candidates_all = build_skill_candidates(db, user, utterance=utterance)
    candidates = [c for c in candidates_all if c.skill_key in set([str(k) for k in keys])]
    if not candidates:
        return {"spoken_reply": "Sorry — please reply with a number from the list.", "fsm": {"mode": "intent_v2", "intent": "disambiguate_retry"}, "gmail": None}

    plan = llm_plan_intent(utterance, candidates=candidates)
    if plan and plan.route == "run_skill" and plan.skill_key:
        chosen = next((c for c in candidates if c.skill_key == plan.skill_key), None)
        if chosen:
            return _execute_candidate(
                chosen,
                utterance=utterance,
                db=db,
                user=user,
                tenant_uuid=tenant_uuid,
                caller_id=caller_id,
                call_id=call_id,
                account_id=account_id,
                context=context,
                reason="disambiguation_llm",
            )

    return {"spoken_reply": "Sorry — please reply with a number from the list.", "fsm": {"mode": "intent_v2", "intent": "disambiguate_retry"}, "gmail": None}


def _execute_candidate(
    cand: SkillCandidate,
    *,
    utterance: str,
    db,
    user,
    tenant_uuid: str | None,
    caller_id: str | None,
    call_id: str | None,
    account_id: str | None = None,
    context: dict | None = None,
    reason: str,
) -> dict | None:
    """Execute a candidate deterministically using existing engines."""
    if cand.skill_key.startswith("websearch_") or cand.skill_key.startswith("dbquery_"):
        # Dynamic skills are executed via the dynamic runtime (records long-term memory, etc.)
        try:
            from services.dynamic_skill_runtime import SkillMatch, execute_dynamic_skill
            cfg_all = get_skills_config(db, user) or {}
            scfg = cfg_all.get(cand.skill_key) if isinstance(cfg_all, dict) else None
            if not isinstance(scfg, dict):
                return None

            if not bool(scfg.get("enabled", True)):
                return {"spoken_reply": "That saved skill is currently disabled.", "fsm": {"mode": "intent_v2", "skill_id": cand.skill_key, "disabled": True}, "gmail": None}

            # These identifiers are only required for memory capture; best-effort.
            t_uuid = str(tenant_uuid or getattr(user, "id", "") or "")
            c_id = str(caller_id or "")
            call_sid = str(call_id or "")
            match = SkillMatch(skill_id=cand.skill_key, skill_cfg=scfg, score=999.0, reason=f"intent_v2:{reason}")
            payload = execute_dynamic_skill(
                db,
                user,
                match=match,
                tenant_uuid=t_uuid,
                caller_id=c_id,
                call_sid=call_sid if call_sid else None,
                input_text=utterance,
            )
            if isinstance(payload, dict):
                payload.setdefault("fsm", {})
                if isinstance(payload.get("fsm"), dict):
                    payload["fsm"]["intent_v2"] = True
                    payload["fsm"]["intent_v2_reason"] = reason
                return payload
        except Exception:
            return None

    # Legacy manifest skills (gmail_summary, investment_reporting)
    try:
        from skills.engine import execute_skill
        out = execute_skill(
            cand.skill_key,
            text=utterance,
            db=db,
            current_user=user,
            account_id=account_id,
            context=context,
        )
        spoken = (out or {}).get("spoken_reply") or (out or {}).get("spoken") or ""
        if not spoken:
            spoken = f"Okay — running {cand.label}."
        return {"spoken_reply": spoken, "fsm": {"mode": "intent_v2", "skill_id": cand.skill_key, "type": "internal", "intent_v2_reason": reason}, "gmail": out.get("gmail") if isinstance(out, dict) else None}
    except Exception:
        return None

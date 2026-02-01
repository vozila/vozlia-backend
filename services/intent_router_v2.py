# services/intent_router_v2.py
"""VOZLIA FILE PURPOSE
Purpose: LLM-first intent routing + (optional) scheduling. Produces strict JSON plans; Python executes deterministically.
Hot path: no (called from /assistant/route, not the realtime WS loop).
Public interfaces: maybe_route_and_execute, maybe_consume_disambiguation_choice.
Reads/Writes: user_settings (skills_config), scheduled_deliveries (via stores), session_store.
Feature flags: INTENT_V2_MODE, INTENT_V2_SCHEDULE_ENABLED, DBQUERY_SCHEDULE_ENABLED.
Failure mode: falls back to legacy routing; scheduling failures return safe errors.
Last touched: 2026-02-01 (normalize schedule destinations; disambiguate missing destination)
"""

# (legacy module docs removed to keep __future__ import valid)


from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Literal

from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from core.logging import logger
from models import DeliveryChannel
from services.session_store import session_store
from services.settings_service import get_skills_config
from services.web_search_skill_store import upsert_daily_schedule
from services.db_query_skill_store import upsert_daily_schedule_dbquery
from services.delivery_destination import resolve_delivery_destination

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore


# ----------------------------
# Env helpers
# ----------------------------

def intent_v2_mode() -> str:
    v = (os.getenv("INTENT_V2_MODE") or "off").strip().lower()
    if v not in ("off", "shadow", "assist", "full"):
        return "off"
    return v


def intent_v2_enabled() -> bool:
    return intent_v2_mode() in ("shadow", "assist", "full")


def intent_v2_debug() -> bool:
    v = (os.getenv("INTENT_V2_DEBUG") or "0").strip().lower()
    return v in ("1", "true", "yes", "on")


def intent_v2_schedule_enabled() -> bool:
    v = (os.getenv("INTENT_V2_SCHEDULE_ENABLED") or "0").strip().lower()
    return v in ("1", "true", "yes", "on")


def dbquery_schedule_enabled() -> bool:
    """Enable scheduling for dbquery_* dynamic skills (opt-in, default OFF)."""
    v = (os.getenv("DBQUERY_SCHEDULE_ENABLED") or "0").strip().lower()
    return v in ("1", "true", "yes", "on")


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


@dataclass(frozen=True)
class SkillCandidate:
    skill_key: str
    label: str
    stype: str  # web_search | db_query | internal
    triggers: List[str]
    category: str = ""  # user-facing grouping label (e.g., sports, parking, finance)
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


# ----------------------------
# Dynamic-skill activation gate
# ----------------------------
#
# Applies ONLY to dynamic skills (websearch_*/dbquery_*).
#
# Env:
#   INTENT_V2_DYNAMIC_ACTIVATION_KEYWORDS="skill,report"
#   - Comma-separated list (recommended).
#   - If single space-separated string with no commas, treat as list of single words.
#   - Empty/unset disables the gate (legacy behavior).
#

def dynamic_skill_activation_keywords() -> List[str]:
    raw = (os.getenv("INTENT_V2_DYNAMIC_ACTIVATION_KEYWORDS") or "").strip()
    if not raw:
        return []
    parts: List[str] = []
    chunks = [c.strip() for c in re.split(r"[,\n]+", raw) if c and c.strip()]
    if len(chunks) == 1 and ("," not in raw) and (" " in chunks[0]):
        parts = [p.strip() for p in re.split(r"\s+", chunks[0]) if p and p.strip()]
    else:
        parts = chunks

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
        return True  # don't block empty-ish

    padded = f" {u} "
    for kw in kws:
        kn = _norm(kw)
        if not kn:
            continue
        if f" {kn} " in padded:
            return True
    return False


def allow_dynamic_skill_candidate(utterance: str, *, score: float, reason: str) -> bool:
    """Return True if a dynamic skill candidate should be considered for this utterance."""
    kws = dynamic_skill_activation_keywords()
    if not kws:
        return True  # gate disabled
    if not (utterance or "").strip():
        return True
    if utterance_has_activation_keyword(utterance):
        return True

    # Without activation keywords, allow only explicit mentions.
    if score >= 90.0:
        return True
    r = (reason or "")
    if r.startswith("label_substring") or r.startswith("trigger_substring"):
        return True
    return False


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
        stype = str((scfg.get("type") or "")).strip()
        if skill_key.startswith("websearch_") and stype != "web_search":
            continue
        if skill_key.startswith("dbquery_") and stype != "db_query":
            continue
        if not (skill_key.startswith("websearch_") or skill_key.startswith("dbquery_")):
            continue

        enabled = bool(scfg.get("enabled", True))
        label = (scfg.get("label") or "").strip() or skill_key
        triggers: List[str] = []
        ep = scfg.get("engagement_phrases")
        if isinstance(ep, list):
            triggers.extend([t for t in ep if isinstance(t, str) and t.strip()])

        score, reason = _candidate_score(utterance, label, triggers + [label])

        out.append(
            SkillCandidate(
                skill_key=skill_key,
                label=label,
                stype=stype,
                triggers=triggers[:5],
                category=str((scfg.get("category") or "")).strip().lower(),
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
            triggers: List[str] = []
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

    out_sorted = sorted(out, key=lambda c: c.score, reverse=True)
    return out_sorted[: _max_candidates()]


# ----------------------------
# LLM plan schema
# ----------------------------

class ScheduleRequest(BaseModel):
    """Structured schedule request extracted by the LLM (MVP: daily schedules)."""

    cadence: Literal["daily"] = "daily"
    hour: int = Field(..., ge=0, le=23)
    minute: int = Field(..., ge=0, le=59)
    timezone: str = Field(..., min_length=1)
    channel: str = Field(..., min_length=1)      # email|sms|whatsapp|call
    destination: str = Field(..., min_length=1)  # email address or phone number


class IntentPlanV2(BaseModel):
    """Strict plan returned by the LLM."""

    route: str = Field(..., description="run_skill|schedule_skill|disambiguate|chitchat")
    skill_key: Optional[str] = Field(None, description="Chosen skill_key if route=run_skill|schedule_skill")
    choices: Optional[List[str]] = Field(None, description="List of skill_keys if route=disambiguate")
    schedule_request: Optional[ScheduleRequest] = Field(
        None,
        description="If schedule-related: cadence/time/channel/destination",
    )
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
    if not text:
        return None
    s = text.strip()
    if s.startswith("{") and s.endswith("}"):
        return s
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

    system = (
        "You are Vozlia's intent router. "
        "The user speaks in natural language and does NOT use standardized phrases. "
        "Choose the BEST route using ONLY the provided candidates list. "
        "Return STRICT JSON ONLY (no markdown, no extra text).\n\n"
        "Routes:\n"
        "- 'run_skill': user wants to run a saved skill now.\n"
        "- 'schedule_skill': user wants to schedule a saved skill (create or modify a daily schedule).\n"
        "- 'disambiguate': multiple skills match and you need the user to choose.\n"
        "- 'chitchat': no skill is appropriate.\n\n"
        "Rules:\n"
        "1) Never invent skill keys; skill_key must be one of the provided candidate.skill_key values.\n"
        "2) For route='run_skill', set skill_key to an exact candidate skill_key.\n"
        "3) For schedule-related requests, populate schedule_request with:\n"
        "   cadence='daily', hour (0-23), minute (0-59), timezone (e.g., 'America/New_York'),\n"
        "   channel ('email' or 'sms' preferred), destination (email or phone number).\n"
        "4) If the user talks about sports, parking, etc., you may use candidate.category to do category-based routing\n"
        "   (e.g., 'sports report' → a sports-category skill).\n"
        "5) If schedule details are ambiguous or missing, you may use route='disambiguate' or 'chitchat' rather than guessing.\n"
    )

    cand_payload = [
        {
            "skill_key": c.skill_key,
            "label": c.label,
            "type": c.stype,
            "triggers": c.triggers[:3],
            "category": c.category or "",
        }
        for c in candidates
    ]

    user_obj = {
        "utterance": (utterance or "")[:800],
        "candidates": cand_payload,
        "output_schema": {
            "route": "run_skill|schedule_skill|disambiguate|chitchat",
            "skill_key": "string (only if route=run_skill or schedule_skill)",
            "choices": "array of skill_key strings (only if route=disambiguate)",
            "schedule_request": {
                "cadence": "daily",
                "hour": "0-23 integer",
                "minute": "0-59 integer",
                "timezone": "IANA timezone string (e.g., America/New_York)",
                "channel": "email|sms|whatsapp|call",
                "destination": "email address or phone number",
            },
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
        if plan.route not in ("run_skill", "schedule_skill", "disambiguate", "chitchat"):
            return None
        return plan
    except Exception:
        return None


# ----------------------------
# Disambiguation helpers
# ----------------------------

def _format_disambiguation_prompt(options: List[SkillCandidate]) -> str:
    lines = ["Which one did you mean?"]
    for i, c in enumerate(options[:6], start=1):
        lines.append(f"{i}. {c.label}")
    lines.append("Reply with the number or the skill name.")
    return "\n".join(lines)


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
    """Handle follow-up replies like '1' or 'Sports Digest' after a disambiguation prompt."""
    if not call_id:
        return None

    bucket = session_store.get(call_id)
    choices_raw = bucket.get("intent_v2_choices") if isinstance(bucket, dict) else None
    pending_action = bucket.get("intent_v2_pending_action") if isinstance(bucket, dict) else None
    if not choices_raw or pending_action not in ("run_skill", "schedule_skill"):
        return None

    try:
        candidates = [SkillCandidate(**c) for c in choices_raw]  # type: ignore[arg-type]
    except Exception:
        return None

    u = (utterance or "").strip()
    if not u:
        return None

    # Numeric selection (1-based)
    if u.isdigit():
        idx = int(u)
        if 1 <= idx <= len(candidates):
            chosen = candidates[idx - 1]
            session_store.set(call_id, "intent_v2_choices", [])
            session_store.set(call_id, "intent_v2_pending_action", None)
            if pending_action == "schedule_skill":
                return {
                    "spoken_reply": "Okay — what time should I run it daily, which timezone, and where should I deliver it (email or SMS)?",
                    "fsm": {"mode": "intent_v2", "intent": "schedule_needs_details"},
                    "gmail": None,
                }
            return _execute_candidate(
                cand=chosen,
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

    # Name-based: match by label substring.
    u_can = _norm(u)
    label_matches = [c for c in candidates if u_can and (u_can in _norm(c.label) or _norm(c.label) in u_can)]
    if len(label_matches) == 1:
        chosen = label_matches[0]
        session_store.set(call_id, "intent_v2_choices", [])
        session_store.set(call_id, "intent_v2_pending_action", None)
        if pending_action == "schedule_skill":
            return {
                "spoken_reply": "Okay — what time should I run it daily, which timezone, and where should I deliver it (email or SMS)?",
                "fsm": {"mode": "intent_v2", "intent": "schedule_needs_details"},
                "gmail": None,
            }
        return _execute_candidate(
            cand=chosen,
            utterance=utterance,
            db=db,
            user=user,
            tenant_uuid=tenant_uuid,
            caller_id=caller_id,
            call_id=call_id,
            account_id=account_id,
            context=context,
            reason="disambiguation_label",
        )

    # If still ambiguous, let LLM replan over the restricted set.
    candidates_all = build_skill_candidates(db, user, utterance=utterance)
    keys = [str(c.skill_key) for c in candidates]
    restricted = [c for c in candidates_all if c.skill_key in set(keys)]
    if not restricted:
        return {
            "spoken_reply": "Sorry — please reply with a number from the list.",
            "fsm": {"mode": "intent_v2", "intent": "disambiguate_retry"},
            "gmail": None,
        }

    plan = llm_plan_intent(utterance, candidates=restricted)
    if plan and plan.route == "run_skill" and plan.skill_key:
        chosen = next((c for c in restricted if c.skill_key == plan.skill_key), None)
        if chosen:
            session_store.set(call_id, "intent_v2_choices", [])
            session_store.set(call_id, "intent_v2_pending_action", None)
            return _execute_candidate(
                cand=chosen,
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

    return {
        "spoken_reply": "Sorry — please reply with a number from the list.",
        "fsm": {"mode": "intent_v2", "intent": "disambiguate_retry"},
        "gmail": None,
    }


# ----------------------------
# Execution glue
# ----------------------------

def _to_delivery_channel(channel: str) -> DeliveryChannel | None:
    c = (channel or "").strip().lower()
    if c in ("email", "e-mail"):
        return DeliveryChannel.email
    if c in ("sms", "text"):
        return DeliveryChannel.sms
    if c in ("whatsapp", "wa"):
        return DeliveryChannel.whatsapp
    if c in ("call", "phone"):
        return DeliveryChannel.call
    return None


def _schedule_candidate(
    db: Session,
    user,
    cand: SkillCandidate,
    sched: ScheduleRequest,
    tenant_uuid: str | None,
    caller_id: str | None,
    call_id: str | None,
) -> dict | None:
    """Create/update a daily schedule for a dynamic skill (websearch_* or dbquery_*).

    Behavior:
      - websearch_* schedules are always supported when INTENT_V2_SCHEDULE_ENABLED is on.
      - dbquery_* schedules are additionally gated by DBQUERY_SCHEDULE_ENABLED (default OFF).
      - Upsert semantics: if a schedule exists for this (tenant, skill), it is updated (time/channel/destination).
    """
    skill_key = (cand.skill_key or "").strip()
    kind: str | None = None
    raw_skill_id = ""

    if skill_key.startswith("websearch_"):
        kind = "websearch"
        raw_skill_id = skill_key.split("websearch_", 1)[1] if "websearch_" in skill_key else ""
    elif skill_key.startswith("dbquery_"):
        if not dbquery_schedule_enabled():
            return {
                "spoken_reply": "Scheduling is not enabled for database metrics yet.",
                "fsm": {"mode": "intent_v2", "intent": "schedule_unsupported", "skill_id": skill_key},
                "gmail": None,
            }
        kind = "dbquery"
        raw_skill_id = skill_key.split("dbquery_", 1)[1] if "dbquery_" in skill_key else ""
    else:
        return {
            "spoken_reply": "Scheduling is only supported for saved WebSearch reports and DB metrics right now.",
            "fsm": {"mode": "intent_v2", "intent": "schedule_unsupported", "skill_id": skill_key},
            "gmail": None,
        }

    if not raw_skill_id:
        return None

    channel = _to_delivery_channel(sched.channel)
    if channel is None:
        return {
            "spoken_reply": "I can only deliver scheduled results by email or SMS right now.",
            "fsm": {"mode": "intent_v2", "intent": "schedule_invalid_channel"},
            "gmail": None,
        }

    if sched.cadence != "daily":
        return {
            "spoken_reply": "I can only schedule daily runs right now.",
            "fsm": {"mode": "intent_v2", "intent": "schedule_invalid_cadence"},
            "gmail": None,
        }


    # Destination safety: prevent placeholder destinations like destination='email'.
    resolved_dest, dest_reason = resolve_delivery_destination(channel=channel, destination=destination, user=user)
    if not resolved_dest:
        # Ask a single clarification question (minimal-question principle).
        if channel == DeliveryChannel.email:
            prompt = "What email address should I send this to?"
        else:
            prompt = "What phone number should I send this to?"
        return {
            "spoken_reply": prompt,
            "fsm": {"mode": "intent_v2", "intent": "schedule_missing_destination", "channel": sched.channel},
            "gmail": None,
        }
    destination = resolved_dest
    try:
        if kind == "websearch":
            upsert_daily_schedule(
                db,
                user,
                web_search_skill_id=raw_skill_id,
                hour=int(sched.hour),
                minute=int(sched.minute),
                timezone=sched.timezone,
                channel=channel,
                destination=destination,
            )
        else:
            upsert_daily_schedule_dbquery(
                db,
                user,
                db_query_skill_id=raw_skill_id,
                hour=int(sched.hour),
                minute=int(sched.minute),
                timezone=sched.timezone,
                channel=channel,
                destination=destination,
            )
    except Exception:
        logger.exception("INTENT_V2_SCHEDULE_UPSERT_FAIL skill_key=%s", skill_key)
        return {
            "spoken_reply": "Sorry — I couldn't save that schedule yet.",
            "fsm": {"mode": "intent_v2", "intent": "schedule_error"},
            "gmail": None,
        }

    if call_id:
        session_store.set(call_id, "intent_v2_last_skill_key", skill_key)

    time_of_day = f"{int(sched.hour):02d}:{int(sched.minute):02d}"
    spoken = (
        f"Okay — scheduled {cand.label} daily at {time_of_day} {sched.timezone} "
        f"and will deliver by {sched.channel} to {destination}."
    )
    return {
        "spoken_reply": spoken,
        "fsm": {
            "mode": "intent_v2",
            "intent": "schedule_created",
            "skill_id": skill_key,
            "schedule_time_of_day": time_of_day,
            "schedule_timezone": sched.timezone,
        },
        "gmail": None,
    }


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
    # Dynamic skills via dynamic runtime
    if cand.skill_key.startswith("websearch_") or cand.skill_key.startswith("dbquery_"):
        try:
            from services.dynamic_skill_runtime import SkillMatch, execute_dynamic_skill
            cfg_all = get_skills_config(db, user) or {}
            scfg = cfg_all.get(cand.skill_key) if isinstance(cfg_all, dict) else None
            if not isinstance(scfg, dict):
                return None

            if not bool(scfg.get("enabled", True)):
                return {
                    "spoken_reply": "That saved skill is currently disabled.",
                    "fsm": {"mode": "intent_v2", "skill_id": cand.skill_key, "disabled": True},
                    "gmail": None,
                }

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
                if call_id:
                    session_store.set(call_id, "intent_v2_last_skill_key", cand.skill_key)
                return payload
        except Exception:
            return None

    # Legacy manifest skills
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
        if call_id:
            session_store.set(call_id, "intent_v2_last_skill_key", cand.skill_key)
        return {
            "spoken_reply": spoken,
            "fsm": {"mode": "intent_v2", "skill_id": cand.skill_key, "type": "internal", "intent_v2_reason": reason},
            "gmail": out.get("gmail") if isinstance(out, dict) else None,
        }
    except Exception:
        return None


# ----------------------------
# Main entrypoint
# ----------------------------

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

    # If we are mid-disambiguation, let that consume the reply first.
    consumed = maybe_consume_disambiguation_choice(
        utterance=utterance,
        db=db,
        user=user,
        tenant_uuid=tenant_uuid,
        caller_id=caller_id,
        call_id=call_id,
        account_id=account_id,
        context=context,
    )
    if consumed is not None:
        return consumed

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
        logger.info(
            "INTENT_V2_CANDIDATES mode=%s n=%s top=%s",
            mode,
            len(candidates),
            [(c.skill_key, round(c.score, 1), c.reason) for c in candidates[:5]],
        )

    # IMPORTANT: when schedule support is enabled, we ALWAYS ask LLM for a plan.
    # We do not fast-path, so that the LLM can attach schedule_request in any
    # natural language time/cadence form.
    plan = llm_plan_intent(utterance, candidates=candidates)
    if plan is None:
        # Optional: simple category-based fallback for dynamic skills.
        if mode == "assist" and call_id and utterance_has_activation_keyword(utterance):
            u_tokens = set(_tokens(utterance))
            categories = {c.category for c in candidates if c.category}
            cat = None
            for c in categories:
                if c and c in u_tokens:
                    cat = c
                    break
            if cat:
                opts = [c for c in candidates if c.enabled and c.category == cat]
                if len(opts) == 1:
                    return _execute_candidate(
                        cand=opts[0],
                        utterance=utterance,
                        db=db,
                        user=user,
                        tenant_uuid=tenant_uuid,
                        caller_id=caller_id,
                        call_id=call_id,
                        account_id=account_id,
                        context=context,
                        reason="category_fallback",
                    )
                if len(opts) > 1:
                    session_store.set(call_id, "intent_v2_choices", [c.__dict__ for c in opts])
                    session_store.set(call_id, "intent_v2_pending_action", "run_skill")
                    return {
                        "spoken_reply": _format_disambiguation_prompt(opts),
                        "fsm": {"mode": "intent_v2", "intent": "disambiguate"},
                        "gmail": None,
                    }
        return None

    schedule_req = plan.schedule_request

    if intent_v2_debug():
        logger.info(
            "LLM_ROUTER_PLAN mode=%s route=%s skill_key=%s conf=%.2f has_schedule=%s",
            mode,
            plan.route,
            plan.skill_key,
            plan.confidence,
            bool(schedule_req),
        )

    # ---- schedule side-effect (LLM-only) ----
    schedule_payload: dict | None = None
    if schedule_req is not None and intent_v2_schedule_enabled():
        # Choose skill_key for scheduling: plan.skill_key or last skill in this call.
        skill_key = plan.skill_key
        if not skill_key and call_id:
            bucket = session_store.get(call_id)
            skill_key = (bucket or {}).get("intent_v2_last_skill_key")

        if skill_key:
            chosen = next((c for c in candidates if c.skill_key == skill_key and c.enabled), None)
            if chosen:
                schedule_payload = _schedule_candidate(
                    db=db,
                    user=user,
                    cand=chosen,
                    sched=schedule_req,
                    tenant_uuid=tenant_uuid,
                    caller_id=caller_id,
                    call_id=call_id,
                )

    # If LLM decided route="schedule_skill", prefer schedule confirmation as the main reply.
    if plan.route == "schedule_skill":
        return schedule_payload

    # 1) Chitchat → fall back to legacy.
    if plan.route == "chitchat":
        return None

    # 2) Run skill once (exact or similar meaning + keyword).
    if plan.route == "run_skill":
        if not plan.skill_key:
            return schedule_payload  # schedule only, if any
        chosen = next((c for c in candidates if c.skill_key == plan.skill_key and c.enabled), None)
        if not chosen:
            return schedule_payload
        if chosen.skill_key.startswith(("websearch_", "dbquery_")) and not allow_dynamic_skill_candidate(
            utterance, score=chosen.score, reason=chosen.reason
        ):
            # dynamic activation gate says no
            return schedule_payload
        run_payload = _execute_candidate(
            cand=chosen,
            utterance=utterance,
            db=db,
            user=user,
            tenant_uuid=tenant_uuid,
            caller_id=caller_id,
            call_id=call_id,
            account_id=account_id,
            context=context,
            reason="plan_run_skill",
        )
        # If we also scheduled, you can choose to append schedule info later; for MVP we just return run_payload.
        return run_payload or schedule_payload

    # 3) Disambiguate among skills.
    if plan.route == "disambiguate":
        if not call_id:
            return schedule_payload
        if not utterance_has_activation_keyword(utterance):
            return schedule_payload
        keys = [k for k in (plan.choices or []) if isinstance(k, str)]
        opts = [next((c for c in candidates if c.skill_key == k), None) for k in keys]
        opts = [c for c in opts if c and c.enabled]
        if not opts:
            return schedule_payload
        session_store.set(call_id, "intent_v2_choices", [c.__dict__ for c in opts])
        session_store.set(call_id, "intent_v2_pending_action", "run_skill")
        return {
            "spoken_reply": _format_disambiguation_prompt(opts),
            "fsm": {"mode": "intent_v2", "intent": "disambiguate"},
            "gmail": None,
        }

    # Unknown route: just return any schedule side-effect, else fall back.
    return schedule_payload

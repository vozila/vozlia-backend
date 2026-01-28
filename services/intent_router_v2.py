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
from models import DeliveryChannel
from services.web_search_skill_store import upsert_daily_schedule

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


class ScheduleRequest(BaseModel):
    """Structured schedule request extracted from a natural-language utterance.

    NOTE: We keep this intentionally small (daily schedules only for MVP).
    Extend later with weekly/monthly cron-like support behind feature flags.
    """

    cadence: Literal["daily"] = "daily"
    hour: int = Field(..., ge=0, le=23)
    minute: int = Field(..., ge=0, le=59)
    timezone: str = Field(..., min_length=1)
    # Delivery channel for scheduled results (MVP: email or sms).
    channel: str = Field(..., min_length=1)
    destination: str = Field(..., min_length=1)

class IntentPlanV2(BaseModel):
    """Strict plan returned by the LLM."""

    route: str = Field(..., description="run_skill|schedule_skill|disambiguate|chitchat")
    skill_key: Optional[str] = Field(None, description="Chosen skill_key if route=run_skill|schedule_skill")
    choices: Optional[List[str]] = Field(None, description="List of skill_keys if route=disambiguate")
    schedule_request: Optional[ScheduleRequest] = Field(None, description="If scheduling: cadence/time/channel/destination")
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


def llm_plan_intent(candidates: List[SkillCandidate], utterance: str, context: Dict[str, Any]) -> IntentPlanV2:
    """Ask the LLM to pick an intent route and (optionally) a skill.

    IMPORTANT:
    - This returns a structured plan ONLY (schema-validated JSON).
    - Python executes deterministically (run skill / schedule / disambiguation).
    """
    tz = (context or {}).get("timezone") or "America/New_York"

    system = f"""You are Vozlia's intent planner.

Your job: decide whether the user wants to (a) run a skill now, (b) schedule a skill,
(c) disambiguate among multiple skills, or (d) just chat.

Return JSON that matches this schema exactly:
{IntentPlanV2.as_json_schema()}

Routes:
- run_skill: user is asking to execute ONE specific skill right now.
- schedule_skill: user is asking to create/update a schedule for a skill (DO NOT run it).
- disambiguate: multiple plausible skills; provide ordered choices (skill_keys).
- chitchat: no tool/skill should run.

SCHEDULING RULES (MVP):
- We only support DAILY schedules right now.
- If the user asks to schedule/recurring/deliver "every day"/"daily", choose route=schedule_skill
  (or route=disambiguate if multiple skills match) and include schedule_request:
  {
    "cadence": "daily",
    "hour": <0-23>,
    "minute": <0-59>,
    "timezone": "<IANA timezone>",
    "channel": "email"|"sms",
    "destination": "<email address or phone number>"
  }
- If timezone is not specified, default to {tz}.
- If the user says "this report" / "this skill", and context implies a recent skill, still schedule it.

CATEGORY RULE:
- If the user is asking for a category (e.g., "sports" / "weather" / "finance") and there are
  multiple skills in that category, choose route=disambiguate with those choices.

CONFIDENCE:
- confidence must be 0.0–1.0.
- If you're not confident, choose chitchat.
"""

    user = json.dumps(
        {
            "utterance": utterance,
            "candidates": [c.model_dump() for c in candidates],
        },
        ensure_ascii=False,
    )
    return openai_chat_json(IntentPlanV2, system, user)
def _detect_category_from_utterance(utterance: str, candidates: List[SkillCandidate]) -> str | None:
    """Best-effort category detection.

    Why this exists
    ---------------
    Users often speak in natural language like "sports update" without naming a specific skill.
    We want to avoid brittle regex/phrase matching and still keep routing deterministic and safe.

    Strategy
    --------
    - Look for a category token present in the utterance (substring + token match)
    - Only consider categories that actually exist in the current candidate snapshot
    - Return the best match, else None
    """
    u = _norm(utterance)
    if not u:
        return None
    u_tokens = set(_tokens(utterance))

    cats = []
    for c in candidates:
        cat = (c.category or "").strip().lower()
        if not cat:
            continue
        cats.append(cat)

    if not cats:
        return None

    # De-dupe while preserving order (stable behavior)
    seen = set()
    unique_cats = []
    for cat in cats:
        if cat in seen:
            continue
        seen.add(cat)
        unique_cats.append(cat)

    best_cat = None
    best_score = 0.0

    for cat in unique_cats:
        cat_norm = _norm(cat)
        if not cat_norm:
            continue

        # Substring match is the strongest signal ("sports" in "sports update")
        if cat_norm in u:
            score = 10.0 + (len(cat_norm) / 10.0)
        else:
            # Token overlap (handles "call metrics" vs category "calls")
            cat_tokens = set([t for t in _tokens(cat_norm) if t])
            inter = len(cat_tokens.intersection(u_tokens))
            if inter <= 0:
                continue
            score = float(inter)

        if score > best_score:
            best_score = score
            best_cat = cat_norm

    return best_cat



def _dynamic_activation_keywords() -> List[str]:
    """Comma-separated keywords required to activate *dynamic* skills from fuzzy/category intent.

    Purpose:
    - Prevent accidental tool execution when the user mentions a topic in passing (e.g. "sports").

    Env:
    - DYNAMIC_SKILL_ACTIVATION_KEYWORDS="skill,report,digest"

    Behavior:
    - If empty/unset → no gating (legacy behavior).
    """
    raw = os.getenv("DYNAMIC_SKILL_ACTIVATION_KEYWORDS", "").strip()
    if not raw:
        return []
    return [k.strip().lower() for k in raw.split(",") if k.strip()]


def _dynamic_activation_keywords_ok(utterance: str) -> bool:
    kws = _dynamic_activation_keywords()
    if not kws:
        return True
    u = utterance.lower()
    return any(k in u for k in kws)


def _utterance_matches_any_phrase(utterance: str, phrases: List[str]) -> bool:
    u = utterance.lower()
    for p in phrases or []:
        p = (p or "").strip().lower()
        if not p:
            continue
        if p in u:
            return True
    return False


def _dynamic_activation_ok_for_candidate(utterance: str, cand: SkillCandidate) -> bool:
    """Allow dynamic skill routing if either:
    - user explicitly used a configured activation keyword, OR
    - utterance directly matches a known trigger phrase for the candidate.
    """
    # Direct trigger phrase match should always win (it is already user-provided disambiguation).
    if _utterance_matches_any_phrase(utterance, cand.trigger_phrases):
        return True
    # Otherwise require activation keywords if configured.
    return _dynamic_activation_keywords_ok(utterance)


def _is_schedule_like_utterance(utterance: str) -> bool:
    u = utterance.lower()
    return any(k in u for k in ["schedule", "every day", "everyday", "daily", "recurring", "deliver every"])


def _parse_schedule_from_utterance(utterance: str, default_timezone: str) -> Optional[ScheduleRequest]:
    """Best-effort deterministic schedule extraction (fallback when LLM doesn't emit schedule_request)."""
    u = utterance.lower()
    if not _is_schedule_like_utterance(utterance):
        return None

    # Cadence (MVP: daily only).
    if "daily" in u or "every day" in u or "everyday" in u:
        cadence = "daily"
    else:
        # Not supported yet (weekly/monthly).
        return None

    # Timezone: first IANA-looking token, else default.
    tz_match = re.search(r"\b([A-Za-z]+\/[A-Za-z_]+)\b", utterance)
    tz = tz_match.group(1) if tz_match else default_timezone

    # Time: look for "at 3:33 AM" or "at 15:33".
    tm = re.search(r"\bat\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b", utterance, flags=re.IGNORECASE)
    if not tm:
        return None
    hour = int(tm.group(1))
    minute = int(tm.group(2) or "0")
    ampm = (tm.group(3) or "").lower()
    if ampm:
        if hour == 12:
            hour = 0
        if ampm == "pm":
            hour = (hour + 12) % 24
    if hour < 0 or hour > 23 or minute < 0 or minute > 59:
        return None

    # Channel + destination.
    channel = "email" if "email" in u else ("sms" if ("sms" in u or "text" in u) else "")
    if not channel:
        return None
    dest = ""
    if channel == "email":
        em = re.search(r"([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})", utterance)
        if em:
            dest = em.group(1)
    elif channel == "sms":
        ph = re.search(r"(\+?\d{10,15})", utterance)
        if ph:
            dest = ph.group(1)
    if not dest:
        return None

    try:
        return ScheduleRequest(cadence=cadence, hour=hour, minute=minute, timezone=tz, channel=channel, destination=dest)
    except Exception:
        return None

def maybe_route_and_execute(
    db: Session,
    user,
    tenant_uuid: str | None,
    caller_id: str | None,
    utterance: str,
    context: Dict[str, Any],
    mode: str,
) -> dict | None:
    """Intent V2: LLM plans (schema-validated), Python executes.

    mode:
      - shadow: plan only (no execution / no schedule writes).
      - assist: execute intent_v2 if confident; else fallback to legacy.
      - strict: always use intent_v2 (not implemented here; handled upstream).
    """
    call_id = (context or {}).get("call_id")

    # 1) If we are waiting on a disambiguation selection, consume it first.
    if call_id:
        consumed = maybe_consume_disambiguation_choice(
            db=db,
            user=user,
            tenant_uuid=tenant_uuid,
            caller_id=caller_id,
            call_id=call_id,
            utterance=utterance,
            context=context,
        )
        if consumed is not None:
            return consumed

    # 2) Build candidates from skills_config (legacy + dynamic).
    skills_cfg = get_skills_config(db, user)
    candidates = _build_candidates(skills_cfg)
    if not candidates:
        return None

    # 3) Deterministic fast-path (exact trigger/name match).
    best, score = _best_match_candidate(utterance, candidates)
    if score >= 0.92 and best.enabled:
        # Optional safety gate for dynamic skills: allow if trigger matched OR activation keyword.
        if best.skill_key.startswith(("websearch_", "dbquery_")) and not _dynamic_activation_ok_for_candidate(utterance, best):
            logger.info("INTENT_V2_FASTPATH_BLOCKED skill=%s", best.skill_key)
        else:
            logger.info("INTENT_V2_FASTPATH skill=%s score=%.2f", best.skill_key, score)
            return _execute_candidate(db, user, best, tenant_uuid, caller_id, call_id)

    # 4) Ask LLM for a plan (schema-validated).
    plan: IntentPlanV2 | None = None
    if mode != "off":
        try:
            plan = llm_plan_intent(candidates=candidates, utterance=utterance, context=context or {})
        except Exception:
            plan = None

    # Shadow mode: never execute / schedule. (We still let legacy routing handle it.)
    if mode == "shadow":
        return None

    # 5) If the utterance is schedule-like, treat scheduling as higher priority than running.
    schedule_like = _is_schedule_like_utterance(utterance)
    default_tz = (context or {}).get("timezone") or "America/New_York"
    schedule_req: ScheduleRequest | None = None
    if schedule_like:
        if plan and plan.schedule_request:
            schedule_req = plan.schedule_request
        else:
            schedule_req = _parse_schedule_from_utterance(utterance, default_timezone=default_tz)

    # 5a) Scheduling path (only if we can extract schedule details).
    if schedule_like and schedule_req is not None:
        # Candidate selection preference:
        # 1) LLM-selected skill_key (even if the LLM mistakenly set route=run_skill).
        # 2) Last executed skill in this call ("schedule this report").
        target_skill_key = (plan.skill_key if plan else None)
        if not target_skill_key and call_id:
            target_skill_key = session_store.get(call_id, "intent_v2_last_skill_key")

        if target_skill_key:
            chosen = next((c for c in candidates if c.skill_key == target_skill_key and c.enabled), None)
            if chosen is not None:
                logger.info("INTENT_V2_SCHEDULE skill=%s", chosen.skill_key)
                out = _schedule_candidate(db, user, chosen, schedule_req, tenant_uuid, caller_id, call_id)
                if out is not None:
                    return out

        # If we couldn't select a single skill, disambiguate among category matches.
        cat = _detect_category_from_utterance(utterance, candidates)
        opts = [c for c in candidates if c.enabled and (cat and c.category == cat)]
        if opts:
            if not _dynamic_activation_keywords_ok(utterance):
                # User mentioned the topic but didn't explicitly ask for a report/skill (configurable gate).
                return None
            session_store.set(call_id, "intent_v2_choices", [c.model_dump() for c in opts])
            session_store.set(call_id, "intent_v2_pending_action", "schedule_skill")
            session_store.set(call_id, "intent_v2_pending_schedule", schedule_req.model_dump())
            return {
                "spoken_reply": _format_disambiguation_prompt(opts),
                "fsm": {"mode": "intent_v2", "intent": "disambiguate"},
                "gmail": None,
            }

        return {
            "spoken_reply": "Which skill should I schedule? You can say: schedule skill <name> daily at <time> <timezone> and deliver by email to <address>.",
            "fsm": {"mode": "intent_v2", "intent": "schedule_needs_skill"},
            "gmail": None,
        }

    # 5b) If user asked to schedule but we couldn't parse schedule details, ask for specifics.
    if schedule_like and schedule_req is None:
        return {
            "spoken_reply": "Sure — what time should I run it daily, which timezone, and where should I deliver it (email or SMS)?",
            "fsm": {"mode": "intent_v2", "intent": "schedule_needs_details"},
            "gmail": None,
        }

    # 6) No plan (LLM failed): optional category fallback in assist mode, gated by activation keywords.
    if plan is None:
        if mode == "assist" and call_id and _dynamic_activation_keywords_ok(utterance):
            cat = _detect_category_from_utterance(utterance, candidates)
            if cat:
                opts = [c for c in candidates if c.enabled and c.category == cat]
                if len(opts) == 1:
                    return _execute_candidate(db, user, opts[0], tenant_uuid, caller_id, call_id)
                if len(opts) > 1:
                    session_store.set(call_id, "intent_v2_choices", [c.model_dump() for c in opts])
                    session_store.set(call_id, "intent_v2_pending_action", "run_skill")
                    return {
                        "spoken_reply": _format_disambiguation_prompt(opts),
                        "fsm": {"mode": "intent_v2", "intent": "disambiguate"},
                        "gmail": None,
                    }
        return None

    # 7) Execute plan routes.
    if plan.route == "chitchat":
        return None

    if plan.route == "run_skill":
        if not plan.skill_key:
            return None
        chosen = next((c for c in candidates if c.skill_key == plan.skill_key and c.enabled), None)
        if not chosen:
            return None
        if chosen.skill_key.startswith(("websearch_", "dbquery_")) and not _dynamic_activation_ok_for_candidate(utterance, chosen):
            return None
        return _execute_candidate(db, user, chosen, tenant_uuid, caller_id, call_id)

    if plan.route == "disambiguate":
        if not call_id:
            return None
        if not _dynamic_activation_keywords_ok(utterance):
            # Topic mentioned without explicit activation keyword; avoid unintended tool prompts.
            return None
        keys = [k for k in (plan.choices or []) if isinstance(k, str)]
        opts = [next((c for c in candidates if c.skill_key == k), None) for k in keys]
        opts = [c for c in opts if c and c.enabled]
        if not opts:
            return None
        session_store.set(call_id, "intent_v2_choices", [c.model_dump() for c in opts])
        session_store.set(call_id, "intent_v2_pending_action", "run_skill")
        return {
            "spoken_reply": _format_disambiguation_prompt(opts),
            "fsm": {"mode": "intent_v2", "intent": "disambiguate"},
            "gmail": None,
        }

    if plan.route == "schedule_skill":
        # Even if the LLM chose schedule_skill, we still need schedule_request; otherwise ask for details.
        if schedule_req is None:
            return {
                "spoken_reply": "Sure — what time should I run it daily, which timezone, and where should I deliver it (email or SMS)?",
                "fsm": {"mode": "intent_v2", "intent": "schedule_needs_details"},
                "gmail": None,
            }
        if not plan.skill_key and call_id:
            plan.skill_key = session_store.get(call_id, "intent_v2_last_skill_key")
        if not plan.skill_key:
            return {
                "spoken_reply": "Which skill should I schedule?",
                "fsm": {"mode": "intent_v2", "intent": "schedule_needs_skill"},
                "gmail": None,
            }
        chosen = next((c for c in candidates if c.skill_key == plan.skill_key and c.enabled), None)
        if not chosen:
            return None
        out = _schedule_candidate(db, user, chosen, schedule_req, tenant_uuid, caller_id, call_id)
        if out is not None:
            return out
        return None

    return None
def maybe_consume_disambiguation_choice(
    db: Session,
    user,
    tenant_uuid: str | None,
    caller_id: str | None,
    call_id: str,
    utterance: str,
    context: Dict[str, Any],
) -> dict | None:
    """If the previous turn asked the user to pick a skill (1/2/3...), consume the choice.

    This supports both:
    - running the selected skill, OR
    - scheduling the selected skill (when a schedule request caused disambiguation).
    """
    stored = session_store.pop(call_id, "intent_v2_choices")
    if not stored:
        return None

    pending_action = session_store.pop(call_id, "intent_v2_pending_action") or "run_skill"
    pending_schedule = session_store.pop(call_id, "intent_v2_pending_schedule")
    schedule_req: ScheduleRequest | None = None
    if pending_schedule:
        try:
            schedule_req = ScheduleRequest.model_validate(pending_schedule)
        except Exception:
            schedule_req = None

    try:
        candidates = [SkillCandidate.model_validate(x) for x in stored]
    except Exception:
        return None

    u = utterance.strip()

    def _restore_and_prompt() -> dict:
        session_store.set(call_id, "intent_v2_choices", [c.model_dump() for c in candidates])
        session_store.set(call_id, "intent_v2_pending_action", pending_action)
        if schedule_req:
            session_store.set(call_id, "intent_v2_pending_schedule", schedule_req.model_dump())
        return {
            "spoken_reply": _format_disambiguation_prompt(candidates),
            "fsm": {"mode": "intent_v2", "intent": "disambiguate"},
            "gmail": None,
        }

    # Numeric choice ("1", "2", ...).
    if u.isdigit():
        idx = int(u) - 1
        if idx < 0 or idx >= len(candidates):
            return _restore_and_prompt()
        chosen = candidates[idx]
        if pending_action == "schedule_skill":
            if not schedule_req:
                return {
                    "spoken_reply": "Okay — what time should I run it daily, which timezone, and where should I deliver it (email or SMS)?",
                    "fsm": {"mode": "intent_v2", "intent": "schedule_needs_details"},
                    "gmail": None,
                }
            out = _schedule_candidate(db, user, chosen, schedule_req, tenant_uuid, caller_id, call_id)
            return out or _restore_and_prompt()
        return _execute_candidate(db, user, chosen, tenant_uuid, caller_id, call_id)

    # Non-numeric: try to match by skill label substring.
    u_can = _canonicalize(u)
    label_matches = [c for c in candidates if u_can and (u_can in _canonicalize(c.label) or _canonicalize(c.label) in u_can)]
    if len(label_matches) == 1:
        chosen = label_matches[0]
        if pending_action == "schedule_skill":
            if not schedule_req:
                return {
                    "spoken_reply": "Okay — what time should I run it daily, which timezone, and where should I deliver it (email or SMS)?",
                    "fsm": {"mode": "intent_v2", "intent": "schedule_needs_details"},
                    "gmail": None,
                }
            out = _schedule_candidate(db, user, chosen, schedule_req, tenant_uuid, caller_id, call_id)
            return out or _restore_and_prompt()
        return _execute_candidate(db, user, chosen, tenant_uuid, caller_id, call_id)

    # Last resort: ask the LLM to map this utterance to one of the candidate skill_keys.
    try:
        plan = llm_plan_intent(candidates=candidates, utterance=utterance, context=context or {})
    except Exception:
        return _restore_and_prompt()

    target_key = plan.skill_key if plan else None
    if target_key:
        chosen = next((c for c in candidates if c.skill_key == target_key), None)
        if chosen:
            if pending_action == "schedule_skill":
                if not schedule_req:
                    return {
                        "spoken_reply": "Okay — what time should I run it daily, which timezone, and where should I deliver it (email or SMS)?",
                        "fsm": {"mode": "intent_v2", "intent": "schedule_needs_details"},
                        "gmail": None,
                    }
                out = _schedule_candidate(db, user, chosen, schedule_req, tenant_uuid, caller_id, call_id)
                return out or _restore_and_prompt()
            return _execute_candidate(db, user, chosen, tenant_uuid, caller_id, call_id)

    return _restore_and_prompt()
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
                # Remember last skill for this call (used for follow-ups like "schedule this report").
                if call_id:
                    session_store.set(call_id, "intent_v2_last_skill_key", cand.skill_key)
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
        if call_id:
            session_store.set(call_id, "intent_v2_last_skill_key", cand.skill_key)
        return {"spoken_reply": spoken, "fsm": {"mode": "intent_v2", "skill_id": cand.skill_key, "type": "internal", "intent_v2_reason": reason}, "gmail": out.get("gmail") if isinstance(out, dict) else None}
    except Exception:
        return None

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
    """Create/update a daily schedule for a dynamic *websearch* skill."""
    if not cand.skill_key.startswith("websearch_"):
        return {
            "spoken_reply": "Scheduling is only supported for WebSearch skills right now.",
            "fsm": {"mode": "intent_v2", "intent": "schedule_unsupported", "skill_id": cand.skill_key},
            "gmail": None,
        }

    # Convert skill_key -> raw uuid for the schedule table.
    raw_skill_id = cand.skill_key.split("websearch_", 1)[1] if "websearch_" in cand.skill_key else ""
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

    try:
        row = upsert_daily_schedule(
            db=db,
            current_user=user,
            skill_id=raw_skill_id,
            hour=int(sched.hour),
            minute=int(sched.minute),
            timezone=sched.timezone,
            channel=channel,
            destination=sched.destination,
            enabled=True,
        )
    except Exception:
        return None

    # Remember last skill to support "schedule this report" follow-ups.
    if call_id:
        session_store.set(call_id, "intent_v2_last_skill_key", cand.skill_key)

    time_of_day = f"{int(sched.hour):02d}:{int(sched.minute):02d}"
    spoken = (
        f"Okay — scheduled {cand.label} daily at {time_of_day} {sched.timezone} "
        f"and will deliver by {sched.channel} to {sched.destination}."
    )
    return {
        "spoken_reply": spoken,
        "fsm": {
            "mode": "intent_v2",
            "intent": "schedule_created",
            "skill_id": cand.skill_key,
            "schedule_id": getattr(row, "id", None),
            "next_run_at": getattr(row, "next_run_at", None),
        },
        "gmail": None,
    }

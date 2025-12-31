# services/memory_llm.py
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from core.logging import logger
from core import config as cfg
from models import CallerMemoryEvent

try:
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore


# -------------------------
# Config (recycle existing env vars)
# -------------------------
CALLER_MEMORY_ENABLED = (os.getenv("CALLER_MEMORY_ENABLED", "0") or "0").strip().lower() in ("1", "true", "yes", "on")
CALLER_MEMORY_DEBUG = (os.getenv("CALLER_MEMORY_DEBUG", "0") or "0").strip().lower() in ("1", "true", "yes", "on")
VOZLIA_DEBUG_MEMORY = (os.getenv("VOZLIA_DEBUG_MEMORY", "0") or "0").strip().lower() in ("1", "true", "yes", "on")

# If you already use these elsewhere, keep them as the primary knobs
DEFAULT_RECALL_LIMIT = int((os.getenv("LONGTERM_MEMORY_RECALL_LIMIT", "50") or "50").strip())
DEFAULT_EVIDENCE_LIMIT = int((os.getenv("LONGTERM_MEMORY_CONTEXT_LIMIT", "8") or "8").strip())

# Optional, but safe defaults if unset
MEMORY_PLANNER_MODEL = (os.getenv("OPENAI_MEMORY_PLANNER_MODEL") or "").strip() or "gpt-4o-mini"
MEMORY_RESPONDER_MODEL = (os.getenv("OPENAI_MEMORY_RESPONDER_MODEL") or "").strip() or "gpt-4o-mini"

# Hard caps to protect cost and rate limits
MAX_CANDIDATES_CAP = int((os.getenv("LONGTERM_MEMORY_MAX_CANDIDATES_CAP", "200") or "200").strip())
MAX_EVIDENCE_CAP = int((os.getenv("LONGTERM_MEMORY_MAX_EVIDENCE_CAP", "20") or "20").strip())

# Retry/backoff for OpenAI calls (not hot path, but keep bounded)
OPENAI_RETRIES = int((os.getenv("OPENAI_RETRIES", "2") or "2").strip())
OPENAI_BACKOFF_S = float((os.getenv("OPENAI_BACKOFF_S", "0.5") or "0.5").strip())


def _get_openai_client() -> Optional[Any]:
    if OpenAI is None:
        return None
    if not getattr(cfg, "OPENAI_API_KEY", None):
        return None
    return OpenAI(api_key=cfg.OPENAI_API_KEY)


_CLIENT = _get_openai_client()


@dataclass
class MemoryPlan:
    intent: str  # "memory_recall" | "summarize_window" | "normal"
    # window in UTC
    start_ts: datetime
    end_ts: datetime
    scope: str  # "caller" | "tenant"
    keywords: List[str]
    entities: List[str]
    fact_keys: List[str]
    candidates_limit: int
    evidence_limit: int
    conflict_mode: str  # "ask_if_multiple" | "choose_most_recent"
    max_choices: int = 3


_COLOR_WORDS = {
    "red","orange","yellow","green","blue","purple","violet","pink","brown","black","white","gray","grey",
    "gold","silver","beige","tan","navy","teal","cyan","magenta","maroon","olive",
}

_RELATIVE_HINTS = ("minute", "minutes", "hour", "hours", "day", "days", "week", "weeks", "month", "months")
_TIME_WORDS = ("yesterday", "today", "last", "previous", "ago", "earlier", "recent")


def looks_like_memory_question(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return False
    # broad (prefer false positives over no-hits)
    triggers = [
        "what did i say",
        "what did i tell",
        "did i say",
        "did i tell",
        "remember",
        "remind me",
        "last time",
        "previous call",
        "earlier",
        "yesterday",
        "last week",
        "what was i talking about",
        "what were we talking about",
        "what did we discuss",
        "my favorite",
    ]
    if any(x in t for x in triggers):
        return True
    # time-ish + question mark
    if "?" in t and any(w in t for w in _TIME_WORDS):
        return True
    return False


def _retention_days() -> int:
    try:
        return int((os.getenv("LONGTERM_MEMORY_RETENTION_DAYS", "30") or "30").strip())
    except Exception:
        return 30


def _clamp_window(now: datetime, start_ts: datetime, end_ts: datetime) -> Tuple[datetime, datetime]:
    # Ensure UTC
    if start_ts.tzinfo is None:
        start_ts = start_ts.replace(tzinfo=timezone.utc)
    if end_ts.tzinfo is None:
        end_ts = end_ts.replace(tzinfo=timezone.utc)

    if end_ts > now:
        end_ts = now
    if start_ts > end_ts:
        start_ts = end_ts - timedelta(minutes=10)

    max_days = max(1, _retention_days())
    earliest = now - timedelta(days=max_days)
    if start_ts < earliest:
        start_ts = earliest
    return start_ts, end_ts


def _safe_json_loads(s: str) -> Optional[dict]:
    try:
        return json.loads(s)
    except Exception:
        return None


def _strip_code_fences(s: str) -> str:
    s = (s or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", s)
        s = re.sub(r"\n?```$", "", s)
    return s.strip()


def plan_memory_request(
    *,
    question: str,
    now_utc: Optional[datetime] = None,
    profile: Optional[dict] = None,
) -> MemoryPlan:
    """
    LLM planner that converts a user question into a structured plan.

    Safety/robustness:
      - Executor enforces caps, clamps time window to retention, and ignores unknown fields.
      - If the LLM is unavailable or returns invalid JSON, we fall back to a permissive heuristic plan.
    """
    now = now_utc or datetime.now(timezone.utc)
    profile = profile or {}

    # fallback defaults (broad)
    fallback = MemoryPlan(
        intent="memory_recall" if looks_like_memory_question(question) else "normal",
        start_ts=now - timedelta(hours=24),
        end_ts=now,
        scope="caller",
        keywords=[],
        entities=[],
        fact_keys=[],
        candidates_limit=min(DEFAULT_RECALL_LIMIT, MAX_CANDIDATES_CAP),
        evidence_limit=min(DEFAULT_EVIDENCE_LIMIT, MAX_EVIDENCE_CAP),
        conflict_mode="ask_if_multiple",
        max_choices=3,
    )

    if not CALLER_MEMORY_ENABLED:
        return fallback

    if _CLIENT is None:
        return fallback

    # If not memory-ish, do not spend tokens
    if not looks_like_memory_question(question):
        return MemoryPlan(**{**fallback.__dict__, "intent": "normal"})

    system = (
        "You are a query planner for a voice assistant's long-term memory database.\n"
        "Return ONLY valid JSON. No markdown.\n"
        "Your job: interpret the question and produce a query plan.\n"
        "Constraints:\n"
        "- Prefer false positives over false negatives.\n"
        "- Use a time window. If the question specifies time (e.g., '5 minutes ago', 'yesterday', 'last week'), use it.\n"
        "- If time is unspecified, default to 24 hours.\n"
        "- Scope defaults to 'caller'. Only use 'tenant' if user explicitly asks about 'all callers' or 'everyone'.\n"
        "- Include keywords and entities helpful for search.\n"
        "- If question implies a fact (favorite color, phone number, email, address, name), include a fact_keys hint.\n"
        "- conflict_mode: 'ask_if_multiple' unless user asked for the latest.\n"
        "Output schema:\n"
        "{\n"
        '  "intent": "memory_recall" | "summarize_window" | "normal",\n'
        '  "lookback_s": <int seconds>,\n'
        '  "scope": "caller" | "tenant",\n'
        '  "keywords": [<strings>],\n'
        '  "entities": [<strings>],\n'
        '  "fact_keys": [<strings>],\n'
        '  "candidates_limit": <int>,\n'
        '  "evidence_limit": <int>,\n'
        '  "conflict_mode": "ask_if_multiple" | "choose_most_recent",\n'
        '  "max_choices": <int>\n'
        "}\n"
    )

    user = {
        "question": question,
        "now_utc": now.isoformat(),
        "profile": profile,
    }

    last_err = None
    for attempt in range(OPENAI_RETRIES + 1):
        try:
            resp = _CLIENT.chat.completions.create(
                model=MEMORY_PLANNER_MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
                ],
                temperature=0.1,
            )
            content = (resp.choices[0].message.content or "").strip()
            content = _strip_code_fences(content)
            data = _safe_json_loads(content)
            if not isinstance(data, dict):
                raise ValueError("planner returned non-dict json")

            intent = str(data.get("intent") or "").strip() or fallback.intent
            lookback_s = int(data.get("lookback_s") or 86400)
            lookback_s = max(60, min(lookback_s, _retention_days() * 86400))
            start_ts = now - timedelta(seconds=lookback_s)
            end_ts = now

            scope = str(data.get("scope") or "caller").strip().lower()
            if scope not in ("caller", "tenant"):
                scope = "caller"

            keywords = [str(x).strip().lower() for x in (data.get("keywords") or []) if str(x).strip()]
            entities = [str(x).strip() for x in (data.get("entities") or []) if str(x).strip()]
            fact_keys = [str(x).strip().lower() for x in (data.get("fact_keys") or []) if str(x).strip()]

            # small heuristic: if question mentions a color word, add it
            qlow = (question or "").lower()
            for c in _COLOR_WORDS:
                if re.search(rf"\b{re.escape(c)}\b", qlow):
                    if c not in keywords:
                        keywords.append(c)
                    if "favorite_color" not in fact_keys and "color" in qlow:
                        fact_keys.append("favorite_color")

            candidates_limit = int(data.get("candidates_limit") or fallback.candidates_limit)
            evidence_limit = int(data.get("evidence_limit") or fallback.evidence_limit)

            candidates_limit = max(10, min(candidates_limit, MAX_CANDIDATES_CAP))
            evidence_limit = max(3, min(evidence_limit, MAX_EVIDENCE_CAP))

            conflict_mode = str(data.get("conflict_mode") or "ask_if_multiple").strip()
            if conflict_mode not in ("ask_if_multiple", "choose_most_recent"):
                conflict_mode = "ask_if_multiple"

            max_choices = int(data.get("max_choices") or 3)
            max_choices = max(2, min(max_choices, 6))

            start_ts, end_ts = _clamp_window(now, start_ts, end_ts)

            plan = MemoryPlan(
                intent=intent,
                start_ts=start_ts,
                end_ts=end_ts,
                scope=scope,
                keywords=keywords,
                entities=entities,
                fact_keys=fact_keys,
                candidates_limit=candidates_limit,
                evidence_limit=evidence_limit,
                conflict_mode=conflict_mode,
                max_choices=max_choices,
            )
            if CALLER_MEMORY_DEBUG or VOZLIA_DEBUG_MEMORY:
                logger.info(
                    "MEM_PLANNER_OK intent=%s scope=%s lookback_s=%s kws=%s facts=%s cand=%s evid=%s conflict=%s",
                    plan.intent, plan.scope, lookback_s, plan.keywords[:8], plan.fact_keys[:6],
                    plan.candidates_limit, plan.evidence_limit, plan.conflict_mode,
                )
            return plan
        except Exception as e:
            last_err = e
            if CALLER_MEMORY_DEBUG or VOZLIA_DEBUG_MEMORY:
                logger.warning("MEM_PLANNER_FAIL attempt=%s err=%s", attempt + 1, e)
            if attempt < OPENAI_RETRIES:
                time.sleep(OPENAI_BACKOFF_S * (2 ** attempt))
            continue

    if CALLER_MEMORY_DEBUG or VOZLIA_DEBUG_MEMORY:
        logger.warning("MEM_PLANNER_FALLBACK err=%s", last_err)
    return fallback


def execute_memory_query(
    db: Any,
    *,
    tenant_id: str,
    caller_id: Optional[str],
    plan: MemoryPlan,
) -> List[CallerMemoryEvent]:
    """
    Deterministic DB query for memory events matching the plan.
    Broad by design: within time window, then optional keyword/entity filters.
    """
    q = db.query(CallerMemoryEvent).filter(CallerMemoryEvent.tenant_id == str(tenant_id))

    if plan.scope == "caller" and caller_id:
        q = q.filter(CallerMemoryEvent.caller_id == str(caller_id))

    q = q.filter(CallerMemoryEvent.created_at >= plan.start_ts).filter(CallerMemoryEvent.created_at <= plan.end_ts)

    # Keyword/entity filters (OR). Keep broad.
    ors = []
    kws = []
    for x in (plan.keywords or [])[:12]:
        x = (x or "").strip()
        if x:
            kws.append(x)
    for x in (plan.entities or [])[:12]:
        x = (x or "").strip()
        if x:
            kws.append(x)

    if kws:
        from sqlalchemy import or_
        for kw in kws[:12]:
            ors.append(CallerMemoryEvent.text.ilike(f"%{kw}%"))
        q = q.filter(or_(*ors))

    rows = q.order_by(CallerMemoryEvent.created_at.desc()).limit(plan.candidates_limit).all()
    if CALLER_MEMORY_DEBUG or VOZLIA_DEBUG_MEMORY:
        logger.info("MEM_QUERY_OK tenant_id=%s caller_id=%s rows=%s", tenant_id, caller_id, len(rows or []))
    return rows or []


def _event_to_evidence(ev: CallerMemoryEvent) -> dict:
    try:
        return {
            "id": getattr(ev, "id", None),
            "created_at": (getattr(ev, "created_at", None).isoformat() if getattr(ev, "created_at", None) else None),
            "skill_key": getattr(ev, "skill_key", None),
            "kind": getattr(ev, "kind", None),
            "text": getattr(ev, "text", None),
            "tags": getattr(ev, "tags_json", None),
            "data": getattr(ev, "data_json", None),
        }
    except Exception:
        return {"id": None, "text": str(ev)}


def generate_memory_response(
    *,
    question: str,
    plan: MemoryPlan,
    evidence_rows: List[CallerMemoryEvent],
) -> Dict[str, Any]:
    """
    LLM responder: produces a spoken answer and optional structured metadata.
    """
    if not evidence_rows:
        return {
            "spoken_reply": "I couldn’t find anything in that time window. Can you narrow the time or tell me a keyword?",
            "needs_clarification": True,
            "choices": [],
        }

    # Trim evidence to limit
    evidence = [_event_to_evidence(e) for e in (evidence_rows or [])[: plan.evidence_limit]]

    # If no OpenAI, respond with simple fallback
    if _CLIENT is None:
        # best-effort: show most recent snippet
        top = evidence[0]
        return {
            "spoken_reply": f"I found this from {top.get('created_at')}: {top.get('text')}",
            "needs_clarification": False,
            "choices": [top],
        }

    system = (
        "You are Vozlia, a phone assistant. You are answering a memory question.\n"
        "Use ONLY the provided evidence. Do NOT invent facts.\n"
        "If there are multiple conflicting candidate values, follow conflict_policy:\n"
        f"- conflict_mode={plan.conflict_mode}\n"
        "If conflict_mode=ask_if_multiple, ask ONE short clarifying question and list up to max_choices candidates.\n"
        "If conflict_mode=choose_most_recent, choose the most recent and mention that it's the most recent.\n"
        "Return ONLY valid JSON with this schema:\n"
        "{\n"
        '  "spoken_reply": <string>,\n'
        '  "needs_clarification": <bool>,\n'
        '  "choices": [ {"id":..., "created_at":..., "text":..., "value":...} ]\n'
        "}\n"
        "Where 'value' is optional: include it when the question asks for a specific fact (color, phone, email, address, name).\n"
    )

    payload = {
        "question": question,
        "plan": {
            "intent": plan.intent,
            "start_ts": plan.start_ts.isoformat(),
            "end_ts": plan.end_ts.isoformat(),
            "scope": plan.scope,
            "keywords": plan.keywords,
            "entities": plan.entities,
            "fact_keys": plan.fact_keys,
            "max_choices": plan.max_choices,
            "conflict_mode": plan.conflict_mode,
        },
        "evidence": evidence,
    }

    last_err = None
    for attempt in range(OPENAI_RETRIES + 1):
        try:
            resp = _CLIENT.chat.completions.create(
                model=MEMORY_RESPONDER_MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
                temperature=0.2,
            )
            content = _strip_code_fences((resp.choices[0].message.content or "").strip())
            data = _safe_json_loads(content)
            if not isinstance(data, dict) or not data.get("spoken_reply"):
                raise ValueError("responder returned invalid json")
            # clamp choices
            choices = data.get("choices") or []
            if not isinstance(choices, list):
                choices = []
            data["choices"] = choices[: max(1, plan.max_choices)]
            if CALLER_MEMORY_DEBUG or VOZLIA_DEBUG_MEMORY:
                logger.info("MEM_RESP_OK clarify=%s choices=%s", bool(data.get("needs_clarification")), len(data["choices"]))
            return data
        except Exception as e:
            last_err = e
            if CALLER_MEMORY_DEBUG or VOZLIA_DEBUG_MEMORY:
                logger.warning("MEM_RESP_FAIL attempt=%s err=%s", attempt + 1, e)
            if attempt < OPENAI_RETRIES:
                time.sleep(OPENAI_BACKOFF_S * (2 ** attempt))
            continue

    if CALLER_MEMORY_DEBUG or VOZLIA_DEBUG_MEMORY:
        logger.warning("MEM_RESP_FALLBACK err=%s", last_err)

    top = evidence[0]
    return {
        "spoken_reply": f"I found this: {top.get('text')}",
        "needs_clarification": False,
        "choices": [top],
    }

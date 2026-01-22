
# services/assistant_service.py
from skills.engine import skills_engine_enabled, match_skill_id, match_skill_ids, execute_skill
from services.settings_service import (
    get_agent_greeting,
    gmail_summary_enabled,
    gmail_summary_add_to_greeting,
    get_gmail_summary_engagement_phrases,
    shortterm_memory_enabled,
    longterm_memory_enabled,
    get_memory_engagement_phrases,
    get_investment_reporting_config,
    get_investment_reporting_tickers,
)


from services.investment_service import get_investment_reports

import os
import re
import json
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from typing import Any
from core.logging import logger
from skills.registry import skill_registry
from sqlalchemy.orm import Session
from models import User
from models import CallerMemoryEvent
from vozlia_fsm import VozliaFSM

def _maybe_answer_history_count(
    db: Session,
    *,
    tenant_id: str,
    caller_id: str,
    text: str,
) -> str | None:
    """Deterministic counts for questions like:
    - 'how many times did I request email summaries'
    - 'how often did I ask for my email summary this week'
    """
    t = (text or "").strip().lower()
    if not t:
        return None
    if not (("how many" in t) or ("number of" in t) or ("how often" in t) or ("times" in t)):
        return None
    if ("email" not in t) and ("gmail" not in t):
        return None
    # Restrict to summary-request counts (avoid hijacking random email questions)
    if ("summary" not in t and "summaries" not in t and "summar" not in t and "request" not in t and "requested" not in t):
        if "my email" not in t:
            return None

    now_utc = datetime.now(timezone.utc)
    # Interpret "today/this week/this month" in America/New_York, then convert to UTC for DB filtering.
    tz = ZoneInfo(os.getenv("APP_TZ", "America/New_York"))
    now_local = now_utc.astimezone(tz)

    start_dt = None
    scope = ""

    if "today" in t:
        start_local = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
        start_dt = start_local.astimezone(timezone.utc).replace(tzinfo=None)
        scope = " today"
    elif "this week" in t:
        # Week starts Monday 00:00 local time
        start_local = (now_local - timedelta(days=now_local.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
        start_dt = start_local.astimezone(timezone.utc).replace(tzinfo=None)
        scope = " this week"
    elif "past week" in t or "last week" in t:
        start_dt = (now_utc - timedelta(days=7)).replace(tzinfo=None)
        scope = " in the past week"
    elif "this month" in t:
        start_local = now_local.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        start_dt = start_local.astimezone(timezone.utc).replace(tzinfo=None)
        scope = " this month"
    elif "past month" in t or "last month" in t:
        start_dt = (now_utc - timedelta(days=30)).replace(tzinfo=None)
        scope = " in the past month"

    # Caller-id normalization: match either "+1..." or "1..." variants to avoid drift.
    cid = str(caller_id)
    cid_digits = "".join([c for c in cid if c.isdigit()])
    caller_variants = {cid}
    if cid.startswith("+"):
        caller_variants.add(cid[1:])
    else:
        caller_variants.add("+" + cid)
    if cid_digits:
        caller_variants.add(cid_digits)
        caller_variants.add("+" + cid_digits)

    qy = db.query(CallerMemoryEvent).filter(
        CallerMemoryEvent.tenant_id == str(tenant_id),
        CallerMemoryEvent.caller_id.in_(sorted(caller_variants)),
        CallerMemoryEvent.skill_key == "gmail_summary",
    )
    if start_dt is not None:
        qy = qy.filter(CallerMemoryEvent.created_at >= start_dt)

    cnt = int(qy.count() or 0)
    if cnt == 0:
        return f"You haven't requested email summaries{scope}."
    suffix = "time" if cnt == 1 else "times"
    return f"In total, {cnt} {suffix}{scope}."

from services.memory_facade import (
    memory,
    make_skill_cache_key_hash,
    SESSION_MEMORY_ENABLED,
    SESSION_MEMORY_TTL_S,
)
from services.caller_cache import (
    CALLER_MEMORY_ENABLED,
    CALLER_MEMORY_TTL_S,
    get_caller_cache,
    put_caller_cache,
    normalize_caller_id,
)
from services.longterm_memory import (
    record_turn_event,
    longterm_memory_enabled_for_tenant,
    fetch_recent_memory_text,
    record_skill_result,
)

# ----------------------------
# LLM Router (intermediate step)
# ----------------------------
# Modes:
#   - off (default): no additional OpenAI call
#   - shadow: ask LLM for an intent/tool plan and LOG it (no behavior change)
#   - assist: use the LLM plan to set force_* flags (still uses existing execution paths)
#
# This is intentionally small and feature-flagged so we can migrate from keyword/FSM triggers
# to LLM-first tool planning without destabilizing production.
try:
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore

_ROUTER_CLIENT = None

def _is_junk_transcript(t: str) -> bool:
    s = (t or "").strip().lower()
    if len(s) < 5:
        return True
    if s in {"you", "ok", "okay", "yeah", "yes", "no", "um", "uh"}:
        return True
    return False

def _tool_to_canonical_phrase(tool: str) -> str | None:
    tool = (tool or "").strip().lower()
    if tool == "gmail_summary":
        return "Email summary"
    if tool == "investment_reporting":
        return "Investment reporting"
    if tool == "memory_lookup":
        return None
    return None


# ----------------------------
# Investment Reporting: dynamic ticker extraction helpers
# ----------------------------
# We want to support caller-requested stocks (e.g., "price on TSLA" or "price on Tesla"),
# while preserving the existing "investment report" (portal-configured tickers) behavior.
#
# Extraction priority:
#   1) LLM router plan tool_args.tickers (best for company-name -> ticker mapping)
#   2) Regex extraction for explicit ticker mentions ($TSLA, ticker TSLA, TSLA)
#
# NOTE: We intentionally keep regex extraction conservative to avoid false positives.
_TICKER_STOPWORDS = {
    # Common words that can look like tickers in ALL CAPS transcripts
    "I", "A", "AN", "AND", "OR", "THE", "TO", "OF", "IN", "ON", "AT", "FOR", "FROM",
    "AS", "IS", "ARE", "AM", "BE", "WAS", "WERE", "IT", "ITS", "THIS", "THAT", "THESE", "THOSE",
    "USA", "US", "UAE", "EU", "UK", "IRS", "SEC", "CEO", "CFO", "AI", "OK", "YES", "NO",
}

# $TSLA, ticker: TSLA, symbol TSLA, etc.
_TICKER_EXPLICIT_RE = re.compile(
    r"""(?ix)
    (?:\$|\b(?:ticker|symbol)\b\s*[:=]?\s*)
    (?P<sym>[A-Za-z]{1,8}(?:[.\-][A-Za-z]{1,6})?(?:\-[A-Za-z]{1,6})?)
    \b
    """
)

# Plain uppercase tokens (TSLA, AAPL, BRK.B). We only accept length>=2 here; 1-letter tickers
# must be explicit ($F or "ticker F") to reduce false positives.
_TICKER_TOKEN_RE = re.compile(r"\b[A-Z]{2,8}(?:[.\-][A-Z]{1,6})?(?:\-[A-Z]{1,6})?\b")

def _normalize_ticker_symbol(sym: str) -> str | None:
    if not sym or not isinstance(sym, str):
        return None
    s = sym.strip()
    if not s:
        return None
    # Strip leading '$', and normalize class share separators for Yahoo (BRK.B -> BRK-B)
    s = s.lstrip("$").strip().upper().replace(".", "-")
    # Keep only [A-Z0-9-]
    s = re.sub(r"[^A-Z0-9\-]", "", s)
    if not s:
        return None
    # Guard against obvious non-tickers
    if s in _TICKER_STOPWORDS:
        return None
    return s

def _normalize_ticker_symbols(raw: Any) -> list[str]:
    if not raw:
        return []
    if isinstance(raw, str):
        raw_list = [raw]
    elif isinstance(raw, list):
        raw_list = raw
    else:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for x in raw_list:
        s = _normalize_ticker_symbol(str(x)) if x is not None else None
        if not s:
            continue
        if s not in seen:
            out.append(s)
            seen.add(s)
    return out

def _extract_tickers_from_text(text: str) -> list[str]:
    t = (text or "")
    if not t:
        return []
    out: list[str] = []
    seen: set[str] = set()

    # Explicit forms: $TSLA / ticker TSLA / symbol TSLA
    for m in _TICKER_EXPLICIT_RE.finditer(t):
        s = _normalize_ticker_symbol(m.group("sym"))
        if s and s not in seen:
            out.append(s)
            seen.add(s)

    # Plain ALL-CAPS tokens. Only run if the user is already in a stock context elsewhere.
    for m in _TICKER_TOKEN_RE.finditer(t):
        s = _normalize_ticker_symbol(m.group(0))
        if not s:
            continue
        if s not in seen:
            out.append(s)
            seen.add(s)

    return out[:8]

# --- Investment Reporting: ticker resolution helpers ---
# Users often say company names ("Cisco") instead of ticker symbols ("CSCO").
# We resolve simple company-name mentions to tickers via Yahoo Finance search (no auth),
# and use those as an override for the investment_reporting tool.

_YAHOO_SEARCH_CACHE: dict[str, str] = {}

def _yahoo_search_first_symbol(query: str) -> str | None:
    q = (query or "").strip()
    if not q:
        return None
    key = q.lower()
    if key in _YAHOO_SEARCH_CACHE:
        return _YAHOO_SEARCH_CACHE[key]

    try:
        import httpx  # already in requirements via gmail/oauth stack
    except Exception:
        return None

    url = "https://query2.finance.yahoo.com/v1/finance/search"
    params = {
        "q": q,
        "quotesCount": 6,
        "newsCount": 0,
        "listsCount": 0,
        "enableFuzzyQuery": "true",
    }
    headers = {
        # A basic UA avoids occasional 403s from some edge networks.
        "User-Agent": "Mozilla/5.0 (compatible; Vozlia/1.0)",
        "Accept": "application/json",
    }

    try:
        r = httpx.get(url, params=params, headers=headers, timeout=3.0)
        r.raise_for_status()
        data = r.json() if r.content else {}
    except Exception:
        return None

    quotes = data.get("quotes") or []
    best: str | None = None
    for item in quotes:
        try:
            sym = item.get("symbol") or ""
            qt = (item.get("quoteType") or "").upper()
            if not sym:
                continue
            # Prefer common tradable instruments. (Cisco => EQUITY)
            if qt not in ("EQUITY", "ETF", "MUTUALFUND", "INDEX"):
                continue
            norm = _normalize_ticker_symbol(sym)
            if not norm:
                continue
            best = norm
            break
        except Exception:
            continue

    if best:
        _YAHOO_SEARCH_CACHE[key] = best
    return best


_STOCK_ENTITY_RE_1 = re.compile(
    r"\b(?:stock|share)\b[^\n]{0,40}?\b(?:price|quote|ticker)\b[^\n]{0,20}?\b(?:of|for|to)\b\s+(?P<name>[^\n\r\t\.,;!?]{2,80})",
    re.IGNORECASE,
)
_STOCK_ENTITY_RE_2 = re.compile(
    r"\b(?P<name>[A-Za-z][A-Za-z0-9&\-\. ]{1,60}?)(?:'s)?\s+stock\b",
    re.IGNORECASE,
)

def _extract_stock_entity_terms(text: str) -> list[str]:
    """Best-effort extraction of company-ish terms from a 'stock' request."""
    t = (text or "").strip()
    if not t:
        return []

    candidates: list[str] = []

    for rx in (_STOCK_ENTITY_RE_1, _STOCK_ENTITY_RE_2):
        for m in rx.finditer(t):
            name = (m.group("name") or "").strip()
            if not name:
                continue
            candidates.append(name)

    # If we captured a tail like "Cisco and Apple", split it.
    out: list[str] = []
    seen: set[str] = set()
    for c in candidates:
        # Strip trailing connectors / filler
        c = re.sub(r"\b(please|thanks|thank you)\b", "", c, flags=re.IGNORECASE).strip()
        parts = re.split(r"\s+(?:and|or)\s+|\s*,\s*", c)
        for p in parts:
            p = (p or "").strip(" \t\r\n\"'()[]{}")
            if not p:
                continue
            # Avoid pulling in other intents (email, inbox, etc.)
            if re.search(r"\b(email|emails|gmail|inbox|mailbox|summary|summaries)\b", p, flags=re.IGNORECASE):
                continue
            key = p.lower()
            if key not in seen:
                out.append(p)
                seen.add(key)

    return out[:4]


def _invrep_requested_tickers(raw_text: str, llm_plan: dict | None) -> tuple[list[str], str | None]:
    """Return (tickers_override, mode_override).

    Priority:
      1) Router plan tool_args (if provided)
      2) Explicit ticker symbols in the user's text (e.g. $CSCO, ticker CSCO)
      3) Company-name resolution via Yahoo Finance search (e.g. "Cisco" -> CSCO)
    """
    tickers: list[str] = []
    mode: str | None = None

    # (1) Prefer router plan tool_args, if present.
    try:
        if isinstance(llm_plan, dict):
            tool_args = llm_plan.get("tool_args")
            if not isinstance(tool_args, dict):
                # Back-compat in case older router outputs "args"
                tool_args = llm_plan.get("args") if isinstance(llm_plan.get("args"), dict) else {}
            tickers = _normalize_ticker_symbols(tool_args.get("tickers"))
            m = (tool_args.get("mode") or "").strip().lower()
            if m in ("brief", "full"):
                mode = m
    except Exception:
        pass

    # (2) Extract explicit ticker tokens from raw text.
    try:
        extracted = _extract_tickers_from_text(raw_text or "")
        if extracted:
            # preserve order: router tickers first, then extracted additions
            seen = set(tickers)
            for s in extracted:
                if s not in seen:
                    tickers.append(s)
                    seen.add(s)
    except Exception:
        pass

    # (3) If we still have no tickers, try resolving company names to tickers.
    if not tickers:
        try:
            terms = _extract_stock_entity_terms(raw_text or "")
            for term in terms:
                sym = _yahoo_search_first_symbol(term)
                if sym and sym not in tickers:
                    tickers.append(sym)
        except Exception:
            pass

    return (tickers[:8], mode)

def _emit_await_more_enabled() -> bool:
    return (os.getenv("ROUTER_EMIT_AWAIT_MORE", "0") or "").strip().lower() in ("1","true","yes","on")

def _await_more_default_ms() -> int:
    try:
        return int(os.getenv("ROUTER_AWAIT_MORE_DEFAULT_MS", "650") or 650)
    except Exception:
        return 650

def _await_more_max_ms() -> int:
    try:
        return int(os.getenv("ROUTER_AWAIT_MORE_MAX_MS", "1400") or 1400)
    except Exception:
        return 1400

def _should_await_more_fragment(text: str) -> tuple[bool, int, str]:
    """Heuristic: return (should_hold, ms, reason) when a transcript looks incomplete."""
    t = (text or "").strip()
    if not t:
        return (False, 0, "")
    s = t.lower()

    # Never hold on explicit questions or "check in" phrases.
    if "?" in s:
        return (False, 0, "")
    if s.startswith(("hello", "hi", "hey")):
        return (False, 0, "")
    if "are you there" in s or "you there" in s or "still there" in s:
        return (False, 0, "")
    if any(k in s for k in ("how many", "number of", "how often", "times")):
        return (False, 0, "")

    words = [w for w in re.split(r"\s+", s) if w]
    wc = len(words)

    if wc <= 4:
        return (True, _await_more_default_ms(), "short_fragment")
    if s.endswith(("...", ",", "—", "-", ":")):
        return (True, _await_more_default_ms(), "trailing_punct")

    trailing_words = ("when", "what", "why", "how", "because", "so", "and", "or", "but")
    if any(s.endswith(" " + tw) or s.endswith(tw) for tw in trailing_words):
        return (True, _await_more_default_ms(), "trailing_word")

    # leading openers (common split)
    leading_openers = ("can you", "could you", "would you", "tell me", "i was wondering", "i'm thinking", "im thinking")
    if any(s.startswith(lo) for lo in leading_openers) and wc <= 8:
        return (True, _await_more_default_ms(), "leading_opener")

    return (False, 0, "")


def _router_mode() -> str:
    return (os.getenv("LLM_ROUTER_MODE", "off") or "off").strip().lower()

def _router_enabled() -> bool:
    """Return True when the LLM router should be consulted.

    NOTE: In earlier builds, LLM_ROUTER_MODE=tools accidentally bypassed the router entirely
    because this helper only enabled assist/shadow.

    We include 'tools' here so tools-mode can still use the LLM planner.
    """
    return _router_mode() in ("shadow", "assist", "tools")

def _get_router_client():
    global _ROUTER_CLIENT
    if _ROUTER_CLIENT is not None:
        return _ROUTER_CLIENT
    if OpenAI is None:
        return None
    # Prefer OPENAI_API_KEY; fall back to legacy naming if present
    api_key = (os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY") or "").strip()
    if not api_key:
        return None
    try:
        _ROUTER_CLIENT = OpenAI(api_key=api_key)
        return _ROUTER_CLIENT
    except Exception:
        return None
        
def llm_plan_tools_multi(text: str, *, ctx: dict | None = None) -> dict | None:
    """
    TRUE multi-tool planner.
    Returns STRICT JSON with a list of tools to execute in THIS turn.

    Schema:
      {
        "tools": [
          {"tool": "gmail_summary", "args": {"query": "optional string"}},
          {"tool": "investment_reporting", "args": {"tickers": ["CSCO"], "mode": "brief"}},
          {"tool": "memory_lookup", "args": {"query": "...", "time_range": "last_call|last_7_days|last_30_days|all", "scope": "caller"}}
        ]
      }

    Notes:
    - Keep it short and deterministic (voice latency).
    - If user asks multiple things, include multiple tools, ordered logically.
    """
    client = _get_router_client()
    if client is None:
        return None

    model = (os.getenv("OPENAI_ROUTER_MODEL") or "").strip() or "gpt-4o-mini"
    timeout_s = float((os.getenv("OPENAI_ROUTER_TIMEOUT_S", "4.0") or "4.0").strip())
    max_tokens = int((os.getenv("OPENAI_ROUTER_MAX_TOKENS", "220") or "220").strip())
    max_tools = int((os.getenv("LLM_ROUTER_MAX_TOOLS", "3") or "3").strip())

    system = (
        "You are a routing planner for a phone voice assistant.\n"
        "The user may ask for multiple actions in one sentence.\n"
        "Return STRICT JSON only. No markdown.\n"
        f"Return at most {max_tools} tool calls.\n\n"
        "Available tools:\n"
        "1) gmail_summary: user asks to check inbox/emails/unread/summarize emails.\n"
        "   args: {\"query\": \"optional\"}\n"
        "2) investment_reporting: user asks about stocks/tickers/portfolio/market.\n"
        "   args: {\"tickers\": [\"CSCO\"], \"mode\": \"brief\"|\"full\"}\n"
        "   If user names a company, infer a well-known ticker when reasonable.\n"
        "3) memory_lookup: user asks about something previously discussed.\n"
        "   args: {\"query\": \"string\", \"time_range\": \"last_call|last_7_days|last_30_days|all\", \"scope\": \"caller\"}\n\n"
        "Rules:\n"
        "- If user asks for multiple things, include multiple tools.\n"
        "- Preserve user intent; do not invent tools.\n"
        "- If none apply, return {\"tools\": []}.\n"
    )

    user = {"text": text or "", "ctx": (ctx or {})}

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user)},
            ],
            max_tokens=max_tokens,
            temperature=0.0,
            timeout=timeout_s,
        )
        content = (resp.choices[0].message.content or "").strip()
        plan = json.loads(content)
        if not isinstance(plan, dict):
            return None

        tools = plan.get("tools")
        if not isinstance(tools, list):
            plan["tools"] = []
            return plan

        normalized = []
        for item in tools[:max_tools]:
            if not isinstance(item, dict):
                continue
            tool = str(item.get("tool") or "").strip().lower()
            args = item.get("args") if isinstance(item.get("args"), dict) else {}
            if tool not in ("gmail_summary", "investment_reporting", "memory_lookup"):
                continue
            normalized.append({"tool": tool, "args": args})

        plan["tools"] = normalized
        return plan
    except Exception:
        return None

def llm_plan_route(text: str, *, ctx: dict | None = None) -> dict | None:
    """Return a lightweight intent/tool plan (JSON) for routing.

    This is NOT a tool-execution loop yet. It is an intermediate step to:
      - validate LLM intent classification quality on real calls
      - de-risk replacing Engagement Phrase keyword triggers
    """
    client = _get_router_client()
    if client is None:
        return None

    model = (os.getenv("OPENAI_ROUTER_MODEL") or "").strip() or "gpt-4o-mini"
    timeout_s = float((os.getenv("OPENAI_ROUTER_TIMEOUT_S", "4.0") or "4.0").strip())
    max_tokens = int((os.getenv("OPENAI_ROUTER_MAX_TOKENS", "220") or "220").strip())

    # Keep prompts short (voice hot path). Shadow mode should be used sparingly in production.
    system = (
        "You are an intent router for a real-time voice assistant. "
        "Given the user's utterance, choose ONE tool to call next (or 'none'). "
        "Be decisive: if the user asks to check emails/inbox/unread emails, choose gmail_summary. "
        "If the user asks about stocks/tickers/portfolio/market performance, choose investment_reporting. "
        "If the user asks about something previously discussed (last call/last week/before), choose memory_lookup. "
        "IMPORTANT: If you choose investment_reporting and the user mentioned a specific stock/company, "
        "set tool_args.tickers to the ticker symbols (e.g. ['TSLA'], ['AAPL']). "
        "If the user did NOT mention a specific ticker/company and wants a general investment report, "
        "set tool_args.tickers to [] (empty) so the server can use the configured tickers. "
        "If the company name clearly maps to a well-known ticker, infer it (Tesla->TSLA, Apple->AAPL, Microsoft->MSFT). "
        "Return STRICT JSON only (no markdown)."
    )
    tools = [
        {
            "tool": "memory_lookup",
            "when": "User asks about something previously discussed (last call, last week, before).",
            "args": {"query": "string", "time_range": "last_call|last_7_days|last_30_days|all", "scope": "caller"},
        },
        {
            "tool": "gmail_summary",
            "when": "User asks to check emails / inbox / unread emails / latest emails / email summary.",
            "args": {"query": "string optional"},
        },
        {
            "tool": "investment_reporting",
            "when": "User asks to check stocks / tickers / portfolio / market performance / investment report.",
            "args": {"tickers": ["string"], "mode": "brief|full"},
        },
        {
            "tool": "none",
            "when": "Normal conversation or the assistant can answer without tools.",
            "args": {},
        },
    ]

    user = {
        "utterance": (text or "")[:800],
        "context_keys": sorted(list((ctx or {}).keys()))[:20] if isinstance(ctx, dict) else [],
        "available_tools": tools,
        "output_schema": {
            "intent": "short snake_case label",
            "tool": "memory_lookup|gmail_summary|investment_reporting|none",
            "tool_args": "object",
            "confidence": "0.0-1.0",
        },
    }

    try:
        # Use chat.completions for compatibility with existing repo usage.
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user)},
            ],
            max_tokens=max_tokens,
            temperature=0.0,
            timeout=timeout_s,
        )
        content = (resp.choices[0].message.content or "").strip()
        # Best effort strict JSON parse
        plan = json.loads(content)
        if not isinstance(plan, dict):
            return None
        # Basic normalization
        tool = str(plan.get("tool") or "").strip().lower()
        if tool not in ("memory_lookup", "gmail_summary", "investment_reporting", "none"):
            plan["tool"] = "none"
        return plan
    except Exception:
        return None



from services.gmail_service import get_default_gmail_account_id, summarize_gmail_for_assistant


def _looks_like_memory_question(text: str) -> bool:
    """Heuristic: does the user appear to be asking about prior calls / notes / what was said?

    This should be permissive—it's okay to attempt memory recall and return silence if no evidence.
    """
    t = (text or "").strip().lower()
    if not t:
        return False

    # Common temporal / recall cues
    triggers = (
        "previous call",
        "last call",
        "last time",
        "earlier",
        "before",
        "previously",
        "did i mention",
        "did i say",
        "what did i say",
        "what did i tell",
        "what did we talk",
        "we talked about",
        "we spoke about",
        "you remember",
        "do you remember",
        "remind me",
        "from my notes",
        "in my notes",
        "my notes",
        "from our conversation",
        "from our last conversation",
        "what was i talking about",
        "what was i speaking about",
        "what did i mention",
        "what animal",
        "what was the animal",
        "what color",
        "what was the color",
    )
    if any(x in t for x in triggers):
        return True

    # Question-shaped: asking "what/which" about something mentioned, without explicit trigger.
    if "mention" in t and any(q in t for q in ("what", "which", "who")):
        return True

    return False


# -----------------------------
# Investment report navigation
# -----------------------------
_NEXT_PHRASES = {
    "next", "skip", "continue", "go on", "next stock", "next ticker", "next one",
}
_STOP_PHRASES = {
    "stop", "cancel", "done", "end", "that's all", "thats all", "quit", "exit",
}

def _normalize_cmd(text: str) -> str:
    t = (text or "").strip().lower()
    # strip common punctuation
    t = re.sub(r"[^a-z0-9\s']", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def _is_next_cmd(text: str) -> bool:
    t = _normalize_cmd(text)
    return t in _NEXT_PHRASES

def _is_stop_cmd(text: str) -> bool:
    t = _normalize_cmd(text)
    return t in _STOP_PHRASES

def _answer_favorite_color(rows) -> str | None:
    # rows are CallerMemoryEvent ordered newest-first
    # Prefer structured facts, but fall back to parsing raw text to avoid "no hits" for common phrasings.
    fav_re = re.compile(r"\bfavorite\s+color\b(?:\s+\w+){0,4}\s+(?:is|was)\s+([A-Za-z]+)\b", re.I)
    for r in rows or []:
        try:
            # 1) Tags
            tags = getattr(r, "tags_json", None) or []
            if isinstance(tags, list):
                for tag in tags:
                    if isinstance(tag, str) and tag.startswith("fact:favorite_color="):
                        v = tag.split("=", 1)[1].strip()
                        if v:
                            return v

            # 2) data_json facts
            data = getattr(r, "data_json", None) or {}
            if isinstance(data, dict):
                facts = data.get("facts") or {}
                if isinstance(facts, dict) and facts.get("favorite_color"):
                    v = str(facts["favorite_color"]).strip()
                    if v:
                        return v

            # 3) Fallback: parse the stored text
            body = getattr(r, "text", None) or ""
            if isinstance(body, str) and body:
                m = fav_re.search(body)
                if m:
                    v = (m.group(1) or "").strip()
                    if v:
                        return v.lower()
        except Exception:
            continue
    return None


def llm_answer_from_memory(question: str, evidence_lines: list[str]) -> str:
    """LLM compose a fluent answer grounded in retrieved memory notes."""
    client = _get_router_client()
    if client is None:
        # Fail-soft: return a minimal grounded response without hallucinating.
        return ""

    model = (os.getenv("MEMORY_ANSWER_MODEL") or os.getenv("OPENAI_ROUTER_MODEL") or "gpt-4o-mini").strip()
    max_tokens = int(os.getenv("MEMORY_ANSWER_MAX_TOKENS", "220") or 220)
    temperature = float(os.getenv("MEMORY_ANSWER_TEMPERATURE", "0.2") or 0.2)

    # Keep context compact and grounded.
    evidence_txt = "\n".join(evidence_lines[:12])
    sys = (
        "You are Vozlia, a voice assistant. You can answer questions using ONLY the provided MEMORY NOTES. "
        "If the MEMORY NOTES do not contain the answer, return an empty string. "
        "Do NOT ask clarifying questions. Do NOT mention limitations like 'I can't remember' or 'I don't have memory'. "
        "Be concise and grounded."
    )

    user = f"USER QUESTION:\n{question}\n\nMEMORY NOTES (most relevant first):\n{evidence_txt}"

    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": sys},
                {"role": "user", "content": user},
            ],
        )
        out = (resp.choices[0].message.content or "").strip()
        return out or ""
    except Exception:
        logger.exception("MEMORY_ANSWER_LLM_FAIL")
        return ""

def run_assistant_route(
    text: str,
    db: Session,
    current_user: User,
    account_id: str | None = None,
    context: dict | None = None,
) -> dict:
    """
    Routing order (safe / behavior-preserving):
      1) Always run FSM first
      2) If FSM requests backend_call=gmail_summary -> execute existing gmail summary logic (current behavior)
      3) Else if SKILLS_ENGINE_ENABLED=true and FSM did not request a backend call:
           - phrase-match manifest triggers
           - if gmail_summary matched -> call existing gmail summary helper and return it
      4) Otherwise return FSM response
    """
    import time as _time
    debug = (os.getenv("ASSISTANT_ROUTE_DEBUG_LOGS") or "").strip().lower() in ("1","true","yes","on")
    t0 = _time.perf_counter()
    if debug:
        text_snip = (text[:200] + "…") if len(text or "") > 200 else (text or "")
    else:
        text_snip = ""

    # Preserve original user text before any router canonicalization
    raw_user_text = text

    text_l = (text or "").lower()

    ctx_flags = context if isinstance(context, dict) else {}
    is_auto_execute = bool(ctx_flags.get("auto_execute") or ctx_flags.get("auto_exec") or ctx_flags.get("autoExecute"))
    is_offer_followup = bool(ctx_flags.get("offer_followup") or ctx_flags.get("offerFollowup"))
    wants_standby_ack = bool(is_auto_execute or is_offer_followup or bool(ctx_flags.get("forced_skill_id")))

    def _standby_phrase() -> str:
        import random as _random
        forced = (ctx_flags.get("forced_skill_id") if isinstance(ctx_flags, dict) else None) or ""
        if forced == "investment_reporting":
            choices = [
                "One moment — I'm pulling up your stock report now.",
                "Sure — please hold while I fetch the latest prices and news.",
                "Okay — give me a second to pull the market update.",
            ]
        else:
            choices = [
                "One moment — I'm pulling up your email summaries now.",
                "Sure — please hold while I retrieve your latest emails.",
                "Okay — give me a second to fetch your email summaries.",
                "Please stand by while I check your inbox.",
            ]
        return _random.choice(choices)
        return _random.choice(choices)
    # Skill engagement phrases (Gmail Summary)
    force_gmail_summary = False
    try:
        phrases = get_gmail_summary_engagement_phrases(db, current_user)
        force_gmail_summary = any((p or "").strip().lower() in text_l for p in phrases)
    except Exception:
        force_gmail_summary = False

    # Memory engagement phrases (force memory routing even if heuristics miss)

    # Skill engagement phrases (Investment Reporting)
    force_investment_reporting = False
    try:
        inv_cfg = get_investment_reporting_config(db, current_user) or {}
        inv_enabled = bool(inv_cfg.get("enabled", False))
        inv_phrases = inv_cfg.get("engagement_phrases") or []
        if inv_enabled and isinstance(inv_phrases, list):
            force_investment_reporting = any((p or "").strip().lower() in text_l for p in inv_phrases if isinstance(p, str))
    except Exception:
        force_investment_reporting = False

    force_memory = False
    try:
        mphrases = get_memory_engagement_phrases(db, current_user)
        force_memory = any((p or "").strip().lower() in text_l for p in mphrases)
    except Exception:
        force_memory = False

    # ----------------------------
    # (0) LLM router plan
    #   - assist/shadow: legacy single-tool behavior    
    #   - tools: TRUE multi-tool plan (execute multiple)
    # ----------------------------
    llm_plan = None
    llm_tools: list[dict] = []

    if _router_enabled():
        if not _is_junk_transcript(text or ""):
            mode = _router_mode()

            if mode == "tools":
                llm_plan = llm_plan_tools_multi(text or "", ctx=ctx_flags if isinstance(ctx_flags, dict) else None)
                if llm_plan and isinstance(llm_plan.get("tools"), list):
                    llm_tools = list(llm_plan.get("tools") or [])
                    logger.info("LLM_ROUTER_TOOLS plan_tools=%s", [t.get("tool") for t in llm_tools])
            else:
                llm_plan = llm_plan_route(text or "", ctx=ctx_flags if isinstance(ctx_flags, dict) else None)
                if llm_plan:
                    logger.info(
                        "LLM_ROUTER_PLAN mode=%s tool=%s conf=%s intent=%s",
                        mode,
                        llm_plan.get("tool"),
                        llm_plan.get("confidence"),
                        llm_plan.get("intent"),
                    )

                    # Existing assist steering (single-tool)
                    if mode == "assist":
                        tool = str(llm_plan.get("tool") or "").strip().lower()
                        conf = float(llm_plan.get("confidence") or 0.0)
                        min_conf = float(os.getenv("LLM_ROUTER_ASSIST_MIN_CONF", "0.65") or "0.65")
                        if conf >= min_conf:
                            if tool == "gmail_summary":
                                force_gmail_summary = True
                            elif tool == "investment_reporting":
                                force_investment_reporting = True
                            elif tool == "memory_lookup":
                                force_memory = True

                            canonical = _tool_to_canonical_phrase(tool)
                            if canonical and tool != "memory_lookup":
                                logger.info("LLM_ROUTER_ASSIST_CANONICALIZE from=%r to=%r", text, canonical)
                                text = canonical
    else:
        if debug:
            logger.info("LLM_ROUTER_SKIP_JUNK transcript=%r", (text or ""))

        else:
            # Still useful to see this in debug traces, but avoid noisy info logs.
            if debug:
                logger.info("LLM_ROUTER_SKIP_JUNK transcript=%r", (text or ""))

    if debug:
        logger.info(
            "ASSISTANT_ROUTE_START user_id=%s account_id=%s text=%s ctx_keys=%s",
            getattr(current_user, "id", None),
            account_id,
            text_snip,
            sorted(list((context or {}).keys())),
        )

    # Investment Reporting: allow dynamic tickers from the caller's utterance.
    # This supports both explicit tickers ("TSLA") and company names ("Tesla") via the LLM router.
    inv_tickers_override: list[str] = []
    inv_mode_override: str | None = None
    try:
        inv_tickers_override, inv_mode_override = _invrep_requested_tickers(raw_user_text, llm_plan if isinstance(llm_plan, dict) else None)
        if inv_tickers_override:
            logger.info("INVREP_TICKERS_OVERRIDE tickers=%s raw=%r", inv_tickers_override, raw_user_text)
    except Exception:
        inv_tickers_override, inv_mode_override = [], None


    # Memory identifiers (used for session caching on transcript turns)
    tenant_uuid = getattr(current_user, "id", None)
    tenant_id = str(tenant_uuid or "")
    ctx = context or {}
    call_id = None
    if isinstance(ctx, dict):
        call_id = ctx.get("call_sid") or ctx.get("stream_sid") or ctx.get("call_id")
        # Short-term memory toggle: disable session cache without touching hot paths
        try:
            if not shortterm_memory_enabled(db, current_user):
                call_id = None
        except Exception:
            pass

    # Caller identifier (for Postgres TTL cache across calls)
    from_number = None
    if isinstance(ctx, dict):
        from_number = ctx.get("from_number") or ctx.get("from") or ctx.get("From")
    caller_id = normalize_caller_id(from_number)
    gmail_data_fresh = False  # safe default; set True only when Gmail fetch occurs

    # Turn capture helper must be defined before early-return paths (e.g., investment reporting next/stop).
    # Defaults are set here; later logic may update longterm_enabled/capture_turns based on tenant settings.
    longterm_enabled = False
    capture_turns = False

    def _capture_turn(role: str, msg: str | None) -> None:
        if not longterm_enabled or not capture_turns:
            return
        if not tenant_uuid or not caller_id:
            return
        body = (msg or "").strip()
        if not body:
            return
        try:
            record_turn_event(
                db,
                tenant_uuid=str(tenant_uuid),
                caller_id=str(caller_id),
                call_sid=(str(call_id) if call_id else None),
                role=str(role or "user"),
                text=body,
            )
        except Exception:
            logger.exception("TURN_CAPTURE_FAIL tenant_id=%s caller_id=%s role=%s", tenant_id, caller_id, role)
    # If enabled, do NOT persist memory-recall questions or memory-recall answers as turn events.
    exclude_mem_recall_turns = (os.getenv("LONGTERM_MEMORY_EXCLUDE_MEMORY_RECALL_TURNS", "1") or "1").strip().lower() in ("1", "true", "yes", "on")

    def _is_memory_recall_turn(user_text: str) -> bool:
        # Use the same detection that triggers memory routing
        try:
            return bool(force_memory or _looks_like_memory_question(user_text))
        except Exception:
            return False


    # --------------------------------------------
    # Investment Reporting: next/stop navigation
    # If a stock report is in progress, allow the caller to say
    # “next/skip/continue” to advance, or “stop/done” to end.
    # --------------------------------------------
    if call_id and tenant_id:
        try:
            inv_queue = memory.get_handle(tenant_id=tenant_id, call_id=call_id, name="invrep_spoken_queue", default=None)
            inv_idx = memory.get_handle(tenant_id=tenant_id, call_id=call_id, name="invrep_index", default=0)
        except Exception:
            inv_queue, inv_idx = None, 0

        if isinstance(inv_queue, list) and inv_queue:
            if _is_stop_cmd(text):
                # Clear queue
                try:
                    memory.set_handle(tenant_id=tenant_id, call_id=call_id, name="invrep_spoken_queue", value=[], ttl_s=SESSION_MEMORY_TTL_S)
                    memory.set_handle(tenant_id=tenant_id, call_id=call_id, name="invrep_index", value=0, ttl_s=SESSION_MEMORY_TTL_S)
                except Exception:
                    pass
                payload = {"spoken_reply": "Okay — ending the stock report.", "fsm": {"mode": "investment_reporting", "action": "stop"}, "gmail": None}
                _capture_turn("assistant", payload.get("spoken_reply"))
                return payload

            if _is_next_cmd(text):
                try:
                    i2 = int(inv_idx or 0) + 1
                except Exception:
                    i2 = 1
                if i2 >= len(inv_queue):
                    # End of queue
                    try:
                        memory.set_handle(tenant_id=tenant_id, call_id=call_id, name="invrep_spoken_queue", value=[], ttl_s=SESSION_MEMORY_TTL_S)
                        memory.set_handle(tenant_id=tenant_id, call_id=call_id, name="invrep_index", value=0, ttl_s=SESSION_MEMORY_TTL_S)
                    except Exception:
                        pass
                    payload = {"spoken_reply": "That’s the end of your stock report.", "fsm": {"mode": "investment_reporting", "action": "done"}, "gmail": None}
                    _capture_turn("assistant", payload.get("spoken_reply"))
                    return payload

                # Advance
                try:
                    memory.set_handle(tenant_id=tenant_id, call_id=call_id, name="invrep_index", value=i2, ttl_s=SESSION_MEMORY_TTL_S)
                except Exception:
                    pass
                nxt = inv_queue[i2]
                payload = {"spoken_reply": str(nxt), "fsm": {"mode": "investment_reporting", "action": "next", "index": i2}, "gmail": None}
                _capture_turn("assistant", payload.get("spoken_reply"))
                return payload

    # -------------------------
    # LONGTERM MEMORY: config + context + capture turns (Option A)
    # -------------------------
    memory_context = ""
    # Longterm memory toggle: DB overrides, env fallback happens inside settings_service
    try:
        longterm_enabled = longterm_memory_enabled(db, current_user)
    except Exception as e:
        longterm_enabled = False
        if debug:
            logger.exception("LONGTERM_MEMORY_ENABLED_CHECK_FAIL tenant_id=%s err=%s", tenant_id, e)

    capture_turns = (os.getenv("LONGTERM_MEMORY_CAPTURE_TURNS", "1") or "1").strip().lower() in ("1", "true", "yes", "on")
    if debug:
        logger.info(
            "LONGTERM_MEMORY_CONFIG enabled=%s capture_turns=%s tenant_id=%s caller_id=%s call_sid=%s",
            longterm_enabled,
            capture_turns,
            tenant_id,
            caller_id,
            call_id,
        )

        # -------------------------

    # Deterministic history counts: avoid LLM guessing for "how many times ... email summaries" questions.
    # This runs before FSM so it works even when phrasing is inconsistent.
    if caller_id and tenant_uuid:
        try:
            _hc = _maybe_answer_history_count(db, tenant_id=str(tenant_uuid), caller_id=str(caller_id), text=raw_user_text)
        except Exception:
            _hc = None
        if _hc:
            return {"spoken_reply": _hc, "fsm": {"mode": "history_count", "key": "gmail_summary_count"}, "gmail": None}

    
    # ---------------------------------------------------------
    # Layer C (emitter): if the transcript looks incomplete, ask stream.py to hold the floor briefly.
    # Feature-flagged: ROUTER_EMIT_AWAIT_MORE=1
    if _emit_await_more_enabled():
        try:
            hold, ms, reason = _should_await_more_fragment(raw_user_text)
        except Exception:
            hold, ms, reason = (False, 0, "")
        if hold:
            max_ms = _await_more_max_ms()
            if ms > max_ms:
                ms = max_ms
            logger.info("ROUTER_AWAIT_MORE_EMIT ms=%s reason=%s text=%r", ms, reason, (raw_user_text or "")[:120])
            return {
                "spoken_reply": "",
                "fsm": {
                    "mode": "await_more",
                    "await_more": True,
                    "await_more_ms": ms,
                    "await_reason": reason,
                    "suppress_response": True,
                    "suppress_reason": "incomplete_fragment",
                },
                "gmail": None,
            }

# Turn capture (so call summaries include USER questions)
    # -------------------------
    def _wrap_reply(payload: dict) -> dict:
        """Sanitize + capture assistant spoken_reply (if present), then return payload."""

        def _truthy_env(name: str, default: str = "1") -> bool:
            v = (os.getenv(name, default) or default).strip().lower()
            return v in ("1", "true", "yes", "on")

        # If either flag is enabled (default ON), suppress "uncertain / clarify" filler replies.
        _silence_enabled = _truthy_env("VOICE_SILENT_ON_UNCERTAIN", "1") or _truthy_env("VOICE_SILENCE_FSM_FALLBACK", "1")

        _fallback_re = re.compile(
            r"""(
                i['’]?m\s+not\s+sure
                |not\s+sure\s+what\s+you\s+mean
                |not\s+sure\s+what\s+you\s+meant
                |can\s+you\s+rephrase
                |could\s+you\s+rephrase
                |give\s+me\s+(?:a\s+bit\s+)?more\s+detail
                |can\s+you\s+give\s+me\s+(?:a\s+bit\s+)?more\s+detail
                |can\s+you\s+clarify
                |could\s+you\s+clarify
                |i\s+didn['’]?t\s+understand
                |i\s+don['’]?t\s+understand
                |i\s+am\s+confused
                |i['’]?m\s+confused
                |please\s+repeat
                |say\s+that\s+again
            )""",
            re.IGNORECASE | re.VERBOSE,
        )

        def _sanitize_spoken_reply(s: str | None) -> str:
            if not s:
                return ""
            if _silence_enabled and _fallback_re.search(s):
                return ""
            return s

        try:
            if isinstance(payload, dict) and "spoken_reply" in payload:
                payload["spoken_reply"] = _sanitize_spoken_reply(payload.get("spoken_reply"))
            if isinstance(payload, dict):
                _capture_turn("assistant", payload.get("spoken_reply"))
        except Exception:
            logger.exception("TURN_CAPTURE_WRAP_FAIL tenant_id=%s caller_id=%s", tenant_id, caller_id)
        return payload

    # Capture the user turn early so even early returns preserve the question.
    if not (exclude_mem_recall_turns and _is_memory_recall_turn(raw_user_text)):
        _capture_turn("user", raw_user_text)
    else:
        if debug:
            logger.info("TURN_CAPTURE_SKIP role=user reason=memory_recall tenant_id=%s caller_id=%s", tenant_id, caller_id)


# Pull small recent context for prompt grounding (keep short; no hot-path bloat)

    # -------------------------
    # AUTO memory question handling (summaries + vector first)
    # -------------------------
    if longterm_enabled and caller_id and tenant_uuid and (force_memory or _looks_like_memory_question(raw_user_text)):
        from services.memory_controller import (
            parse_memory_query,
            search_memory_events,
            vector_search_call_summaries,
        )

        q_raw = raw_user_text or ""
        qmem = None
        rows: list[Any] = []
        # Deterministic previous-call shortcut (skip vector)
        det_prev_call = os.getenv("LONGTERM_MEMORY_DETERMINISTIC_PREV_CALL", "0").strip() == "1"
        ql = (q_raw or "").lower()
        if det_prev_call and any(p in ql for p in ("previous call", "last call", "prior call", "our last call", "the call before")):
            try:
                exclude_sid = str(call_id) if call_id else ""
                rows = (
                    db.query(CallerMemoryEvent)
                    .filter(CallerMemoryEvent.tenant_id == str(tenant_uuid))
                    .filter(CallerMemoryEvent.caller_id == str(caller_id))
                    .filter(CallerMemoryEvent.skill_key == "call_summary")
                    .filter(CallerMemoryEvent.call_sid.isnot(None))
                    .filter(CallerMemoryEvent.call_sid != exclude_sid)
                    .order_by(CallerMemoryEvent.created_at.desc())
                    .limit(1)
                    .all()
                )
                if debug:
                    picked = rows[0].call_sid if rows else None
                    logger.info(
                        "AUTO_MEMORY_PREV_CALL_PICK tenant_id=%s caller_id=%s picked_call_sid=%s found=%s",
                        tenant_id, caller_id, picked, bool(rows)
                    )
            except Exception:
                logger.exception("AUTO_MEMORY_PREV_CALL_FAIL tenant_id=%s caller_id=%s", tenant_id, caller_id)
                rows = []

        use_turns_bridge = bool(int(os.getenv("LONGTERM_MEMORY_USE_TURNS_FOR_RECALL", "0") or "0"))
        use_vector = os.getenv("VECTOR_MEMORY_ENABLED", "0").strip() == "1"

        try:
            qmem = parse_memory_query(q_raw)
        except Exception:
            logger.exception("AUTO_MEMORY_PARSE_FAIL tenant_id=%s caller_id=%s", tenant_id, caller_id)

        # Optional: LLM timeframe intent → override qmem window (no behavior change unless enabled)
        try:
            use_timeframe_llm = os.getenv("TIMEFRAME_INTENT_LLM", "0").strip() == "1"
            if use_timeframe_llm and qmem is not None:
                from services.timeframe_intent import (
                    looks_like_timeframe,
                    extract_timeframe_intent_llm,
                    resolve_timeframe_intent,
                )

                if looks_like_timeframe(q_raw):
                    tz_name = (os.getenv("APP_TZ") or "America/New_York").strip()
                    spec = extract_timeframe_intent_llm(q_raw, tz_name=tz_name)
                    win = resolve_timeframe_intent(spec, tz_name=tz_name)
                    
                    # Trace the time-intent response (spec + resolved window) in one line
                    if os.getenv("TIMEFRAME_INTENT_TRACE", "0").strip() == "1":
                        try:
                            resolved = None
                            if win:
                                _s, _e, _label = win
                                resolved = {
                                    "label": _label,
                                    "start_utc": _s.isoformat(timespec="seconds"),
                                    "end_utc": _e.isoformat(timespec="seconds"),
                                }
                            trace = {
                                "tenant_id": tenant_id,
                                "caller_id": caller_id,
                                "q_preview": (q_raw or "")[:160],
                                "tz": tz_name,
                                "spec": spec,
                                "resolved_window": resolved,
                            }
                            logger.info("TIMEFRAME_INTENT_TRACE %s", json.dumps(trace, ensure_ascii=False))
                        except Exception:
                            logger.exception(
                                "TIMEFRAME_INTENT_TRACE_FAIL tenant_id=%s caller_id=%s",
                                tenant_id,
                                caller_id,
                            )
                    if win:
                        start_utc, end_utc, label = win
                        qmem.start_ts = start_utc
                        qmem.end_ts = end_utc
                        if debug:
                            logger.info(
                                "AUTO_TIMEFRAME_LLM_OK tenant_id=%s caller_id=%s label=%s start=%s end=%s spec=%s",
                                tenant_id,
                                caller_id,
                                label,
                                start_utc.isoformat(timespec="seconds"),
                                end_utc.isoformat(timespec="seconds"),
                                spec,
                            )
                    else:
                        if debug and spec:
                            logger.info(
                                "AUTO_TIMEFRAME_LLM_NO_WINDOW tenant_id=%s caller_id=%s spec=%s",
                                tenant_id,
                                caller_id,
                                spec,
                            )
        except Exception:
            logger.exception("AUTO_TIMEFRAME_LLM_FAIL tenant_id=%s caller_id=%s", tenant_id, caller_id)

        # 1) Vector search over call_summary rows (best quality)

        # 1) Vector search over call_summary rows (best quality)
        if use_vector and qmem is not None and not rows:
            try:
                from services.embeddings_service import embed_texts
                emb = embed_texts([q_raw])[0]
                rows = vector_search_call_summaries(
                    db,
                    tenant_id=str(tenant_uuid),
                    caller_id=str(caller_id),
                    query_embedding=emb,
                    start_ts=qmem.start_ts,
                    end_ts=qmem.end_ts,
                    limit=int(os.getenv("LONGTERM_MEMORY_VECTOR_LIMIT", "8") or 8),
                )
                if debug:
                    logger.info("AUTO_MEMORY_VECTOR_HITS tenant_id=%s caller_id=%s n=%s", tenant_id, caller_id, len(rows))
            except Exception:
                logger.exception("AUTO_MEMORY_VECTOR_FAIL tenant_id=%s caller_id=%s", tenant_id, caller_id)

        # 2) If no vector hits (or vector disabled), prefer summaries by filtering skill_key='call_summary'
        if not rows and qmem is not None:
            try:
                # First try: summaries only
                qmem2 = qmem
                qmem2.skill_key = "call_summary"
                rows = search_memory_events(
                    db,
                    tenant_id=str(tenant_uuid),
                    caller_id=str(caller_id),
                    q=qmem2,
                    include_turns=False,
                    limit=int(os.getenv("LONGTERM_MEMORY_SUMMARY_LIMIT", "12") or 12),
                )
                if debug:
                    logger.info("AUTO_MEMORY_SUMMARY_HITS tenant_id=%s caller_id=%s n=%s", tenant_id, caller_id, len(rows))
            except Exception:
                logger.exception("AUTO_MEMORY_SUMMARY_SEARCH_FAIL tenant_id=%s caller_id=%s", tenant_id, caller_id)

        # 3) Bridge fallback: include turns (temporary, until call summaries are consistently written)
        if not rows and qmem is not None and use_turns_bridge:
            try:
                rows = search_memory_events(
                    db,
                    tenant_id=str(tenant_uuid),
                    caller_id=str(caller_id),
                    q=qmem,
                    include_turns=True,
                    limit=int(os.getenv("LONGTERM_MEMORY_RECALL_LIMIT", "60") or 60),
                )
                if debug:
                    logger.info("AUTO_MEMORY_TURN_HITS tenant_id=%s caller_id=%s n=%s", tenant_id, caller_id, len(rows))
            except Exception as e:
                logger.exception("AUTO_MEMORY_RECALL_FAIL tenant_id=%s caller_id=%s err=%s", tenant_id, caller_id, e)
                rows = []

        if not rows:
            if debug:
                logger.info("AUTO_MEMORY_NO_HITS tenant_id=%s caller_id=%s q=%s", tenant_id, caller_id, (q_raw or "")[:200])
            payload = {
                "spoken_reply": "I couldn’t find that in my notes yet. Can you tell me roughly when we talked about it?",
                "fsm": {"mode": "memory_recall", "status": "no_hits"},
                "gmail": None,
            }
            if not exclude_mem_recall_turns:
                _capture_turn("assistant", payload.get("spoken_reply"))
            else:
                if debug:
                    logger.info("TURN_CAPTURE_SKIP role=assistant reason=memory_recall tenant_id=%s caller_id=%s", tenant_id, caller_id)
            return payload


        
        # Optional: persist an audit row that records which memory entries were provided
        # to the LLM for this "memory recall" answer. This is intended for debugging
        # via the Admin → Memory Bank UI (search: "memory_recall_audit").
        if os.getenv("MEMORY_AUDIT_DB", "0").strip() == "1":
            try:
                # Local imports so audit is truly optional and doesn't affect cold start.
                # Also ensures failures here never poison the main request DB session.
                from db import SessionLocal
                from fastapi.encoders import jsonable_encoder

                def _iso_dt(dt: Any) -> str | None:
                    try:
                        return dt.isoformat()
                    except Exception:
                        return None

                max_audit_rows = int(os.getenv("MEMORY_AUDIT_MAX_ROWS", "12") or 12)
                include_text = os.getenv("MEMORY_AUDIT_INCLUDE_TEXT", "1").strip() == "1"

                returned: list[dict[str, Any]] = []
                for rr in rows[:max_audit_rows]:
                    rid = getattr(rr, "id", None)
                    item: dict[str, Any] = {
                        "id": str(rid) if rid is not None else None,
                        "created_at": _iso_dt(getattr(rr, "created_at", None)),
                        "call_sid": (str(getattr(rr, "call_sid")) if getattr(rr, "call_sid", None) is not None else None),
                        "skill_key": getattr(rr, "skill_key", None),
                        "kind": getattr(rr, "kind", None),
                    }
                    if include_text:
                        t = (getattr(rr, "text", "") or "").replace("\n", " ").strip()
                        item["text_preview"] = t[:380]
                    returned.append(item)

                audit_payload: dict[str, Any] = {
                    "q": (q_raw or ""),
                    "window": {
                        "start_ts": _iso_dt(getattr(qmem, "start_ts", None)),
                        "end_ts": _iso_dt(getattr(qmem, "end_ts", None)),
                        "skill_key": getattr(qmem, "skill_key", None),
                        "keywords": getattr(qmem, "keywords", None),
                    },
                    "hits": len(rows),
                    "returned": returned,
                }

                audit = CallerMemoryEvent(
                    tenant_id=str(tenant_id or ""),
                    caller_id=str(caller_id or ""),
                    call_sid=(str(call_id) if call_id else None),
                    kind="event",
                    skill_key="memory_recall_audit",
                    text=f"memory_recall_audit hits={len(rows)} q={(q_raw or '')[:220]}",
                    data_json=jsonable_encoder(audit_payload),
                    tags_json=jsonable_encoder(["trace:memory_recall_audit", "trace:llm_context"]),
                )

                # Write audit in a separate DB session so an audit failure can never break /assistant/route.
                audit_db = SessionLocal()
                try:
                    audit_db.add(audit)
                    audit_db.commit()
                finally:
                    audit_db.close()
            except Exception:
                logger.exception("MEMORY_AUDIT_DB_WRITE_FAIL tenant_id=%s caller_id=%s", tenant_id, caller_id)
# Build compact evidence lines for the LLM
        evidence_lines: list[str] = []
        for r in rows[:12]:
            try:
                ts = r.created_at.isoformat()
            except Exception:
                ts = ""
            label = (getattr(r, "skill_key", None) or getattr(r, "kind", None) or "note")
            txt = (getattr(r, "text", "") or "").strip().replace("\n", " ")
            if len(txt) > 420:
                txt = txt[:420] + "…"

            include_sid = os.getenv("MEMORY_EVIDENCE_INCLUDE_CALL_SID", "1").strip() == "1"
            sid = ""
            if include_sid:
                try:
                    sid_val = getattr(r, "call_sid", None)
                    sid = f" sid={sid_val}" if sid_val else ""
                except Exception:
                    sid = ""
            evidence_lines.append(f"- ({ts}){sid} [{label}] {txt}")

        # Optional: trace what we send into the memory-answer LLM (can contain PII; keep OFF by default)
        if os.getenv("MEMORY_EVIDENCE_TRACE", "0").strip() == "1":
            preview = "\n".join(evidence_lines[:5])
            logger.info(
                "MEMORY_EVIDENCE_TRACE tenant_id=%s caller_id=%s q=%r evidence_top5=%s",
                tenant_id,
                caller_id,
                (q_raw or "")[:160],
                preview[:1800],
            )

        # Debug: trace DB rows used for memory answers (single-line JSON).
        if os.getenv("MEMORY_DB_TRACE", "0").strip() == "1":
            try:
                include_text = os.getenv("MEMORY_DB_TRACE_INCLUDE_TEXT", "0").strip() == "1"

                def _iso(dt) -> str | None:
                    try:
                        return dt.isoformat()
                    except Exception:
                        return None

                top_rows = []
                for rr in rows[:8]:
                    item = {
                        "created_at": _iso(getattr(rr, "created_at", None)),
                        "call_sid": getattr(rr, "call_sid", None),
                        "skill_key": getattr(rr, "skill_key", None),
                        "kind": getattr(rr, "kind", None),
                    }
                    if include_text:
                        t = (getattr(rr, "text", "") or "").replace("\n", " ").strip()
                        item["text_preview"] = t[:220]
                    top_rows.append(item)

                trace_obj = {
                    "tenant_id": tenant_id,
                    "caller_id": caller_id,
                    "q_preview": (q_raw or "")[:200],
                    "window": {
                        "start_ts": _iso(getattr(qmem, "start_ts", None)),
                        "end_ts": _iso(getattr(qmem, "end_ts", None)),
                        "skill_key": getattr(qmem, "skill_key", None),
                        "keywords": getattr(qmem, "keywords", None),
                    },
                    "hits": len(rows),
                    "top_rows": top_rows,
                }
                logger.info("MEMORY_DB_TRACE %s", json.dumps(trace_obj, ensure_ascii=False))
            except Exception:
                logger.exception("MEMORY_DB_TRACE_FAIL tenant_id=%s caller_id=%s", tenant_id, caller_id)
        if os.getenv("MEMORY_LLM_ANSWER_ENABLED", "1").strip() == "1":
            spoken = llm_answer_from_memory(q_raw, evidence_lines)
            if not (spoken or "").strip():
                # Hard grounded fallback: read the top note instead of sending empty output.
                top = evidence_lines[0] if evidence_lines else ""
                spoken = top.replace("- ", "", 1) if top else "I couldn’t find that in my notes yet."
                if debug:
                    logger.info("AUTO_MEMORY_EMPTY_LLM_FALLBACK tenant_id=%s caller_id=%s", tenant_id, caller_id)

            if os.getenv("MEMORY_SPOKEN_TRACE", "0").strip() == "1":
                try:
                    logger.info(
                        "MEMORY_SPOKEN_TRACE tenant_id=%s caller_id=%s spoken_len=%s spoken_preview=%r",
                        tenant_id,
                        caller_id,
                        len(spoken or ''),
                        (spoken or '')[:260],
                    )
                except Exception:
                    logger.exception("MEMORY_SPOKEN_TRACE_FAIL tenant_id=%s caller_id=%s", tenant_id, caller_id)

            payload = {
                "spoken_reply": spoken,
                "fsm": {
                    "mode": "memory_recall",
                    "has_evidence": True,
                    "hits": len(rows),
                    "vector": bool(use_vector),
                    "used_turns": bool(use_turns_bridge and any((getattr(r, "kind", "") == "turn") for r in rows)),
                },
                "gmail": None,
            }
            if not exclude_mem_recall_turns:
                _capture_turn("assistant", payload.get("spoken_reply"))
            else:
                if debug:
                    logger.info("TURN_CAPTURE_SKIP role=assistant reason=memory_recall tenant_id=%s caller_id=%s", tenant_id, caller_id)
            return payload


        # Fail-soft fallback: old snippet style (should be rarely used)
        spoken = "Here’s what I found in your notes: " + " ".join([ln.split('] ',1)[-1] for ln in evidence_lines[:3]])
        payload = {
            "spoken_reply": spoken,
            "fsm": {"mode": "memory_recall", "has_evidence": True, "hits": len(rows)},
            "gmail": None,
        }
        if not exclude_mem_recall_turns:
            _capture_turn("assistant", payload.get("spoken_reply"))
        else:
            if debug:
                logger.info("TURN_CAPTURE_SKIP role=assistant reason=memory_recall tenant_id=%s caller_id=%s", tenant_id, caller_id)
        return payload



        # Evidence-first summary (MVP, deterministic)
        # We return a natural sentence + include evidence snippets for debugging / future improvements.
        snippets = []
        for r in rows[:6]:
            ts = (r.created_at.isoformat(timespec="seconds") if getattr(r, "created_at", None) else "")
            snippets.append(f"[{ts}] {r.skill_key}: {r.text}")

        spoken = "Here’s what I found from your history: " + " ".join(snippets[:3])
        if len(snippets) > 3:
            spoken += " …and I found a few more related notes."

        payload = {
            "spoken_reply": spoken,
            "fsm": {
                "mode": "memory_recall",
                "has_evidence": True,
                "hits": len(rows),
                "skill_key": qmem.skill_key,
                "keywords": qmem.keywords,
                "evidence": snippets,
            },
            "gmail": None,
        }
        _capture_turn("assistant", payload.get("spoken_reply"))
        return payload

    fsm = VozliaFSM()

    base_greeting = get_agent_greeting(db, current_user)
    greeting = base_greeting
    # Returning caller preface (previous call summary)
    try:
        # Only attempt at call start: if we have a call_id and no meaningful user text yet,
        # or if the caller just said hello (keeps latency minimal).
        is_call_startish = bool(call_id) and (not (raw_user_text or "").strip() or (raw_user_text or "").strip().lower() in ("hi", "hello", "hey"))
        if is_call_startish:
            from services.greeting_memory import build_prev_call_preface
            pre = build_prev_call_preface(
                db,
                tenant_id=str(tenant_uuid),
                caller_id=str(caller_id),
                current_call_sid=str(call_id) if call_id else None,
            )
            if pre:
                greeting = (pre + " " + (greeting or "")).strip()
                if debug:
                    logger.info("GREETING_PREV_CALL_USED tenant_id=%s caller_id=%s call_sid=%s", tenant_id, caller_id, call_id)
    except Exception:
        logger.exception("GREETING_PREV_CALL_FAIL tenant_id=%s caller_id=%s", tenant_id, caller_id)

    # Optional: append the Gmail Summary skill greeting line to the greeting
    try:
        if gmail_summary_enabled(db, current_user) and gmail_summary_add_to_greeting(db, current_user):
            sk = skill_registry.get("gmail_summary")
            add = (getattr(sk, "greeting", "") or "").strip() if sk else ""
            if add:
                # keep spacing clean
                greeting = (base_greeting or "").strip()
                if greeting:
                    greeting = greeting + (" " if not greeting.endswith(("!", ".", "?")) else " ") + add
                else:
                    greeting = add
    except Exception:
        pass

    fsm.greeting_text = greeting


    fsm_context = context or {}
    if isinstance(fsm_context, dict) and memory_context:
        fsm_context.setdefault("memory_context", memory_context)

    fsm_context.setdefault("user_id", current_user.id)
    fsm_context.setdefault("channel", "phone")

    fsm_result: dict = fsm.handle_utterance(text, context=fsm_context)

    spoken_reply: str = fsm_result.get("spoken_reply") or ""
    backend_call: dict | None = fsm_result.get("backend_call") or None

    # Silence annoying clarifying / uncertain fallbacks (user preference).
    def _truthy_env(name: str, default: str = "1") -> bool:
        v = (os.getenv(name, default) or default).strip().lower()
        return v in ("1", "true", "yes", "on")

    _silence_enabled = _truthy_env("VOICE_SILENT_ON_UNCERTAIN", "1") or _truthy_env("VOICE_SILENCE_FSM_FALLBACK", "1")
    if _silence_enabled and spoken_reply:
        _fallback_re = re.compile(
            r"""(
                i['’]?m\s+not\s+sure
                |not\s+sure\s+what\s+you\s+mean
                |not\s+sure\s+what\s+you\s+meant
                |can\s+you\s+rephrase
                |could\s+you\s+rephrase
                |give\s+me\s+(?:a\s+bit\s+)?more\s+detail
                |can\s+you\s+give\s+me\s+(?:a\s+bit\s+)?more\s+detail
                |can\s+you\s+clarify
                |could\s+you\s+clarify
                |i\s+didn['’]?t\s+understand
                |i\s+don['’]?t\s+understand
                |i\s+am\s+confused
                |i['’]?m\s+confused
                |please\s+repeat
                |say\s+that\s+again
            )""",
            re.IGNORECASE | re.VERBOSE,
        )
        if _fallback_re.search(spoken_reply):
            spoken_reply = ""
            fsm_result["spoken_reply"] = ""
            fsm_result["suppress_response"] = True
    if debug:
        bc_type = backend_call.get('type') if isinstance(backend_call, dict) else None
        logger.info(
            "ASSISTANT_ROUTE_FSM spoken_len=%s backend_call=%s fsm_keys=%s dt_ms=%s",
            len(spoken_reply or ''),
            bc_type,
            sorted(list(fsm_result.keys())) if isinstance(fsm_result, dict) else None,
            int((_time.perf_counter() - t0) * 1000),
        )
    gmail_data: dict | None = None
    if (not backend_call) and force_gmail_summary:
        backend_call = {
            "type": "gmail_summary",
            "params": {"query": "is:unread", "max_results": 20},
        }
        # Keep behavior consistent with FSM email intent reply
        spoken_reply = _standby_phrase() if wants_standby_ack else "Sure, I'll take a quick look at your recent unread emails."
        try:
            fsm_result = dict(fsm_result)
            fsm_result["backend_call"] = backend_call
        except Exception:
            pass



    if (not backend_call) and force_investment_reporting:
        inv_params: dict = {}
        if inv_tickers_override:
            inv_params["tickers"] = inv_tickers_override
            # Default brief for 1 ticker unless router specified otherwise
            if inv_mode_override in ("brief", "full"):
                inv_params["mode"] = inv_mode_override
            else:
                inv_params["mode"] = "brief" if len(inv_tickers_override) <= 1 else "full"

        backend_call = {
            "type": "investment_reporting",
            "params": inv_params,
        }

        if inv_tickers_override:
            spoken_reply = _standby_phrase() if wants_standby_ack else f"One moment — I'm pulling up {', '.join(inv_tickers_override)}."
        else:
            spoken_reply = _standby_phrase() if wants_standby_ack else "Sure — I can give you a stock report."
        try:
            fsm_result = dict(fsm_result)
            fsm_result["backend_call"] = backend_call
        except Exception:
            pass

    # If the FSM (or another mechanism) already set investment_reporting, attach ticker overrides if present.
    if backend_call and backend_call.get("type") == "investment_reporting" and inv_tickers_override:
        try:
            bc_params = backend_call.get("params") if isinstance(backend_call, dict) else {}
            if not isinstance(bc_params, dict):
                bc_params = {}
            if not bc_params.get("tickers"):
                bc_params = dict(bc_params)
                bc_params["tickers"] = inv_tickers_override
                if inv_mode_override in ("brief", "full"):
                    bc_params["mode"] = inv_mode_override
                elif "mode" not in bc_params:
                    bc_params["mode"] = "brief" if len(inv_tickers_override) <= 1 else "full"
                backend_call = dict(backend_call)
                backend_call["params"] = bc_params
                try:
                    fsm_result = dict(fsm_result)
                    fsm_result["backend_call"] = backend_call
                except Exception:
                    pass
        except Exception:
            pass

    # ----------------------------
    # (1) Existing FSM backend call behavior (no change)


    # (1.5) True multi-tool routing (same turn)
    #
    # If the user asks for multiple tools/skills in ONE utterance (e.g. "summarize my emails and what's Cisco's stock price"),
    # run them sequentially and combine their spoken replies into a single assistant turn.
    #
    # IMPORTANT:
    # - We keep this fail-open: if anything errors, we log and fall back to the single-tool routing below.
    # - This runs before the single `backend_call` handlers, because `backend_call` can only represent one action.
    try:
        multi_tools_enabled = os.getenv("VOICE_MULTI_TOOLS_ENABLED", "1").lower() in ("1", "true", "yes", "on")
        if multi_tools_enabled:
            user_text = raw_user_text if "raw_user_text" in locals() else (text or "")
            user_text_l = (user_text or "").lower()

            # Build ordered tool/skill list for this turn
            requested: list[str] = []

            # 1) Forced tool/skill(s) from the voice layer (string or list)
            forced = None
            if isinstance(ctx_flags, dict):
                forced = ctx_flags.get("forced_skill_id") or ctx_flags.get("forced_tool_id")
            if isinstance(forced, str) and forced:
                requested.append(forced)
            elif isinstance(forced, (list, tuple)):
                requested.extend([x for x in forced if isinstance(x, str) and x])

            # 2) Phrase-based skill matches (Skills Engine registry)
            try:
                requested.extend(match_skill_ids(user_text, limit=8))
            except Exception:
                pass

            # 3) Heuristic tool matches (built-ins)
            if re.search(r"\b(email|emails|gmail|inbox|mailbox)\b", user_text_l):
                requested.append("gmail_summary")

            # Investment reporting intent: explicit stock/price language OR ticker(s) present
            try:
                inv_tickers, _inv_mode = _invrep_requested_tickers(user_text, llm_plan)
            except Exception:
                inv_tickers, _inv_mode = ([], None)

            if inv_tickers or re.search(r"\b(stock|stocks|share price|quote|ticker|market cap|price target)\b", user_text_l):
                requested.append("investment_reporting")

            # 4) Router/FSM hints (single-tool planners still contribute one)
            if isinstance(llm_plan, dict):
                tool = llm_plan.get("tool")
                if isinstance(tool, str) and tool:
                    requested.append(tool)
            if isinstance(backend_call, dict):
                bc_type = backend_call.get("type") or backend_call.get("skill_id")
                if isinstance(bc_type, str) and bc_type:
                    requested.append(bc_type)

            # Deduplicate while preserving order
            seen: set[str] = set()
            requested = [sid for sid in requested if sid and not (sid in seen or seen.add(sid))]

            # Filter: keep only supported built-ins or registered Skills Engine skills
            builtin_tools: set[str] = {"gmail_summary", "investment_reporting"}
            requested = [sid for sid in requested if (sid in builtin_tools) or bool(skill_registry.get(sid))]

            # Enforce enable gates:
            # - For built-ins, use their existing settings toggles.
            # - For Skills Engine skills, if the tenant has an explicit enabled flag, respect it;
            #   otherwise default allow (fail-open).
            skill_cfg = {}
            try:
                if tenant_uuid:
                    skill_cfg = get_skills_config(db, current_user) or {}
            except Exception:
                skill_cfg = {}

            def _tool_enabled(sid: str) -> bool:
                if sid == "gmail_summary":
                    return gmail_summary_enabled(db, current_user)
                if sid == "investment_reporting":
                    cfg = get_investment_reporting_config(db, current_user) or {}
                    return bool(cfg.get("enabled", False))
                if isinstance(skill_cfg, dict) and sid in skill_cfg:
                    try:
                        return bool((skill_cfg.get(sid) or {}).get("enabled", False))
                    except Exception:
                        return False
                return True

            requested = [sid for sid in requested if _tool_enabled(sid)]

            # Clamp how many tools we will execute in one turn (default: 2)
            max_tools = int(os.getenv("VOICE_MULTI_TOOLS_MAX") or os.getenv("VOICE_MULTI_SKILLS_MAX") or "2")
            if max_tools < 2:
                max_tools = 2
            requested = requested[:max_tools]

            def _extract_tool_params(tool_id: str) -> dict:
                """Best-effort param extraction for a tool id (router/FSM hints)."""
                params: dict = {}
                # llm_plan args if this tool was selected (single-tool planner)
                if isinstance(llm_plan, dict) and llm_plan.get("tool") == tool_id:
                    a = llm_plan.get("args") or {}
                    if isinstance(a, dict):
                        params.update(a)
                # backend_call params if matches
                if isinstance(backend_call, dict):
                    bc_type = backend_call.get("type") or backend_call.get("skill_id")
                    if bc_type == tool_id and isinstance(backend_call.get("params"), dict):
                        params.update(backend_call.get("params") or {})
                return params

            # Single-tool fallback:
            # If we detected exactly one actionable tool/skill but FSM didn't emit a backend_call,
            # promote it into backend_call so the existing single-skill execution path runs.
            #
            # This prevents "silent" turns when VOICE_SILENT_ON_UNCERTAIN=1 (FSM fallback is suppressed),
            # especially after barge-in where the user may only ask for one thing (e.g., "stock quote for Cisco").
            if (not backend_call) and len(requested) == 1:
                sid0 = requested[0]
                if sid0 in ("gmail_summary", "investment_reporting"):
                    params0 = _extract_tool_params(sid0)
                    backend_call = {"type": sid0, "params": params0}
                    try:
                        fsm_result = dict(fsm_result)
                        fsm_result["backend_call"] = backend_call
                    except Exception:
                        pass
                    # Optional short ack for auto-exec contexts; normal voice turns will speak the final skill output.
                    if wants_standby_ack:
                        spoken_reply = _standby_phrase()
                        try:
                            fsm_result["spoken_reply"] = spoken_reply
                        except Exception:
                            pass

            if len(requested) >= 2:
                logger.info("MULTI_TOOL detected requested=%s", requested)

                combined_spoken_parts: list[str] = []
                merged: dict = {"skills": []}

                for sid in requested:
                    params = _extract_tool_params(sid)

                    if sid == "gmail_summary":
                        if not gmail_summary_enabled(db, current_user):
                            part = "Email summaries are currently turned off in your settings."
                            combined_spoken_parts.append(part)
                            merged["skills"].append({"id": sid, "ok": False, "error": "disabled"})
                            continue

                        account_id_effective = params.get("account_id") or account_id or get_default_gmail_account_id(current_user, db)
                        gmail_query = (params.get("query") or "is:unread").strip()
                        gmail_max_results = int(params.get("max_results") or 20)

                        if not account_id_effective:
                            part = "I don't see a Gmail account connected for you yet."
                            combined_spoken_parts.append(part)
                            merged["skills"].append({"id": sid, "ok": False, "error": "no_account"})
                            continue

                        cache_hash = None
                        cached = None
                        gmail_data: dict = {}

                        # Session (call) cache
                        try:
                            if SESSION_MEMORY_ENABLED and call_id and tenant_id:
                                cache_hash = make_skill_cache_key_hash(
                                    "gmail_summary",
                                    account_id_effective,
                                    gmail_query,
                                    gmail_max_results,
                                )
                                cached = memory.get_cached_skill_result(
                                    tenant_id=tenant_id,
                                    call_id=call_id,
                                    skill_key="gmail_summary",
                                    cache_key_hash=cache_hash,
                                )
                        except Exception:
                            cached = None

                        if cached and isinstance(cached.result, dict) and isinstance((cached.result or {}).get("gmail"), dict):
                            gmail_data = dict((cached.result or {}).get("gmail") or {})
                        else:
                            # Caller-level cache across calls
                            caller_cached = None
                            if CALLER_MEMORY_ENABLED and caller_id and tenant_uuid and cache_hash:
                                try:
                                    caller_cached = get_caller_cache(
                                        db,
                                        tenant_id=tenant_uuid,
                                        caller_id=caller_id,
                                        skill_key="gmail_summary",
                                        cache_key_hash=cache_hash,
                                    )
                                except Exception:
                                    caller_cached = None

                            if caller_cached and isinstance(caller_cached.get("gmail"), dict):
                                gmail_data = dict(caller_cached.get("gmail") or {})
                            else:
                                gmail_data = summarize_gmail_for_assistant(
                                    account_id_effective,
                                    current_user,
                                    db,
                                    max_results=gmail_max_results,
                                    query=gmail_query,
                                ) or {}

                                # Put session cache
                                try:
                                    if SESSION_MEMORY_ENABLED and call_id and tenant_id and cache_hash and isinstance(gmail_data, dict):
                                        memory.put_cached_skill_result(
                                            tenant_id=tenant_id,
                                            call_id=call_id,
                                            skill_key="gmail_summary",
                                            cache_key_hash=cache_hash,
                                            result={"gmail": gmail_data},
                                            ttl_s=SESSION_MEMORY_TTL_S,
                                        )
                                except Exception:
                                    pass

                                # Put caller cache
                                try:
                                    if CALLER_MEMORY_ENABLED and caller_id and tenant_uuid and cache_hash and isinstance(gmail_data, dict):
                                        put_caller_cache(
                                            db,
                                            tenant_id=tenant_uuid,
                                            caller_id=caller_id,
                                            skill_key="gmail_summary",
                                            cache_key_hash=cache_hash,
                                            payload={"gmail": gmail_data},
                                            ttl_s=CALLER_MEMORY_TTL_S,
                                        )
                                except Exception:
                                    pass

                        if isinstance(gmail_data, dict):
                            gmail_data["used_account_id"] = account_id_effective

                        summary = (gmail_data.get("summary") or "").strip() if isinstance(gmail_data, dict) else ""
                        part = summary or "I couldn't generate an email summary right now."
                        combined_spoken_parts.append(part)
                        merged["gmail"] = gmail_data
                        merged["skills"].append({"id": sid, "ok": bool(summary), "query": gmail_query, "max_results": gmail_max_results})

                    elif sid == "investment_reporting":
                        cfg = get_investment_reporting_config(db, current_user) or {}
                        if not bool(cfg.get("enabled", False)):
                            part = "Investment reporting is currently turned off in your settings."
                            combined_spoken_parts.append(part)
                            merged["skills"].append({"id": sid, "ok": False, "error": "disabled"})
                            continue

                        override_tickers = _normalize_ticker_symbols(params.get("tickers")) if isinstance(params, dict) else []
                        tickers = override_tickers or (inv_tickers or []) or get_investment_reporting_tickers(db, current_user)
                        tickers_source = "override" if (override_tickers or inv_tickers) else "configured"

                        if not tickers:
                            part = "I can do stock updates, but no tickers are configured yet."
                            combined_spoken_parts.append(part)
                            merged["skills"].append({"id": sid, "ok": False, "error": "no_tickers"})
                            continue

                        llm_prompt = (cfg.get("llm_prompt") or "").strip()
                        if not llm_prompt:
                            if tickers_source == "override" and len(tickers) == 1:
                                llm_prompt = (
                                    "You are Vozlia, a concise voice assistant delivering a stock update for the requested ticker. "
                                    "Include: current price, previous close, percent change, 1–3 key news items from the last 24 hours, "
                                    "and any analyst upgrades/downgrades or new price targets if available. "
                                    "Keep it under 20 seconds and do not mention saying next."
                                )
                            else:
                                llm_prompt = (
                                    "You are Vozlia, a concise voice assistant delivering a stock report. "
                                    "For each ticker: current price, previous close, percent change, 1–3 key news items from the last 24 hours, "
                                    "and any analyst upgrades/downgrades or new price targets if available. "
                                    "Keep each ticker under 20 seconds. After each ticker say: 'Say next to continue, or stop to end.'"
                                )

                        try:
                            rep = get_investment_reports(tickers, llm_prompt=llm_prompt)
                            logger.info("INVREP_FETCH_OK tickers=%s source=%s", tickers, tickers_source)
                        except Exception as e:
                            logger.exception("INVREP_FETCH_FAIL tickers=%s err=%s", tickers, e)
                            part = "Sorry — I couldn't fetch stock data right now."
                            combined_spoken_parts.append(part)
                            merged["skills"].append({"id": sid, "ok": False, "error": "fetch_failed"})
                            continue

                        spoken_list = rep.get("spoken_reports") or []
                        part = str(spoken_list[0]) if isinstance(spoken_list, list) and spoken_list else "Sorry — I couldn't generate a stock report right now."
                        combined_spoken_parts.append(part)
                        merged["investment_reporting"] = rep
                        merged["skills"].append({"id": sid, "ok": bool(spoken_list), "tickers": tickers, "source": tickers_source})

                        # Seed next/stop queue for investment reporting turns (optional)
                        if call_id and tenant_id and isinstance(spoken_list, list) and spoken_list:
                            try:
                                memory.set_handle(
                                    tenant_id=tenant_id,
                                    call_id=call_id,
                                    name="invrep_spoken_queue",
                                    value=spoken_list,
                                    ttl_s=SESSION_MEMORY_TTL_S,
                                )
                                memory.set_handle(
                                    tenant_id=tenant_id,
                                    call_id=call_id,
                                    name="invrep_index",
                                    value=0,
                                    ttl_s=SESSION_MEMORY_TTL_S,
                                )
                            except Exception:
                                pass

                    else:
                        # Generic Skills Engine skill (best-effort)
                        sk = skill_registry.get(sid)
                        if not sk:
                            continue
                        try:
                            res = execute_skill(
                                sid,
                                text=user_text,
                                db=db,
                                current_user=current_user,
                                account_id=account_id,
                                context=ctx_flags,
                            )
                            part = (res or {}).get("spoken_reply") or ""
                            if part:
                                combined_spoken_parts.append(str(part))
                            merged["skills"].append({"id": sid, "ok": True})
                        except Exception as e:
                            logger.exception("MULTI_TOOL_EXEC_FAIL sid=%s err=%s", sid, e)
                            merged["skills"].append({"id": sid, "ok": False, "error": str(e)})

                spoken_reply = "\n\n".join([p.strip() for p in combined_spoken_parts if p and str(p).strip()]).strip()
                if spoken_reply:
                    payload = {"spoken_reply": spoken_reply, "fsm": fsm_result, "gmail": merged.get("gmail")}
                    payload.update({k: v for k, v in merged.items() if k not in ("gmail",)})
                    _capture_turn("assistant", payload.get("spoken_reply"))
                    return payload
    except Exception as e:
        logger.exception("MULTI_TOOL_FAIL err=%s", e)
    if backend_call and backend_call.get("type") == "gmail_summary":
        # ✅ Skill toggle gate (portal-controlled)
        if not gmail_summary_enabled(db, current_user):
            payload = {
                "spoken_reply": "Email summaries are currently turned off in your settings.",
                "fsm": fsm_result,
                "gmail": None,
            }
            _capture_turn("assistant", payload.get("spoken_reply"))
            return payload

        # If this was initiated via auto-exec/offer-followup, use a neutral standby phrase instead of a confirmation.
        if wants_standby_ack:
            spoken_reply = _standby_phrase()

        params = backend_call.get("params") or {}
        account_id_effective = params.get("account_id") or account_id or get_default_gmail_account_id(current_user, db)
        if debug:
            logger.info(
                "ASSISTANT_GMAIL_CALL params=%s account_id_effective=%s",
                {k: v for k, v in params.items() if k in ('account_id','query','max_results')},
                account_id_effective,
            )


        if not account_id_effective:
            spoken_reply = (
                (spoken_reply.rstrip(". ") + " However, I don't see a Gmail account connected for you yet.")
                if spoken_reply
                else "I tried to check your email, but I don't see a Gmail account connected for you yet."
            )
        else:
            gmail_query = params.get("query")
            gmail_max_results = params.get("max_results", 20)

            gmail_data_fresh = False

            cache_hash = None
            cached = None
            if SESSION_MEMORY_ENABLED and call_id and tenant_id:
                cache_hash = make_skill_cache_key_hash(
                    "gmail_summary",
                    account_id_effective,
                    (gmail_query or ""),
                    int(gmail_max_results),
                )
                cached = memory.get_cached_skill_result(
                    tenant_id=tenant_id,
                    call_id=call_id,
                    skill_key="gmail_summary",
                    cache_key_hash=cache_hash,
                )

            # If session cache miss, try caller-scoped cache (Postgres TTL) across calls
            caller_cached = None
            if (not cached) and CALLER_MEMORY_ENABLED and caller_id and tenant_uuid and cache_hash:
                caller_cached = get_caller_cache(
                    db,
                    tenant_id=tenant_uuid,
                    caller_id=caller_id,
                    skill_key="gmail_summary",
                    cache_key_hash=cache_hash,
                )

            if caller_cached and isinstance(caller_cached.get("gmail"), dict):
                gmail_data = dict(caller_cached.get("gmail") or {})
                if debug:
                    logger.info(
                        "ASSISTANT_GMAIL_CALLER_CACHE_HIT tenant_id=%s caller_id=%s hash=%s used_account_id=%s",
                        tenant_id,
                        caller_id,
                        (cache_hash or "")[:8],
                        account_id_effective,
                    )
                # Warm session cache for the rest of this call
                if SESSION_MEMORY_ENABLED and call_id and tenant_id and cache_hash:
                    memory.put_cached_skill_result(
                        tenant_id=tenant_id,
                        call_id=call_id,
                        skill_key="gmail_summary",
                        cache_key_hash=cache_hash,
                        result={"gmail": gmail_data},
                        ttl_s=SESSION_MEMORY_TTL_S,
                    )
                    msgs = (gmail_data.get("messages") or []) if isinstance(gmail_data, dict) else []
                    ids = [m.get("id") for m in msgs if isinstance(m, dict) and m.get("id")]
                    memory.set_handle(tenant_id=tenant_id, call_id=call_id, name="last_gmail_message_ids", value=ids)
                    memory.set_handle(tenant_id=tenant_id, call_id=call_id, name="last_gmail_messages", value=msgs[:20])

            elif cached:

                gmail_data = dict((cached.result or {}).get("gmail") or {})
                if debug:
                    logger.info(
                        "ASSISTANT_GMAIL_CACHE_HIT tenant_id=%s call_id=%s hash=%s used_account_id=%s",
                        tenant_id,
                        call_id,
                        (cache_hash or "")[:8],
                        account_id_effective,
                    )
            else:
                t_g = _time.perf_counter()
                gmail_data = summarize_gmail_for_assistant(
                    account_id_effective,
                    current_user,
                    db,
                    max_results=gmail_max_results,
                    query=gmail_query,
                )
                gmail_data_fresh = True
                if debug:
                    logger.info(
                        "ASSISTANT_GMAIL_DONE dt_ms=%s got_messages=%s summary_len=%s used_account_id=%s",
                        int((_time.perf_counter() - t_g) * 1000),
                        len((gmail_data.get('messages') or [])) if isinstance(gmail_data, dict) else None,
                        len((gmail_data.get('summary') or '')) if isinstance(gmail_data, dict) else None,
                        account_id_effective,
                    )

                # Cache for follow-ups within this call
                if SESSION_MEMORY_ENABLED and call_id and tenant_id and cache_hash and isinstance(gmail_data, dict):
                    memory.put_cached_skill_result(
                        tenant_id=tenant_id,
                        call_id=call_id,
                        skill_key="gmail_summary",
                        cache_key_hash=cache_hash,
                        result={"gmail": gmail_data},
                        ttl_s=SESSION_MEMORY_TTL_S,
                    )
                    msgs = (gmail_data.get("messages") or []) if isinstance(gmail_data, dict) else []
                    ids = [m.get("id") for m in msgs if isinstance(m, dict) and m.get("id")]
                    memory.set_handle(tenant_id=tenant_id, call_id=call_id, name="last_gmail_message_ids", value=ids)
                    memory.set_handle(tenant_id=tenant_id, call_id=call_id, name="last_gmail_messages", value=msgs[:20])

                    # Also cache across calls for this caller (Postgres TTL), if enabled
                    if CALLER_MEMORY_ENABLED and caller_id and tenant_uuid and cache_hash and isinstance(gmail_data, dict):
                        put_caller_cache(
                            db,
                            tenant_id=tenant_uuid,
                            caller_id=caller_id,
                            skill_key="gmail_summary",
                            cache_key_hash=cache_hash,
                            result_json={"gmail": gmail_data},
                            ttl_s=CALLER_MEMORY_TTL_S,
                        )

            if gmail_data.get("summary"):
                spoken_reply = (
                    (spoken_reply.strip() + " " + gmail_data["summary"].strip()).strip()
                    if spoken_reply
                    else gmail_data["summary"].strip()
                )

            # Long-term memory write (per tenant + caller) - fail-open
            if longterm_enabled and caller_id and tenant_uuid:
                try:
                    record_skill_result(
                        db,
                        tenant_uuid=tenant_uuid,
                        caller_id=caller_id,
                        call_sid=str(call_id) if call_id else None,
                        skill_key="gmail_summary",
                        input_text=raw_user_text,
                        memory_text=(gmail_data.get("summary") or "").strip()[:900],
                        data_json={
                            "account_id": account_id_effective,
                            "query": gmail_query,
                            "max_results": gmail_max_results,
                            "used_account_id": gmail_data.get("used_account_id"),
                            "fresh": bool(gmail_data_fresh),
                        },
                        expires_in_s=int(os.getenv("LONGTERM_MEMORY_SKILL_EVENT_TTL_S", "0") or 0) or None,
                    )
                except Exception:
                    logger.exception("LONGTERM_MEM_RECORD_GMAIL_FAILED")
            gmail_data["used_account_id"] = account_id_effective

        payload = {"spoken_reply": spoken_reply, "fsm": fsm_result, "gmail": gmail_data}
        _capture_turn("assistant", payload.get("spoken_reply"))
        return payload

    # ----------------------------
    # (2) Skills Engine fallback (feature-flagged)
    # Only runs when FSM did NOT request a backend call.
    # ----------------------------
    skills_enabled = (os.getenv("SKILLS_ENGINE_ENABLED") or "").strip().lower() in ("1", "true", "yes", "on")

    if skills_enabled and not backend_call:
        try:
            text_l = (text or "").lower()

            matched_skill = None
            for s in skill_registry.all():
                for phrase in (s.trigger.phrases or []):
                    p = (phrase or "").strip().lower()
                    if p and p in text_l:
                        matched_skill = s
                        break
                if matched_skill:
                    break

            if matched_skill and matched_skill.id == "gmail_summary":
                logger.info("SkillsEngine matched skill=%s text=%r", matched_skill.id, text)

                account_id_effective = account_id or get_default_gmail_account_id(current_user, db)
                if not account_id_effective:
                    spoken_reply = "I tried to check your email, but I don't see a Gmail account connected for you yet."
                    payload = {"spoken_reply": spoken_reply, "fsm": fsm_result, "gmail": {"summary": None, "used_account_id": None}}
                    _capture_turn("assistant", payload.get("spoken_reply"))
                    return payload

                # Session memory cache (SkillsEngine path)
                cache_hash = None
                cached = None
                if SESSION_MEMORY_ENABLED and call_id and tenant_id:
                    cache_hash = make_skill_cache_key_hash("gmail_summary", account_id_effective, "", 20)
                    cached = memory.get_cached_skill_result(
                        tenant_id=tenant_id,
                        call_id=call_id,
                        skill_key="gmail_summary",
                        cache_key_hash=cache_hash,
                    )

                # If session cache miss, try caller-scoped cache (Postgres TTL) across calls
                caller_cached = None
                if (not cached) and CALLER_MEMORY_ENABLED and caller_id and tenant_uuid and cache_hash:
                    caller_cached = get_caller_cache(
                        db,
                        tenant_id=tenant_uuid,
                        caller_id=caller_id,
                        skill_key="gmail_summary",
                        cache_key_hash=cache_hash,
                    )

                if caller_cached and isinstance(caller_cached.get("gmail"), dict):
                    gmail_data = dict(caller_cached.get("gmail") or {})
                    logger.info(
                        "ASSISTANT_GMAIL_CALLER_CACHE_HIT (skills_engine) tenant_id=%s caller_id=%s hash=%s used_account_id=%s",
                        tenant_id,
                        caller_id,
                        (cache_hash or "")[:8],
                        account_id_effective,
                    )
                    # Warm session cache for this call
                    if SESSION_MEMORY_ENABLED and call_id and tenant_id and cache_hash:
                        memory.put_cached_skill_result(
                            tenant_id=tenant_id,
                            call_id=call_id,
                            skill_key="gmail_summary",
                            cache_key_hash=cache_hash,
                            result={"gmail": gmail_data},
                            ttl_s=SESSION_MEMORY_TTL_S,
                        )
                        msgs = (gmail_data.get("messages") or []) if isinstance(gmail_data, dict) else []
                        ids = [mm.get("id") for mm in msgs if isinstance(mm, dict) and mm.get("id")]
                        memory.set_handle(tenant_id=tenant_id, call_id=call_id, name="last_gmail_message_ids", value=ids)
                        memory.set_handle(tenant_id=tenant_id, call_id=call_id, name="last_gmail_messages", value=msgs[:20])

                elif cached:

                    gmail_data = dict((cached.result or {}).get("gmail") or {})
                    logger.info(
                        "ASSISTANT_GMAIL_CACHE_HIT (skills_engine) tenant_id=%s call_id=%s hash=%s used_account_id=%s",
                        tenant_id,
                        call_id,
                        (cache_hash or "")[:8],
                        account_id_effective,
                    )
                else:
                    gmail_data = summarize_gmail_for_assistant(account_id_effective, current_user, db)
                    # Cache for follow-ups within this call
                    if SESSION_MEMORY_ENABLED and call_id and tenant_id and cache_hash and isinstance(gmail_data, dict):
                        memory.put_cached_skill_result(
                            tenant_id=tenant_id,
                            call_id=call_id,
                            skill_key="gmail_summary",
                            cache_key_hash=cache_hash,
                            result={"gmail": gmail_data},
                            ttl_s=SESSION_MEMORY_TTL_S,
                        )
                        msgs = (gmail_data.get("messages") or []) if isinstance(gmail_data, dict) else []
                        ids = [m.get("id") for m in msgs if isinstance(m, dict) and m.get("id")]
                        memory.set_handle(tenant_id=tenant_id, call_id=call_id, name="last_gmail_message_ids", value=ids)
                        memory.set_handle(tenant_id=tenant_id, call_id=call_id, name="last_gmail_messages", value=msgs[:20])
                        # Also cache across calls for this caller (Postgres TTL), if enabled
                        if CALLER_MEMORY_ENABLED and caller_id and tenant_uuid and cache_hash and isinstance(gmail_data, dict):
                            put_caller_cache(
                                db,
                                tenant_id=tenant_uuid,
                                caller_id=caller_id,
                                skill_key="gmail_summary",
                                cache_key_hash=cache_hash,
                                result_json={"gmail": gmail_data},
                                ttl_s=CALLER_MEMORY_TTL_S,
                            )

                gmail_data["used_account_id"] = account_id_effective

                summary = (gmail_data.get("summary") or "").strip()
                if summary:
                    # minimal template support: only {{summary}}
                    template = None
                    if matched_skill.response and matched_skill.response.speak:
                        template = matched_skill.response.speak
                    spoken_reply = (template or "{{summary}}").replace("{{summary}}", summary).strip()
                else:
                    spoken_reply = "I checked your email, but there wasn’t anything to summarize right now."

                logger.info("SkillsEngine executed skill=%s account_id=%s", matched_skill.id, account_id_effective)
                payload = {"spoken_reply": spoken_reply, "fsm": fsm_result, "gmail": gmail_data}
                _capture_turn("assistant", payload.get("spoken_reply"))
                return payload

        except Exception:
            # Never fail the call path because the skills engine had an issue.
            logger.exception("SkillsEngine failed; falling back to FSM result.")

    # ----------------------------
    # (3) Default: return FSM result (no change)
    # ----------------------------

    # -----------------------------
    # Backend Call: Investment Reporting (wiring validation)
    # -----------------------------
    if backend_call and backend_call.get("type") == "investment_reporting":
        cfg = get_investment_reporting_config(db, current_user) or {}
        if not bool(cfg.get("enabled", False)):
            payload = {"spoken_reply": "Investment reporting is currently turned off in your settings.", "fsm": fsm_result, "gmail": None}
            _capture_turn("assistant", payload.get("spoken_reply"))
            return payload

        # Allow per-call override tickers (e.g., user asked: "price on TSLA").
        inv_params = backend_call.get("params") if isinstance(backend_call, dict) else {}
        override_tickers = _normalize_ticker_symbols(inv_params.get("tickers")) if isinstance(inv_params, dict) else []
        tickers = override_tickers or get_investment_reporting_tickers(db, current_user)
        tickers_source = "override" if override_tickers else "configured"

        if not tickers:
            payload = {"spoken_reply": "Investment reporting is enabled, but no tickers are configured yet.", "fsm": fsm_result, "gmail": None}
            _capture_turn("assistant", payload.get("spoken_reply"))
            return payload
        llm_prompt = (cfg.get("llm_prompt") or "").strip()
        if not llm_prompt:
            if tickers_source == "override" and len(tickers) == 1:
                llm_prompt = (
                    "You are Vozlia, a concise voice assistant delivering a stock update for the requested ticker. "
                    "Include: current price, previous close, percent change, 1–3 key news items from the last 24 hours, "
                    "and any analyst upgrades/downgrades or new price targets if available. "
                    "Keep it under 20 seconds and do not mention saying next."
                )
            else:
                llm_prompt = (
                    "You are Vozlia, a concise voice assistant delivering a stock report. "
                    "For each ticker: current price, previous close, percent change, 1–3 key news items from the last 24 hours, "
                    "and any analyst upgrades/downgrades or new price targets if available. "
                    "Keep each ticker under 20 seconds. After each ticker say: 'Say next to continue, or stop to end.'"
                )

        try:
            rep = get_investment_reports(tickers, llm_prompt=llm_prompt)
            logger.info("INVREP_FETCH_OK tickers=%s source=%s", tickers, tickers_source)
        except Exception as e:
            logger.exception("INVREP_FETCH_FAIL tickers=%s err=%s", tickers, e)
            payload = {"spoken_reply": "Sorry — I couldn’t fetch stock data right now.", "fsm": fsm_result, "gmail": None}
            _capture_turn("assistant", payload.get("spoken_reply"))
            return payload

        spoken_list = rep.get("spoken_reports") or []
        if not isinstance(spoken_list, list) or not spoken_list:
            payload = {"spoken_reply": "Sorry — I couldn’t generate a stock report right now.", "fsm": fsm_result, "gmail": None}
            _capture_turn("assistant", payload.get("spoken_reply"))
            return payload

        # Seed session queue so the caller can say “next” to advance tickers.
        if call_id and tenant_id:
            try:
                memory.set_handle(tenant_id=tenant_id, call_id=call_id, name="invrep_spoken_queue", value=spoken_list, ttl_s=SESSION_MEMORY_TTL_S)
                memory.set_handle(tenant_id=tenant_id, call_id=call_id, name="invrep_index", value=0, ttl_s=SESSION_MEMORY_TTL_S)
            except Exception:
                pass

        # Persist skill outcome (summary + structured items) to DB/memory like other skills
        try:
            if tenant_uuid and caller_id:
                record_skill_result(
                    db,
                    tenant_uuid=str(tenant_uuid),
                    caller_id=str(caller_id),
                    call_sid=str(call_id) if call_id else None,
                    skill_key="investment_reporting",
                    input_text=raw_user_text,
                    memory_text=str(spoken_list[0]),
                    data_json={"tickers": tickers, "report": rep},
                )
        except Exception:
            pass

        payload = {"spoken_reply": str(spoken_list[0]), "fsm": fsm_result, "gmail": None}
        _capture_turn("assistant", payload.get("spoken_reply"))
        return payload

    payload = {"spoken_reply": spoken_reply, "fsm": fsm_result, "gmail": gmail_data}
    _capture_turn("assistant", payload.get("spoken_reply"))
    return payload

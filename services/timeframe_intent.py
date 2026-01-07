# services/timeframe_intent.py
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone, date
from typing import Any

from openai import OpenAI
from zoneinfo import ZoneInfo

from core.logging import logger

_TZ_DEFAULT = "America/New_York"


def looks_like_timeframe(text: str) -> bool:
    """Cheap gate to avoid extra LLM calls when no timeframe is implied."""
    t = (text or "").lower()
    return any(
        w in t
        for w in (
            "today",
            "yesterday",
            "this week",
            "last week",
            "a day ago",
            "days ago",
            "couple days",
            "few days",
            "previous call",
            "last call",
            "prior call",
            "earlier",
            "this morning",
            "tonight",
        )
    )


_ROUTER_CLIENT: OpenAI | None = None


def _get_client() -> OpenAI | None:
    global _ROUTER_CLIENT
    if _ROUTER_CLIENT is not None:
        return _ROUTER_CLIENT
    api_key = (os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY") or "").strip()
    if not api_key:
        return None
    try:
        _ROUTER_CLIENT = OpenAI(api_key=api_key)
        return _ROUTER_CLIENT
    except Exception:
        return None


def extract_timeframe_intent_llm(
    text: str,
    *,
    tz_name: str | None = None,
    now_utc: datetime | None = None,
) -> dict[str, Any] | None:
    """Use LLM to classify timeframe phrases into a small, structured spec.

    IMPORTANT: The LLM does NOT compute timestamps.
    Backend resolves spec into exact (start,end) using calendar semantics in tz.
    """
    client = _get_client()
    if client is None:
        return None

    tz_name = (tz_name or os.getenv("APP_TZ") or _TZ_DEFAULT).strip()
    tz = ZoneInfo(tz_name)

    now_utc = now_utc or datetime.now(timezone.utc)
    now_local = now_utc.astimezone(tz)

    model = (os.getenv("OPENAI_TIMEFRAME_MODEL") or "").strip() or (os.getenv("OPENAI_ROUTER_MODEL") or "").strip() or "gpt-4o-mini"
    timeout_s = float((os.getenv("OPENAI_TIMEFRAME_TIMEOUT_S", "1.8") or "1.8").strip())
    max_tokens = int((os.getenv("OPENAI_TIMEFRAME_MAX_TOKENS", "200") or "200").strip())

    system = (
        "You extract a timeframe intent for a voice assistant.\n"
        "Return STRICT JSON only (no markdown), matching this schema:\n"
        "{"
        '"timeframe_type": '
        '"today|yesterday|this_week|last_week|n_days_ago|last_n_days|couple_or_few_days_ago|previous_call|none", '
        '"n": integer_or_null, '
        '"start_date": "YYYY-MM-DD"_or_null, '
        '"end_date": "YYYY-MM-DD"_or_null, '
        '"confidence": 0.0_to_1.0'
        "}\n"
        "Rules:\n"
        "- 'a day ago' == yesterday\n"
        "- 'a couple days ago' / 'a few days ago' -> couple_or_few_days_ago\n"
        "- 'previous call'/'last call' -> previous_call\n"
        "- If no timeframe is implied, use timeframe_type='none' and confidence>=0.7\n"
        "Do NOT invent dates. If user gives explicit dates, put them in start_date/end_date.\n"
    )

    user = {
        "text": text or "",
        "timezone": tz_name,
        "now_local_iso": now_local.isoformat(timespec="seconds"),
        "today_local_date": now_local.date().isoformat(),
    }

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

        # normalize
        tf = str(spec.get("timeframe_type") or "none").strip().lower()
        spec["timeframe_type"] = tf
        try:
            spec["confidence"] = float(spec.get("confidence") or 0.0)
        except Exception:
            spec["confidence"] = 0.0

        n = spec.get("n")
        if n is not None:
            try:
                spec["n"] = int(n)
            except Exception:
                spec["n"] = None

        for k in ("start_date", "end_date"):
            v = spec.get(k)
            if v is None:
                continue
            v = str(v).strip()
            if not re.match(r"^\d{4}-\d{2}-\d{2}$", v):
                spec[k] = None
            else:
                spec[k] = v

        return spec
    except Exception:
        return None


def resolve_timeframe_intent(
    spec: dict[str, Any] | None,
    *,
    tz_name: str | None = None,
    now_utc: datetime | None = None,
) -> tuple[datetime, datetime, str] | None:
    """Resolve a normalized spec into exact UTC (start,end) using NY-calendar semantics."""
    if not spec or not isinstance(spec, dict):
        return None

    conf = float(spec.get("confidence") or 0.0)
    min_conf = float(os.getenv("TIMEFRAME_INTENT_LLM_MIN_CONF", "0.60") or "0.60")
    if conf < min_conf:
        return None

    tf = str(spec.get("timeframe_type") or "none").strip().lower()
    if tf == "none":
        return None

    tz_name = (tz_name or os.getenv("APP_TZ") or _TZ_DEFAULT).strip()
    tz = ZoneInfo(tz_name)

    now_utc = now_utc or datetime.now(timezone.utc)
    now_local = now_utc.astimezone(tz)

    start_of_today = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
    # Monday 00:00 local
    start_of_week = start_of_today - timedelta(days=start_of_today.weekday())

    label = tf

    # Helpers for explicit date ranges
    def _parse_date(s: str) -> date | None:
        try:
            y, m, d = s.split("-")
            return date(int(y), int(m), int(d))
        except Exception:
            return None

    if tf == "today":
        start_local = start_of_today
        end_local = now_local

    elif tf in ("yesterday",):
        start_local = start_of_today - timedelta(days=1)
        end_local = start_of_today

    elif tf == "this_week":
        # Your stated preference: Mon -> previous day (exclude today)
        exclude_today = os.getenv("TIMEFRAME_THIS_WEEK_EXCLUDE_TODAY", "1").strip() == "1"
        start_local = start_of_week
        end_local = start_of_today if exclude_today else now_local

    elif tf == "last_week":
        # Previous Mon 00:00 -> this Mon 00:00
        start_local = start_of_week - timedelta(days=7)
        end_local = start_of_week

    elif tf == "couple_or_few_days_ago":
        # Your rule: interpret as a 48h window ending at today's midnight
        start_local = start_of_today - timedelta(days=2)
        end_local = start_of_today

    elif tf == "n_days_ago":
        n = spec.get("n")
        if not isinstance(n, int) or n < 1 or n > 30:
            return None
        # Specific calendar day window: that day 00:00 -> next day 00:00
        start_local = start_of_today - timedelta(days=n)
        end_local = start_local + timedelta(days=1)

    elif tf == "last_n_days":
        n = spec.get("n")
        if not isinstance(n, int) or n < 1 or n > 30:
            return None
        # Rolling-ish but calendar aligned start at midnight
        start_local = start_of_today - timedelta(days=n)
        end_local = now_local

    elif tf == "previous_call":
        # We don't return a time window here; caller should handle deterministically
        return None

    else:
        # If user gave explicit dates, we support them as inclusive end_date.
        sd = spec.get("start_date")
        ed = spec.get("end_date")
        if isinstance(sd, str) and isinstance(ed, str):
            sdd = _parse_date(sd)
            edd = _parse_date(ed)
            if not sdd or not edd:
                return None
            start_local = datetime(sdd.year, sdd.month, sdd.day, 0, 0, 0, tzinfo=tz)
            end_local = datetime(edd.year, edd.month, edd.day, 0, 0, 0, tzinfo=tz) + timedelta(days=1)
            label = "between_dates"
        else:
            return None

    start_utc = start_local.astimezone(timezone.utc)
    end_utc = end_local.astimezone(timezone.utc)
    return (start_utc, end_utc, label)

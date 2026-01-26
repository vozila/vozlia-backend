# services/metrics_llm_planner.py
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, ValidationError

from core.logging import logger

try:
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore


# ----------------------------
# Pydantic plan schema
# ----------------------------

class TimeframeSpec(BaseModel):
    # Prefer generic presets to avoid code changes as new queries appear.
    preset: Literal[
        "today",
        "yesterday",
        "this_week",
        "last_week",
        "this_month",
        "last_n_days",
    ] = "today"
    n_days: int | None = Field(default=None, ge=1, le=365)
    timezone: str = "America/New_York"


class SkillRef(BaseModel):
    # Either provide an explicit skill_key, or a human skill name / label.
    skill_key: str | None = None
    name: str | None = None


class MetricsPlan(BaseModel):
    # The plan is *only* for deterministic execution. Never put computed numbers here.
    intent: Literal["metrics_run"] = "metrics_run"

    metric: Literal[
        "calls.count",
        "calls.unique_callers",
        "calls.recent_callers",
        "calls.top_callers",
        "skills.invocations",
        "skills.invocations_timeseries",
        "skills.top_invoked",
        "skills.email_summaries_requested",
    ]

    timeframe: TimeframeSpec = Field(default_factory=TimeframeSpec)

    # Optional parameters depending on metric
    skill: SkillRef | None = None
    bucket: Literal["hour", "day"] | None = None
    limit: int | None = Field(default=None, ge=1, le=500)
    by_distinct_callers: bool | None = None

    # Safety / UX
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    needs_clarification: bool = False
    clarifying_question: str | None = None


def _openai_client() -> OpenAI | None:  # type: ignore
    if OpenAI is None:
        return None
    api_key = (os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY") or "").strip()
    if not api_key:
        return None
    try:
        return OpenAI(api_key=api_key)
    except Exception:
        return None


def metrics_llm_enabled() -> bool:
    return (os.getenv("METRICS_LLM_PLANNER_ENABLED", "0") or "0").strip().lower() in ("1", "true", "yes", "on")


def plan_metrics_question(question: str, *, timezone: str | None = None) -> MetricsPlan | None:
    """Use an LLM to turn a free-text question into a validated MetricsPlan.

    Guardrails:
      - Output must validate against MetricsPlan (schema-validated).
      - The plan contains NO computed numbers.
      - If planning fails, callers should fall back to deterministic heuristics.

    IMPORTANT: This planner is allowed to call OpenAI. Do NOT call it from tight audio loops
    unless gated by feature flags.
    """
    if not metrics_llm_enabled():
        return None

    q = (question or "").strip()
    if not q:
        return None

    client = _openai_client()
    if client is None:
        return None

    model = (os.getenv("METRICS_LLM_MODEL") or "").strip() or "gpt-4o-mini"
    timeout_s = float((os.getenv("METRICS_LLM_TIMEOUT_S") or "6.0").strip() or "6.0")
    max_tokens = int((os.getenv("METRICS_LLM_MAX_TOKENS") or "500").strip() or "500")

    tz = (timezone or "").strip() or "America/New_York"

    system = (
        "You are a metrics planner for Vozlia. "
        "Your job is to convert the user's request into a STRICT JSON plan that matches the provided schema. "
        "You MUST NOT answer the question or invent numbers. "
        "Only output JSON. No markdown. No extra keys. "
        "If the request needs a specific skill but none is provided, set needs_clarification=true and ask ONE short question."
    )

    schema = MetricsPlan.model_json_schema()

    user_payload: dict[str, Any] = {
        "question": q,
        "timezone": tz,
        "supported_metrics": [
            "calls.count",
            "calls.unique_callers",
            "calls.recent_callers",
            "calls.top_callers",
            "skills.invocations",
            "skills.invocations_timeseries",
            "skills.top_invoked",
            "skills.email_summaries_requested",
        ],
        "notes": [
            "Use skills.email_summaries_requested when the user asks about 'email summaries requested' (requests + turns + executions).",
            "Use skills.invocations when the user asks 'how many times was <skill> used/invoked/requested/executed'.",
            "For phrases like 'last 30 days' or 'last 90 days', use timeframe.preset='last_n_days' and timeframe.n_days accordingly.",
            "If the user says: skill 'gmail_summary' or skill 'investment_reporting', set skill.skill_key exactly.",
            "If the user references a saved skill by name (example: 'sports digest'), set skill.name to that phrase.",
            "For 'invocations by hour/day', use metric=skills.invocations_timeseries and set bucket to 'hour' or 'day'.",
            "For lists like 'list the 25 most recent callers', use calls.recent_callers and set limit.",
        ],
        "output_schema": schema,
        "examples": [
            {
                "in": "how many calls did we receive today",
                "out": {
                    "intent": "metrics_run",
                    "metric": "calls.count",
                    "timeframe": {"preset": "today", "timezone": "America/New_York"},
                    "confidence": 0.9,
                    "needs_clarification": False,
                },
            },
            {
                "in": "list the 25 most recent callers yesterday",
                "out": {
                    "intent": "metrics_run",
                    "metric": "calls.recent_callers",
                    "timeframe": {"preset": "yesterday", "timezone": "America/New_York"},
                    "limit": 25,
                    "confidence": 0.85,
                    "needs_clarification": False,
                },
            },
            {
                "in": "How many times was skill 'gmail_summary' yesterday",
                "out": {
                    "intent": "metrics_run",
                    "metric": "skills.invocations",
                    "timeframe": {"preset": "yesterday", "timezone": "America/New_York"},
                    "skill": {"skill_key": "gmail_summary"},
                    "confidence": 0.9,
                    "needs_clarification": False,
                },
            },
            {
                "in": "Show skill 'investment_reporting' invocations by hour last 30 days",
                "out": {
                    "intent": "metrics_run",
                    "metric": "skills.invocations_timeseries",
                    "bucket": "hour",
                    "timeframe": {"preset": "last_n_days", "n_days": 30, "timezone": "America/New_York"},
                    "skill": {"skill_key": "investment_reporting"},
                    "confidence": 0.9,
                    "needs_clarification": False,
                },
            },
        ],
    }

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user_payload)},
            ],
            max_tokens=max_tokens,
            temperature=0.0,
            timeout=timeout_s,
        )
        content = (resp.choices[0].message.content or "").strip()
        plan_obj = json.loads(content)
        plan = MetricsPlan.model_validate(plan_obj)
        # Ensure timezone is always set
        if plan.timeframe and not plan.timeframe.timezone:
            plan.timeframe.timezone = tz
        # Very light logging (no PII)
        logger.info("METRICS_LLM_PLAN metric=%s preset=%s conf=%s", plan.metric, plan.timeframe.preset, plan.confidence)
        return plan
    except ValidationError as ve:
        logger.warning("METRICS_LLM_PLAN_INVALID err=%s", str(ve)[:300])
        return None
    except Exception as e:
        logger.warning("METRICS_LLM_PLAN_FAIL err=%s", str(e)[:300])
        return None

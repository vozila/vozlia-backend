# skills/engine.py
from __future__ import annotations

import os
from typing import Any, Dict, Optional

from core.logging import logger
from skills.registry import skill_registry
from services.gmail_service import get_default_gmail_account_id, summarize_gmail_for_assistant
from services.investment_service import get_investment_reports
from services.settings_service import get_skills_config


def skills_engine_enabled() -> bool:
    return (os.getenv("SKILLS_ENGINE_ENABLED") or "").strip().lower() in ("1", "true", "yes", "on")


def match_skill_id(text: str) -> Optional[str]:
    """
    Very simple phrase matcher:
    - If any trigger.phrases substring matches the utterance, return the first matching skill id.
    """
    text_l = (text or "").lower()
    if not text_l:
        return None

    for skill in skill_registry.all():
        for phrase in (skill.trigger.phrases or []):
            p = (phrase or "").strip().lower()
            if p and p in text_l:
                return skill.id
    return None


def render_template(template: str, vars: Dict[str, Any]) -> str:
    """
    Minimal template renderer supporting {{key}}.
    """
    out = template or ""
    for k, v in (vars or {}).items():
        out = out.replace("{{" + str(k) + "}}", str(v if v is not None else ""))
    return out


def execute_skill(
    skill_id: str,
    *,
    text: str,
    db,
    current_user,
    account_id: str | None = None,
    context: dict | None = None,
) -> Dict[str, Any]:
    """
    Executes a skill. Step 2 only supports:
      - api.type == "internal"
      - handler == "gmail.get_summaries"
    """
    skill = skill_registry.get(skill_id)
    skills_cfg = get_skills_config(db, current_user) or {}
    skill_cfg = skills_cfg.get(skill_id) if isinstance(skills_cfg, dict) else {}
    if not skill:
        return {"ok": False, "error": f"Unknown skill: {skill_id}"}

    if skill.api.type != "internal":
        return {"ok": False, "error": f"Unsupported api.type: {skill.api.type}"}

    if skill.api.handler not in ("gmail.get_summaries", "investment.get_reports"):
        return {"ok": False, "error": f"Unsupported handler: {skill.api.handler}"}

    # Handler: Investment Reporting (Yahoo Finance)
    if skill.api.handler == "investment.get_reports":
        # tickers come from params first, else from per-skill config
        params = (context or {}).get("params") if isinstance(context, dict) else None
        if not isinstance(params, dict):
            params = {}
        tickers = params.get("tickers") or (skill_cfg or {}).get("tickers") or (skill_cfg or {}).get("tickers_csv") or ""
        llm_prompt = (skill_cfg or {}).get("llm_prompt") or ""
        report = get_investment_reports(tickers=tickers, llm_prompt=str(llm_prompt))
        spoken_reports = report.get("spoken_reports") or []
        # combine or return first, letting caller route manage pagination
        combined = " ".join([s for s in spoken_reports if isinstance(s, str) and s.strip()])
        return {
            "ok": True,
            "spoken_reply": combined or "I couldn't fetch stock data right now.",
            "investment_reporting": report,
        }

    # Handler: Gmail Summary
    # Use existing Gmail summary code (no behavior change beyond routing to it)
    account_id_effective = account_id or get_default_gmail_account_id(current_user, db)
    if not account_id_effective:
        return {
            "ok": True,
            "spoken_reply": "I tried to check your email, but I don't see a Gmail account connected for you yet.",
            "gmail": {"summary": None, "used_account_id": None},
        }

    gmail_data = summarize_gmail_for_assistant(account_id_effective, current_user, db)
    summary = (gmail_data.get("summary") or "").strip()

    # If manifest has a speak template, use it; else just speak the summary.
    speak_template = (skill.response.speak if skill.response else None) or "{{summary}}"
    spoken_reply = render_template(speak_template, {"summary": summary}).strip() if summary else ""

    gmail_data["used_account_id"] = account_id_effective

    logger.info("SkillsEngine executed %s (gmail_summary) account_id=%s", skill_id, account_id_effective)
    return {"ok": True, "spoken_reply": spoken_reply, "gmail": gmail_data}

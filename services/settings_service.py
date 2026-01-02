
# services/settings_service.py
from __future__ import annotations

import os
from typing import Any, Optional

from sqlalchemy.orm import Session

from models import User, UserSetting

DEFAULT_GMAIL_SUMMARY_LLM_PROMPT = (
    "You are Vozlia. Given email metadata (subject, sender, snippet, date), "
    "produce a VERY short spoken-style summary (1–3 sentences). "
    "Do NOT read email addresses or long codes out loud."
)

DEFAULTS: dict[str, dict[str, Any]] = {
    "agent_greeting": {"text": "Hello! How can I assist you today?"},
    "gmail_summary_enabled": {"enabled": True},
    "gmail_account_id": {"account_id": "d8c8cd99-c9bc-4e8c-a560-d220782665a1"},
    "realtime_prompt_addendum": {
        "text": (
            "CALL OPENING RULE (FIRST UTTERANCE ONLY): "
            "Greet the caller and introduce yourself as Vozlia in one short sentence. "
            'Example: "Hello, you\'re speaking with Vozlia — how can I help today?" '
            "Do not repeat the brand intro after the first utterance."
        )
    },
    # NEW (modular): per-skill configuration
    "skills_config": {
        "skills": {
            "gmail_summary": {
                "enabled": True,
                "add_to_greeting": False,
                "engagement_phrases": ["email summaries"],
                "llm_prompt": DEFAULT_GMAIL_SUMMARY_LLM_PROMPT,
            }
        }
    },
    "skills_priority_order": {"order": ["gmail_summary", "memory", "sms", "calendar", "web_search", "weather", "investment_reporting"]},
}
DEFAULT_INVESTMENT_REPORTING_LLM_PROMPT = (
    "Summarize stock news in plain language for a caller. "
    "If there is no major news in the last 24 hours, say so. "
    "If an analyst upgrade/downgrade is provided, mention it briefly. "
    "Never mention URLs."
)



def _get_setting_row(db: Session, user: User, key: str) -> Optional[UserSetting]:
    return db.query(UserSetting).filter(UserSetting.user_id == user.id, UserSetting.key == key).first()


def get_setting(db: Session, user: User, key: str) -> dict:
    row = _get_setting_row(db, user, key)
    if row and isinstance(row.value, dict):
        return row.value
    return DEFAULTS.get(key, {})


def set_setting(db: Session, user: User, key: str, value: dict) -> None:
    row = _get_setting_row(db, user, key)
    if row:
        row.value = value or {}
    else:
        row = UserSetting(user_id=user.id, key=key, value=value or {})
        db.add(row)


# -----------------------------
# Core settings
# -----------------------------
def get_agent_greeting(db: Session, user: User) -> str:
    v = get_setting(db, user, "agent_greeting")
    t = (v or {}).get("text")
    if isinstance(t, str) and t.strip():
        return t.strip()
    return str(DEFAULTS["agent_greeting"]["text"])


def gmail_summary_enabled(db: Session, user: User) -> bool:
    v = get_setting(db, user, "gmail_summary_enabled")
    enabled = (v or {}).get("enabled")
    return bool(True if enabled is None else enabled)


def get_selected_gmail_account_id(db: Session, user: User) -> Optional[str]:
    v = get_setting(db, user, "gmail_account_id")
    account_id = (v or {}).get("account_id")
    if isinstance(account_id, str) and account_id.strip():
        return account_id.strip()
    return None


def get_realtime_prompt_addendum(db: Session, user: User) -> str:
    v = get_setting(db, user, "realtime_prompt_addendum")
    t = (v or {}).get("text")
    if isinstance(t, str) and t.strip():
        return t.strip()
    return str(DEFAULTS["realtime_prompt_addendum"]["text"])


# -----------------------------
# NEW: Skill config helpers
# -----------------------------
def get_skills_config(db: Session, user: User) -> dict[str, dict]:
    v = get_setting(db, user, "skills_config")
    skills = (v or {}).get("skills")
    if isinstance(skills, dict):
        out: dict[str, dict] = {}
        for k, cfg in skills.items():
            if isinstance(k, str) and isinstance(cfg, dict):
                out[k] = cfg
        return out
    return dict(DEFAULTS["skills_config"]["skills"])



def get_skills_priority_order(db: Session, user: User) -> list[str]:
    """Returns an ordered list of skill_ids used to organize announcement + auto-exec selection."""
    v = get_setting(db, user, "skills_priority_order")
    order = (v or {}).get("order")
    if isinstance(order, list):
        cleaned: list[str] = []
        for s in order:
            if isinstance(s, str) and s.strip():
                cleaned.append(s.strip())
        if cleaned:
            return cleaned
    # default fallback
    d = DEFAULTS.get("skills_priority_order", {}).get("order")
    if isinstance(d, list):
        return [s for s in d if isinstance(s, str) and s.strip()]
    return []


def set_skills_priority_order(db: Session, user: User, order: list[str]) -> None:
    cleaned: list[str] = []
    for s in (order or []):
        if isinstance(s, str) and s.strip():
            cleaned.append(s.strip())
    set_setting(db, user, "skills_priority_order", {"order": cleaned})


def get_skill_config(db: Session, user: User, skill_id: str) -> dict:
    skills = get_skills_config(db, user)
    base = dict(DEFAULTS["skills_config"]["skills"].get(skill_id, {}))
    override = skills.get(skill_id)
    if isinstance(override, dict):
        base.update(override)
    return base



def get_investment_reporting_config(db: Session, user: User) -> dict:
    cfg = get_skill_config(db, user, "investment_reporting")
    return cfg if isinstance(cfg, dict) else {}


def get_investment_reporting_tickers(db: Session, user: User) -> list[str]:
    cfg = get_investment_reporting_config(db, user)
    raw = cfg.get("tickers") or cfg.get("tickers_csv") or ""
    if isinstance(raw, list):
        out = []
        for t in raw:
            s = str(t).strip().upper()
            if s:
                out.append(s)
        return out
    parts = []
    for chunk in str(raw).replace("\n", ",").split(","):
        s = chunk.strip().upper()
        if s:
            parts.append(s)
    seen=set()
    out=[]
    for s in parts:
        if s not in seen:
            seen.add(s); out.append(s)
    return out
def patch_skill_config(db: Session, user: User, skill_id: str, patch: dict) -> dict[str, dict]:
    """Patch a single skill's config. Accepts both snake_case and portal camelCase keys."""
    current = get_skills_config(db, user)
    base = dict(current.get(skill_id) or DEFAULTS["skills_config"]["skills"].get(skill_id, {}))

    # Normalize engagement prompt
    if "engagementPrompt" in patch and "engagement_phrases" not in patch:
        raw = patch.get("engagementPrompt")
        if isinstance(raw, str):
            phrases = []
            for chunk in raw.replace(",", "\n").splitlines():
                s = chunk.strip()
                if s:
                    phrases.append(s)
            patch["engagement_phrases"] = phrases

    # Normalize tickers
    if "tickers" in patch:
        raw = patch.get("tickers")
        if isinstance(raw, str):
            parts = []
            for chunk in raw.replace("\n", ",").split(","):
                s = chunk.strip().upper()
                if s:
                    parts.append(s)
            # de-dupe preserve order
            seen=set()
            tickers=[]
            for s in parts:
                if s not in seen:
                    seen.add(s); tickers.append(s)
            patch["tickers"] = tickers
    if "tickersCsv" in patch and "tickers" not in patch:
        patch["tickers"] = patch.get("tickersCsv")

    for k in (
        "enabled",
        "add_to_greeting",
        "auto_execute_after_greeting",
        "engagement_phrases",
        "llm_prompt",
        "tickers",
    ):
        if k in patch:
            base[k] = patch[k]

    current[skill_id] = base
    set_setting(db, user, "skills_config", {"skills": current})
    return current



def get_gmail_summary_llm_prompt(db: Session, user: User) -> str:
    cfg = get_skill_config(db, user, "gmail_summary")
    p = (cfg or {}).get("llm_prompt")
    if isinstance(p, str) and p.strip():
        return p.strip()
    return DEFAULT_GMAIL_SUMMARY_LLM_PROMPT


def get_gmail_summary_engagement_phrases(db: Session, user: User) -> list[str]:
    cfg = get_skill_config(db, user, "gmail_summary")
    phrases = (cfg or {}).get("engagement_phrases")
    if isinstance(phrases, list):
        return [str(x).strip() for x in phrases if str(x).strip()]
    return []


def gmail_summary_add_to_greeting(db: Session, user: User) -> bool:
    cfg = get_skill_config(db, user, "gmail_summary")
    return bool((cfg or {}).get("add_to_greeting") or False)


# -----------------------------
# NEW: Memory toggles (DB overrides, env fallback when unset)
# -----------------------------
def shortterm_memory_enabled(db: Session, user: User) -> bool:
    row = _get_setting_row(db, user, "shortterm_memory_enabled")
    if row and isinstance(row.value, dict) and "enabled" in row.value:
        return bool(row.value.get("enabled"))
    # env fallback preserves current behavior (memory_facade defaults to ON)
    return (os.getenv("SESSION_MEMORY_ENABLED") or "1").strip().lower() in ("1", "true", "yes", "on")


def longterm_memory_enabled(db: Session, user: User) -> bool:
    row = _get_setting_row(db, user, "longterm_memory_enabled")
    if row and isinstance(row.value, dict) and "enabled" in row.value:
        return bool(row.value.get("enabled"))
    # env fallback preserves current behavior (longterm defaults to OFF)
    return (os.getenv("LONGTERM_MEMORY_ENABLED_DEFAULT") or "0").strip().lower() in ("1", "true", "yes", "on")


def get_memory_engagement_phrases(db: Session, user: User) -> list[str]:
    v = get_setting(db, user, "memory_engagement_phrases")
    phrases = (v or {}).get("phrases")
    if isinstance(phrases, list):
        return [str(x).strip() for x in phrases if str(x).strip()]
    return []


# -----------------------------
# Legacy class wrapper (kept for compatibility)
# -----------------------------
from db import SessionLocal  # noqa: E402
from services.user_service import get_or_create_primary_user  # noqa: E402


class SettingsService:
    def get_settings(self) -> dict:
        db = SessionLocal()
        try:
            user = get_or_create_primary_user(db)
            return {
                "agent_greeting": get_agent_greeting(db, user),
                "gmail_summary_enabled": gmail_summary_enabled(db, user),
                "gmail_account_id": (get_selected_gmail_account_id(db, user) or DEFAULTS["gmail_account_id"]["account_id"]),
                "realtime_prompt_addendum": get_realtime_prompt_addendum(db, user),
                "skills_config": get_skills_config(db, user),
                "shortterm_memory_enabled": shortterm_memory_enabled(db, user),
                "longterm_memory_enabled": longterm_memory_enabled(db, user),
                "memory_engagement_phrases": get_memory_engagement_phrases(db, user),
            }
        finally:
            db.close()

    def patch_settings(self, patch: dict) -> dict:
        db = SessionLocal()
        try:
            user = get_or_create_primary_user(db)

            if "agent_greeting" in patch and patch["agent_greeting"] is not None:
                set_setting(db, user, "agent_greeting", {"text": str(patch["agent_greeting"]).strip()})

            if "gmail_summary_enabled" in patch:
                set_setting(db, user, "gmail_summary_enabled", {"enabled": bool(patch["gmail_summary_enabled"])})

            if "gmail_account_id" in patch:
                v = patch["gmail_account_id"]
                if v is None:
                    set_setting(db, user, "gmail_account_id", {"account_id": DEFAULTS["gmail_account_id"]["account_id"]})
                else:
                    set_setting(db, user, "gmail_account_id", {"account_id": str(v).strip()})

            if "realtime_prompt_addendum" in patch and patch["realtime_prompt_addendum"] is not None:
                set_setting(db, user, "realtime_prompt_addendum", {"text": str(patch["realtime_prompt_addendum"]).strip()})

            if "skills_config" in patch and isinstance(patch["skills_config"], dict):
                for sid, cfg in patch["skills_config"].items():
                    if isinstance(sid, str) and isinstance(cfg, dict):
                        patch_skill_config(db, user, sid, cfg)

            if "shortterm_memory_enabled" in patch:
                set_setting(db, user, "shortterm_memory_enabled", {"enabled": bool(patch["shortterm_memory_enabled"])})

            if "longterm_memory_enabled" in patch:
                set_setting(db, user, "longterm_memory_enabled", {"enabled": bool(patch["longterm_memory_enabled"])})

            if "memory_engagement_phrases" in patch:
                phrases = patch.get("memory_engagement_phrases")
                cleaned = [str(x).strip() for x in phrases if str(x).strip()] if isinstance(phrases, list) else []
                set_setting(db, user, "memory_engagement_phrases", {"phrases": cleaned})

            db.commit()
            return self.get_settings()
        finally:
            db.close()


settings_service = SettingsService()

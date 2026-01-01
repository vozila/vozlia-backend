# services/settings_service.py
from __future__ import annotations

import os
from typing import Any, Optional

from sqlalchemy.orm import Session

import time
import httpx

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
                "auto_execute_after_greeting": False,
                "engagement_phrases": ["email summaries"],
                "llm_prompt": DEFAULT_GMAIL_SUMMARY_LLM_PROMPT,
            }
        }
    },
}


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
    data = _get_control_plane_settings_cached()
    if isinstance(data, dict):
        v = data.get("agent_greeting")
        if isinstance(v, str) and v.strip():
            return v.strip()

    v = get_setting(db, user, "agent_greeting")
    t = (v or {}).get("text")
    if isinstance(t, str) and t.strip():
        return t.strip()
    return str(DEFAULTS["agent_greeting"]["text"])



def gmail_summary_enabled(db: Session, user: User) -> bool:
    # Prefer Control Plane (single source of truth for Portal)
    data = _get_control_plane_settings_cached()
    if isinstance(data, dict) and "gmail_summary_enabled" in data:
        return bool(data.get("gmail_summary_enabled"))

    v = get_setting(db, user, "gmail_summary_enabled")
    enabled = (v or {}).get("enabled")
    return bool(True if enabled is None else enabled)



def get_gmail_summary_engagement_phrases(db: Session, user: User) -> list[str]:
    cfg = get_skill_config(db, user, "gmail_summary")
    phrases = (cfg or {}).get("engagement_phrases")
    if isinstance(phrases, list):
        return [str(x).strip() for x in phrases if str(x).strip()]
    return list(DEFAULTS["skills_config"]["skills"]["gmail_summary"]["engagement_phrases"])



def gmail_summary_add_to_greeting(db: Session, user: User) -> bool:
    cfg = get_skill_config(db, user, "gmail_summary")
    return bool((cfg or {}).get("add_to_greeting") or False)


def gmail_summary_auto_execute_after_greeting(db: Session, user: User) -> bool:
    cfg = get_skill_config(db, user, "gmail_summary")
    return bool((cfg or {}).get("auto_execute_after_greeting") or False)


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
    data = _get_control_plane_settings_cached()
    if isinstance(data, dict):
        phrases = data.get("memory_engagement_phrases")
        if isinstance(phrases, list):
            cleaned = [str(x).strip() for x in phrases if str(x).strip()]
            return cleaned

    v = get_setting(db, user, "memory_engagement_phrases")
    phrases = (v or {}).get("phrases")
    if isinstance(phrases, list):
        cleaned = [str(x).strip() for x in phrases if str(x).strip()]
        return cleaned
    return list(DEFAULTS["memory_engagement_phrases"]["phrases"])



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
# -----------------------------
# Optional: Control Plane settings bridge (Portal -> Control Plane -> Backend)
# -----------------------------
CONTROL_PLANE_URL = (os.getenv("CONTROL_PLANE_URL") or "").strip().rstrip("/")
CONTROL_PLANE_ADMIN_KEY = (os.getenv("CONTROL_PLANE_ADMIN_KEY") or "").strip()
CONTROL_PLANE_SETTINGS_TTL_S = float((os.getenv("CONTROL_PLANE_SETTINGS_TTL_S") or "5").strip() or "5")

_cp_cache: dict[str, Any] = {"ts": 0.0, "data": None}

def _get_control_plane_settings_cached() -> Optional[dict]:
    if not CONTROL_PLANE_URL or not CONTROL_PLANE_ADMIN_KEY:
        return None
    now = time.time()
    ts = float(_cp_cache.get("ts") or 0.0)
    if _cp_cache.get("data") is not None and (now - ts) < CONTROL_PLANE_SETTINGS_TTL_S:
        d = _cp_cache.get("data")
        return d if isinstance(d, dict) else None

    url = f"{CONTROL_PLANE_URL}/admin/settings"
    headers = {"X-Vozlia-Admin-Key": CONTROL_PLANE_ADMIN_KEY}
    try:
        with httpx.Client(timeout=3.0) as client:
            r = client.get(url, headers=headers)
            if r.status_code == 200:
                data = r.json()
                if isinstance(data, dict):
                    _cp_cache["ts"] = now
                    _cp_cache["data"] = data
                    return data
    except Exception:
        # Fail closed to local DB/env settings
        return None
    return None

def _cp_get_skills_config() -> Optional[dict[str, dict]]:
    data = _get_control_plane_settings_cached()
    if not isinstance(data, dict):
        return None
    sc = data.get("skills_config")
    if isinstance(sc, dict):
        skills = sc.get("skills")
        if isinstance(skills, dict):
            out: dict[str, dict] = {}
            for sid, cfg in skills.items():
                if isinstance(sid, str) and isinstance(cfg, dict):
                    out[sid] = cfg
            return out
    return None



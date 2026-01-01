# services/settings_service.py
from __future__ import annotations

import os
import time
import threading
from typing import Any, Optional

import httpx
from sqlalchemy.orm import Session

from models import User, UserSetting

# -----------------------------
# Control Plane settings bridge
# -----------------------------
# NOTE: The Portal UI writes settings to the Control Plane.
# The backend must read the SAME settings source; otherwise toggles (like "Add to greeting")
# will appear to "not work" at runtime.
#
# This client is cached (TTL) to avoid adding too much latency on /assistant/route.
CONTROL_PLANE_URL = (os.getenv("CONTROL_PLANE_URL") or "").strip().rstrip("/")
CONTROL_PLANE_ADMIN_KEY = (os.getenv("CONTROL_PLANE_ADMIN_KEY") or "").strip()
CONTROL_PLANE_SETTINGS_TTL_S = float(os.getenv("CONTROL_PLANE_SETTINGS_TTL_S", "5"))

_CP_LOCK = threading.Lock()
_CP_CACHE_TS = 0.0
_CP_CACHE_DATA: Optional[dict] = None


def _fetch_control_plane_admin_settings(force: bool = False) -> Optional[dict]:
    global _CP_CACHE_TS, _CP_CACHE_DATA

    if not CONTROL_PLANE_URL or not CONTROL_PLANE_ADMIN_KEY:
        return None

    now = time.monotonic()
    with _CP_LOCK:
        if not force and _CP_CACHE_DATA is not None and (now - _CP_CACHE_TS) < CONTROL_PLANE_SETTINGS_TTL_S:
            return _CP_CACHE_DATA

    try:
        url = f"{CONTROL_PLANE_URL}/admin/settings"
        headers = {"X-Vozlia-Admin-Key": CONTROL_PLANE_ADMIN_KEY}
        with httpx.Client(timeout=2.0) as client:
            resp = client.get(url, headers=headers)
            if resp.status_code != 200:
                return None
            data = resp.json()
    except Exception:
        return None

    if isinstance(data, dict):
        with _CP_LOCK:
            _CP_CACHE_DATA = data
            _CP_CACHE_TS = now
        return data

    return None


def _cp_get(field: str) -> Any:
    data = _fetch_control_plane_admin_settings()
    if isinstance(data, dict):
        return data.get(field)
    return None


# -----------------------------
# Defaults (used when CP not configured or temporarily unavailable)
# -----------------------------
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
    # Skill config (fallback shape)
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
    # Memory defaults (fallback)
    "shortterm_memory_enabled": {"enabled": True},
    "longterm_memory_enabled": {"enabled": False},
    "memory_engagement_phrases": {"phrases": []},
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
# Core settings (prefer Control Plane, fallback to local DB)
# -----------------------------
def get_agent_greeting(db: Session, user: User) -> str:
    cp = _cp_get("agent_greeting")
    if isinstance(cp, str) and cp.strip():
        return cp.strip()

    v = get_setting(db, user, "agent_greeting")
    t = (v or {}).get("text")
    if isinstance(t, str) and t.strip():
        return t.strip()
    return str(DEFAULTS["agent_greeting"]["text"])


def gmail_summary_enabled(db: Session, user: User) -> bool:
    cp = _cp_get("gmail_summary_enabled")
    if isinstance(cp, bool):
        return bool(cp)

    v = get_setting(db, user, "gmail_summary_enabled")
    enabled = (v or {}).get("enabled")
    return bool(True if enabled is None else enabled)


def get_selected_gmail_account_id(db: Session, user: User) -> Optional[str]:
    cp = _cp_get("gmail_account_id")
    if isinstance(cp, str) and cp.strip():
        return cp.strip()

    v = get_setting(db, user, "gmail_account_id")
    account_id = (v or {}).get("account_id")
    if isinstance(account_id, str) and account_id.strip():
        return account_id.strip()
    return None


def get_realtime_prompt_addendum(db: Session, user: User) -> str:
    cp = _cp_get("realtime_prompt_addendum")
    if isinstance(cp, str) and cp.strip():
        return cp.strip()

    v = get_setting(db, user, "realtime_prompt_addendum")
    t = (v or {}).get("text")
    if isinstance(t, str) and t.strip():
        return t.strip()
    return str(DEFAULTS["realtime_prompt_addendum"]["text"])


# -----------------------------
# Skill config helpers (prefer Control Plane)
# -----------------------------
def get_skills_config(db: Session, user: User) -> dict[str, dict]:
    # Control Plane returns: { "gmail_summary": { ... }, ... }
    cp = _cp_get("skills_config")
    if isinstance(cp, dict):
        out: dict[str, dict] = {}
        for k, cfg in cp.items():
            if isinstance(k, str) and isinstance(cfg, dict):
                out[k] = cfg
        if out:
            return out

    # Fallback: local DB stores {"skills": {...}}
    v = get_setting(db, user, "skills_config")
    skills = (v or {}).get("skills")
    if isinstance(skills, dict):
        out: dict[str, dict] = {}
        for k, cfg in skills.items():
            if isinstance(k, str) and isinstance(cfg, dict):
                out[k] = cfg
        return out

    return dict(DEFAULTS["skills_config"]["skills"])


def get_skill_config(db: Session, user: User, skill_id: str) -> dict:
    skills = get_skills_config(db, user)
    base = dict(DEFAULTS["skills_config"]["skills"].get(skill_id, {}))
    override = skills.get(skill_id)
    if isinstance(override, dict):
        base.update(override)
    return base


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
# Memory toggles (prefer Control Plane)
# -----------------------------
def shortterm_memory_enabled(db: Session, user: User) -> bool:
    cp = _cp_get("shortterm_memory_enabled")
    if isinstance(cp, bool):
        return bool(cp)

    row = _get_setting_row(db, user, "shortterm_memory_enabled")
    if row and isinstance(row.value, dict) and "enabled" in row.value:
        return bool(row.value.get("enabled"))

    # env fallback preserves current behavior
    return (os.getenv("SESSION_MEMORY_ENABLED") or "1").strip().lower() in ("1", "true", "yes", "on")


def longterm_memory_enabled(db: Session, user: User) -> bool:
    cp = _cp_get("longterm_memory_enabled")
    if isinstance(cp, bool):
        return bool(cp)

    row = _get_setting_row(db, user, "longterm_memory_enabled")
    if row and isinstance(row.value, dict) and "enabled" in row.value:
        return bool(row.value.get("enabled"))

    return (os.getenv("LONGTERM_MEMORY_ENABLED_DEFAULT") or "0").strip().lower() in ("1", "true", "yes", "on")


def get_memory_engagement_phrases(db: Session, user: User) -> list[str]:
    cp = _cp_get("memory_engagement_phrases")
    if isinstance(cp, list):
        return [str(x).strip() for x in cp if str(x).strip()]

    v = get_setting(db, user, "memory_engagement_phrases")
    phrases = (v or {}).get("phrases")
    if isinstance(phrases, list):
        return [str(x).strip() for x in phrases if str(x).strip()]
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

            # NOTE: Portal writes to Control Plane; backend patch is typically not used.
            # Keeping this for backward compatibility with /me/settings endpoints.
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
                # store in local DB in the legacy shape {"skills": {...}}
                local = get_setting(db, user, "skills_config") or {}
                skills = local.get("skills") if isinstance(local, dict) else None
                skills = skills if isinstance(skills, dict) else {}
                for sid, cfg in patch["skills_config"].items():
                    if isinstance(sid, str) and isinstance(cfg, dict):
                        skills[sid] = cfg
                set_setting(db, user, "skills_config", {"skills": skills})

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

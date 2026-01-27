# services/dynamic_skill_autosync.py
# -----------------------------------------------------------------------------
# Purpose
#   Ensure dynamically-created skills (WebSearchSkill / DBQuerySkill rows) are
#   also registered in user_settings.skills_config so that:
#     - Voice (/assistant/route) and Portal chat can route to them reliably
#     - Intent Router V2 can surface them as candidates
#
# Why
#   We observed "code drift" / rollbacks where dynamic skills existed in DB
#   tables but no longer existed in skills_config (or were missing triggers),
#   causing routing to fall back to chitchat/KB.
#
# Safety
#   - Additive-by-default: only adds missing skills_config entries, and only
#     fills missing required fields on existing entries.
#   - Does NOT delete or reorder existing skills by default.
#
# Rollback
#   Set DYNAMIC_SKILLS_AUTOSYNC=0 to disable autosync.
# -----------------------------------------------------------------------------
from __future__ import annotations

import os
import re
from typing import Any, Tuple

from sqlalchemy.orm import Session

from core.logging import logger
from models import User, WebSearchSkill, DBQuerySkill
from services.settings_service import (
    get_skills_config,
    get_skills_priority_order,
    set_setting,
    set_skills_priority_order,
)


def dynamic_skills_autosync_enabled() -> bool:
    return (os.getenv("DYNAMIC_SKILLS_AUTOSYNC", "1") or "1").strip().lower() in ("1", "true", "yes", "on")


# -----------------------------------------------------------------------------
# Category metadata for dynamic skills
#
# Why
#   Voice/chat intent routing improves when skills carry lightweight category tags
#   (e.g. "sports", "parking", "weather"). Categories are OPTIONAL and must never
#   break execution. If missing, we set a safe default. Optionally, we can
#   heuristically auto-classify at sync-time (NOT on the voice hot path).
#
# Rollback / safety
#   - Set DYNAMIC_SKILL_CATEGORY_AUTO=0 to disable heuristic categorization.
#   - Categories are stored in skills_config only; no DB migration required.
# -----------------------------------------------------------------------------

def dynamic_skill_category_default() -> str:
    return (os.getenv("DYNAMIC_SKILL_CATEGORY_DEFAULT", "general") or "general").strip() or "general"


def dynamic_skill_category_auto_enabled() -> bool:
    # Optional heuristic categorization (safe: only fills missing category).
    return (os.getenv("DYNAMIC_SKILL_CATEGORY_AUTO", "1") or "1").strip().lower() in ("1", "true", "yes", "on")


def _norm(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


_CATEGORY_RULES: list[tuple[str, list[str]]] = [
    ("sports", ["sport", "sports", "nba", "wnba", "nfl", "mlb", "nhl", "soccer", "football", "basketball", "baseball", "hockey", "digest", "score", "scores", "games", "matchup", "matchups"]),
    ("parking", ["alternate side parking", "asp", "parking"]),
    ("weather", ["weather", "forecast", "temperature", "rain", "snow", "humidity", "wind"]),
    ("finance", ["investment", "invest", "portfolio", "stock", "stocks", "crypto", "bitcoin", "btc", "market", "yfinance", "debt", "national debt", "interest rate"]),
    ("email", ["gmail", "email", "inbox", "summary", "summaries"]),
    ("calls", ["caller", "callers", "call", "calls", "voicemail", "missed call"]),
    ("business", ["hours", "open", "close", "appointment", "reservation", "order", "orders", "booking"]),
]


def guess_dynamic_skill_category(*, label: str, query: str = "", entity: str = "", phrases: list[str] | None = None) -> str:
    """Best-effort category guess.

    IMPORTANT: This is a heuristic convenience only. It MUST be safe:
      - never raises
      - never returns empty
      - never overwrites an operator-set category
    """
    try:
        text = " ".join([label or "", query or "", entity or "", " ".join(phrases or [])])
        t = _norm(text)
        if not t:
            return dynamic_skill_category_default()
        for cat, keys in _CATEGORY_RULES:
            for k in keys:
                if _norm(k) and _norm(k) in t:
                    return cat
    except Exception:
        pass
    return dynamic_skill_category_default()


def _clean_label(name: str) -> str:
    s = (name or "").strip()
    if not s:
        return ""
    # Remove common prefixes introduced by UI labeling
    s2 = re.sub(r"^\s*(web\s*search|websearch)\s*[:\-]\s*", "", s, flags=re.I)
    return s2.strip() or s


def _default_phrases(name: str) -> list[str]:
    """Derive reasonable engagement phrases when the owner did not provide triggers."""
    base = _clean_label(name)
    out: list[str] = []
    if base:
        out.append(base)
    # Also keep original name if it differs (helps match UI-prefixed labels)
    if name and name.strip() and name.strip() != base:
        out.append(name.strip())
    # De-dupe while preserving order
    seen: set[str] = set()
    uniq: list[str] = []
    for p in out:
        k = p.lower().strip()
        if k and k not in seen:
            seen.add(k)
            uniq.append(p)
    return uniq


def _ensure_priority_contains(order: list[str], key: str) -> list[str]:
    if key in order:
        return order
    return list(order) + [key]


def _merge_cfg(existing: dict[str, Any], desired: dict[str, Any]) -> Tuple[dict[str, Any], bool]:
    """Merge desired into existing without clobbering operator-edited toggles."""
    if not isinstance(existing, dict):
        return desired, True

    merged = dict(desired)

    # Preserve common operator toggles if present in existing
    for k in (
        "enabled",
        "add_to_greeting",
        "auto_execute_after_greeting",
        "greeting_line",
    ):
        if k in existing:
            merged[k] = existing.get(k)


    # Preserve operator-assigned category if present (do not clobber manual grouping)
    cat = existing.get("category")
    if isinstance(cat, str) and cat.strip():
        merged["category"] = cat.strip()

    # Preserve existing engagement phrases if non-empty; else fill with desired
    ep = existing.get("engagement_phrases")
    if isinstance(ep, list) and any(isinstance(x, str) and x.strip() for x in ep):
        merged["engagement_phrases"] = ep

    changed = merged != existing
    return merged, changed


def autosync_dynamic_skills(db: Session, user: User, *, force: bool = False) -> dict[str, Any]:
    """Sync DB rows into skills_config (best-effort).

    Returns counts for observability:
      {added, updated, total_dynamic, total_config}
    """
    cfg = get_skills_config(db, user) or {}
    if not isinstance(cfg, dict):
        cfg = {}

    added = 0
    updated = 0

    # --- WebSearch skills ---
    ws_rows = (
        db.query(WebSearchSkill)
        .filter(WebSearchSkill.tenant_id == user.id)
        .order_by(WebSearchSkill.created_at.asc())
        .all()
    )
    for row in ws_rows:
        key = f"websearch_{row.id}"
        triggers = row.triggers if isinstance(row.triggers, list) else []
        phrases = [t for t in triggers if isinstance(t, str) and t.strip()] or _default_phrases(row.name)
        # Category: preserve operator-set category if present; else default/auto-classify (safe).
        existing = cfg.get(key) if isinstance(cfg.get(key), dict) else None
        existing_cat = (existing.get("category") if isinstance(existing, dict) else None)
        if isinstance(existing_cat, str) and existing_cat.strip():
            category = existing_cat.strip()
        else:
            if dynamic_skill_category_auto_enabled():
                category = guess_dynamic_skill_category(label=str(row.name or ""), query=str(row.query or ""), phrases=phrases)
            else:
                category = dynamic_skill_category_default()
        desired = {
            "enabled": bool(getattr(row, "enabled", True)),
            "label": str(row.name or "Saved Web Search"),
            "category": str(category),
            "type": "web_search",
            "query": str(row.query or ""),
            "engagement_phrases": phrases,
            "add_to_greeting": False,
            "auto_execute_after_greeting": False,
            "greeting_line": f"I can run '{_clean_label(row.name)}'. Just ask: {phrases[0]}" if phrases else "",
            "web_search_skill_id": str(row.id),
        }

        if force or key not in cfg:
            existed = key in cfg
            cfg[key] = desired
            added += 0 if existed else 1
            updated += 1 if existed else 0
        else:
            merged, changed = _merge_cfg(cfg.get(key) or {}, desired)
            if changed:
                cfg[key] = merged
                updated += 1

    # --- DBQuery skills ---
    dq_rows = (
        db.query(DBQuerySkill)
        .filter(DBQuerySkill.tenant_id == user.id)
        .order_by(DBQuerySkill.created_at.asc())
        .all()
    )
    for row in dq_rows:
        key = f"dbquery_{row.id}"
        triggers = row.triggers if isinstance(row.triggers, list) else []
        phrases = [t for t in triggers if isinstance(t, str) and t.strip()] or _default_phrases(row.name)
        # Category: preserve operator-set category if present; else default/auto-classify (safe).
        existing = cfg.get(key) if isinstance(cfg.get(key), dict) else None
        existing_cat = (existing.get("category") if isinstance(existing, dict) else None)
        if isinstance(existing_cat, str) and existing_cat.strip():
            category = existing_cat.strip()
        else:
            if dynamic_skill_category_auto_enabled():
                category = guess_dynamic_skill_category(label=str(row.name or ""), entity=str(row.entity or ""), phrases=phrases)
            else:
                category = dynamic_skill_category_default()
        desired = {
            "enabled": bool(getattr(row, "enabled", True)),
            "label": str(row.name or "DB Query"),
            "category": str(category),
            "type": "db_query",
            "entity": str(row.entity or "caller_memory_events"),
            "spec": row.spec if isinstance(row.spec, dict) else {},
            "engagement_phrases": phrases,
            "add_to_greeting": False,
            "auto_execute_after_greeting": False,
            "greeting_line": f"I can run '{_clean_label(row.name)}'. Just ask: {phrases[0]}" if phrases else "",
            "db_query_skill_id": str(row.id),
        }

        if force or key not in cfg:
            existed = key in cfg
            cfg[key] = desired
            added += 0 if existed else 1
            updated += 1 if existed else 0
        else:
            merged, changed = _merge_cfg(cfg.get(key) or {}, desired)
            if changed:
                cfg[key] = merged
                updated += 1

    # Write back
    set_setting(db, user, "skills_config", cfg)

    # Ensure priority contains all dynamic skills (append only; do not reorder)
    try:
        order = get_skills_priority_order(db, user) or []
        if not isinstance(order, list):
            order = []
        for k in cfg.keys():
            if isinstance(k, str) and (k.startswith("websearch_") or k.startswith("dbquery_")):
                order = _ensure_priority_contains(order, k)
        set_skills_priority_order(db, user, order)
    except Exception:
        logger.exception("DYNAMIC_SKILLS_AUTOSYNC_PRIORITY_FAIL")

    total_dynamic = len(ws_rows) + len(dq_rows)
    logger.info(
        "DYNAMIC_SKILLS_AUTOSYNC ok=1 added=%s updated=%s total_dynamic=%s total_config=%s",
        added,
        updated,
        total_dynamic,
        len(cfg),
    )
    return {
        "added": added,
        "updated": updated,
        "total_dynamic": total_dynamic,
        "total_config": len(cfg),
    }

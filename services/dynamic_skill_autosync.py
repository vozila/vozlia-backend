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
        desired = {
            "enabled": bool(getattr(row, "enabled", True)),
            "label": str(row.name or "Saved Web Search"),
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
        desired = {
            "enabled": bool(getattr(row, "enabled", True)),
            "label": str(row.name or "DB Query"),
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

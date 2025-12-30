# services/caller_cache.py
"""Caller-scoped TTL cache backed by Postgres.

Use case:
- Persist short-term memory across calls for the same caller phone number.
- Avoid re-running expensive skills (e.g., Gmail summaries) when the user calls back soon.

Key:
  (tenant_id, caller_id, skill_key, cache_key_hash)

TTL:
- Enforced by expires_at comparisons at read time.
- Optional opportunistic cleanup on write.

Safety:
- Feature-flagged (CALLER_MEMORY_ENABLED).
- Skill allowlist (CALLER_CACHE_SKILLS) to avoid caching sensitive/side-effecting skills by default.
"""

from __future__ import annotations

import hashlib
import os
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from core.logging import logger
from models import CallerSkillCache

CALLER_MEMORY_ENABLED = (os.getenv("CALLER_MEMORY_ENABLED") or "0").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)

CALLER_MEMORY_TTL_S = int((os.getenv("CALLER_MEMORY_TTL_S") or "21600").strip() or "21600")  # 6 hours
CALLER_MEMORY_DEBUG = (os.getenv("CALLER_MEMORY_DEBUG") or "0").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)

# Comma-separated allowlist of skills that may use caller cache
_CALLER_CACHE_SKILLS_RAW = (os.getenv("CALLER_CACHE_SKILLS") or "gmail_summary").strip()
CALLER_CACHE_SKILLS = {s.strip() for s in _CALLER_CACHE_SKILLS_RAW.split(",") if s.strip()}




# Log configuration once at import time so env issues are visible in Render logs.
try:
    logger.info(
        "CALLER_CACHE_CONFIG enabled=%s debug=%s ttl_s=%s skills=%s",
        CALLER_MEMORY_ENABLED,
        CALLER_MEMORY_DEBUG,
        CALLER_MEMORY_TTL_S,
        sorted(CALLER_CACHE_SKILLS),
    )
except Exception:
    pass

def normalize_caller_id(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    s = str(raw).strip()
    if not s:
        return None
    # keep leading '+' if present; remove spaces
    s = s.replace(" ", "")
    return s


def is_skill_allowed(skill_key: str) -> bool:
    if not skill_key:
        return False
    return skill_key in CALLER_CACHE_SKILLS


def make_cache_key_hash(*parts: Any) -> str:
    joined = "|".join([str(p) for p in parts])
    return hashlib.sha1(joined.encode("utf-8", errors="ignore")).hexdigest()


def get_caller_cache(
    db,
    *,
    tenant_id,
    caller_id: str,
    skill_key: str,
    cache_key_hash: str,
) -> Optional[Dict[str, Any]]:
    if not (CALLER_MEMORY_ENABLED and tenant_id and caller_id and skill_key and cache_key_hash):
        if CALLER_MEMORY_DEBUG:
            logger.info(
                "CALLER_MEM_SKIP preconditions enabled=%s tenant=%s caller=%s skill=%s hash=%s",
                CALLER_MEMORY_ENABLED,
                bool(tenant_id),
                bool(caller_id),
                bool(skill_key),
                bool(cache_key_hash),
            )
        return None
    if not is_skill_allowed(skill_key):
        if CALLER_MEMORY_DEBUG:
            logger.info("CALLER_MEM_SKIP skill_not_allowed skill=%s allowed=%s", skill_key, sorted(CALLER_CACHE_SKILLS))
        return None

    now = datetime.utcnow()
    caller_id_norm = normalize_caller_id(caller_id)
    if not caller_id_norm:
        return None

    row = (
        db.query(CallerSkillCache)
        .filter(
            CallerSkillCache.tenant_id == tenant_id,
            CallerSkillCache.caller_id == caller_id_norm,
            CallerSkillCache.skill_key == skill_key,
            CallerSkillCache.cache_key_hash == cache_key_hash,
            CallerSkillCache.expires_at > now,
        )
        .order_by(CallerSkillCache.updated_at.desc())
        .first()
    )

    if not row:
        if CALLER_MEMORY_DEBUG:
            logger.info(
                "CALLER_MEM_MISS skill=%s hash=%s tenant_id=%s caller_id=%s",
                skill_key,
                cache_key_hash[:8],
                str(tenant_id),
                caller_id_norm,
            )
        return None

    if CALLER_MEMORY_DEBUG:
        try:
            age_s = int((now - (row.updated_at or row.created_at)).total_seconds())
        except Exception:
            age_s = -1

        # If the TTL env changed since this cache row was written, enforce the CURRENT TTL
        # using age_s as a fallback guard (prevents old rows with long expires_at lingering).
        if age_s >= 0 and age_s > CALLER_MEMORY_TTL_S:
            if CALLER_MEMORY_DEBUG:
                logger.info(
                    "CALLER_MEM_TTL_EXPIRED_BY_AGE skill=%s hash=%s age_s=%s ttl_s=%s tenant_id=%s caller_id=%s",
                    skill_key,
                    cache_key_hash[:8],
                    age_s,
                    CALLER_MEMORY_TTL_S,
                    str(tenant_id),
                    caller_id_norm,
                )
            try:
                db.query(CallerSkillCache).filter(CallerSkillCache.id == row.id).delete()
                db.commit()
            except Exception:
                db.rollback()
            return None

        logger.info(
            "CALLER_MEM_HIT skill=%s hash=%s age_s=%s tenant_id=%s caller_id=%s",
            skill_key,
            cache_key_hash[:8],
            age_s,
            str(tenant_id),
            caller_id_norm,
        )

    try:
        return dict(row.result_json or {})
    except Exception:
        logger.exception("CALLER_MEM_CORRUPT skill=%s hash=%s", skill_key, cache_key_hash[:8])
        return None


def put_caller_cache(
    db,
    *,
    tenant_id,
    caller_id: str,
    skill_key: str,
    cache_key_hash: str,
    result_json: Dict[str, Any],
    ttl_s: int = CALLER_MEMORY_TTL_S,
) -> None:
    """Write caller-scoped cache entry.

    Important: We opportunistically delete *expired* rows **before** we upsert. This avoids a race where
    an expired row is loaded + updated in-memory, but then a bulk delete (based on stale DB state)
    deletes it before flush/commit, which can trigger SQLAlchemy's StaleDataError.
    """
    if not (CALLER_MEMORY_ENABLED and tenant_id and caller_id and skill_key and cache_key_hash):
        if CALLER_MEMORY_DEBUG:
            logger.info(
                "CALLER_MEM_SKIP_WRITE preconditions enabled=%s tenant=%s caller=%s skill=%s hash=%s",
                CALLER_MEMORY_ENABLED,
                bool(tenant_id),
                bool(caller_id),
                bool(skill_key),
                bool(cache_key_hash),
            )
        return
    if not is_skill_allowed(skill_key):
        return

    caller_id_norm = normalize_caller_id(caller_id)
    if not caller_id_norm:
        return

    ttl_s_i = max(60, int(ttl_s or CALLER_MEMORY_TTL_S))  # at least 60s
    now = datetime.utcnow()
    expires_at = now + timedelta(seconds=ttl_s_i)

    # Opportunistic cleanup of old rows (cheap, bounded to this tenant+caller)
    # NOTE: Do this BEFORE reading/updating the target row to avoid StaleDataError.
    try:
        db.query(CallerSkillCache).filter(
            CallerSkillCache.tenant_id == tenant_id,
            CallerSkillCache.caller_id == caller_id_norm,
            CallerSkillCache.expires_at <= now,
        ).delete(synchronize_session=False)
    except Exception:
        # Cleanup failures shouldn't block writes.
        logger.debug("CALLER_MEM_CLEANUP_FAILED", exc_info=True)

    row = (
        db.query(CallerSkillCache)
        .filter(
            CallerSkillCache.tenant_id == tenant_id,
            CallerSkillCache.caller_id == caller_id_norm,
            CallerSkillCache.skill_key == skill_key,
            CallerSkillCache.cache_key_hash == cache_key_hash,
        )
        .first()
    )

    if row:
        row.result_json = result_json or {}
        row.updated_at = now
        row.expires_at = expires_at
    else:
        row = CallerSkillCache(
            tenant_id=tenant_id,
            caller_id=caller_id_norm,
            skill_key=skill_key,
            cache_key_hash=cache_key_hash,
            result_json=result_json or {},
            created_at=now,
            updated_at=now,
            expires_at=expires_at,
        )
        db.add(row)

    try:
        db.commit()
        if CALLER_MEMORY_DEBUG:
            logger.info(
                "CALLER_MEM_WRITE_OK skill=%s hash=%s tenant_id=%s caller_id=%s ttl_s=%s",
                skill_key,
                cache_key_hash[:8],
                str(tenant_id),
                caller_id_norm,
                ttl_s_i,
            )
    except Exception:
        db.rollback()
        logger.exception("CALLER_MEM_WRITE_FAILED skill=%s hash=%s", skill_key, cache_key_hash[:8])
        return

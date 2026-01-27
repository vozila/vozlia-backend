# api/routers/admin_dynamic_skills.py
# -----------------------------------------------------------------------------
# Troubleshooting endpoint to (re)sync DB dynamic skills into skills_config.
# This helps after rollbacks / DB restores where skills exist in tables but
# routing fails because skills_config lost entries.
#
# Rollback: you can ignore this endpoint; it is additive only.
# -----------------------------------------------------------------------------
from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from api.deps.admin_key import require_admin_key
from deps import get_db
from services.user_service import get_or_create_primary_user
from services.dynamic_skill_autosync import autosync_dynamic_skills, dynamic_skills_autosync_enabled

router = APIRouter(
    prefix="/admin/dynamic-skills",
    tags=["admin-dynamic-skills"],
    dependencies=[Depends(require_admin_key)],
)


@router.post("/sync")
def sync_dynamic_skills(db: Session = Depends(get_db)):
    user = get_or_create_primary_user(db)
    # Even if DYNAMIC_SKILLS_AUTOSYNC=0, allow a manual sync via this endpoint.
    out = autosync_dynamic_skills(db, user, force=False)
    db.commit()
    return {"ok": True, "enabled": bool(dynamic_skills_autosync_enabled()), **out}

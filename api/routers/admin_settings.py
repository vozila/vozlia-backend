from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import Any

from api.deps.admin_key import require_admin_key
from services.settings_service import settings_service

router = APIRouter(prefix="/admin", tags=["admin-settings"])


class AdminSettingsPatch(BaseModel):
    # Basic
    agent_greeting: str | None = None
    gmail_summary_enabled: bool | None = None
    gmail_account_id: str | None = None
    realtime_prompt_addendum: str | None = None

    # Modular skill config + priority
    skills_config: dict[str, dict[str, Any]] | None = None
    skills_priority_order: list[str] | None = None

    # Memory toggles
    shortterm_memory_enabled: bool | None = None
    longterm_memory_enabled: bool | None = None
    memory_engagement_phrases: list[str] | None = None


@router.get("/settings", dependencies=[Depends(require_admin_key)])
def admin_get_settings():
    # Back-compat: settings_service exposes get_settings()
    return settings_service.get_settings()


@router.patch("/settings", dependencies=[Depends(require_admin_key)])
def admin_patch_settings(payload: AdminSettingsPatch):
    # Back-compat: settings_service exposes patch_settings()
    return settings_service.patch_settings(payload.model_dump(exclude_none=True))

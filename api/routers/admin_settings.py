from fastapi import APIRouter, Depends
from pydantic import BaseModel
from api.deps.admin_key import require_admin_key
from services.settings_service import settings_service

router = APIRouter(prefix="/admin", tags=["admin-settings"])

class AdminSettingsPatch(BaseModel):
    agent_greeting: str | None = None
    gmail_summary_enabled: bool | None = None
    # add more fields later safely

@router.get("/settings", dependencies=[Depends(require_admin_key)])
def admin_get_settings():
    return settings_service.get_current_settings()

@router.patch("/settings", dependencies=[Depends(require_admin_key)])
def admin_patch_settings(payload: AdminSettingsPatch):
    return settings_service.update_settings(payload.model_dump(exclude_none=True))

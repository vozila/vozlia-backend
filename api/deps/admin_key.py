import os
from fastapi import Header, HTTPException
from fastapi import APIRouter, Depends
from api.deps.admin_key import require_admin_key
from services.settings_service import settings_service  # whatever you already use

def require_admin_key(x_vozlia_admin_key: str | None = Header(default=None)):
    expected = os.getenv("ADMIN_API_KEY")
    if not expected or x_vozlia_admin_key != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")



router = APIRouter(prefix="/admin", tags=["admin"])

@router.get("/settings", dependencies=[Depends(require_admin_key)])
def admin_get_settings():
    # reuse your existing settings read logic
    return settings_service.get_current_settings()

@router.patch("/settings", dependencies=[Depends(require_admin_key)])
def admin_patch_settings(payload: dict):
    # reuse your existing settings write logic
    return settings_service.update_settings(payload)

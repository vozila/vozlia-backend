from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import RedirectResponse
import httpx
from sqlalchemy.orm import Session

from core.logging import logger
from core import config as cfg
from deps import get_db
from models import User, EmailAccount
from main import get_current_user  # if this causes circular imports, see Option 2 below

router = APIRouter(prefix="/auth/google", tags=["oauth-google"])

@router.get("/start")
def google_auth_start(current_user: User = Depends(get_current_user)):
    if not cfg.GOOGLE_CLIENT_ID or not cfg.GOOGLE_REDIRECT_URI:
        raise HTTPException(status_code=500, detail="Google OAuth not configured")

    state = f"user-{current_user.id}"
    params = {
        "client_id": cfg.GOOGLE_CLIENT_ID,
        "redirect_uri": cfg.GOOGLE_REDIRECT_URI,
        "response_type": "code",
        "scope": cfg.GOOGLE_GMAIL_SCOPE,
        "access_type": "offline",
        "include_granted_scopes": "true",
        "prompt": "consent",
        "state": state,
    }
    url = httpx.URL("https://accounts.google.com/o/oauth2/v2/auth", params=params)
    return RedirectResponse(str(url))

@router.get("/callback")
async def google_auth_callback(
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    # keep your existing callback logic here later
    logger.info("Google OAuth callback hit (stub). params=%s", dict(request.query_params))
    raise HTTPException(status_code=501, detail="OAuth callback not wired yet")

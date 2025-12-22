import os
from urllib.parse import urlencode

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import RedirectResponse, HTMLResponse

from admin_auth import set_admin_session, clear_admin_session, require_admin, admin_enabled


router = APIRouter(prefix="/admin", tags=["admin"])


def _env(name: str, default: str = "") -> str:
    return (os.getenv(name) or default).strip()


GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://openidconnect.googleapis.com/v1/userinfo"


def _oauth_config_ok() -> bool:
    return all(
        _env(k)
        for k in (
            "GOOGLE_OAUTH_CLIENT_ID",
            "GOOGLE_OAUTH_CLIENT_SECRET",
            "GOOGLE_OAUTH_REDIRECT_URI",
            "SESSION_SECRET",
            "ADMIN_EMAIL",
        )
    )


@router.get("")
async def admin_home(request: Request):
    # If admin disabled, show a plain message (don’t leak details)
    if not admin_enabled():
        return HTMLResponse("<h3>Admin disabled</h3>", status_code=403)

    try:
        email = require_admin(request)
        return HTMLResponse(
            f"<h2>Vozlia Admin</h2><p>Logged in as <b>{email}</b></p>"
            "<form method='post' action='/admin/logout'><button type='submit'>Logout</button></form>"
        )
    except Exception:
        # Not logged in -> show login link
        return HTMLResponse(
            "<h2>Vozlia Admin</h2><p><a href='/admin/login'>Login with Google</a></p>",
            status_code=401,
        )


@router.get("/login")
async def admin_login():
    if not admin_enabled():
        return HTMLResponse("<h3>Admin disabled</h3>", status_code=403)
    if not _oauth_config_ok():
        return HTMLResponse("<h3>Admin OAuth not configured</h3>", status_code=500)

    params = {
        "client_id": _env("GOOGLE_OAUTH_CLIENT_ID"),
        "redirect_uri": _env("GOOGLE_OAUTH_REDIRECT_URI"),
        "response_type": "code",
        "scope": "openid email profile",
        "access_type": "online",
        "prompt": "select_account",
    }
    return RedirectResponse(f"{GOOGLE_AUTH_URL}?{urlencode(params)}")


@router.get("/oauth/callback")
async def admin_oauth_callback(request: Request, code: str):
    if not admin_enabled():
        return HTMLResponse("<h3>Admin disabled</h3>", status_code=403)
    if not _oauth_config_ok():
        return HTMLResponse("<h3>Admin OAuth not configured</h3>", status_code=500)

    client_id = _env("GOOGLE_OAUTH_CLIENT_ID")
    client_secret = _env("GOOGLE_OAUTH_CLIENT_SECRET")
    redirect_uri = _env("GOOGLE_OAUTH_REDIRECT_URI")

    # Exchange code for tokens
    async with httpx.AsyncClient(timeout=12) as client:
        token_resp = await client.post(
            GOOGLE_TOKEN_URL,
            data={
                "code": code,
                "client_id": client_id,
                "client_secret": client_secret,
                "redirect_uri": redirect_uri,
                "grant_type": "authorization_code",
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
    token_resp.raise_for_status()
    tokens = token_resp.json()
    access_token = tokens.get("access_token")
    if not access_token:
        return HTMLResponse("<h3>Missing access token</h3>", status_code=500)

    # Fetch user profile (email)
    async with httpx.AsyncClient(timeout=12) as client:
        me_resp = await client.get(
            GOOGLE_USERINFO_URL,
            headers={"Authorization": f"Bearer {access_token}"},
        )
    me_resp.raise_for_status()
    profile = me_resp.json()
    email = (profile or {}).get("email", "")

    # Enforce single-admin email allowlist
    if email.lower() != _env("ADMIN_EMAIL").lower():
        # Don’t create session; show forbidden
        return HTMLResponse("<h3>Forbidden</h3>", status_code=403)

    resp = RedirectResponse("/admin", status_code=302)
    set_admin_session(resp, email=email)
    return resp


@router.post("/logout")
async def admin_logout():
    resp = RedirectResponse("/admin", status_code=302)
    clear_admin_session(resp)
    return resp

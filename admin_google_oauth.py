import os
from urllib.parse import urlencode
from datetime import datetime, timedelta

import httpx
from fastapi import APIRouter, Request, Depends
from fastapi.responses import RedirectResponse, HTMLResponse
from sqlalchemy.orm import Session

from deps import get_db
from models import User, EmailAccount
from core.security import encrypt_str
from admin_auth import set_admin_session, clear_admin_session, require_admin, admin_enabled
from core.logging import logger

router = APIRouter(prefix="/admin", tags=["admin"])

GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://openidconnect.googleapis.com/v1/userinfo"

GMAIL_CONNECT_STATE = "admin_gmail_connect"



@router.get("/_debug/env")
async def admin_debug_env():
    # Do NOT return secrets. Only return presence + redirect URI for sanity.
    payload = {
        "ADMIN_ENABLED": admin_enabled(),
        "GOOGLE_OAUTH_CLIENT_ID_set": bool(_env("GOOGLE_OAUTH_CLIENT_ID")),
        "GOOGLE_OAUTH_CLIENT_SECRET_set": bool(_env("GOOGLE_OAUTH_CLIENT_SECRET")),
        "GOOGLE_OAUTH_REDIRECT_URI": _env("GOOGLE_OAUTH_REDIRECT_URI"),
        "SESSION_SECRET_set": bool(_env("SESSION_SECRET")),
        "ADMIN_EMAIL": _env("ADMIN_EMAIL"),
    }
    logger.info("ADMIN_DEBUG_ENV %s", payload)
    return payload


def _env(name: str, default: str = "") -> str:
    return (os.getenv(name) or default).strip()


def _oauth_config_ok() -> bool:
    required = [
        "GOOGLE_OAUTH_CLIENT_ID",
        "GOOGLE_OAUTH_CLIENT_SECRET",
        "GOOGLE_OAUTH_REDIRECT_URI",
        "SESSION_SECRET",
        "ADMIN_EMAIL",
    ]
    missing = [k for k in required if not _env(k)]
    if missing:
        from core.logging import logger
        logger.error("Admin OAuth missing env vars: %s", missing)
        return False
    return True


def _gmail_oauth_ok() -> bool:
    return all(
        _env(k)
        for k in (
            "GMAIL_OAUTH_CLIENT_ID",
            "GMAIL_OAUTH_CLIENT_SECRET",
            "GMAIL_OAUTH_REDIRECT_URI",
            "ENCRYPTION_KEY",
            "ADMIN_EMAIL",
        )
    )


def _gmail_scope() -> str:
    return "https://www.googleapis.com/auth/gmail.readonly"


def _get_or_create_user_by_email(db: Session, email: str) -> User:
    user = db.query(User).filter(User.email == email).first()
    if user:
        return user
    user = User(email=email)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@router.get("")
async def admin_home(request: Request):
    if not admin_enabled():
        return HTMLResponse("<h3>Admin disabled</h3>", status_code=403)

    try:
        email = require_admin(request)
        # Add Connect Gmail link for operator convenience
        return HTMLResponse(
            f"<h2>Vozlia Admin</h2>"
            f"<p>Logged in as <b>{email}</b></p>"
            "<p><a href='/admin/gmail/connect'>Connect Gmail</a></p>"
            "<form method='post' action='/admin/logout'><button type='submit'>Logout</button></form>"
        )
    except Exception:
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

    if token_resp.status_code != 200:
        return HTMLResponse(
            f"<h3>Admin token exchange failed</h3><pre>{token_resp.text}</pre>",
            status_code=500,
        )

    tokens = token_resp.json()
    access_token = tokens.get("access_token")
    if not access_token:
        return HTMLResponse("<h3>Missing access token</h3>", status_code=500)

    async with httpx.AsyncClient(timeout=12) as client:
        me_resp = await client.get(
            GOOGLE_USERINFO_URL,
            headers={"Authorization": f"Bearer {access_token}"},
        )

    if me_resp.status_code != 200:
        return HTMLResponse(
            f"<h3>Failed to fetch user profile</h3><pre>{me_resp.text}</pre>",
            status_code=500,
        )

    profile = me_resp.json()
    email = (profile or {}).get("email", "")

    if email.lower() != _env("ADMIN_EMAIL").lower():
        return HTMLResponse("<h3>Forbidden</h3>", status_code=403)

    resp = RedirectResponse("/admin", status_code=302)
    set_admin_session(resp, email=email)
    return resp


@router.post("/logout")
async def admin_logout():
    resp = RedirectResponse("/admin", status_code=302)
    clear_admin_session(resp)
    return resp


@router.get("/gmail/connect")
async def admin_gmail_connect(request: Request):
    admin_email = require_admin(request)

    if not _gmail_oauth_ok():
        return HTMLResponse("<h3>Gmail OAuth not configured</h3>", status_code=500)

    params = {
        "client_id": _env("GMAIL_OAUTH_CLIENT_ID"),
        "redirect_uri": _env("GMAIL_OAUTH_REDIRECT_URI"),
        "response_type": "code",
        "scope": _gmail_scope(),
        "access_type": "offline",
        "prompt": "consent",
        "include_granted_scopes": "true",
        "state": GMAIL_CONNECT_STATE,
    }
    url = f"{GOOGLE_AUTH_URL}?{urlencode(params)}"
    return RedirectResponse(url)


@router.get("/gmail/callback")
async def admin_gmail_callback(
    request: Request,
    code: str,
    state: str | None = None,
    db: Session = Depends(get_db),
):
    admin_email = require_admin(request)

    if not _gmail_oauth_ok():
        return HTMLResponse("<h3>Gmail OAuth not configured</h3>", status_code=500)

    if state != GMAIL_CONNECT_STATE:
        return HTMLResponse("<h3>Invalid state</h3>", status_code=400)

    client_id = _env("GMAIL_OAUTH_CLIENT_ID")
    client_secret = _env("GMAIL_OAUTH_CLIENT_SECRET")
    redirect_uri = _env("GMAIL_OAUTH_REDIRECT_URI")

    async with httpx.AsyncClient(timeout=15) as client:
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

    if token_resp.status_code != 200:
        return HTMLResponse(
            f"<h3>Token exchange failed</h3><pre>{token_resp.text}</pre>",
            status_code=500,
        )

    tokens = token_resp.json()
    access_token = tokens.get("access_token")
    refresh_token = tokens.get("refresh_token")
    expires_in = tokens.get("expires_in", 3600)

    if not refresh_token:
        return HTMLResponse(
            "<h3>No refresh_token returned.</h3>"
            "<p>Try again. If it still fails, revoke access in your Google Account and retry.</p>",
            status_code=500,
        )

    user = _get_or_create_user_by_email(db, admin_email)

    db.query(EmailAccount).filter(
        EmailAccount.user_id == user.id,
        EmailAccount.provider_type == "gmail",
    ).update({"is_primary": False})

    acct = EmailAccount(
        user_id=user.id,
        provider_type="gmail",
        oauth_provider="google",
        oauth_access_token=encrypt_str(access_token),
        oauth_refresh_token=encrypt_str(refresh_token),
        oauth_expires_at=datetime.utcnow() + timedelta(seconds=int(expires_in)),
        is_primary=True,
        is_active=True,
    )
    db.add(acct)
    db.commit()

    return RedirectResponse("/admin", status_code=302)

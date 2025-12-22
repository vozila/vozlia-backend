import os
from typing import Optional, Dict, Any

from fastapi import Request, HTTPException
from itsdangerous import URLSafeSerializer, BadSignature


def _env(name: str, default: str = "") -> str:
    return (os.getenv(name) or default).strip()


def admin_enabled() -> bool:
    return _env("ADMIN_ENABLED", "false").lower() in ("1", "true", "yes", "on")


def _serializer() -> URLSafeSerializer:
    secret = _env("SESSION_SECRET")
    if not secret:
        # Don't crash the whole app; admin routes will refuse.
        raise RuntimeError("SESSION_SECRET is not set")
    return URLSafeSerializer(secret, salt="vozlia-admin-session-v1")


SESSION_COOKIE = "vozlia_admin_session"


def set_admin_session(response, email: str) -> None:
    s = _serializer()
    token = s.dumps({"email": email})
    # secure cookie settings
    response.set_cookie(
        SESSION_COOKIE,
        token,
        httponly=True,
        secure=True,
        samesite="lax",
        max_age=60 * 60 * 24 * 7,  # 7 days
        path="/",
    )


def clear_admin_session(response) -> None:
    response.delete_cookie(SESSION_COOKIE, path="/")


def get_session_email(request: Request) -> Optional[str]:
    token = request.cookies.get(SESSION_COOKIE)
    if not token:
        return None
    try:
        s = _serializer()
        data = s.loads(token)
        return (data or {}).get("email")
    except (BadSignature, RuntimeError):
        return None


def require_admin(request: Request) -> str:
    if not admin_enabled():
        raise HTTPException(status_code=403, detail="Admin is disabled.")

    expected = _env("ADMIN_EMAIL")
    if not expected:
        raise HTTPException(status_code=500, detail="ADMIN_EMAIL not configured.")

    email = get_session_email(request)
    if not email:
        raise HTTPException(status_code=401, detail="Not logged in.")

    if email.lower() != expected.lower():
        raise HTTPException(status_code=403, detail="Forbidden.")
    return email

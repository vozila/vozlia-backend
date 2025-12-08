import os
import json
import asyncio
import logging
import base64
import time
from typing import List
from datetime import datetime, timedelta

import httpx  # <-- for Google OAuth + Gmail API

from fastapi import (
    FastAPI,
    Request,
    WebSocket,
    WebSocketDisconnect,
    Depends,
    HTTPException,
)
from fastapi.responses import PlainTextResponse, Response, JSONResponse, RedirectResponse

from sqlalchemy.orm import Session

from twilio.twiml.voice_response import VoiceResponse, Connect
from openai import OpenAI
import websockets

from cryptography.fernet import Fernet  # centralized crypto

from db import Base, engine
from models import User, EmailAccount
from schemas import EmailAccountCreate, EmailAccountRead
from deps import get_db

# ⬇️ NEW: bring in task engine API routes
from api.routes import tasks as tasks_routes
from api.routes import task_debug as task_debug_routes


# ---------- Logging ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vozlia")
logger.setLevel(logging.INFO)

# ---------- FastAPI app ----------
app = FastAPI()

# ⬇️ NEW: include task engine routers
app.include_router(tasks_routes.router)
app.include_router(task_debug_routes.router)


# ---------- Internal brain wiring (call /debug/brain from inside the app) ----------

VOICE_USER_ID_CACHE: str | None = None


def _get_internal_base_url() -> str:
    """
    Internal base URL to call this same service from inside the container.
    On Render, uvicorn listens on PORT (e.g. 10000) on 0.0.0.0.
    """
    port = os.getenv("PORT", "8000")
    return f"http://127.0.0.1:{port}"


async def get_voice_user_id() -> str:
    """
    Resolve which user_id to use for phone calls.

    Strategy:
      1) If VOZLIA_VOICE_USER_ID env var is set, use that.
      2) Otherwise, call /debug/me once and cache the resulting user id.
    """
    global VOICE_USER_ID_CACHE

    if VOICE_USER_ID_CACHE:
        return VOICE_USER_ID_CACHE

    env_user_id = os.getenv("VOZLIA_VOICE_USER_ID")
    if env_user_id:
        VOICE_USER_ID_CACHE = env_user_id
        logger.info(f"Using VOZLIA_VOICE_USER_ID={env_user_id} for voice calls")
        return env_user_id

    base = _get_internal_base_url()
    url = f"{base}/debug/me"

    try:
        async with httpx.AsyncClient(timeout=5.0) as client_http:
            resp = await client_http.get(url)
        resp.raise_for_status()
        data = resp.json()
        user_id = data.get("id")
        if not user_id:
            raise RuntimeError("debug/me did not return an 'id' field")

        VOICE_USER_ID_CACHE = user_id
        logger.info(f"Resolved voice user_id via /debug/me: {user_id}")
        return user_id

    except Exception as e:
        logger.error(f"Failed to resolve voice user id via /debug/me: {e}")
        # Fallback: dummy id (tasks will still work but not tied cleanly)
        fallback_id = "voice-demo-user"
        VOICE_USER_ID_CACHE = fallback_id
        return fallback_id


async def run_brain_on_transcript(transcript: str) -> str | None:
    """
    Call the existing /debug/brain endpoint with the transcript text and
    return the 'speech' string that should be spoken back to the caller.

    This reuses the same intent detection + task engine you already have.
    """
    base = _get_internal_base_url()
    url = f"{base}/debug/brain"

    user_id = await get_voice_user_id()

    payload = {
        "user_id": user_id,
        "text": transcript,
    }

    try:
        async with httpx.AsyncClient(timeout=8.0) as client_http:
            resp = await client_http.post(url, json=payload)
        if resp.status_code != 200:
            logger.error(
                f"/debug/brain HTTP {resp.status_code}: {resp.text}"
            )
            return None

        data = resp.json()
        speech = data.get("speech")
        if not speech:
            logger.warning(f"/debug/brain response missing 'speech': {data}")
            return None

        logger.info(f"[BRAIN] Got speech from /debug/brain: {speech!r}")
        return speech

    except Exception as e:
        logger.error(f"Error calling internal /debug/brain: {e}")
        return None


# ---------- Crypto helpers (for passwords & OAuth tokens) ----------
def get_fernet() -> Fernet:
    key = os.getenv("ENCRYPTION_KEY")
    if not key:
        raise RuntimeError("ENCRYPTION_KEY is not configured")
    # ENCRYPTION_KEY should be something like Fernet.generate_key().decode()
    return Fernet(key.encode() if not key.startswith("gAAAA") else key)


def encrypt_str(value: str | None) -> str | None:
    if value is None:
        return None
    f = get_fernet()
    return f.encrypt(value.encode()).decode()


def decrypt_str(value: str | None) -> str | None:
    if not value:
        return None
    f = get_fernet()
    return f.decrypt(value.encode()).decode()


# ---------- Google / Gmail OAuth config ----------
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
GOOGLE_REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI")

# For now: read-only Gmail scope (you can expand later)
GOOGLE_GMAIL_SCOPE = "https://www.googleapis.com/auth/gmail.readonly"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GMAIL_API_BASE = "https://gmail.googleapis.com/gmail/v1"


# Create tables on startup (simple MVP, later use Alembic)
@app.on_event("startup")
def on_startup() -> None:
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables ensured (create_all).")


# ---------- Auth stub / current user ----------
def get_current_user(db: Session = Depends(get_db)) -> User:
    """
    TEMP: For now, just returns the first user or creates a demo user.
    Later this will be replaced with real auth / tenant logic.
    """
    user = db.query(User).first()
    if not user:
        user = User(email="demo@vozlia.com")
        db.add(user)
        db.commit()
        db.refresh(user)
        logger.info(f"Created demo user with id={user.id} email={user.email}")
    return user


# ---------- Email account endpoints ----------
@app.get("/email/accounts", response_model=List[EmailAccountRead])
def list_email_accounts(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> List[EmailAccountRead]:
    accounts = (
        db.query(EmailAccount)
        .filter(EmailAccount.user_id == current_user.id)
        .all()
    )
    return accounts


@app.post("/email/accounts", response_model=EmailAccountRead)
def create_email_account(
    payload: EmailAccountCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> EmailAccountRead:
    """
    Creates an email account record for the current user.
    For provider_type == 'imap_custom', password (if provided) is encrypted
    before storing in password_enc.
    """
    password_enc = None
    if payload.provider_type == "imap_custom" and payload.password:
        key = os.getenv("ENCRYPTION_KEY")
        if not key:
            raise HTTPException(
                status_code=500,
                detail="ENCRYPTION_KEY not configured on server",
            )
        # Use shared helper for consistency
        password_enc = encrypt_str(payload.password)

    account = EmailAccount(
        user_id=current_user.id,
        provider_type=payload.provider_type,
        email_address=str(payload.email_address),
        display_name=payload.display_name,
        is_primary=payload.is_primary,
        is_active=payload.is_active,
        imap_host=payload.imap_host,
        imap_port=payload.imap_port,
        imap_ssl=payload.imap_ssl,
        smtp_host=payload.smtp_host,
        smtp_port=payload.smtp_port,
        smtp_ssl=payload.smtp_ssl,
        username=payload.username,
        password_enc=password_enc,
    )

    db.add(account)
    db.commit()
    db.refresh(account)

    return account


# ---------- Google OAuth: start flow ----------
@app.get("/auth/google/start")
def google_auth_start(
    current_user: User = Depends(get_current_user),
):
    """
    Starts Gmail OAuth flow for the current user.
    Returns a redirect to Google's consent screen.
    """
    if not GOOGLE_CLIENT_ID or not GOOGLE_REDIRECT_URI:
        raise HTTPException(
            status_code=500,
            detail="Google OAuth not configured on server",
        )

    # For MVP, embed user id in state (no DB-backed state yet)
    state = f"user-{current_user.id}"

    params = {
        "client_id": GOOGLE_CLIENT_ID,
        "redirect_uri": GOOGLE_REDIRECT_URI,
        "response_type": "code",
        "scope": GOOGLE_GMAIL_SCOPE,
        "access_type": "offline",  # ask for refresh_token
        "include_granted_scopes": "true",
        "prompt": "consent",       # force showing consent to reliably get refresh_token
        "state": state,
    }

    url = httpx.URL("https://accounts.google.com/o/oauth2/v2/auth", params=params)
    return RedirectResponse(str(url))


# ---------- Google OAuth: callback handler ----------
@app.get("/auth/google/callback")
async def google_auth_callback(
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Handles Google's OAuth callback.
    Exchanges 'code' for tokens, gets Gmail profile, and stores/updates EmailAccount.
    """
    if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET or not GOOGLE_REDIRECT_URI:
        raise HTTPException(
            status_code=500,
            detail="Google OAuth not configured on server",
        )

    params = dict(request.query_params)
    error = params.get("error")
    if error:
        raise HTTPException(status_code=400, detail=f"Google OAuth error: {error}")

    code = params.get("code")
    if not code:
        raise HTTPException(status_code=400, detail="Missing 'code' in callback")

    # 1. Exchange code for tokens
    data = {
        "code": code,
        "client_id": GOOGLE_CLIENT_ID,
        "client_secret": GOOGLE_CLIENT_SECRET,
        "redirect_uri": GOOGLE_REDIRECT_URI,
        "grant_type": "authorization_code",
    }

    async with httpx.AsyncClient() as client_http:
        token_resp = await client_http.post(GOOGLE_TOKEN_URL, data=data)
        if token_resp.status_code != 200:
            logger.error(
                "Google token exchange failed: %s %s",
                token_resp.status_code,
                token_resp.text,
            )
            try:
                google_error = token_resp.json()
            except Exception:
                google_error = {"raw": token_resp.text}

            raise HTTPException(
                status_code=500,
                detail={
                    "message": "Failed to exchange code for tokens with Google",
                    "google_status": token_resp.status_code,
                    "google_error": google_error,
                },
            )

        token_data = token_resp.json()

        access_token = token_data.get("access_token")
        refresh_token = token_data.get("refresh_token")  # may be None on re-consent
        expires_in = token_data.get("expires_in")

        if not access_token:
            raise HTTPException(
                status_code=500,
                detail="No access_token returned from Google",
            )

        # 2. Use access token to get Gmail profile (email address)
        profile_url = f"{GMAIL_API_BASE}/users/me/profile"
        profile_resp = await client_http.get(
            profile_url,
            headers={"Authorization": f"Bearer {access_token}"},
        )

        if profile_resp.status_code != 200:
            logger.error(
                "Gmail profile request failed: %s %s",
                profile_resp.status_code,
                profile_resp.text,
            )
            try:
                google_error = profile_resp.json()
            except Exception:
                google_error = {"raw": profile_resp.text}

            raise HTTPException(
                status_code=500,
                detail={
                    "message": "Failed to fetch Gmail profile",
                    "google_status": profile_resp.status_code,
                    "google_error": google_error,
                },
            )

        profile = profile_resp.json()
        email_address = profile.get("emailAddress")
        if not email_address:
            raise HTTPException(
                status_code=500,
                detail="Gmail profile did not include emailAddress",
            )

    oauth_expires_at = (
        datetime.utcnow() + timedelta(seconds=expires_in)
        if expires_in
        else None
    )

    # 3. Store or update EmailAccount row for this Gmail address
    account = (
        db.query(EmailAccount)
        .filter(
            EmailAccount.user_id == current_user.id,
            EmailAccount.email_address == email_address,
            EmailAccount.provider_type == "gmail",
        )
        .first()
    )

    if not account:
        account = EmailAccount(
            user_id=current_user.id,
            provider_type="gmail",
            oauth_provider="google",
            email_address=email_address,
            display_name=email_address,
            is_primary=False,  # you can adjust via UI later
            is_active=True,
        )
        db.add(account)

    # Encrypt tokens and store
    account.oauth_access_token = encrypt_str(access_token)
    if refresh_token:
        account.oauth_refresh_token = encrypt_str(refresh_token)
    account.oauth_expires_at = oauth_expires_at

    db.commit()
    db.refresh(account)

    logger.info(
        f"Linked Gmail account for user_id={current_user.id}, email={email_address}"
    )

    # For now, just return JSON; later redirect back to your landing page/front-end
    return {
        "status": "ok",
        "message": "Gmail account connected",
        "email_address": email_address,
        "account_id": str(account.id),
    }


# ---------- Gmail helpers (token refresh + account fetch) ----------
def _get_gmail_account_or_404(
    account_id: str,
    current_user: User,
    db: Session,
) -> EmailAccount:
    account = (
        db.query(EmailAccount)
        .filter(
            EmailAccount.id == account_id,
            EmailAccount.user_id == current_user.id,
        )
        .first()
    )
    if not account:
        raise HTTPException(status_code=404, detail="Email account not found")

    if account.provider_type != "gmail" or account.oauth_provider != "google":
        raise HTTPException(
            status_code=400,
            detail="Email account is not a Gmail account linked via Google OAuth",
        )

    if not account.oauth_access_token:
        raise HTTPException(
            status_code=400,
            detail="No OAuth access token stored for this account",
        )

    return account


def ensure_gmail_access_token(account: EmailAccount, db: Session) -> str:
    """
    Returns a valid Gmail access token, refreshing with the stored refresh token
    if needed. Updates DB if a refresh occurs.
    """
    if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
        raise HTTPException(
            status_code=500,
            detail="Google OAuth not configured on server",
        )

    access_token = decrypt_str(account.oauth_access_token)
    refresh_token = decrypt_str(account.oauth_refresh_token)
    now = datetime.utcnow()

    # If token is still valid for at least 60s, reuse it
    if account.oauth_expires_at and access_token:
        if account.oauth_expires_at > now + timedelta(seconds=60):
            return access_token

    # Need to refresh
    if not refresh_token:
        raise HTTPException(
            status_code=401,
            detail=(
                "Gmail access token expired and no refresh token available. "
                "Please reconnect your Gmail account."
            ),
        )

    data = {
        "client_id": GOOGLE_CLIENT_ID,
        "client_secret": GOOGLE_CLIENT_SECRET,
        "refresh_token": refresh_token,
        "grant_type": "refresh_token",
    }

    with httpx.Client(timeout=10.0) as client_http:
        resp = client_http.post(GOOGLE_TOKEN_URL, data=data)
        if resp.status_code != 200:
            logger.error(
                f"Failed to refresh Gmail token: {resp.status_code} {resp.text}"
            )
            raise HTTPException(
                status_code=502,
                detail="Failed to refresh Gmail access token with Google",
            )

        token_data = resp.json()
        new_access_token = token_data.get("access_token")
        expires_in = token_data.get("expires_in")

        if not new_access_token:
            raise HTTPException(
                status_code=500,
                detail="Google did not return a new access token during refresh",
            )

        account.oauth_access_token = encrypt_str(new_access_token)
        if expires_in:
            account.oauth_expires_at = now + timedelta(seconds=expires_in)
        db.commit()
        db.refresh(account)

        return new_access_token


def _extract_headers(message_json: dict) -> dict:
    headers_list = (
        message_json.get("payload", {})
        .get("headers", [])
    )
    h = {hdr.get("name", "").lower(): hdr.get("value", "") for hdr in headers_list}
    return {
        "subject": h.get("subject"),
        "from": h.get("from"),
        "to": h.get("to"),
        "date": h.get("date"),
    }


# ---------- Core Gmail listing logic (reusable by endpoints and helper) ----------
def _gmail_list_messages_core(
    account_id: str,
    max_results: int,
    query: str | None,
    db: Session,
    current_user: User,
):
    """
    Core logic to list Gmail messages for an account.
    Returns the same JSON structure used by the /messages endpoint.
    """
    # Safety clamp: avoid accidentally slamming Gmail with huge requests
    if max_results <= 0:
        max_results = 1
    if max_results > 50:
        max_results = 50

    account = _get_gmail_account_or_404(account_id, current_user, db)
    access_token = ensure_gmail_access_token(account, db)

    params = {
        "maxResults": max_results,
    }
    if query:
        params["q"] = query

    with httpx.Client(timeout=10.0) as client_http:
        # 1) List message IDs
        list_url = f"{GMAIL_API_BASE}/users/me/messages"
        list_resp = client_http.get(
            list_url,
            headers={"Authorization": f"Bearer {access_token}"},
            params=params,
        )
        if list_resp.status_code != 200:
            logger.error(
                f"Gmail list messages failed: {list_resp.status_code} {list_resp.text}"
            )
            raise HTTPException(
                status_code=502,
                detail="Failed to list Gmail messages",
            )

        list_data = list_resp.json()
        messages = list_data.get("messages", [])
        size_estimate = list_data.get("resultSizeEstimate", len(messages))

        # 2) Fetch details for each message (up to max_results)
        detailed = []
        for msg in messages:
            msg_id = msg.get("id")
            if not msg_id:
                continue

            msg_url = f"{GMAIL_API_BASE}/users/me/messages/{msg_id}"
            msg_resp = client_http.get(
                msg_url,
                headers={"Authorization": f"Bearer {access_token}"},
                params={"format": "metadata"},
            )
            if msg_resp.status_code != 200:
                logger.warning(
                    f"Failed to fetch Gmail message {msg_id}: "
                    f"{msg_resp.status_code} {msg_resp.text}"
                )
                continue

            msg_json = msg_resp.json()
            headers = _extract_headers(msg_json)
            detailed.append(
                {
                    "id": msg_json.get("id"),
                    "threadId": msg_json.get("threadId"),
                    "snippet": msg_json.get("snippet"),
                    "subject": headers.get("subject"),
                    "from": headers.get("from"),
                    "to": headers.get("to"),
                    "date": headers.get("date"),
                }
            )

    return {
        "account_id": account_id,
        "email_address": account.email_address,
        "query": query,
        "resultSizeEstimate": size_estimate,
        "messages": detailed,
    }


# ---------- Gmail API usage endpoints ----------

@app.get("/email/accounts/{account_id}/messages")
def list_gmail_messages(
    account_id: str,
    max_results: int = 20,
    query: str | None = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    List recent Gmail messages for this account with basic metadata + snippet.

    Example queries:
      - /email/accounts/{account_id}/messages
      - /email/accounts/{account_id}/messages?max_results=10
      - /email/accounts/{account_id}/messages?query=from:amazon OR subject:invoice
    """
    return _gmail_list_messages_core(
        account_id=account_id,
        max_results=max_results,
        query=query,
        db=db,
        current_user=current_user,
    )


@app.get("/email/accounts/{account_id}/stats")
def gmail_stats(
    account_id: str,
    window_days: int = 1,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Approximate count of messages received in the last N days.
    Uses Gmail search 'newer_than:{N}d'.

    Examples:
      - /email/accounts/{account_id}/stats?window_days=1   -> "today-ish"
      - /email/accounts/{account_id}/stats?window_days=7   -> last week
      - /email/accounts/{account_id}/stats?window_days=30  -> last month-ish
    """
    if window_days <= 0:
        raise HTTPException(
            status_code=400,
            detail="window_days must be >= 1",
        )

    account = _get_gmail_account_or_404(account_id, current_user, db)
    access_token = ensure_gmail_access_token(account, db)

    query = f"newer_than:{window_days}d"

    with httpx.Client(timeout=10.0) as client_http:
        list_url = f"{GMAIL_API_BASE}/users/me/messages"
        list_resp = client_http.get(
            list_url,
            headers={"Authorization": f"Bearer {access_token}"},
            params={"q": query, "maxResults": 1},
        )
        if list_resp.status_code != 200:
            logger.error(
                f"Gmail stats list failed: {list_resp.status_code} {list_resp.text}"
            )
            raise HTTPException(
                status_code=502,
                detail="Failed to compute Gmail stats",
            )

        data = list_resp.json()
        size_estimate = data.get("resultSizeEstimate", 0)

    return {
        "account_id": account_id,
        "email_address": account.email_address,
        "window_days": window_days,
        "query": query,
        "approx_message_count": size_estimate,
    }


# ---------- Gmail summary helper for Vozlia (core logic) ----------
def summarize_gmail_messages_for_assistant(
    account_id: str,
    db: Session,
    current_user: User,
    max_results: int = 20,
    query: str | None = None,
) -> dict:
    """
    Helper for Vozlia's voice logic.

    Returns a dict:
      {
        "summary": "<short spoken-style summary>",
        "messages": [... up to max_results messages ...],
        "account_id": ...,
        "email_address": ...,
        "query": ...
      }
    """
    # Clamp here too, in case it's called directly
    if max_results <= 0:
        max_results = 1
    if max_results > 50:
        max_results = 50

    if not OPENAI_API_KEY or client is None:
        # Fallback: just return a plain-text description using subjects
        data = _gmail_list_messages_core(
            account_id=account_id,
            max_results=max_results,
            query=query,
            db=db,
            current_user=current_user,
        )
        messages = data.get("messages", [])
        if not messages:
            summary = "You have no recent emails matching that filter."
        else:
            subjects = [m.get("subject") or "(no subject)" for m in messages]
            joined = "; ".join(subjects[:5])
            summary = (
                f"You have {len(messages)} recent emails. "
                f"Some subjects include: {joined}."
            )
        data["summary"] = summary
        return data

    # Use OpenAI to create a tight, spoken-style summary.
    data = _gmail_list_messages_core(
        account_id=account_id,
        max_results=max_results,
        query=query,
        db=db,
        current_user=current_user,
    )
    messages = data.get("messages", [])

    if not messages:
        summary_text = "You have no recent emails matching that filter."
    else:
        # Truncate for prompt safety
        messages_for_prompt = messages[: max_results]

        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are Vozlia, an AI phone secretary. "
                            "Given a list of email metadata (subject, sender, snippet, date), "
                            "produce a VERY short spoken-style summary for the caller. "
                            "1–3 short sentences. Mention approximate counts and the most important themes, "
                            "like bills, important notices, or personal messages. "
                            "Do NOT read email addresses or long codes out loud."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            "Here is the list of recent emails:\n"
                            + json.dumps(messages_for_prompt, indent=2)
                            + "\n\nRespond with a short spoken summary only."
                        ),
                    },
                ],
            )
            summary_text = resp.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating Gmail summary via OpenAI: {e}")
            # Fallback to simple subject-based summary
            subjects = [m.get("subject") or "(no subject)" for m in messages]
            joined = "; ".join(subjects[:5])
            summary_text = (
                f"You have {len(messages)} recent emails. "
                f"Some subjects include: {joined}."
            )

    data["summary"] = summary_text
    return data


@app.get("/email/accounts/{account_id}/summary")
def gmail_summary(
    account_id: str,
    max_results: int = 20,
    query: str | None = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    High-level Gmail summary for this account, suitable for Vozlia to speak.

    Example:
      - /email/accounts/{account_id}/summary
      - /email/accounts/{account_id}/summary?query=is:unread&max_results=10
    """
    data = summarize_gmail_messages_for_assistant(
        account_id=account_id,
        db=db,
        current_user=current_user,
        max_results=max_results,
        query=query,
    )

    # Optionally trim the messages list to keep response lighter
    data["messages"] = data.get("messages", [])[: max_results]
    return data


# ---------- OpenAI config ----------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY is not set. GPT / Realtime calls will fail.")

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

OPENAI_REALTIME_MODEL = os.getenv(
    "OPENAI_REALTIME_MODEL",
    "gpt-4o-mini-realtime-preview-2024-12-17",
)
OPENAI_REALTIME_URL = f"wss://api.openai.com/v1/realtime?model={OPENAI_REALTIME_MODEL}"

OPENAI_REALTIME_HEADERS = {
    "Authorization": f"Bearer {OPENAI_API_KEY}" if OPENAI_API_KEY else "",
    "OpenAI-Beta": "realtime=v1",
}

SUPPORTED_VOICES = {
    "alloy",
    "ash",
    "ballad",
    "coral",
    "echo",
    "sage",
    "shimmer",
    "verse",
    "marin",
    "cedar",
}

voice_env = os.getenv("OPENAI_REALTIME_VOICE", "coral")
if voice_env not in SUPPORTED_VOICES:
    logger.warning(
        f"Unsupported voice '{voice_env}' for Realtime. "
        "Falling back to 'coral'. Supported voices: "
        + ", ".join(sorted(SUPPORTED_VOICES))
    )
    voice_env = "coral"

VOICE_NAME = voice_env

SYSTEM_PROMPT = (
    "You are Vozlia, a warm, friendly, highly capable AI phone assistant using the "
    "Coral voice persona. Speak naturally, confidently, and in a professional tone, "
    "similar to a helpful customer support agent. "
    "You greet callers immediately at the start of the call, introduce yourself as "
    "Vozlia, and invite them to describe what they need help with. "
    "Keep your responses very concise: usually no more than 2–3 short sentences "
    "(around 5–7 seconds of speech) before pausing. "
    "If the caller asks for a long explanation or story, summarize the key points "
    "briefly and offer to go deeper only if they ask. "
    "Be attentive to interruptions: if the caller starts speaking while you are "
    "talking, immediately stop and listen. "
    "Your goal is to make callers feel welcome, understood, and supported."
)

# ---------- Audio framing for G.711 μ-law ----------
SAMPLE_RATE = 8000        # Hz
FRAME_MS = 20             # 20 ms per frame
BYTES_PER_FRAME = int(SAMPLE_RATE * FRAME_MS / 1000)  # 160 bytes

# Real-time pacing: one 20ms frame every 20ms
FRAME_INTERVAL = FRAME_MS / 1000.0  # 0.020 seconds

# Prebuffer at start of each utterance to smooth jitter
# 4 frames = 80 ms
PREBUFFER_FRAMES = 4
PREBUFFER_BYTES = PREBUFFER_FRAMES * BYTES_PER_FRAME

# Limit how far ahead we've sent audio to Twilio (in seconds)
# This directly bounds the maximum "tail" after barge-in.
MAX_TWILIO_BACKLOG_SECONDS = 1.0


# ---------- Basic health endpoints ----------
@app.get("/")
async def root():
    return PlainTextResponse("OK")


@app.get("/health")
async def health():
    return {"status": "ok"}


# ---------- Debug GPT (text only) ----------
async def generate_gpt_reply(text: str) -> str:
    logger.info(f"/debug/gpt called with text: {text!r}")

    if not OPENAI_API_KEY or client is None:
        return "OpenAI API key is not configured on the server."

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are Vozlia, a helpful AI assistant."},
                {"role": "user", "content": text},
            ],
        )
        return resp.choices[0].message.content
    except Exception as e:
        logger.error(f"Error calling OpenAI chat.completions: {e}")
        return f"Error talking to GPT: {e}"


@app.get("/debug/gpt")
async def debug_gpt(text: str = "Hello Vozlia"):
    reply = await generate_gpt_reply(text)
    return JSONResponse({"reply": reply})


@app.get("/debug/me")
def debug_me(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    return {
        "id": str(current_user.id),
        "email": current_user.email,
    }


# ---------- Twilio inbound → TwiML ----------
@app.post("/twilio/inbound")
async def twilio_inbound(request: Request):
    form = await request.form()
    from_number = form.get("From")
    to_number = form.get("To")
    call_sid = form.get("CallSid")

    logger.info(
        f"Incoming call: From={from_number}, To={to_number}, CallSid={call_sid}"
    )

    resp = VoiceResponse()
    connect = Connect()

    stream_url = os.getenv(
        "TWILIO_STREAM_URL",
        "wss://vozlia-backend.onrender.com/twilio/stream",
    )
    connect.stream(url=stream_url)
    resp.append(connect)

    xml = str(resp)
    logger.debug(f"Generated TwiML:\n{xml}")

    return Response(content=xml, media_type="application/xml")


# ---------- Helper: initial greeting via Realtime ----------
async def send_initial_greeting(openai_ws):
    try:
        convo_item = {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "Greet the caller as Vozlia, a friendly AI phone assistant using "
                            "the Coral voice. Briefly introduce yourself and invite them to "
                            "say what they need help with."
                        ),
                    }
                ],
            },
        }
        await openai_ws.send(json.dumps(convo_item))
        await openai_ws.send(json.dumps({"type": "response.create"}))
        logger.info("Sent initial greeting request to OpenAI Realtime")
    except Exception as e:
        logger.error(f"Error sending initial greeting: {e}")


# ---------- Helper: OpenAI Realtime session via websockets ----------
async def create_realtime_session():
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set; cannot use Realtime API.")

    logger.info("Connecting to OpenAI Realtime WebSocket via websockets...")

    openai_ws = await websockets.connect(
        OPENAI_REALTIME_URL,
        extra_headers=OPENAI_REALTIME_HEADERS,
    )

    # IMPORTANT:
    #  - modalities MUST be ["audio", "text"] if you want audio.
    #  - input/output formats stay g711_ulaw for Twilio.
    #  - input_audio_transcription only takes a 'model' field (no 'enabled').
    session_update = {
        "type": "session.update",
        "session": {
            "instructions": SYSTEM_PROMPT,
            "voice": VOICE_NAME,
            "modalities": ["audio", "text"],
            "input_audio_format": "g711_ulaw",
            "output_audio_format": "g711_ulaw",
            "turn_detection": {
                "type": "server_vad",
            },
            "input_audio_transcription": {
                "model": "gpt-4o-mini-transcribe",
                # optional later:
                # "language": "en",
                # "prompt": "Phone call with a user talking to a virtual assistant.",
            },
        },
    }

    await openai_ws.send(json.dumps(session_update))
    logger.info("Sent session.update to OpenAI Realtime (with transcription enabled)")

    await send_initial_greeting(openai_ws)

    return openai_ws


# ---------- Twilio media stream ↔ OpenAI Realtime ----------
@app.websocket("/twilio/stream")
async def twilio_stream(websocket: WebSocket):
    await websocket.accept()
    logger.info("Twilio media stream connected")

    openai_ws = None
    stream_sid = None

    # AI / call state
    barge_in_enabled = False  # after first full response
    cancel_in_progress = False

    # Server VAD-based user speech flag
    user_speaking_vad = False

    # Outgoing assistant audio buffer (raw μ-law bytes)
    audio_buffer = bytearray()
    assistant_last_audio_time = 0.0

    # Prebuffer state: we hold back sending until we have PREBUFFER_BYTES
    prebuffer_active = False

    # Per-call transcript history (last few utterances)
    call_transcripts: list[str] = []

    def assistant_actively_speaking() -> bool:
        # If there's buffered audio, or we've sent audio very recently,
        # treat the assistant as actively speaking.
        if audio_buffer:
            return True
        return (time.monotonic() - assistant_last_audio_time) < 0.6

    try:
        openai_ws = await create_realtime_session()

        async def twilio_to_openai():
            nonlocal stream_sid, openai_ws
            while True:
                try:
                    msg_text = await websocket.receive_text()
                except WebSocketDisconnect:
                    logger.info("Twilio WebSocket disconnected")
                    break

                try:
                    data = json.loads(msg_text)
                except json.JSONDecodeError:
                    logger.warning(f"Non-JSON message from Twilio: {msg_text!r}")
                    continue

                event_type = data.get("event")

                if event_type != "media":
                    logger.info(f"Twilio stream event: {event_type}")

                if event_type == "start":
                    stream_sid = data.get("start", {}).get("streamSid")
                    logger.info(f"Stream started: {stream_sid}")

                elif event_type == "media":
                    payload_b64 = data.get("media", {}).get("payload")
                    if not payload_b64:
                        continue

                    # Always forward μ-law audio to OpenAI
                    audio_event = {
                        "type": "input_audio_buffer.append",
                        "audio": payload_b64,
                    }
                    await openai_ws.send(json.dumps(audio_event))

                elif event_type == "stop":
                    logger.info("Twilio sent stop; closing call.")
                    break

                elif event_type == "connected":
                    logger.info("Twilio reports call connected")

        async def openai_to_twilio():
            nonlocal stream_sid, barge_in_enabled, cancel_in_progress
            nonlocal user_speaking_vad, assistant_last_audio_time, audio_buffer, prebuffer_active, call_transcripts

            async for msg in openai_ws:
                try:
                    event = json.loads(msg)
                except json.JSONDecodeError:
                    logger.warning(f"Non-JSON message from OpenAI: {msg!r}")
                    continue

                etype = event.get("type")

                # ----- AUDIO FROM OPENAI → BUFFER FOR TWILIO -----
                if etype == "response.audio.delta":
                    audio_chunk_b64 = event.get("delta")
                    if not audio_chunk_b64:
                        continue

                    try:
                        raw_bytes = base64.b64decode(audio_chunk_b64)
                    except Exception as e:
                        logger.error(f"Error decoding audio delta: {e}")
                        continue

                    # If we were idle (no buffered audio), this is a new utterance:
                    if len(audio_buffer) == 0:
                        prebuffer_active = True

                    audio_buffer.extend(raw_bytes)

                elif etype == "response.audio.done":
                    logger.info("OpenAI finished an audio response")
                    cancel_in_progress = False
                    # After the first full response, allow barge-in.
                    if not barge_in_enabled:
                        barge_in_enabled = True
                        logger.info("Barge-in is now ENABLED for subsequent responses.")

                # ----- TEXT LOGGING (optional) -----
                elif etype == "response.text.delta":
                    delta = event.get("delta", "")
                    if delta:
                        logger.info(f"AI (text delta): {delta}")

                elif etype == "response.text.done":
                    text = event.get("text", "")
                    if text:
                        logger.info(f"AI full text response: {text}")

                # ----- USER TRANSCRIPTS (ASR from caller audio) -----
                elif etype == "conversation.item.input_audio_transcription.completed":
                    transcript = event.get("transcript")
                    item_id = event.get("item_id")
                    if transcript:
                        logger.info(
                            f"[ASR] User said (item_id={item_id}): {transcript!r}"
                        )

                        # Keep a short rolling history for better NLU
                        call_transcripts.append(transcript)
                        context_text = " ".join(call_transcripts[-5:])
                        logger.info(
                            f"[ASR] Combined context for brain: {context_text!r}"
                        )

                        # 🔁 Route the combined text through your text brain
                        speech = await run_brain_on_transcript(context_text)

                        if speech:
                            # Feed the brain's response back into the Realtime session
                            # so Coral will speak it to the caller.
                            try:
                                convo_item = {
                                    "type": "conversation.item.create",
                                    "item": {
                                        "type": "message",
                                        "role": "user",
                                        "content": [
                                            {
                                                "type": "input_text",
                                                "text": (
                                                    "Say the following sentence to the caller "
                                                    "exactly as written, without changing the meaning "
                                                    "and without adding any commentary: "
                                                    f"\"{speech}\""
                                                ),
                                            }
                                        ],
                                    },
                                }
                                await openai_ws.send(json.dumps(convo_item))
                                await openai_ws.send(
                                    json.dumps({"type": "response.create"})
                                )
                                logger.info(
                                    "[BRAIN] Sent brain-generated speech back to Realtime."
                                )
                            except Exception as e:
                                logger.error(
                                    f"Error sending brain response into Realtime: {e}"
                                )

                # ----- VAD EVENTS: drive user_speaking_vad & barge-in -----
                elif etype == "input_audio_buffer.speech_started":
                    user_speaking_vad = True
                    logger.info("OpenAI VAD: user speech START")

                    if (
                        barge_in_enabled
                        and assistant_actively_speaking()
                        and not cancel_in_progress
                    ):
                        try:
                            await openai_ws.send(
                                json.dumps({"type": "response.cancel"})
                            )
                            cancel_in_progress = True

                            # Drop any queued assistant audio so Twilio tail is minimal
                            audio_buffer.clear()
                            prebuffer_active = False

                            logger.info(
                                "BARGE-IN: user speech started while AI speaking; "
                                "sent response.cancel and cleared audio buffer."
                            )
                        except Exception as e:
                            logger.error(f"Error sending response.cancel: {e}")

                elif etype == "input_audio_buffer.speech_stopped":
                    user_speaking_vad = False
                    logger.info("OpenAI VAD: user speech STOP")

                # ----- ERROR EVENTS -----
                elif etype == "error":
                    logger.error(f"OpenAI error event: {event}")
                    err = event.get("error") or {}
                    code = err.get("code")

                    if code == "response_cancel_not_active":
                        # harmless: we tried to cancel after it already finished
                        cancel_in_progress = False

        async def twilio_audio_sender():
            """
            Convert the continuous μ-law audio buffer into fixed 20 ms frames
            and send them to Twilio at a steady cadence, with:
              - a small prebuffer per utterance to avoid choppiness
              - an explicit cap on how far ahead we've sent audio (backlog)
            """
            nonlocal assistant_last_audio_time, stream_sid, audio_buffer, prebuffer_active

            # Real-time pacing variables
            next_send_time = time.monotonic()
            call_start_time = time.monotonic()
            frames_sent = 0

            try:
                while True:
                    now = time.monotonic()

                    # Compute how far ahead Twilio is scheduled to play vs. wall clock.
                    audio_sent_duration = frames_sent * (FRAME_MS / 1000.0)
                    call_elapsed = max(0.0, now - call_start_time)
                    backlog_seconds = audio_sent_duration - call_elapsed

                    if backlog_seconds > MAX_TWILIO_BACKLOG_SECONDS:
                        # We've sent too far ahead. Pause sending and let real time catch up.
                        await asyncio.sleep(0.01)
                        # Reset next_send_time to "now" so we don't try to rush later.
                        next_send_time = time.monotonic()
                        continue

                    if stream_sid and len(audio_buffer) >= BYTES_PER_FRAME:
                        # If we're still prebuffering this utterance, wait until we have enough
                        if prebuffer_active and len(audio_buffer) < PREBUFFER_BYTES:
                            await asyncio.sleep(0.005)
                            continue
                        else:
                            prebuffer_active = False

                        # Take the next 20ms frame
                        frame_bytes = audio_buffer[:BYTES_PER_FRAME]
                        del audio_buffer[:BYTES_PER_FRAME]

                        frame_b64 = base64.b64encode(frame_bytes).decode("utf-8")

                        twilio_msg = {
                            "event": "media",
                            "streamSid": stream_sid,
                            "media": {"payload": frame_b64},
                        }

                        try:
                            await websocket.send_text(json.dumps(twilio_msg))
                            assistant_last_audio_time = time.monotonic()
                            frames_sent += 1
                        except WebSocketDisconnect:
                            logger.info("WebSocket disconnected while sending audio")
                            return
                        except Exception as e:
                            logger.error(f"Error sending audio frame to Twilio: {e}")

                        # Schedule next frame at real-time
                        next_send_time += FRAME_INTERVAL
                        sleep_for = max(0.0, next_send_time - time.monotonic())
                        await asyncio.sleep(sleep_for)
                    else:
                        # Nothing (or not enough) to send, avoid busy loop and
                        # keep next_send_time aligned with "now"
                        next_send_time = time.monotonic()
                        await asyncio.sleep(0.005)
            except asyncio.CancelledError:
                logger.info("twilio_audio_sender cancelled")
            except Exception as e:
                logger.exception(f"Error in twilio_audio_sender: {e}")

        await asyncio.gather(
            twilio_to_openai(),
            openai_to_twilio(),
            twilio_audio_sender(),
        )

    except Exception as e:
        logger.exception(f"Error in /twilio/stream: {e}")
    finally:
        if openai_ws is not None:
            try:
                await openai_ws.close()
            except Exception:
                pass

        try:
            await websocket.close()
        except Exception:
            pass

        logger.info("Twilio stream closed")

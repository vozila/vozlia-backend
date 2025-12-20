import os
import json
import asyncio
import logging
import base64
import time
from typing import List, Optional
from datetime import datetime, timedelta

from fastapi import (
    FastAPI,
    Request,
    WebSocket,
    WebSocketDisconnect,
    Depends,
    HTTPException,
)
from fastapi.responses import PlainTextResponse, Response, JSONResponse, RedirectResponse

from pydantic import BaseModel
from sqlalchemy.orm import Session

import httpx  # <-- for Google OAuth + Gmail API
from openai import OpenAI
import websockets

from cryptography.fernet import Fernet  # centralized crypto
from vozlia_twilio.stream import twilio_stream

from core.obs import maybe_record_event
from core.logging import logger
from core import config as cfg

from vozlia_fsm import VozliaFSM  # Flow B FSM
from vozlia_twilio.inbound import router as twilio_inbound_router

from db import Base, engine
from models import User, EmailAccount
from schemas import EmailAccountCreate, EmailAccountRead
from deps import get_db


# ===============================
# LOGGING
# ===============================
# NOTE: Logging is centralized in core/logging.py.
# Do NOT call logging.basicConfig() here or re-create the logger.


# ---------- FastAPI app ----------
app = FastAPI()

# Register routers AFTER app is defined
app.include_router(twilio_inbound_router)

app.add_api_websocket_route("/twilio/stream", twilio_stream)

# ---------- Crypto helpers (for passwords & OAuth tokens) ----------

def get_fernet() -> Fernet:
    key = os.getenv("ENCRYPTION_KEY")
    if not key:
        raise RuntimeError("ENCRYPTION_KEY is not configured")
    # ENCRYPTION_KEY should be something like Fernet.generate_key().decode()
    return Fernet(key.encode())



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


# ---------- FSM debug endpoint (text-only) ----------
@app.post("/fsm/debug")
async def fsm_debug(request: Request):
    """
    Simple text-only endpoint to exercise the VozliaFSM.

    Body example:
      { "text": "Do I have any unread emails?" }

    Returns whatever the FSM decides:
      {
        "intent": "...",
        "state": "...",
        "backend_call": { ... },
        "spoken_reply": "...",
        "raw": { ... }   # optional extra info
      }
    """
    body = await request.json()
    text = (body.get("text") or "").strip()

    if not text:
        raise HTTPException(status_code=400, detail="Missing 'text' in request body")

    # New FSM instance per request for now (stateless behavior)
    fsm = VozliaFSM()

    try:
        fsm_result = fsm.handle_utterance(text)
    except Exception as e:
        logger.exception("Error running VozliaFSM")
        raise HTTPException(
            status_code=500,
            detail=f"FSM error: {e}",
        )

    # You can normalize the shape here if needed:
    return {
        "input": text,
        "fsm_result": fsm_result,
    }


# ---------- Google / Gmail OAuth config ----------
# (extracted to core/config.py)
GOOGLE_CLIENT_ID = cfg.GOOGLE_CLIENT_ID
GOOGLE_CLIENT_SECRET = cfg.GOOGLE_CLIENT_SECRET
GOOGLE_REDIRECT_URI = cfg.GOOGLE_REDIRECT_URI
GOOGLE_GMAIL_SCOPE = cfg.GOOGLE_GMAIL_SCOPE
GOOGLE_TOKEN_URL = cfg.GOOGLE_TOKEN_URL
GMAIL_API_BASE = cfg.GMAIL_API_BASE


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


def _get_default_gmail_account_id(current_user: User, db: Session) -> str | None:
    """
    Returns the id of the user's primary Gmail account if present,
    otherwise the first active Gmail account. Returns None if none exist.
    """
    q = (
        db.query(EmailAccount)
        .filter(
            EmailAccount.user_id == current_user.id,
            EmailAccount.provider_type == "gmail",
            EmailAccount.oauth_provider == "google",
            EmailAccount.is_active == True,  # noqa: E712
        )
    )

    primary = q.filter(EmailAccount.is_primary == True).first()  # noqa: E712
    if primary:
        return str(primary.id)

    first = q.first()
    if first:
        return str(first.id)

    return None


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


# ---------- Assistant routing models (FSM entrypoint) ----------
class AssistantRouteIn(BaseModel):
    """
    Input body when the assistant (phone or ChatGPT) asks the backend
    what to do with a specific user utterance.
    """
    text: str
    account_id: str | None = None  # optional: explicit Gmail account
    context: dict | None = None    # optional: extra metadata (channel, etc.)


class AssistantRouteOut(BaseModel):
    """
    What the backend returns to the assistant:
      - spoken_reply: what Vozlia should actually say
      - fsm: raw FSM decision payload (intent, next_state, etc.)
      - gmail: optional Gmail summary data (if used)
    """
    spoken_reply: str
    fsm: dict
    gmail: dict | None = None


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


# ---------- Core assistant routing logic (FSM + skills) ----------
def _run_fsm_and_backend(
    text: str,
    db: Session,
    current_user: User,
    account_id: str | None = None,
    context: dict | None = None,
) -> AssistantRouteOut:
    """
    Single entrypoint for:
      - Running VozliaFSM on the caller's utterance.
      - Optionally calling backend skills (currently: Gmail summary).
      - Producing a spoken reply plus structured data.
    """
    fsm = VozliaFSM()

    # You can pass extra context if your FSM needs it (channel, user id, etc.)
    fsm_context = context or {}
    fsm_context.setdefault("user_id", current_user.id)
    fsm_context.setdefault("channel", "phone")  # or "chat" for GPT-side usage

    fsm_result: dict = fsm.handle_utterance(text, context=fsm_context)

    spoken_reply: str = fsm_result.get("spoken_reply") or ""
    backend_call: dict | None = fsm_result.get("backend_call") or None

    gmail_data: dict | None = None

    # --- Handle Gmail summary skill, if requested by FSM ---
    if backend_call and backend_call.get("type") == "gmail_summary":
        params = backend_call.get("params") or {}

        # Priority: explicit account_id in params > request body > default Gmail account
        account_id_effective = (
            params.get("account_id")
            or account_id
            or _get_default_gmail_account_id(current_user, db)
        )

        if not account_id_effective:
            # No Gmail account available – append a brief explanation.
            if spoken_reply:
                spoken_reply = (
                    spoken_reply.rstrip(". ")
                    + " However, I don't see a Gmail account connected for you yet."
                )
            else:
                spoken_reply = (
                    "I tried to check your email, but I don't see a Gmail "
                    "account connected for you yet."
                )
        else:
            gmail_query = params.get("query")
            gmail_max_results = params.get("max_results", 20)

            gmail_data = summarize_gmail_messages_for_assistant(
                account_id=account_id_effective,
                db=db,
                current_user=current_user,
                max_results=gmail_max_results,
                query=gmail_query,
            )

            gmail_summary = gmail_data.get("summary")
            if gmail_summary:
                # Combine FSM reply + Gmail summary in a natural way
                if spoken_reply:
                    spoken_reply = f"{spoken_reply.strip()} {gmail_summary.strip()}"
                else:
                    spoken_reply = gmail_summary.strip()

            # Make sure we expose which account_id was used
            gmail_data["used_account_id"] = account_id_effective

    # TODO: In the future, add other backend_call types here:
    #   - weather lookup
    #   - nearby_location search
    #   - task engine, calendar, etc.

    return AssistantRouteOut(
        spoken_reply=spoken_reply,
        fsm=fsm_result,
        gmail=gmail_data,
    )


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


# ---------- Assistant router endpoint (for phone + ChatGPT) ----------

# ===============================
# FLOW B HANDLER
# ===============================

@app.post("/assistant/route", response_model=AssistantRouteOut)
def assistant_route(
    payload: AssistantRouteIn,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Unified router for Vozlia.

    This is what:
      - The phone agent (OpenAI Realtime) can call via tools, and
      - Your future custom GPT can call as a tool,

    whenever the assistant needs to:
      - Interpret the caller's request (via FSM), and
      - Optionally use backend skills (Gmail, etc.)

    It returns:
      - spoken_reply: what Vozlia should actually say
      - fsm: raw FSM decision info
      - gmail: optional Gmail summary block (if used)
    """
    result = _run_fsm_and_backend(
        text=payload.text,
        db=db,
        current_user=current_user,
        account_id=payload.account_id,
        context=payload.context,
    )
    return result


# ---------- FSM router helper for external callers (Twilio, ChatGPT tools, etc.) ----------
# (extracted to core/config.py)
VOZLIA_BACKEND_BASE_URL = cfg.VOZLIA_BACKEND_BASE_URL


async def call_fsm_router(
    text: str,
    context: Optional[dict] = None,
    account_id: Optional[str] = None,
) -> dict:
    """
    Helper to call the unified /assistant/route endpoint from other parts of
    the system (e.g., Twilio calls, future ChatGPT tools).

    This keeps Twilio <-> Realtime code decoupled from the internal FSM wiring.
    """
    if not text:
        return {"spoken_reply": "", "fsm": {}, "gmail": None}

    url = VOZLIA_BACKEND_BASE_URL.rstrip("/") + "/assistant/route"
    payload: dict = {"text": text}
    if context is not None:
        payload["context"] = context
    if account_id is not None:
        payload["account_id"] = account_id

    try:
        async with httpx.AsyncClient(timeout=10.0) as client_http:
            resp = await client_http.post(url, json=payload)
            resp.raise_for_status()
            return resp.json()
    except Exception as e:
        logger.exception("Error calling /assistant/route at %s: %s", url, e)
        # Fallback: simple spoken error
        return {
            "spoken_reply": (
                "I tried to check that information in the backend, "
                "but something went wrong. Please try again in a moment."
            ),
            "fsm": {"error": str(e)},
            "gmail": None,
        }


# ---------- OpenAI config ----------
# (env-backed config extracted to core/config.py)
OPENAI_API_KEY = cfg.OPENAI_API_KEY
OPENAI_REALTIME_MODEL = cfg.OPENAI_REALTIME_MODEL
OPENAI_REALTIME_URL = cfg.OPENAI_REALTIME_URL

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

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

voice_env = cfg.OPENAI_REALTIME_VOICE
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
    "GENERAL KNOWLEDGE & ‘REAL-TIME’ QUESTIONS\n"
    "- You DO NOT mention that you are an AI model, or that you lack real-time internet, "
    "  GPS, or live data access.\n"
    "- When callers ask for things like the current temperature, nearby restaurants, "
    "  store hours, traffic, or other time-sensitive information, you still try to be "
    "  as helpful and concrete as possible.\n"
    "- Use your general knowledge and typical patterns to give a reasonable, clearly "
    "  framed estimate or guidance.\n"
    "- Briefly signal that it is approximate, for example by saying things like "
    "  'roughly', 'typically', or 'around this time of year'. Avoid long disclaimers.\n"
    "- Then give one simple, practical next step. For example:\n"
    "    * Weather: 'In early December, Queens is usually around the 30s to 40s "
    "      Fahrenheit, so you’ll probably want a warm jacket. For the exact temperature, "
    "      a quick weather app check will give you the current reading.'\n"
    "    * Nearby restaurants: 'If you open your maps app and search for "
    "      \"restaurants near me\", sorting by rating will show you the best options "
    "      close by.'\n"
    "- Avoid saying 'I cannot access the internet', 'I don’t have real-time data', "
    "  or directly telling them to 'look it up yourself'. Always add some concrete, "
    "  useful guidance or an approximate answer first.\n\n"
)

# 🔹 Realtime-specific rules about Gmail / email access
REALTIME_SYSTEM_PROMPT = SYSTEM_PROMPT + (
    "EMAIL & GMAIL ACCESS RULES\n"
    "- You DO have help with email and calendar via a secure backend brain.\n"
    "- The server may send you short EMAIL SUMMARIES that you must read to the caller.\n"
    "- NEVER say you 'can’t access email', 'can’t read your Gmail', or similar.\n"
    "- When you receive an EMAIL SUMMARY, you:\n"
    "    • Trust it as accurate.\n"
    "    • Read it out to the caller in natural speech.\n"
    "    • You may lightly rephrase for flow, but keep counts, topics, and deadlines.\n"
    "- If the caller asks about email before the backend has sent a summary, say a short "
    "  bridging phrase like 'Let me check that for you.' and pause while the backend "
    "  prepares the summary.\n"
    "- Do not mention tools, APIs, or backend details to the caller.\n"
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


# ---------- Twilio inbound → TwiML ----------

# ===============================
# TWILIO ENDPOINTS
# ===============================





# ===============================
# OBS
# ===============================
# NOTE: Observability is intentionally gated behind env flags. Keep it out of latency-sensitive paths.


# services/gmail_service.py

import base64
import json
import os
import time
from datetime import datetime, timedelta
from email.message import EmailMessage
from typing import Any, Dict, Optional

import httpx
from fastapi import HTTPException
from sqlalchemy.orm import Session

from core import config as cfg
from core.logging import logger
from core.security import decrypt_str, encrypt_str
from models import EmailAccount, User
from openai import OpenAI
from services.settings_service import get_selected_gmail_account_id

# Flags
GMAIL_DEBUG = os.getenv("GMAIL_DEBUG", "0") == "1"
OAUTH_DEBUG_LOGS = os.getenv("OAUTH_DEBUG_LOGS", "0") == "1"

# OpenAI (used only for summarization; safe if missing)
_client: Optional[OpenAI] = OpenAI(api_key=cfg.OPENAI_API_KEY) if getattr(cfg, "OPENAI_API_KEY", None) else None

# Config
GMAIL_API_BASE = getattr(cfg, "GMAIL_API_BASE", "https://gmail.googleapis.com/gmail/v1")
GOOGLE_TOKEN_URL = getattr(cfg, "GOOGLE_TOKEN_URL", "https://oauth2.googleapis.com/token")
GOOGLE_CLIENT_ID = getattr(cfg, "GOOGLE_CLIENT_ID", None)
GOOGLE_CLIENT_SECRET = getattr(cfg, "GOOGLE_CLIENT_SECRET", None)


def get_gmail_account_or_404(account_id: str, current_user: User, db: Session) -> EmailAccount:
    account = (
        db.query(EmailAccount)
        .filter(EmailAccount.id == account_id, EmailAccount.user_id == current_user.id)
        .first()
    )
    if not account:
        raise HTTPException(status_code=404, detail="Email account not found")

    if account.provider_type != "gmail" or account.oauth_provider != "google":
        raise HTTPException(
            status_code=400,
            detail="Email account is not a Gmail account linked via Google OAuth",
        )

    # Access token might be absent if user disconnected or token was reset
    if not account.oauth_access_token:
        raise HTTPException(status_code=400, detail="No OAuth access token stored for this account")

    return account


def get_default_gmail_account_id(current_user: User, db: Session) -> Optional[str]:
    """
    Determines the default Gmail account for the user.
    Priority:
      1) Explicit selection (settings_service)
      2) Primary Gmail account
      3) First active Gmail account
    """
    selected = get_selected_gmail_account_id(db, current_user)
    if selected:
        row = (
            db.query(EmailAccount)
            .filter(
                EmailAccount.id == selected,
                EmailAccount.user_id == current_user.id,
                EmailAccount.provider_type == "gmail",
                EmailAccount.oauth_provider == "google",
                EmailAccount.is_active == True,  # noqa: E712
            )
            .first()
        )
        if row:
            return str(row.id)

    if OAUTH_DEBUG_LOGS:
        logger.info(
            "GMAIL_DEFAULT_ACCOUNT_LOOKUP user_id=%s email=%s",
            getattr(current_user, "id", None),
            getattr(current_user, "email", None),
        )

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
    return str(first.id) if first else None


def ensure_gmail_access_token(account: EmailAccount, db: Session) -> str:
    """
    Returns a valid access token (refreshing if needed).
    Raises HTTPException on failure so callers can bubble it up cleanly.
    """
    if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
        raise HTTPException(status_code=500, detail="Google OAuth not configured on server")

    # decrypt safely
    access_token = decrypt_str(account.oauth_access_token) if account.oauth_access_token else None
    refresh_token = decrypt_str(account.oauth_refresh_token) if account.oauth_refresh_token else None

    now = datetime.utcnow()

    # If token is still valid for at least 60 seconds, reuse it
    if account.oauth_expires_at and access_token and account.oauth_expires_at > now + timedelta(seconds=60):
        return access_token

    if not refresh_token:
        raise HTTPException(
            status_code=401,
            detail="Gmail token expired and no refresh token available. Reconnect Gmail.",
        )

    data = {
        "client_id": GOOGLE_CLIENT_ID,
        "client_secret": GOOGLE_CLIENT_SECRET,
        "refresh_token": refresh_token,
        "grant_type": "refresh_token",
    }

    with httpx.Client(timeout=10.0) as client_http:
        if OAUTH_DEBUG_LOGS:
            logger.info(
                "GMAIL_REFRESH_USING_CLIENT client_id_suffix=%s token_url=%s",
                (GOOGLE_CLIENT_ID[-8:] if GOOGLE_CLIENT_ID else None),
                GOOGLE_TOKEN_URL,
            )
            logger.info(
                "GMAIL_REFRESH_CLIENT_CHECK client_id_suffix=%s secret_present=%s secret_len=%s",
                (GOOGLE_CLIENT_ID[-12:] if GOOGLE_CLIENT_ID else None),
                bool(GOOGLE_CLIENT_SECRET),
                (len(GOOGLE_CLIENT_SECRET) if GOOGLE_CLIENT_SECRET else 0),
            )

        resp = client_http.post(GOOGLE_TOKEN_URL, data=data)

        if resp.status_code != 200:
            # Always log enough to debug refresh failures
            logger.error(
                "GMAIL_REFRESH_FAILED status=%s body=%s account_id=%s user_id=%s email_address=%s expires_at=%s",
                resp.status_code,
                resp.text,
                getattr(account, "id", None),
                getattr(account, "user_id", None),
                getattr(account, "email_address", None),
                getattr(account, "oauth_expires_at", None),
            )
            raise HTTPException(status_code=502, detail="Failed to refresh Gmail access token with Google")

        token_data = resp.json()
        new_access_token = token_data.get("access_token")
        expires_in = token_data.get("expires_in")

        if not new_access_token:
            raise HTTPException(status_code=500, detail="Google did not return a new access token during refresh")

        account.oauth_access_token = encrypt_str(new_access_token)
        if expires_in:
            try:
                account.oauth_expires_at = now + timedelta(seconds=int(expires_in))
            except Exception:
                account.oauth_expires_at = now + timedelta(seconds=3600)

        db.commit()
        db.refresh(account)
        return new_access_token


def _headers_to_dict(headers_list: Any) -> Dict[str, str]:
    headers: Dict[str, str] = {}
    if not isinstance(headers_list, list):
        return headers
    for hdr in headers_list:
        if not isinstance(hdr, dict):
            continue
        name = (hdr.get("name") or "").lower()
        value = hdr.get("value") or ""
        if name:
            headers[name] = value
    return headers


def gmail_list_messages(
    account_id: str,
    current_user: User,
    db: Session,
    max_results: int = 20,
    query: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Lists recent Gmail messages with basic metadata (subject/from/to/date/snippet).
    """
    t0 = time.perf_counter()

    if max_results <= 0:
        max_results = 1
    if max_results > 50:
        max_results = 50

    if GMAIL_DEBUG:
        logger.info(
            "GMAIL_LIST_START account_id=%s user_id=%s max_results=%s query=%r",
            account_id,
            getattr(current_user, "id", None),
            max_results,
            query,
        )

    account = get_gmail_account_or_404(account_id, current_user, db)
    access_token = ensure_gmail_access_token(account, db)

    params: Dict[str, Any] = {"maxResults": max_results}
    if query:
        params["q"] = query

    detailed: list[Dict[str, Any]] = []
    size_estimate: int = 0

    with httpx.Client(timeout=10.0) as client_http:
        list_url = f"{GMAIL_API_BASE}/users/me/messages"
        list_resp = client_http.get(
            list_url,
            headers={"Authorization": f"Bearer {access_token}"},
            params=params,
        )
        if list_resp.status_code != 200:
            logger.error("GMAIL_LIST_FAILED status=%s body=%s", list_resp.status_code, list_resp.text)
            raise HTTPException(status_code=502, detail="Failed to list Gmail messages")

        list_data = list_resp.json()
        messages = list_data.get("messages", []) or []
        size_estimate = int(list_data.get("resultSizeEstimate", len(messages)) or 0)

        for msg in messages:
            if not isinstance(msg, dict):
                continue
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
                    "GMAIL_MESSAGE_METADATA_FAILED message_id=%s status=%s body=%s",
                    msg_id,
                    msg_resp.status_code,
                    msg_resp.text,
                )
                continue

            msg_json = msg_resp.json() or {}
            headers = _headers_to_dict((msg_json.get("payload") or {}).get("headers"))

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

    if GMAIL_DEBUG:
        dt_ms = int((time.perf_counter() - t0) * 1000)
        logger.info(
            "GMAIL_LIST_OK account_id=%s ms=%s size_estimate=%s returned=%s",
            account_id,
            dt_ms,
            size_estimate,
            len(detailed),
        )

    return {
        "account_id": account_id,
        "email_address": account.email_address,
        "query": query,
        "resultSizeEstimate": size_estimate,
        "messages": detailed,
    }


def summarize_gmail_for_assistant(
    account_id: str,
    current_user: User,
    db: Session,
    max_results: int = 20,
    query: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Produces a short spoken-style summary of recent emails.
    Falls back to a simple subject list if OpenAI isn't configured or errors.
    """
    t0 = time.perf_counter()

    if GMAIL_DEBUG:
        logger.info(
            "GMAIL_SUMMARY_START account_id=%s user_id=%s max_results=%s query=%r",
            account_id,
            getattr(current_user, "id", None),
            max_results,
            query,
        )

    data = gmail_list_messages(account_id, current_user, db, max_results=max_results, query=query)
    messages = data.get("messages", []) or []

    if not messages:
        data["summary"] = "You have no recent emails matching that filter."
        return data

    # Fallback summary if OpenAI not configured
    if not getattr(cfg, "OPENAI_API_KEY", None) or _client is None:
        subjects = [m.get("subject") or "(no subject)" for m in messages if isinstance(m, dict)]
        data["summary"] = (
            f"You have {len(messages)} recent emails. Some subjects include: " + "; ".join(subjects[:5]) + "."
        )
        return data

    try:
        resp = _client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are Vozlia. Given email metadata (subject, sender, snippet, date), "
                        "produce a VERY short spoken-style summary (1–3 sentences). "
                        "Do NOT read email addresses or long codes out loud."
                    ),
                },
                {"role": "user", "content": "Recent emails:\n" + json.dumps(messages[:max_results], indent=2)},
            ],
        )
        data["summary"] = resp.choices[0].message.content
        used_openai = True
    except Exception as e:
        logger.error("GMAIL_SUMMARY_OPENAI_ERROR err=%s", e)
        subjects = [m.get("subject") or "(no subject)" for m in messages if isinstance(m, dict)]
        data["summary"] = (
            f"You have {len(messages)} recent emails. Some subjects include: " + "; ".join(subjects[:5]) + "."
        )
        used_openai = False

    if GMAIL_DEBUG:
        dt_ms = int((time.perf_counter() - t0) * 1000)
        summary = data.get("summary")
        logger.info(
            "GMAIL_SUMMARY_OK account_id=%s ms=%s messages=%s summary_len=%s used_openai=%s",
            account_id,
            dt_ms,
            len(messages),
            (len(summary) if isinstance(summary, str) else None),
            used_openai,
        )

    return data


def gmail_get_message(
    account_id: str,
    current_user: User,
    db: Session,
    message_id: str,
    format: str = "full",
) -> Dict[str, Any]:
    """
    Fetch a single Gmail message by ID.
    Used by /email API routes; does not affect voice flow unless called.
    """
    account = get_gmail_account_or_404(account_id, current_user, db)
    access_token = ensure_gmail_access_token(account, db)

    with httpx.Client(timeout=15.0) as client_http:
        url = f"{GMAIL_API_BASE}/users/me/messages/{message_id}"
        resp = client_http.get(
            url,
            headers={"Authorization": f"Bearer {access_token}"},
            params={"format": format},
        )
        if resp.status_code != 200:
            logger.error(
                "GMAIL_GET_MESSAGE_FAILED status=%s body=%s account_id=%s message_id=%s",
                resp.status_code,
                resp.text,
                account_id,
                message_id,
            )
            raise HTTPException(status_code=resp.status_code, detail="Failed to fetch Gmail message")
        return resp.json()


def gmail_reply_to_message(
    account_id: str,
    current_user: User,
    db: Session,
    message_id: str,
    reply_text: str,
    *,
    subject_prefix: str = "Re: ",
) -> Dict[str, Any]:
    """
    Send a basic reply to an existing Gmail message.
    Minimal implementation: fetch metadata, reply in same thread.
    """
    # Ensure account belongs to user and tokens are valid
    account = get_gmail_account_or_404(account_id, current_user, db)
    access_token = ensure_gmail_access_token(account, db)

    # Fetch metadata for recipient/subject/threading
    meta = gmail_get_message(account_id, current_user, db, message_id, format="metadata")
    thread_id = meta.get("threadId")

    headers = _headers_to_dict((meta.get("payload") or {}).get("headers"))
    to_addr = headers.get("reply-to") or headers.get("from")
    orig_subject = headers.get("subject") or ""
    orig_msgid = headers.get("message-id")

    if not to_addr:
        raise HTTPException(status_code=400, detail="Cannot determine recipient for reply (missing From/Reply-To).")

    subject = orig_subject
    if subject and not subject.lower().startswith("re:"):
        subject = f"{subject_prefix}{subject}"
    elif not subject:
        subject = f"{subject_prefix}(no subject)"

    msg = EmailMessage()
    msg["To"] = to_addr
    msg["Subject"] = subject
    if orig_msgid:
        msg["In-Reply-To"] = orig_msgid
        msg["References"] = orig_msgid
    msg.set_content(reply_text)

    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode("utf-8")
    payload: Dict[str, Any] = {"raw": raw}
    if thread_id:
        payload["threadId"] = thread_id

    with httpx.Client(timeout=15.0) as client_http:
        url = f"{GMAIL_API_BASE}/users/me/messages/send"
        resp = client_http.post(
            url,
            headers={"Authorization": f"Bearer {access_token}"},
            json=payload,
        )
        if resp.status_code not in (200, 201):
            logger.error(
                "GMAIL_REPLY_FAILED status=%s body=%s account_id=%s message_id=%s",
                resp.status_code,
                resp.text,
                account_id,
                message_id,
            )
            raise HTTPException(status_code=resp.status_code, detail="Failed to send Gmail reply")
        return resp.json()

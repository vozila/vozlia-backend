# services/gmail_service.py
import json
import httpx
import os
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from fastapi import HTTPException
from services.settings_service import get_selected_gmail_account_id

from core.logging import logger
from core import config as cfg
from core.security import encrypt_str, decrypt_str
from models import EmailAccount, User
from openai import OpenAI

client = OpenAI(api_key=cfg.OPENAI_API_KEY) if cfg.OPENAI_API_KEY else None

GMAIL_API_BASE = cfg.GMAIL_API_BASE
GOOGLE_TOKEN_URL = cfg.GOOGLE_TOKEN_URL
GOOGLE_CLIENT_ID = cfg.GOOGLE_CLIENT_ID
GOOGLE_CLIENT_SECRET = cfg.GOOGLE_CLIENT_SECRET


def get_gmail_account_or_404(account_id: str, current_user: User, db: Session) -> EmailAccount:
    account = (
        db.query(EmailAccount)
        .filter(EmailAccount.id == account_id, EmailAccount.user_id == current_user.id)
        .first()
    )
    if not account:
        raise HTTPException(status_code=404, detail="Email account not found")

    if account.provider_type != "gmail" or account.oauth_provider != "google":
        raise HTTPException(status_code=400, detail="Email account is not a Gmail account linked via Google OAuth")

    if not account.oauth_access_token:
        raise HTTPException(status_code=400, detail="No OAuth access token stored for this account")

    return account


def get_default_gmail_account_id(current_user: User, db: Session) -> str | None:
    import os
    selected = get_selected_gmail_account_id(db, current_user)
    if selected:
        # verify it still exists & belongs to user & is gmail/google & active
        row = (
            db.query(EmailAccount)
            .filter(
                EmailAccount.id == selected,
                EmailAccount.user_id == current_user.id,
                EmailAccount.provider_type == "gmail",
                EmailAccount.oauth_provider == "google",
                EmailAccount.is_active == True,  # noqa
            )
            .first()
        )
        if row:
            return str(row.id)

    OAUTH_DEBUG_LOGS = os.getenv("OAUTH_DEBUG_LOGS", "0") == "1"
    if OAUTH_DEBUG_LOGS:
        logger.info("GMAIL_DEFAULT_ACCOUNT_LOOKUP user_id=%s email=%s", current_user.id, getattr(current_user, "email", None))

    q = (
        db.query(EmailAccount)
        .filter(
            EmailAccount.user_id == current_user.id,
            EmailAccount.provider_type == "gmail",
            EmailAccount.oauth_provider == "google",
            EmailAccount.is_active == True,  # noqa
        )
    )
    primary = q.filter(EmailAccount.is_primary == True).first()  # noqa
    if primary:
        return str(primary.id)
    first = q.first()
    return str(first.id) if first else None


def ensure_gmail_access_token(account: EmailAccount, db: Session) -> str:
    import os  # safe even if also imported globally

    if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
        raise HTTPException(status_code=500, detail="Google OAuth not configured on server")

    OAUTH_DEBUG_LOGS = os.getenv("OAUTH_DEBUG_LOGS", "0") == "1"

    access_token = decrypt_str(account.oauth_access_token)
    refresh_token = decrypt_str(account.oauth_refresh_token)
    now = datetime.utcnow()

    if account.oauth_expires_at and access_token and account.oauth_expires_at > now + timedelta(seconds=60):
        return access_token

    if not refresh_token:
        raise HTTPException(status_code=401, detail="Gmail token expired and no refresh token available. Reconnect Gmail.")

    data = {
        "client_id": GOOGLE_CLIENT_ID,
        "client_secret": GOOGLE_CLIENT_SECRET,
        "refresh_token": refresh_token,
        "grant_type": "refresh_token",
    }

    with httpx.Client(timeout=10.0) as client_http:
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
            if OAUTH_DEBUG_LOGS:
                logger.error(
                    "GMAIL_REFRESH_FAILED status=%s body=%s account_id=%s user_id=%s email_address=%s expires_at=%s",
                    resp.status_code,
                    resp.text,
                    getattr(account, "id", None),
                    getattr(account, "user_id", None),
                    getattr(account, "email_address", None),
                    getattr(account, "oauth_expires_at", None),
                )
            else:
                logger.error("Failed to refresh Gmail token: %s %s", resp.status_code, resp.text)

            # ALWAYS raise on refresh failure
            raise HTTPException(status_code=502, detail="Failed to refresh Gmail access token with Google")

        token_data = resp.json()
        new_access_token = token_data.get("access_token")
        expires_in = token_data.get("expires_in")

        if not new_access_token:
            raise HTTPException(status_code=500, detail="Google did not return a new access token during refresh")

        account.oauth_access_token = encrypt_str(new_access_token)
        if expires_in:
            account.oauth_expires_at = now + timedelta(seconds=expires_in)

        db.commit()
        db.refresh(account)
        return new_access_token


def gmail_list_messages(account_id: str, current_user: User, db: Session, max_results: int = 20, query: str | None = None) -> dict:
    if max_results <= 0:
        max_results = 1
    if max_results > 50:
        max_results = 50

    account = get_gmail_account_or_404(account_id, current_user, db)
    access_token = ensure_gmail_access_token(account, db)

    params = {"maxResults": max_results}
    if query:
        params["q"] = query

    with httpx.Client(timeout=10.0) as client_http:
        list_url = f"{GMAIL_API_BASE}/users/me/messages"
        list_resp = client_http.get(list_url, headers={"Authorization": f"Bearer {access_token}"}, params=params)
        if list_resp.status_code != 200:
            logger.error("Gmail list messages failed: %s %s", list_resp.status_code, list_resp.text)
            raise HTTPException(status_code=502, detail="Failed to list Gmail messages")

        list_data = list_resp.json()
        messages = list_data.get("messages", [])
        size_estimate = list_data.get("resultSizeEstimate", len(messages))

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
                logger.warning("Failed to fetch Gmail message %s: %s %s", msg_id, msg_resp.status_code, msg_resp.text)
                continue

            msg_json = msg_resp.json()
            headers_list = msg_json.get("payload", {}).get("headers", [])
            h = {hdr.get("name", "").lower(): hdr.get("value", "") for hdr in headers_list}
            detailed.append({
                "id": msg_json.get("id"),
                "threadId": msg_json.get("threadId"),
                "snippet": msg_json.get("snippet"),
                "subject": h.get("subject"),
                "from": h.get("from"),
                "to": h.get("to"),
                "date": h.get("date"),
            })

    return {
        "account_id": account_id,
        "email_address": account.email_address,
        "query": query,
        "resultSizeEstimate": size_estimate,
        "messages": detailed,
    }


def summarize_gmail_for_assistant(account_id: str, current_user: User, db: Session, max_results: int = 20, query: str | None = None) -> dict:
    data = gmail_list_messages(account_id, current_user, db, max_results=max_results, query=query)
    messages = data.get("messages", [])

    if not cfg.OPENAI_API_KEY or client is None:
        # fallback summary
        if not messages:
            data["summary"] = "You have no recent emails matching that filter."
            return data
        subjects = [m.get("subject") or "(no subject)" for m in messages]
        data["summary"] = f"You have {len(messages)} recent emails. Some subjects include: " + "; ".join(subjects[:5]) + "."
        return data

    if not messages:
        data["summary"] = "You have no recent emails matching that filter."
        return data

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": (
                    "You are Vozlia. Given email metadata (subject, sender, snippet, date), "
                    "produce a VERY short spoken-style summary (1–3 sentences). "
                    "Do NOT read email addresses or long codes out loud."
                )},
                {"role": "user", "content": "Recent emails:\n" + json.dumps(messages[:max_results], indent=2)},
            ],
        )
        data["summary"] = resp.choices[0].message.content
    except Exception as e:
        logger.error("Error generating Gmail summary via OpenAI: %s", e)
        subjects = [m.get("subject") or "(no subject)" for m in messages]
        data["summary"] = f"You have {len(messages)} recent emails. Some subjects include: " + "; ".join(subjects[:5]) + "."

    return data

# ---------------------------------------------------------------------------
# Expanded Gmail adapter endpoints
# ---------------------------------------------------------------------------

import base64
from email.message import EmailMessage


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("utf-8").rstrip("=")


def _extract_text_from_payload(payload: dict) -> str:
    """Best-effort extraction of readable text from Gmail 'full' payload.

    Strategy:
    - Prefer text/plain parts
    - Fall back to text/html stripped
    - Fall back to snippet
    """
    if not payload:
        return ""

    def decode_body(body_b64: str) -> str:
        if not body_b64:
            return ""
        # Gmail uses base64url without padding sometimes
        padding = "=" * (-len(body_b64) % 4)
        raw = base64.urlsafe_b64decode((body_b64 + padding).encode("utf-8"))
        try:
            return raw.decode("utf-8", errors="ignore")
        except Exception:
            return raw.decode(errors="ignore")

    def walk_parts(part: dict):
        if not part:
            return
        yield part
        for child in part.get("parts") or []:
            yield from walk_parts(child)

    # Collect candidate bodies
    plain_texts = []
    html_texts = []
    for part in walk_parts(payload):
        mime = (part.get("mimeType") or "").lower()
        body = part.get("body") or {}
        data = body.get("data")
        if not data:
            continue
        txt = decode_body(data)
        if not txt:
            continue
        if mime == "text/plain":
            plain_texts.append(txt)
        elif mime == "text/html":
            html_texts.append(txt)

    if plain_texts:
        return "\n\n".join(plain_texts).strip()

    if html_texts:
        # super light strip: remove tags
        import re
        s = "\n\n".join(html_texts)
        s = re.sub(r"<\s*br\s*/?>", "\n", s, flags=re.I)
        s = re.sub(r"<[^>]+>", "", s)
        return s.strip()

    return (payload.get("snippet") or "").strip()


def gmail_get_message(account_id: str, user: User, db: Session, message_id: str) -> dict:
    """Fetch a specific Gmail message with extracted body text."""
    account = get_gmail_account_or_404(account_id, user, db)
    access_token = ensure_gmail_access_token(account, db)

    url = f"{GMAIL_API_BASE}/users/me/messages/{message_id}"
    params = {"format": "full"}
    with httpx.Client(timeout=20.0) as client:
        resp = client.get(url, headers={"Authorization": f"Bearer {access_token}"}, params=params)
        resp.raise_for_status()
        msg = resp.json()

    payload = msg.get("payload") or {}
    headers = { (h.get("name") or "").lower(): (h.get("value") or "") for h in (payload.get("headers") or []) }
    body_text = _extract_text_from_payload(payload)

    return {
        "account_id": account_id,
        "email_address": account.email_address,
        "id": msg.get("id"),
        "threadId": msg.get("threadId"),
        "snippet": msg.get("snippet"),
        "internalDate": msg.get("internalDate"),
        "headers": headers,
        "body_text": body_text,
    }


def gmail_reply_to_message(account_id: str, user: User, db: Session, message_id: str, body: str, reply_all: bool = False) -> dict:
    """Reply to an existing message in-thread.

    Requires OAuth scope: https://www.googleapis.com/auth/gmail.send
    """
    # Fetch original message to get thread, subject, from/to, Message-ID header
    original = gmail_get_message(account_id, user, db, message_id)
    thread_id = original.get("threadId")
    headers = original.get("headers") or {}

    subj = headers.get("subject") or ""
    if subj and not subj.lower().startswith("re:"):
        subj = "Re: " + subj

    from_addr = original.get("email_address")
    to_addr = headers.get("reply-to") or headers.get("from") or ""
    if reply_all:
        # naive reply-all: include original To + Cc if present
        # (Gmail will also thread by In-Reply-To/References)
        pass

    message_id_hdr = headers.get("message-id") or ""

    msg = EmailMessage()
    msg["To"] = to_addr
    msg["From"] = from_addr
    msg["Subject"] = subj
    if message_id_hdr:
        msg["In-Reply-To"] = message_id_hdr
        msg["References"] = message_id_hdr
    msg.set_content(body or "")

    raw = _b64url_encode(msg.as_bytes())

    account = get_gmail_account_or_404(account_id, user, db)
    access_token = ensure_gmail_access_token(account, db)
    url = f"{GMAIL_API_BASE}/users/me/messages/send"
    payload = {"raw": raw}
    if thread_id:
        payload["threadId"] = thread_id

    with httpx.Client(timeout=20.0) as client:
        resp = client.post(url, headers={"Authorization": f"Bearer {access_token}"}, json=payload)
        resp.raise_for_status()
        sent = resp.json()

    return {
        "account_id": account_id,
        "email_address": account.email_address,
        "sent": sent,
    }

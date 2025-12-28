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
    import os as _os
    import time as _time
    api_debug = _os.getenv('GMAIL_API_DEBUG_LOGS', '0') == '1'
    t0 = _time.perf_counter()

    if max_results <= 0:
        max_results = 1
    if max_results > 50:
        max_results = 50

    account = get_gmail_account_or_404(account_id, current_user, db)
    access_token = ensure_gmail_access_token(account, db)
    if api_debug:
        logger.info('GMAIL_LIST_START account_id=%s email=%s max_results=%s query=%r', account_id, getattr(account, 'email_address', None), max_results, query)

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
        if api_debug:
            logger.info('GMAIL_LIST_OK dt_ms=%s size_estimate=%s ids=%s', int((_time.perf_counter()-t0)*1000), size_estimate, len(messages))

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

    if api_debug:
        logger.info('GMAIL_LIST_DONE dt_ms=%s detailed=%s', int((_time.perf_counter()-t0)*1000), len(detailed))

    return {
        "account_id": account_id,
        "email_address": account.email_address,
        "query": query,
        "resultSizeEstimate": size_estimate,
        "messages": detailed,
    }


def summarize_gmail_for_assistant(account_id: str, current_user: User, db: Session, max_results: int = 20, query: str | None = None) -> dict:
    import os as _os
    import time as _time
    summary_debug = _os.getenv('GMAIL_SUMMARY_DEBUG_LOGS', '0') == '1'
    t0 = _time.perf_counter()

    data = gmail_list_messages(account_id, current_user, db, max_results=max_results, query=query)
    if summary_debug:
        logger.info('GMAIL_SUMMARY_START account_id=%s max_results=%s query=%r', account_id, max_results, query)
    messages = data.get("messages", [])

    if not cfg.OPENAI_API_KEY or client is None:
        # fallback summary
        if not messages:
            data["summary"] = "You have no recent emails matching that filter."
            return data
        subjects = [m.get("subject") or "(no subject)" for m in messages]
        data["summary"] = f"You have {len(messages)} recent emails. Some subjects include: " + "; ".join(subjects[:5]) + "."
        if summary_debug:
            logger.info('GMAIL_SUMMARY_FALLBACK dt_ms=%s messages=%s', int((_time.perf_counter()-t0)*1000), len(messages))
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
        if summary_debug:
            logger.info('GMAIL_SUMMARY_OPENAI_OK dt_ms=%s summary_len=%s', int((_time.perf_counter()-t0)*1000), len((data.get('summary') or '')))
    except Exception as e:
        logger.error("Error generating Gmail summary via OpenAI: %s", e)
        subjects = [m.get("subject") or "(no subject)" for m in messages]
        data["summary"] = f"You have {len(messages)} recent emails. Some subjects include: " + "; ".join(subjects[:5]) + "."
        if summary_debug:
            logger.info('GMAIL_SUMMARY_FALLBACK dt_ms=%s messages=%s', int((_time.perf_counter()-t0)*1000), len(messages))

    return data

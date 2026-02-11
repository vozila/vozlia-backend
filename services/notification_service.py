# services/notification_service.py
from __future__ import annotations

import os
import smtplib
import time
from email.message import EmailMessage
from dataclasses import dataclass

from twilio.rest import Client

from core.logging import logger

def _env_flag(name: str, default: str = "0") -> bool:
    v = (os.getenv(name, default) or default).strip().lower()
    return v in ("1", "true", "yes", "y", "on")


SMTP_CIRCUIT_BREAKER_ENABLED = _env_flag("SMTP_CIRCUIT_BREAKER_ENABLED", "1")
SMTP_CIRCUIT_BREAKER_S = int((os.getenv("SMTP_CIRCUIT_BREAKER_S") or "900").strip() or "900")
SMTP_CIRCUIT_BREAKER_LOG_S = int((os.getenv("SMTP_CIRCUIT_BREAKER_LOG_S") or "60").strip() or "60")

_SMTP_BREAKER_OPEN_UNTIL: float = 0.0
_SMTP_BREAKER_LAST_LOG: float = 0.0

from services.sms_service import send_sms as _send_sms


@dataclass(frozen=True)
class SmtpConfig:
    host: str
    port: int
    username: str | None
    password: str | None
    use_tls: bool
    from_email: str


@dataclass(frozen=True)
class TwilioConfig:
    account_sid: str
    auth_token: str
    from_number: str
    whatsapp_from: str | None = None


def _load_twilio() -> TwilioConfig:
    sid = (os.getenv("TWILIO_ACCOUNT_SID") or "").strip()
    tok = (os.getenv("TWILIO_AUTH_TOKEN") or "").strip()
    frm = (os.getenv("TWILIO_FROM_NUMBER") or "").strip()
    wa = (os.getenv("TWILIO_WHATSAPP_FROM") or "").strip() or None
    if not sid or not tok or not frm:
        raise RuntimeError("Twilio env vars missing: TWILIO_ACCOUNT_SID/TWILIO_AUTH_TOKEN/TWILIO_FROM_NUMBER")
    return TwilioConfig(account_sid=sid, auth_token=tok, from_number=frm, whatsapp_from=wa)


def _load_smtp() -> SmtpConfig:
    host = (os.getenv("SMTP_HOST") or "").strip()
    port = int((os.getenv("SMTP_PORT") or "587").strip() or "587")
    user = (os.getenv("SMTP_USERNAME") or "").strip() or None
    pwd = (os.getenv("SMTP_PASSWORD") or "").strip() or None
    use_tls = (os.getenv("SMTP_TLS") or "1").strip().lower() in ("1", "true", "yes", "on")
    from_email = (os.getenv("SMTP_FROM_EMAIL") or os.getenv("ADMIN_EMAIL") or "").strip()
    if not host or not from_email:
        raise RuntimeError("SMTP env vars missing: SMTP_HOST and SMTP_FROM_EMAIL (or ADMIN_EMAIL)")
    return SmtpConfig(host=host, port=port, username=user, password=pwd, use_tls=use_tls, from_email=from_email)


def send_sms(to_number: str, body: str) -> str:
    return _send_sms(to_number, body)


def send_whatsapp(to_number: str, body: str) -> str:
    cfg = _load_twilio()
    client = Client(cfg.account_sid, cfg.auth_token)

    to = to_number.strip()
    if not to.startswith("whatsapp:"):
        to = "whatsapp:" + to

    frm = (cfg.whatsapp_from or "").strip() or (("whatsapp:" + cfg.from_number) if cfg.from_number else "")
    if not frm.startswith("whatsapp:"):
        frm = "whatsapp:" + frm

    msg = client.messages.create(from_=frm, to=to, body=body)
    logger.info("WHATSAPP_SENT to=%s sid=%s", to_number, msg.sid)
    return msg.sid


def send_email(to_email: str, subject: str, body: str) -> str:
    smtp = _load_smtp()

    global _SMTP_BREAKER_OPEN_UNTIL, _SMTP_BREAKER_LAST_LOG
    now = time.time()
    if SMTP_CIRCUIT_BREAKER_ENABLED and now < _SMTP_BREAKER_OPEN_UNTIL:
        # Throttle logs while breaker is open.
        if (now - _SMTP_BREAKER_LAST_LOG) >= max(1, SMTP_CIRCUIT_BREAKER_LOG_S):
            _SMTP_BREAKER_LAST_LOG = now
            logger.warning(
                "SMTP_BREAKER_OPEN remaining_s=%s host=%s",
                round(_SMTP_BREAKER_OPEN_UNTIL - now, 1),
                smtp.host,
            )
        raise RuntimeError("SMTP temporarily disabled due to repeated authentication errors. Retry later.")

    msg = EmailMessage()
    msg["From"] = smtp.from_email
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.set_content(body or "")

    try:
        with smtplib.SMTP(smtp.host, smtp.port, timeout=20) as server:
            if smtp.use_tls:
                server.starttls()
            if smtp.username and smtp.password:
                server.login(smtp.username, smtp.password)
            server.send_message(msg)
    except smtplib.SMTPAuthenticationError as e:
        if SMTP_CIRCUIT_BREAKER_ENABLED:
            _SMTP_BREAKER_OPEN_UNTIL = time.time() + max(5, SMTP_CIRCUIT_BREAKER_S)
            _SMTP_BREAKER_LAST_LOG = 0.0
            logger.error(
                "SMTP_AUTH_FAIL breaker_open_s=%s host=%s err=%s",
                SMTP_CIRCUIT_BREAKER_S,
                smtp.host,
                e,
            )
        raise
    except Exception as e:
        # Do not open the breaker on all failures; log and bubble up.
        logger.exception("SMTP_SEND_FAIL host=%s err=%s", smtp.host, e)
        raise

    logger.info("EMAIL_SENT to=%s subject_len=%s", to_email, len(subject or ""))
    return "ok"
def make_phone_call(to_number: str, body: str) -> str:
    cfg = _load_twilio()
    client = Client(cfg.account_sid, cfg.auth_token)

    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Say voice="alice">{_escape_for_twiml(body or "")}</Say>
</Response>"""

    call = client.calls.create(from_=cfg.from_number, to=to_number, twiml=twiml)
    logger.info("CALL_PLACED to=%s sid=%s", to_number, call.sid)
    return call.sid


def _escape_for_twiml(text: str) -> str:
    return (
        (text or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )

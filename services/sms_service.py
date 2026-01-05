# services/sms_service.py
from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from typing import Optional

from twilio.rest import Client

logger = logging.getLogger("vozlia")


@dataclass(frozen=True)
class TwilioSmsConfig:
    account_sid: str
    auth_token: str
    from_number: str


def load_twilio_sms_config() -> TwilioSmsConfig:
    sid = (os.getenv("TWILIO_ACCOUNT_SID") or "").strip()
    tok = (os.getenv("TWILIO_AUTH_TOKEN") or "").strip()
    frm = (os.getenv("TWILIO_FROM_NUMBER") or "").strip()
    if not sid or not tok or not frm:
        raise RuntimeError("Twilio SMS env vars missing: TWILIO_ACCOUNT_SID/TWILIO_AUTH_TOKEN/TWILIO_FROM_NUMBER")
    return TwilioSmsConfig(account_sid=sid, auth_token=tok, from_number=frm)


def send_sms(to_number: str, body: str, *, cfg: Optional[TwilioSmsConfig] = None) -> str:
    cfg = cfg or load_twilio_sms_config()
    client = Client(cfg.account_sid, cfg.auth_token)
    msg = client.messages.create(from_=cfg.from_number, to=to_number, body=body)
    logger.info("SMS sent to=%s sid=%s", to_number, msg.sid)
    return msg.sid

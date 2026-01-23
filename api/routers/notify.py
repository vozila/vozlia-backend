# api/routers/notify.py
from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from api.deps.admin_key import require_admin_key
from services.notification_service import send_sms, send_whatsapp, send_email, make_phone_call


router = APIRouter(prefix="/notify", tags=["notify"], dependencies=[Depends(require_admin_key)])


class SmsIn(BaseModel):
    to: str = Field(..., min_length=3)
    body: str = Field(..., min_length=1)


class WhatsappIn(BaseModel):
    to: str = Field(..., min_length=3)
    body: str = Field(..., min_length=1)


class EmailIn(BaseModel):
    to: str = Field(..., min_length=3)
    subject: str = Field(..., min_length=1)
    body: str = Field(..., min_length=1)


class CallIn(BaseModel):
    to: str = Field(..., min_length=3)
    body: str = Field(..., min_length=1)


@router.post("/sms")
def notify_sms(payload: SmsIn):
    sid = send_sms(payload.to, payload.body)
    return {"ok": True, "sid": sid}


@router.post("/whatsapp")
def notify_whatsapp(payload: WhatsappIn):
    sid = send_whatsapp(payload.to, payload.body)
    return {"ok": True, "sid": sid}


@router.post("/email")
def notify_email(payload: EmailIn):
    out = send_email(payload.to, payload.subject, payload.body)
    return {"ok": True, "result": out}


@router.post("/call")
def notify_call(payload: CallIn):
    sid = make_phone_call(payload.to, payload.body)
    return {"ok": True, "sid": sid}

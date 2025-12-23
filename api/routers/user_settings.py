# api/routers/user_settings.py
from __future__ import annotations

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from deps import get_db
from services.user_service import get_or_create_primary_user
from services.settings_service import (
    get_agent_greeting,
    gmail_summary_enabled,
    get_selected_gmail_account_id,
    set_setting,
)
from models import EmailAccount

router = APIRouter(prefix="/me", tags=["me-settings"])

class MeSettingsOut(BaseModel):
    agent_greeting: str
    gmail_summary_enabled: bool
    gmail_account_id: Optional[str] = None

@router.get("/settings", response_model=MeSettingsOut)
def me_get_settings(db: Session = Depends(get_db)):
    user = get_or_create_primary_user(db)
    return MeSettingsOut(
        agent_greeting=get_agent_greeting(db, user),
        gmail_summary_enabled=gmail_summary_enabled(db, user),
        gmail_account_id=get_selected_gmail_account_id(db, user),
    )

class UpdateGreetingIn(BaseModel):
    text: str = Field(..., min_length=1, max_length=500)

@router.put("/settings/greeting", response_model=MeSettingsOut)
def me_set_greeting(payload: UpdateGreetingIn, db: Session = Depends(get_db)):
    user = get_or_create_primary_user(db)
    set_setting(db, user, "agent_greeting", {"text": payload.text.strip()})
    return me_get_settings(db)

class UpdateGmailEnabledIn(BaseModel):
    enabled: bool

@router.put("/settings/gmail-summary/enabled", response_model=MeSettingsOut)
def me_set_gmail_enabled(payload: UpdateGmailEnabledIn, db: Session = Depends(get_db)):
    user = get_or_create_primary_user(db)
    set_setting(db, user, "gmail_summary_enabled", {"enabled": bool(payload.enabled)})
    return me_get_settings(db)

class EmailAccountOut(BaseModel):
    id: str
    provider_type: str
    oauth_provider: Optional[str] = None
    email_address: str
    is_primary: bool
    is_active: bool

@router.get("/email-accounts", response_model=list[EmailAccountOut])
def me_list_email_accounts(db: Session = Depends(get_db)):
    user = get_or_create_primary_user(db)
    rows = (
        db.query(EmailAccount)
        .filter(EmailAccount.user_id == user.id, EmailAccount.is_active == True)  # noqa
        .order_by(EmailAccount.created_at.desc())
        .all()
    )
    return [
        EmailAccountOut(
            id=str(r.id),
            provider_type=r.provider_type,
            oauth_provider=r.oauth_provider,
            email_address=r.email_address,
            is_primary=bool(r.is_primary),
            is_active=bool(r.is_active),
        )
        for r in rows
    ]

class SelectEmailAccountIn(BaseModel):
    account_id: str

@router.put("/settings/gmail-account", response_model=MeSettingsOut)
def me_select_gmail_account(payload: SelectEmailAccountIn, db: Session = Depends(get_db)):
    user = get_or_create_primary_user(db)

    row = (
        db.query(EmailAccount)
        .filter(
            EmailAccount.id == payload.account_id,
            EmailAccount.user_id == user.id,
            EmailAccount.provider_type == "gmail",
            EmailAccount.oauth_provider == "google",
            EmailAccount.is_active == True,  # noqa
        )
        .first()
    )
    if not row:
        raise HTTPException(status_code=404, detail="Gmail account not found for this user")

    set_setting(db, user, "gmail_account_id", {"account_id": str(row.id)})
    return me_get_settings(db)

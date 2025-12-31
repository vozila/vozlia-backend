# schemas.py
from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, EmailStr


class EmailAccountBase(BaseModel):
    provider_type: str
    email_address: EmailStr
    display_name: Optional[str] = None
    is_primary: bool = False
    is_active: bool = True

    # Optional IMAP/SMTP fields for custom providers
    imap_host: Optional[str] = None
    imap_port: Optional[int] = None
    imap_ssl: Optional[bool] = None
    smtp_host: Optional[str] = None
    smtp_port: Optional[int] = None
    smtp_ssl: Optional[bool] = None
    username: Optional[str] = None


class EmailAccountCreate(EmailAccountBase):
    # For custom providers, frontend would send password; you encrypt it before saving
    password: Optional[str] = None


class EmailAccountRead(EmailAccountBase):
    id: UUID
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


"""VOZLIA FILE PURPOSE
Purpose: Normalize and safely resolve delivery destinations (email/phone) for scheduled deliveries.
Hot path: no
Public interfaces: resolve_delivery_destination
Reads/Writes: none (pure functions)
Feature flags: none
Failure mode: returns (None, reason) when destination cannot be resolved safely.
Last touched: 2026-02-01 (add destination normalization to prevent placeholder destinations)
"""

from __future__ import annotations

import re
from typing import Optional, Tuple

from models import DeliveryChannel, User

_EMAIL_PLACEHOLDERS = {
    "email",
    "e-mail",
    "my email",
    "my_email",
    "owner email",
    "owner_email",
    "primary email",
    "primary_email",
    "default",
    "me",
}

_PHONE_PLACEHOLDERS = {
    "sms",
    "text",
    "phone",
    "call",
    "my phone",
    "my_phone",
    "default",
    "me",
}

# Lightweight: accept common RFC-ish emails (not fully RFC compliant; just to catch obvious placeholders).
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _looks_like_email(s: str) -> bool:
    s = (s or "").strip()
    if not s:
        return False
    if "@" not in s:
        return False
    return bool(_EMAIL_RE.match(s))


def _digits_only(s: str) -> str:
    return "".join([c for c in (s or "") if c.isdigit()])


def _looks_like_phone(s: str) -> bool:
    # Loose validation: at least 8 digits.
    d = _digits_only(s)
    return len(d) >= 8


def resolve_delivery_destination(
    *,
    channel: DeliveryChannel,
    destination: str | None,
    user: User,
) -> Tuple[Optional[str], str]:
    """
    Resolve a destination string for a given channel.

    Key goal: prevent placeholder destinations like destination="email" from being persisted.

    Returns:
      (resolved_destination, reason)

    Reasons:
      - "provided_valid"
      - "defaulted_to_user_email"
      - "invalid_missing_email"
      - "invalid_phone"
      - "missing_destination"
    """
    raw = (destination or "").strip()
    raw_l = raw.lower()

    if channel == DeliveryChannel.email:
        # If LLM/UX sends a placeholder (or anything not email-ish), default to owner's email if available.
        if (not raw) or (raw_l in _EMAIL_PLACEHOLDERS) or (not _looks_like_email(raw)):
            owner = (getattr(user, "email", "") or "").strip()
            if _looks_like_email(owner):
                return owner, "defaulted_to_user_email"
            return None, "invalid_missing_email"
        return raw, "provided_valid"

    # SMS / WhatsApp / Call
    if not raw:
        return None, "missing_destination"
    if raw_l in _PHONE_PLACEHOLDERS:
        return None, "missing_destination"

    if not _looks_like_phone(raw):
        return None, "invalid_phone"

    return raw, "provided_valid"

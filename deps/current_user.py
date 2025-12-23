# deps/current_user.py
from __future__ import annotations

from fastapi import Depends
from sqlalchemy.orm import Session

from deps import get_db
from models import User
from services.user_service import get_or_create_primary_user


def get_current_user(db: Session = Depends(get_db)) -> User:
    """
    Single-tenant MVP behavior:
    Treat the primary admin user (ADMIN_EMAIL) as the current user for all requests
    that use get_current_user().
    """
    return get_or_create_primary_user(db)

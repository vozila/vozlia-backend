# deps/current_user.py
from __future__ import annotations

from fastapi import Depends
from sqlalchemy.orm import Session

from deps import get_db
from models import User
from services.user_service import get_or_create_demo_user


def get_current_user(db: Session = Depends(get_db)) -> User:
    # MVP behavior: demo user
    return get_or_create_demo_user(db)

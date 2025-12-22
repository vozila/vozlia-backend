# services/user_service.py
import os
from sqlalchemy.orm import Session

from core.logging import logger
from models import User


def get_or_create_primary_user(db: Session) -> User:
    """
    Deterministic primary user:
    - If ADMIN_EMAIL is set, use that user (create if missing)
    - Else fallback to first user / demo user behavior
    """
    admin_email = os.getenv("ADMIN_EMAIL")

    if admin_email:
        user = db.query(User).filter(User.email == admin_email).first()
        if not user:
            user = User(email=admin_email)
            db.add(user)
            db.commit()
            db.refresh(user)
            logger.info("Created primary user from ADMIN_EMAIL id=%s email=%s", user.id, user.email)
        return user

    # Legacy fallback
    user = db.query(User).first()
    if not user:
        user = User(email="demo@vozlia.com")
        db.add(user)
        db.commit()
        db.refresh(user)
        logger.info("Created demo user with id=%s email=%s", user.id, user.email)
    return user


def get_or_create_demo_user(db: Session) -> User:
    """
    Backwards-compatible function name.
    Some routers still import this symbol. Do not remove yet.
    """
    return get_or_create_primary_user(db)

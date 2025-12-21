# services/user_service.py
from sqlalchemy.orm import Session
from core.logging import logger
from models import User


def get_or_create_demo_user(db: Session) -> User:
    user = db.query(User).first()
    if not user:
        user = User(email="demo@vozlia.com")
        db.add(user)
        db.commit()
        db.refresh(user)
        logger.info("Created demo user with id=%s email=%s", user.id, user.email)
    return user

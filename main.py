import os
import json
import asyncio
import logging
import base64
import time
from typing import List

from fastapi import (
    FastAPI,
    Request,
    WebSocket,
    WebSocketDisconnect,
    Depends,
    HTTPException,
)
from fastapi.responses import PlainTextResponse, Response, JSONResponse

from sqlalchemy.orm import Session

from twilio.twiml.voice_response import VoiceResponse, Connect
from openai import OpenAI
import websockets

from db import Base, engine
from models import User, EmailAccount
from schemas import EmailAccountCreate, EmailAccountRead
from deps import get_db


# ---------- Logging ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vozlia")
logger.setLevel(logging.INFO)

# ---------- FastAPI app ----------
app = FastAPI()


# Create tables on startup (simple MVP, later use Alembic)
@app.on_event("startup")
def on_startup() -> None:
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables ensured (create_all).")


# ---------- Auth stub / current user ----------
def get_current_user(db: Session = Depends(get_db)) -> User:
    """
    TEMP: For now, just returns the first user or creates a demo user.
    Later this will be replaced with real auth / tenant logic.
    """
    user = db.query(User).first()
    if not user:
        user = User(email="demo@vozlia.com")
        db.add(user)
        db.commit()
        db.refresh(user)
        logger.info(f"Created demo user with id={user.id} email={user.email}")
    return user


# ---------- Email account endpoints ----------
@app.get("/email/accounts", response_model=List[EmailAccountRead])
def list_email_accounts(

# api/routers/assistant.py
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from deps import get_db
from services.user_service import get_or_create_primary_user
from services.assistant_service import run_assistant_route

router = APIRouter()

class AssistantRouteIn(BaseModel):
    backend_call: str | None = None
    text: str
    account_id: str | None = None
    context: dict | None = None

class AssistantRouteOut(BaseModel):
    spoken_reply: str
    fsm: dict
    gmail: dict | None = None

@router.post("/assistant/route", response_model=AssistantRouteOut)
def assistant_route(payload: AssistantRouteIn, db: Session = Depends(get_db)):
    current_user = get_or_create_primary_user(db)
    ctx = payload.context or {}
    if isinstance(ctx, dict) and payload.backend_call:
        ctx.setdefault('backend_call', payload.backend_call)
    return run_assistant_route(payload.text, db, current_user, account_id=payload.account_id, context=ctx)

# api/routers/assistant.py
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from deps import get_db
from services.user_service import get_or_create_demo_user
from services.assistant_service import run_assistant_route

router = APIRouter()

class AssistantRouteIn(BaseModel):
    text: str
    account_id: str | None = None
    context: dict | None = None

class AssistantRouteOut(BaseModel):
    spoken_reply: str
    fsm: dict
    gmail: dict | None = None

@router.post("/assistant/route", response_model=AssistantRouteOut)
def assistant_route(payload: AssistantRouteIn, db: Session = Depends(get_db)):
    current_user = get_or_create_demo_user(db)
    return run_assistant_route(payload.text, db, current_user, account_id=payload.account_id, context=payload.context)

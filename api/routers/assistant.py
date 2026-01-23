# api/routers/assistant.py
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from deps import get_db
from services.user_service import get_or_create_primary_user
import os
from services.assistant_service import run_assistant_route
try:
    from services.assistant_websearch_wrapper import run_assistant_route_with_websearch
except Exception:  # pragma: no cover
    run_assistant_route_with_websearch = None  # type: ignore

router = APIRouter()

class AssistantRouteIn(BaseModel):
    text: str
    backend_call: str | None = None
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
    if payload.backend_call:
        # Option A: stream.py unchanged; backend_call travels through this payload
        if isinstance(ctx, dict):
            ctx = dict(ctx)
            ctx["backend_call"] = payload.backend_call
    use_web = (os.getenv("ASSISTANT_WEBSEARCH_WRAPPER") or "0").strip() in ("1","true","yes","on")
    if use_web and run_assistant_route_with_websearch is not None:
        return run_assistant_route_with_websearch(payload.text, db, current_user, account_id=payload.account_id, context=ctx)
    return run_assistant_route(payload.text, db, current_user, account_id=payload.account_id, context=ctx)

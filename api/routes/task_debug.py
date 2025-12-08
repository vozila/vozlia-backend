from __future__ import annotations

import json
import os
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from sqlalchemy.orm import Session

from deps import get_db
from models import Task as TaskORM
from tasks.interpreter import NLUEvent, handle_nlu_event
from openai import OpenAI


# All routes in this file will be under /debug/...
router = APIRouter(prefix="/debug", tags=["task-debug"])

# ---------- OpenAI client for brain/NLU ----------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
_openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


@router.get("/tasks")
def list_all_tasks(db: Session = Depends(get_db)) -> List[Dict[str, Any]]:
    """
    Developer-only view of all tasks and their state.
    NOTE: This returns JSON-encoded SQLAlchemy objects for debugging.
    """
    tasks = db.query(TaskORM).all()
    return jsonable_encoder(tasks)


@router.get("/tasks/summary")
def tasks_summary(db: Session = Depends(get_db)) -> Dict[str, Any]:
    tasks = db.query(TaskORM).all()
    by_status: Dict[str, int] = {}
    for t in tasks:
        key = t.status.value if hasattr(t.status, "value") else str(t.status)
        by_status[key] = by_status.get(key, 0) + 1

    return {
        "total": len(tasks),
        "by_status": by_status,
    }


@router.post("/tasks/intent")
def debug_tasks_intent(
    event: NLUEvent,
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """
    Simulate 'brain' behavior:
      - take an intent + entities
      - run it through the task engine
      - return what Vozlia should say + task_id
    """
    directive = handle_nlu_event(db, event)
    return {
        "speech": directive.speech,
        "task_id": directive.task_id,
        "meta": directive.meta,
    }


# ---------- Natural-language brain → NLUEvent → task engine ----------

class BrainRequest(BaseModel):
    user_id: str
    text: str


class BrainResponse(BaseModel):
    raw_event: Dict[str, Any]
    nlu_event: Dict[str, Any]
    speech: str
    task_id: str | None = None
    meta: Dict[str, Any]


@router.post("/brain", response_model=BrainResponse)
def debug_brain(
    payload: BrainRequest,
    db: Session = Depends(get_db),
) -> BrainResponse:
    """
    End-to-end simulator:

      caller text → GPT builds an NLUEvent JSON → task engine → assistant speech

    This is the exact flow we will later embed inside /twilio/stream,
    except there we'll feed in transcribed audio instead of plain text.
    """
    if not OPENAI_API_KEY or _openai_client is None:
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY is not configured on the server.",
        )

    # 1) Ask GPT to turn free-form text into an NLUEvent JSON
    system_prompt = (
        "You are the NLU 'brain' for Vozlia, an AI phone assistant.\n"
        "Given what the caller says, you must output a single JSON object ONLY, "
        "with no extra text. The JSON must match this schema:\n\n"
        "{\n"
        '  \"event\": \"task.intent\",               // literal string\n'
        "  \"intent\": string,                      // one of: set_reminder, start_timer, "
        "take_note, check_email, start_counting, continue_task, cancel_task, "
        "what_number_were_you_on\n"
        "  \"text\": string | null,                 // the original caller text\n"
        "  \"user_id\": string,                     // passed through from the request\n"
        "  \"task_id\": string | null,              // if referring to an existing task, else null\n"
        "  \"entities\": {                          // key-value pairs extracted from text\n"
        "      // for set_reminder: time, message\n"
        "      // for start_timer: duration\n"
        "      // for take_note: text\n"
        "      // for check_email: query, max_results\n"
        "

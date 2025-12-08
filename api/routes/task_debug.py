from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from sqlalchemy.orm import Session

from deps import get_db
from models import Task as TaskORM
from tasks.interpreter import NLUEvent, handle_nlu_event
from openai import OpenAI
from tasks.brain import run_brain_on_text


# All routes in this file will be under /debug/...
router = APIRouter(prefix="/debug", tags=["task-debug"])

# ---------- OpenAI client for brain/NLU ----------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
_openai_client: Optional[OpenAI] = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


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
    task_id: Optional[str] = None
    meta: Dict[str, Any]


@router.post("/brain", response_model=BrainResponse)
def debug_brain(
    payload: BrainRequest,
    db: Session = Depends(get_db),
) -> BrainResponse:
    """
    End-to-end simulator:

      caller text → brain (NLU + tasks) → assistant speech

    This uses the same core logic that we will later invoke from /twilio/stream.
    """
    result = run_brain_on_text(
        db,
        user_id=payload.user_id,
        text=payload.text,
    )

    return BrainResponse(
        raw_event=result["raw_event"],
        nlu_event=jsonable_encoder(result["nlu_event"]),
        speech=result["speech"],
        task_id=result["task_id"],
        meta=result["meta"],
    )


    # 1) Ask GPT to turn free-form text into an NLUEvent JSON
    system_prompt = (
        "You are the NLU 'brain' for Vozlia, an AI phone assistant.\n"
        "Given what the caller says, you must output a single JSON object ONLY, "
        "with no extra text.\n"
        "The JSON must match this schema:\n"
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
        "      // for start_counting: start, end\n"
        "  }\n"
        "}\n\n"
        "Rules:\n"
        "- Respond with JSON ONLY. No explanations, no comments.\n"
        "- If the user asks to continue a previous task, use intent=\"continue_task\" and "
        "set task_id if mentioned; otherwise leave it null.\n"
        "- For simple yes/no or small-talk that does not map to a task, "
        "pick the closest intent or 'take_note' with a generic text.\n"
    )

    user_message = (
        f"Caller text: {payload.text}\n\n"
        f"User id: {payload.user_id}\n\n"
        "Produce ONE JSON object matching the schema."
    )

    try:
        resp = _openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0,
        )
        raw_content = resp.choices[0].message.content
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"Error calling OpenAI for NLU: {e}",
        )

    # 2) Parse the JSON the model returned
    try:
        nlu_dict = json.loads(raw_content)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Failed to parse NLU JSON from model.",
                "raw_content": raw_content,
                "error": str(e),
            },
        )

    # Ensure user_id/text are set from request (even if model forgot)
    nlu_dict["user_id"] = payload.user_id
    if "text" not in nlu_dict or nlu_dict["text"] is None:
        nlu_dict["text"] = payload.text

    # 3) Validate/normalize via NLUEvent model
    try:
        nlu_event = NLUEvent(**nlu_dict)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "NLUEvent validation failed.",
                "nlu_dict": nlu_dict,
                "error": str(e),
            },
        )

    # 4) Run it through your existing task engine
    directive = handle_nlu_event(db, nlu_event)

    return BrainResponse(
        raw_event=nlu_dict,
        nlu_event=jsonable_encoder(nlu_event),
        speech=directive.speech,
        task_id=directive.task_id,
        meta=directive.meta,
    )

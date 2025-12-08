from __future__ import annotations

import json
import os
from typing import Any, Dict

from sqlalchemy.orm import Session
from openai import OpenAI
from fastapi import HTTPException

from tasks.interpreter import NLUEvent, handle_nlu_event

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


SYSTEM_PROMPT = (
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
    "  \"user_id\": string,                     // passed in from the caller context\n"
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


def run_brain_on_text(
    db: Session,
    *,
    user_id: str,
    text: str,
) -> Dict[str, Any]:
    """
    Core brain pipeline:

      raw text -> OpenAI NLU JSON -> NLUEvent -> task engine -> directive

    Returns a dict:
      {
        "raw_event": <dict>,
        "nlu_event": <NLUEvent>,
        "speech": <str>,
        "task_id": <str or None>,
        "meta": <dict>
      }
    """
    if not OPENAI_API_KEY or _client is None:
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY is not configured on the server.",
        )

    user_message = (
        f"Caller text: {text}\n\n"
        f"User id: {user_id}\n\n"
        "Produce ONE JSON object matching the schema."
    )

    # 1) LLM → NLU JSON
    resp = _client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0,
    )
    raw_content = resp.choices[0].message.content

    # 2) Parse JSON
    try:
        nlu_dict = json.loads(raw_content)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Failed to parse NLU JSON from model.",
                "raw_content": raw_content,
                "error": str(e),
            },
        )

    # Ensure proper fields
    nlu_dict["user_id"] = user_id
    if "text" not in nlu_dict or nlu_dict["text"] is None:
        nlu_dict["text"] = text

    # 3) Validate with NLUEvent
    try:
        nlu_event = NLUEvent(**nlu_dict)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(
            status_code=400,
            detail={
                "message": "NLUEvent validation failed.",
                "nlu_dict": nlu_dict,
                "error": str(e),
            },
        )

    # 4) Run through task engine
    directive = handle_nlu_event(db, nlu_event)

    return {
        "raw_event": nlu_dict,
        "nlu_event": nlu_event,
        "speech": directive.speech,
        "task_id": directive.task_id,
        "meta": directive.meta,
    }

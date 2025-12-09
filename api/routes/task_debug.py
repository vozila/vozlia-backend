import os
import json
from collections import defaultdict
from typing import Dict, List, Literal, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from openai import OpenAI

router = APIRouter()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    # We won't crash the app, but /debug/brain will 500 if called.
    client: Optional[OpenAI] = None
else:
    client = OpenAI(api_key=OPENAI_API_KEY)


# ---------- Simple in-memory "brain" state per user ----------
class UserBrainState(BaseModel):
    utterances: List[str] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)
    reminders: List[str] = Field(default_factory=list)


# ⚠️ This is process-local, non-persistent.
# It's enough for MVP + testing, but not durable across restarts / replicas.
USER_STATE: Dict[str, UserBrainState] = defaultdict(UserBrainState)


# ---------- Request/response models ----------
class BrainRequest(BaseModel):
    user_id: str
    # This is the latest recognized text from the caller
    text: str


class BrainResponse(BaseModel):
    # What Vozlia should SAY out loud back to the caller
    speech: str


# ---------- Internal helper to call OpenAI as a "brain" ----------
async def call_brain_llm(
    state: UserBrainState,
    latest_text: str,
) -> Dict:
    """
    Call GPT in JSON mode to:
      - detect intent (NOTE, REMINDER, RECALL, SMALL_TALK)
      - decide what to store/update in memory
      - generate a short 'speech' answer
    Returns a parsed dict.
    """

    if client is None:
        raise HTTPException(
            status_code=500,
            detail="OpenAI API key not configured on server.",
        )

    # Build a compact history summary for the model.
    # We keep only the last ~10 user utterances to avoid prompt bloat.
    recent_utterances = state.utterances[-10:]
    history_text = " | ".join(recent_utterances) if recent_utterances else ""

    notes_text = "; ".join(state.notes) if state.notes else ""
    reminders_text = "; ".join(state.reminders) if state.reminders else ""

    system_instructions = (
        "You are the internal 'task brain' for Vozlia, a phone-based assistant.\n"
        "You do NOT speak directly to the caller; you ONLY return JSON that tells "
        "Vozlia what to say and what to store in its short-term memory.\n\n"
        "You receive:\n"
        "  - prior_utterances: a compact string of what the caller has said so far\n"
        "  - notes: existing simple notes already saved\n"
        "  - reminders: existing simple reminders already saved\n"
        "  - latest_text: the most recent thing the caller just said.\n\n"
        "Your job:\n"
        "  1) Detect the user's intent. Use one of these types:\n"
        "     - 'NOTE'       : user wants to jot something down (e.g., 'tell the plumber not to make a mess').\n"
        "     - 'REMINDER'   : user wants a reminder (e.g., 'remind me tomorrow at 8am to take a shower').\n"
        "     - 'RECALL'     : user asks you to repeat or recall something from earlier in the call "
        "                       or from saved notes/reminders "
        "                       (e.g., 'what was the first question I asked you?', "
        "                             'what note did I leave about the plumber?').\n"
        "     - 'SMALL_TALK' : anything else (chitchat, thank you, unclear).\n\n"
        "  2) Decide what to store in memory:\n"
        "     - For NOTE: extract a short clean 'note_text' string.\n"
        "     - For REMINDER: extract a short clean 'reminder_text' that combines time + action in natural language.\n"
        "       You do NOT have to compute exact timestamps. Just capture e.g. "
        "       'Tomorrow at 8 AM: take a shower'.\n"
        "     - For RECALL: look at prior_utterances, notes, and reminders to answer the question.\n"
        "       For example, if the first thing they said was 'Remind me tomorrow at 8am to take a shower', "
        "       you should be able to answer that explicitly.\n\n"
        "  3) Generate a very short 'speech' string that Vozlia will say back to the caller.\n"
        "     - Keep it natural and concise (1–2 sentences).\n"
        "     - For REMINDER: sound confident that you understood and 'will remember it', "
        "       even if the actual push notification is not implemented.\n"
        "     - For NOTE: confirm you saved the note and maybe paraphrase it.\n"
        "     - For RECALL: answer clearly based on memory; if you truly don't know, admit it.\n"
        "     - For SMALL_TALK: just respond kindly.\n\n"
        "CRITICAL:\n"
        " - DO NOT tell the user to 'use their phone reminders' or that 'you cannot set reminders'.\n"
        "   Treat reminders as something you can remember internally.\n"
        " - You MUST respond in valid JSON only.\n"
        " - Your JSON MUST have this shape exactly:\n"
        "   {\n"
        "     \"mode\": \"NOTE\" | \"REMINDER\" | \"RECALL\" | \"SMALL_TALK\",\n"
        "     \"speech\": \"string\",\n"
        "     \"note_text\": \"string or null\",\n"
        "     \"reminder_text\": \"string or null\"\n"
        "   }\n"
        " - Never include extra top-level keys.\n"
    )

    user_payload = {
        "prior_utterances": history_text,
        "notes": notes_text,
        "reminders": reminders_text,
        "latest_text": latest_text,
    }

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_instructions},
                {
                    "role": "user",
                    "content": json.dumps(user_payload, ensure_ascii=False),
                },
            ],
        )
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"Error calling OpenAI for brain: {e}",
        )

    try:
        content = resp.choices[0].message.content
        # content is already a JSON string because of response_format
        data = json.loads(content)
        return data
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Brain JSON parse error: {e}",
        )


# ---------- Main /debug/brain endpoint ----------
@router.post("/debug/brain", response_model=BrainResponse)
async def debug_brain(request: BrainRequest) -> BrainResponse:
    """
    Internal brain endpoint for voice calls.

    Input:  { "user_id": "...", "text": "..." }
    Output: { "speech": "..." }

    It also updates per-user short-term memory:
      - utterances[]
      - notes[]
      - reminders[]
    """
    if not request.text.strip():
        return BrainResponse(speech="I'm here and listening whenever you're ready.")

    state = USER_STATE[request.user_id]

    # Append latest utterance to history
    state.utterances.append(request.text.strip())
    # Cap history length
    if len(state.utterances) > 50:
        state.utterances = state.utterances[-50:]

    brain_data = await call_brain_llm(state, request.text.strip())

    mode = brain_data.get("mode", "SMALL_TALK")
    speech = brain_data.get("speech") or "I'm here and listening."

    note_text = brain_data.get("note_text")
    reminder_text = brain_data.get("reminder_text")

    # Persist to the in-memory state depending on mode
    if mode == "NOTE" and note_text:
        state.notes.append(note_text.strip())
    elif mode == "REMINDER" and reminder_text:
        state.reminders.append(reminder_text.strip())
    elif mode == "RECALL":
        # nothing to store, but speech may mention prior notes/reminders
        pass
    else:
        # SMALL_TALK or unknown; no storage change
        pass

    # Log for server-side debugging
    # (Safe to log; no PII more than user's speech.)
    print(f"[BRAIN] mode={mode}  speech={speech!r}")
    if note_text:
        print(f"[BRAIN] saved note: {note_text!r}")
    if reminder_text:
        print(f"[BRAIN] saved reminder: {reminder_text!r}")

    return BrainResponse(speech=speech)

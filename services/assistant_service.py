# services/assistant_service.py
from sqlalchemy.orm import Session
from models import User
from vozlia_fsm import VozliaFSM

from services.gmail_service import get_default_gmail_account_id, summarize_gmail_for_assistant


def run_assistant_route(text: str, db: Session, current_user: User, account_id: str | None = None, context: dict | None = None) -> dict:
    fsm = VozliaFSM()

    fsm_context = context or {}
    fsm_context.setdefault("user_id", current_user.id)
    fsm_context.setdefault("channel", "phone")

    fsm_result: dict = fsm.handle_utterance(text, context=fsm_context)

    spoken_reply: str = fsm_result.get("spoken_reply") or ""
    backend_call: dict | None = fsm_result.get("backend_call") or None
    gmail_data: dict | None = None

    if backend_call and backend_call.get("type") == "gmail_summary":
        params = backend_call.get("params") or {}
        account_id_effective = params.get("account_id") or account_id or get_default_gmail_account_id(current_user, db)

        if not account_id_effective:
            spoken_reply = (spoken_reply.rstrip(". ") + " However, I don't see a Gmail account connected for you yet.") if spoken_reply else \
                "I tried to check your email, but I don't see a Gmail account connected for you yet."
        else:
            gmail_query = params.get("query")
            gmail_max_results = params.get("max_results", 20)
            gmail_data = summarize_gmail_for_assistant(account_id_effective, current_user, db, max_results=gmail_max_results, query=gmail_query)
            if gmail_data.get("summary"):
                spoken_reply = (spoken_reply.strip() + " " + gmail_data["summary"].strip()).strip() if spoken_reply else gmail_data["summary"].strip()
            gmail_data["used_account_id"] = account_id_effective

    return {"spoken_reply": spoken_reply, "fsm": fsm_result, "gmail": gmail_data}

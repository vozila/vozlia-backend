# services/assistant_service.py
from skills.engine import skills_engine_enabled, match_skill_id, execute_skill
from services.settings_service import get_agent_greeting, get_enabled_gmail_account_ids
from services.session_store import session_store
import os
from core.logging import logger
from skills.registry import skill_registry
from sqlalchemy.orm import Session
from models import User, EmailAccount
from vozlia_fsm import VozliaFSM
from services.settings_service import gmail_summary_enabled


from services.gmail_service import get_default_gmail_account_id, summarize_gmail_for_assistant


def _gmail_selection_call_id(context: dict | None) -> str:
    ctx = context or {}
    return str(ctx.get("streamSid") or ctx.get("callSid") or ctx.get("CallSid") or "").strip()


def _list_enabled_active_gmail_accounts(db: Session, user: User) -> list[EmailAccount]:
    enabled_ids = get_enabled_gmail_account_ids(db, user)
    q = (
        db.query(EmailAccount)
        .filter(
            EmailAccount.user_id == user.id,
            EmailAccount.provider_type == "gmail",
            EmailAccount.oauth_provider == "google",
            EmailAccount.is_active == True,  # noqa
        )
    )
    accounts = q.all() or []
    if enabled_ids:
        enabled_set = set(enabled_ids)
        accounts = [a for a in accounts if str(a.id) in enabled_set]
    accounts.sort(key=lambda a: (0 if a.is_primary else 1, (a.email_address or "").lower()))
    return accounts


def _parse_inbox_choice(choice_text: str, accounts: list[EmailAccount]) -> EmailAccount | None:
    t = (choice_text or "").strip().lower()
    if not t or not accounts:
        return None
    if t.isdigit():
        n = int(t)
        if 1 <= n <= len(accounts):
            return accounts[n - 1]
    mapping = {
        "first": 1, "1st": 1, "one": 1,
        "second": 2, "2nd": 2, "two": 2,
        "third": 3, "3rd": 3, "three": 3,
    }
    if t in mapping and 1 <= mapping[t] <= len(accounts):
        return accounts[mapping[t] - 1]
    for a in accounts:
        hay = f"{a.email_address or ''} {a.display_name or ''}".lower()
        if t in hay:
            return a
    return None


def _build_inbox_prompt(accounts: list[EmailAccount]) -> str:
    parts = []
    for i, a in enumerate(accounts, start=1):
        label = a.display_name or a.email_address or str(a.id)
        parts.append(f"{i}) {label}")
    return "I see multiple inboxes connected. " + " ".join(parts) + " Which one should I check?"


def run_assistant_route(
    text: str,
    db: Session,
    current_user: User,
    account_id: str | None = None,
    context: dict | None = None,
) -> dict:
    """
    Routing order (safe / behavior-preserving):
      1) Always run FSM first
      2) If FSM requests backend_call=gmail_summary -> execute existing gmail summary logic (current behavior)
      3) Else if SKILLS_ENGINE_ENABLED=true and FSM did not request a backend call:
           - phrase-match manifest triggers
           - if gmail_summary matched -> call existing gmail summary helper and return it
      4) Otherwise return FSM response
    """
    fsm = VozliaFSM()
    # Portal-controlled greeting (admin-configurable)
    fsm.greeting_text = get_agent_greeting(db, current_user)

    fsm_context = context or {}
    fsm_context.setdefault("user_id", current_user.id)
    fsm_context.setdefault("channel", "phone")

    fsm_result: dict = fsm.handle_utterance(text, context=fsm_context)

    spoken_reply: str = fsm_result.get("spoken_reply") or ""
    backend_call: dict | None = fsm_result.get("backend_call") or None
    gmail_data: dict | None = None
    # -----------------------------------------------------------
    # Gmail inbox selection (multi-inbox) - FSM layer only (safe).
    # -----------------------------------------------------------
    call_id = _gmail_selection_call_id(context)
    pending = session_store.get(call_id).get("awaiting_gmail_inbox") if call_id else None
    if pending:
        accounts = _list_enabled_active_gmail_accounts(db, current_user)
        chosen = _parse_inbox_choice(text, accounts)
        if not chosen:
            return {
                "spoken_reply": _build_inbox_prompt(accounts) if accounts else "I don't see any active Gmail inboxes connected right now.",
                "fsm": fsm_result,
                "gmail": None,
            }
        session_store.set(call_id, "selected_gmail_account_id", str(chosen.id))
        session_store.pop(call_id, "awaiting_gmail_inbox", None)
        backend_call = {"type": "gmail_summary", "params": {"account_id": str(chosen.id)}}


    # ----------------------------
    # (1) Existing FSM backend call behavior (no change)
    # ----------------------------
    if backend_call and backend_call.get("type") == "gmail_summary":
        # ✅ Skill toggle gate (portal-controlled)
        if not gmail_summary_enabled(db, current_user):
            return {
                "spoken_reply": "Email summaries are currently turned off in your settings.",
                "fsm": fsm_result,
                "gmail": None,
            }

        params = backend_call.get("params") or {}
        account_id_effective = (
            params.get("account_id")
            or (session_store.get(call_id).get("selected_gmail_account_id") if call_id else None)
            or account_id
            or get_default_gmail_account_id(current_user, db)
        )

        accounts_enabled = _list_enabled_active_gmail_accounts(db, current_user)
        if len(accounts_enabled) > 1 and not (session_store.get(call_id).get("selected_gmail_account_id") if call_id else None) and not params.get("account_id") and not account_id:
            if call_id:
                session_store.set(call_id, "awaiting_gmail_inbox", True)
            return {
                "spoken_reply": _build_inbox_prompt(accounts_enabled),
                "fsm": fsm_result,
                "gmail": None,
            }


        if not account_id_effective:
            spoken_reply = (
                (spoken_reply.rstrip(". ") + " However, I don't see a Gmail account connected for you yet.")
                if spoken_reply
                else "I tried to check your email, but I don't see a Gmail account connected for you yet."
            )
        else:
            gmail_query = params.get("query")
            gmail_max_results = params.get("max_results", 20)
            gmail_data = summarize_gmail_for_assistant(
                account_id_effective,
                current_user,
                db,
                max_results=gmail_max_results,
                query=gmail_query,
            )
            if gmail_data.get("summary"):
                spoken_reply = (
                    (spoken_reply.strip() + " " + gmail_data["summary"].strip()).strip()
                    if spoken_reply
                    else gmail_data["summary"].strip()
                )
            gmail_data["used_account_id"] = account_id_effective

        return {"spoken_reply": spoken_reply, "fsm": fsm_result, "gmail": gmail_data}

    # ----------------------------
    # (2) Skills Engine fallback (feature-flagged)
    # Only runs when FSM did NOT request a backend call.
    # ----------------------------
    skills_enabled = (os.getenv("SKILLS_ENGINE_ENABLED") or "").strip().lower() in ("1", "true", "yes", "on")

    if skills_enabled and not backend_call:
        try:
            text_l = (text or "").lower()

            matched_skill = None
            for s in skill_registry.all():
                for phrase in (s.trigger.phrases or []):
                    p = (phrase or "").strip().lower()
                    if p and p in text_l:
                        matched_skill = s
                        break
                if matched_skill:
                    break

            if matched_skill and matched_skill.id == "gmail_summary":
                logger.info("SkillsEngine matched skill=%s text=%r", matched_skill.id, text)

                account_id_effective = account_id or get_default_gmail_account_id(current_user, db)
                if not account_id_effective:
                    spoken_reply = "I tried to check your email, but I don't see a Gmail account connected for you yet."
                    return {"spoken_reply": spoken_reply, "fsm": fsm_result, "gmail": {"summary": None, "used_account_id": None}}

                gmail_data = summarize_gmail_for_assistant(account_id_effective, current_user, db)
                gmail_data["used_account_id"] = account_id_effective

                summary = (gmail_data.get("summary") or "").strip()
                if summary:
                    # minimal template support: only {{summary}}
                    template = None
                    if matched_skill.response and matched_skill.response.speak:
                        template = matched_skill.response.speak
                    spoken_reply = (template or "{{summary}}").replace("{{summary}}", summary).strip()
                else:
                    spoken_reply = "I checked your email, but there wasn’t anything to summarize right now."

                logger.info("SkillsEngine executed skill=%s account_id=%s", matched_skill.id, account_id_effective)
                return {"spoken_reply": spoken_reply, "fsm": fsm_result, "gmail": gmail_data}

        except Exception:
            # Never fail the call path because the skills engine had an issue.
            logger.exception("SkillsEngine failed; falling back to FSM result.")

    # ----------------------------
    # (3) Default: return FSM result (no change)
    # ----------------------------
    return {"spoken_reply": spoken_reply, "fsm": fsm_result, "gmail": gmail_data}


# services/assistant_service.py
from skills.engine import skills_engine_enabled, match_skill_id, execute_skill
from services.settings_service import get_agent_greeting
import os
from core.logging import logger
from skills.registry import skill_registry
from sqlalchemy.orm import Session
from models import User
from vozlia_fsm import VozliaFSM
from services.settings_service import gmail_summary_enabled


from services.gmail_service import get_default_gmail_account_id, summarize_gmail_for_assistant


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
    import time as _time
    debug = (os.getenv("ASSISTANT_ROUTE_DEBUG_LOGS") or "").strip().lower() in ("1","true","yes","on")
    t0 = _time.perf_counter()
    if debug:
        text_snip = (text[:200] + "…") if len(text or "") > 200 else (text or "")
        logger.info("ASSISTANT_ROUTE_START user_id=%s account_id=%s text=%r context_keys=%s", getattr(current_user, "id", None), account_id, text_snip, sorted(list((context or {}).keys())))

    fsm = VozliaFSM()
    # Portal-controlled greeting (admin-configurable)
    fsm.greeting_text = get_agent_greeting(db, current_user)

    fsm_context = context or {}
    fsm_context.setdefault("user_id", current_user.id)
    fsm_context.setdefault("channel", "phone")

    fsm_result: dict = fsm.handle_utterance(text, context=fsm_context)

    spoken_reply: str = fsm_result.get("spoken_reply") or ""
    if debug:
        bc_type = backend_call.get('type') if isinstance(backend_call, dict) else None
        logger.info(
            "ASSISTANT_ROUTE_FSM spoken_len=%s backend_call=%s fsm_keys=%s dt_ms=%s",
            len(spoken_reply or ''),
            bc_type,
            sorted(list(fsm_result.keys())) if isinstance(fsm_result, dict) else None,
            int((_time.perf_counter() - t0) * 1000),
        )
    backend_call: dict | None = fsm_result.get("backend_call") or None
    gmail_data: dict | None = None

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
        account_id_effective = params.get("account_id") or account_id or get_default_gmail_account_id(current_user, db)
        if debug:
            logger.info(
                "ASSISTANT_GMAIL_CALL params=%s account_id_effective=%s",
                {k: v for k, v in params.items() if k in ('account_id','query','max_results')},
                account_id_effective,
            )


        if not account_id_effective:
            spoken_reply = (
                (spoken_reply.rstrip(". ") + " However, I don't see a Gmail account connected for you yet.")
                if spoken_reply
                else "I tried to check your email, but I don't see a Gmail account connected for you yet."
            )
        else:
            gmail_query = params.get("query")
            gmail_max_results = params.get("max_results", 20)
            t_g = _time.perf_counter()
            gmail_data = summarize_gmail_for_assistant(
                account_id_effective,
                current_user,
                db,
                max_results=gmail_max_results,
                query=gmail_query,
            )
            if debug:
                logger.info(
                    "ASSISTANT_GMAIL_DONE dt_ms=%s got_messages=%s summary_len=%s used_account_id=%s",
                    int((_time.perf_counter() - t_g) * 1000),
                    len((gmail_data.get('messages') or [])) if isinstance(gmail_data, dict) else None,
                    len((gmail_data.get('summary') or '')) if isinstance(gmail_data, dict) else None,
                    account_id_effective,
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


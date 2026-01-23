# services/assistant_websearch_wrapper.py
from __future__ import annotations

import os
import re
import uuid
from datetime import datetime

from sqlalchemy.orm import Session

from core.logging import logger
from models import User, DeliveryChannel, WebSearchSkill
from services.assistant_service import run_assistant_route as _run_assistant_route_core
from services.memory_facade import memory, SESSION_MEMORY_TTL_S
from services.web_search_service import run_web_search
from services.web_search_skill_store import (
    parse_time_of_day,
    create_web_search_skill,
    upsert_daily_schedule,
    disable_all_schedules,
)

_STATE_KEY = "websearch_state_v1"


def run_assistant_route_with_websearch(
    text: str,
    db: Session,
    current_user: User,
    *,
    account_id: str | None = None,
    context: dict | None = None,
) -> dict:
    ctx = context if isinstance(context, dict) else {}
    tenant_id = str(getattr(current_user, "id", "") or "")
    call_id = _infer_call_id(ctx, tenant_id)

    forced = (ctx.get("forced_skill_id") or "").strip()
    forced_reply = _maybe_execute_forced_websearch_skill(forced, db, current_user)
    if forced_reply:
        return forced_reply

    cancel_reply = _maybe_handle_cancel(text, db, current_user)
    if cancel_reply:
        return cancel_reply

    handled = _maybe_handle_stateful_flow(text, db, current_user, tenant_id=tenant_id, call_id=call_id, ctx=ctx)
    if handled:
        return handled

    if _explicit_web_search_request(text):
        res = run_web_search(text)
        spoken = _format_web_answer_for_voice(res.answer)
        out = {"spoken_reply": spoken, "fsm": {"intent": "web_search", "mode": "direct"}, "gmail": None}
        if _should_offer_save_as_skill(text):
            _set_state(tenant_id, call_id, {"stage": "offer_save", "query": text, "last_answer": res.answer})
            out["spoken_reply"] = spoken + " Would you like to save this as a dedicated skill and optionally schedule it?"
        return out

    core_payload = _run_assistant_route_core(text, db, current_user, account_id=account_id, context=context)

    if _should_offer_web_fallback(user_text=text, core_payload=core_payload):
        if _web_fallback_allowed(ctx):
            _set_state(tenant_id, call_id, {"stage": "ask_permission", "query": text})
            prompt = (
                "I couldn't find that information in my current skills. "
                "Would you like me to search the internet for it?"
            )
            return {"spoken_reply": prompt, "fsm": {"intent": "web_search", "mode": "ask_permission"}, "gmail": core_payload.get("gmail")}
    return core_payload


def _maybe_execute_forced_websearch_skill(forced_skill_id: str, db: Session, user: User) -> dict | None:
    if not forced_skill_id or not forced_skill_id.startswith("websearch_"):
        return None

    raw = forced_skill_id.replace("websearch_", "", 1).strip()
    try:
        skill_uuid = uuid.UUID(raw)
    except Exception:
        return {"spoken_reply": "I couldn't run that web search skill because its ID looks invalid.", "fsm": {"intent": "web_search", "mode": "forced_invalid"}, "gmail": None}

    skill = (
        db.query(WebSearchSkill)
        .filter(WebSearchSkill.tenant_id == user.id, WebSearchSkill.id == skill_uuid, WebSearchSkill.enabled == True)  # noqa: E712
        .first()
    )
    if not skill:
        return {"spoken_reply": "I couldn't find that saved web search skill.", "fsm": {"intent": "web_search", "mode": "forced_missing"}, "gmail": None}

    res = run_web_search(skill.query)
    spoken = _format_web_answer_for_voice(res.answer)

    today = datetime.utcnow().strftime("%Y-%m-%d")
    header = f"{skill.name} for {today}. "
    return {"spoken_reply": header + spoken, "fsm": {"intent": "web_search", "mode": "forced_exec", "skill_id": forced_skill_id}, "gmail": None}


def _infer_call_id(ctx: dict, tenant_id: str) -> str:
    for k in ("stream_sid", "call_sid", "session_id", "conversation_id", "from_number"):
        v = ctx.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return tenant_id or "global"


def _get_state(tenant_id: str, call_id: str) -> dict | None:
    st = memory.get_handle(tenant_id=tenant_id, call_id=call_id, name=_STATE_KEY, default=None)
    return st if isinstance(st, dict) else None


def _set_state(tenant_id: str, call_id: str, state: dict | None) -> None:
    if state is None:
        memory.set_handle(tenant_id=tenant_id, call_id=call_id, name=_STATE_KEY, value=None, ttl_s=1)
        return
    memory.set_handle(tenant_id=tenant_id, call_id=call_id, name=_STATE_KEY, value=state, ttl_s=SESSION_MEMORY_TTL_S)


def _clear_state(tenant_id: str, call_id: str) -> None:
    _set_state(tenant_id, call_id, None)


def _maybe_handle_stateful_flow(text: str, db: Session, user: User, *, tenant_id: str, call_id: str, ctx: dict) -> dict | None:
    st = _get_state(tenant_id, call_id)
    if not st:
        return None

    stage = str(st.get("stage") or "").strip()

    if stage == "ask_permission":
        if _is_yes(text):
            query = str(st.get("query") or "").strip() or (text or "").strip()
            res = run_web_search(query)
            spoken = _format_web_answer_for_voice(res.answer)
            if _should_offer_save_as_skill(query):
                _set_state(tenant_id, call_id, {"stage": "offer_save", "query": query, "last_answer": res.answer})
                spoken += " Would you like to save this as a dedicated skill and optionally schedule it?"
            else:
                _clear_state(tenant_id, call_id)
            return {"spoken_reply": spoken, "fsm": {"intent": "web_search", "mode": "answer"}, "gmail": None}

        if _is_no(text):
            _clear_state(tenant_id, call_id)
            return {"spoken_reply": "Okay — I won't search the internet right now.", "fsm": {"intent": "web_search", "mode": "declined"}, "gmail": None}

        return {
            "spoken_reply": "Just to confirm — should I search the internet for that? Please say yes or no.",
            "fsm": {"intent": "web_search", "mode": "clarify_permission"},
            "gmail": None,
        }

    if stage == "offer_save":
        query = str(st.get("query") or "").strip()
        if _is_no(text):
            _clear_state(tenant_id, call_id)
            return {"spoken_reply": "No problem. If you need it again, just ask.", "fsm": {"intent": "web_search", "mode": "no_save"}, "gmail": None}

        if _is_yes(text):
            name = _suggest_skill_name(query)
            triggers = _suggest_triggers(query)
            skill = create_web_search_skill(db, user, name=name, query=query, triggers=triggers)
            _set_state(tenant_id, call_id, {"stage": "ask_auto_execute", "skill_id": str(skill.id), "query": query})
            return {
                "spoken_reply": f"Done. I created the skill '{name}'. Would you like to auto-execute it daily and receive a report?",
                "fsm": {"intent": "web_search", "mode": "skill_created", "skill_id": str(skill.id)},
                "gmail": None,
            }

        return {
            "spoken_reply": "Would you like me to save this as a dedicated skill? Please say yes or no.",
            "fsm": {"intent": "web_search", "mode": "clarify_save"},
            "gmail": None,
        }

    if stage == "ask_auto_execute":
        if _is_no(text):
            _clear_state(tenant_id, call_id)
            return {"spoken_reply": "Okay. The skill is saved, but it won't run automatically.", "fsm": {"intent": "web_search", "mode": "no_auto_execute"}, "gmail": None}
        if _is_yes(text):
            _set_state(tenant_id, call_id, {"stage": "ask_time", "skill_id": st.get("skill_id"), "query": st.get("query")})
            return {"spoken_reply": "Great. What time each day should I send the report?", "fsm": {"intent": "web_search", "mode": "ask_time"}, "gmail": None}
        return {"spoken_reply": "Would you like this to run automatically every day? Please say yes or no.", "fsm": {"intent": "web_search", "mode": "clarify_auto_execute"}, "gmail": None}

    if stage == "ask_time":
        parsed = parse_time_of_day(text or "")
        if not parsed:
            return {"spoken_reply": "What time should I send it? For example, 8 AM or 6:30 PM.", "fsm": {"intent": "web_search", "mode": "clarify_time"}, "gmail": None}
        hour, minute = parsed
        timezone = (os.getenv("DEFAULT_TIMEZONE") or os.getenv("APP_TZ") or "America/New_York").strip() or "America/New_York"
        _set_state(
            tenant_id,
            call_id,
            {
                "stage": "ask_channel",
                "skill_id": st.get("skill_id"),
                "hour": hour,
                "minute": minute,
                "timezone": timezone,
            },
        )
        return {
            "spoken_reply": "How would you like to receive it — email, SMS text, WhatsApp, or a phone call?",
            "fsm": {"intent": "web_search", "mode": "ask_channel"},
            "gmail": None,
        }

    if stage == "ask_channel":
        chan = _parse_channel(text or "")
        if not chan:
            return {
                "spoken_reply": "Please choose one: email, SMS text, WhatsApp, or phone call.",
                "fsm": {"intent": "web_search", "mode": "clarify_channel"},
                "gmail": None,
            }
        _set_state(
            tenant_id,
            call_id,
            {
                "stage": "ask_destination",
                "skill_id": st.get("skill_id"),
                "hour": st.get("hour"),
                "minute": st.get("minute"),
                "timezone": st.get("timezone"),
                "channel": chan.value,
            },
        )
        default_hint = _default_destination_hint(chan, ctx)
        return {
            "spoken_reply": "Where should I send it? " + default_hint,
            "fsm": {"intent": "web_search", "mode": "ask_destination"},
            "gmail": None,
        }

    if stage == "ask_destination":
        chan_s = str(st.get("channel") or "").strip()
        chan = DeliveryChannel(chan_s) if chan_s in [c.value for c in DeliveryChannel] else None
        if chan is None:
            _clear_state(tenant_id, call_id)
            return {"spoken_reply": "Something went wrong setting the delivery channel. Please try again.", "fsm": {"intent": "web_search", "mode": "error"}, "gmail": None}

        dest = _parse_destination(text or "", chan, ctx)
        if not dest:
            return {
                "spoken_reply": "I didn't catch the destination. Please say the phone number or email address.",
                "fsm": {"intent": "web_search", "mode": "clarify_destination"},
                "gmail": None,
            }

        hour = int(st.get("hour") or 0)
        minute = int(st.get("minute") or 0)
        timezone = str(st.get("timezone") or "America/New_York")

        skill_id = st.get("skill_id")
        if not skill_id:
            _clear_state(tenant_id, call_id)
            return {"spoken_reply": "I lost track of which skill to schedule. Please try again.", "fsm": {"intent": "web_search", "mode": "error"}, "gmail": None}

        row = upsert_daily_schedule(
            db,
            user,
            web_search_skill_id=skill_id,
            hour=hour,
            minute=minute,
            timezone=timezone,
            channel=chan,
            destination=dest,
        )
        _clear_state(tenant_id, call_id)

        return {
            "spoken_reply": f"All set. I'll send that report every day at {row.time_of_day} via {chan.value}.",
            "fsm": {"intent": "web_search", "mode": "scheduled", "schedule_id": str(row.id)},
            "gmail": None,
        }

    _clear_state(tenant_id, call_id)
    return None


def _maybe_handle_cancel(text: str, db: Session, user: User) -> dict | None:
    t = (text or "").strip().lower()
    if not t:
        return None

    if any(k in t for k in ("cancel my report", "stop my report", "cancel the report", "stop the report", "stop sending", "cancel auto execute", "stop auto execute")):
        n = disable_all_schedules(db, user)
        if n <= 0:
            return {"spoken_reply": "I don't see any active scheduled reports right now.", "fsm": {"intent": "web_search", "mode": "cancel_none"}, "gmail": None}
        return {"spoken_reply": f"Okay. I disabled {n} scheduled report{'s' if n != 1 else ''}.", "fsm": {"intent": "web_search", "mode": "cancelled"}, "gmail": None}

    return None


def _explicit_web_search_request(text: str) -> bool:
    t = (text or "").lower()
    return any(k in t for k in ("search the internet", "search the web", "look it up online", "google", "can you look up"))


def _looks_like_lookup(text: str) -> bool:
    t = (text or "").lower()
    if not t:
        return False
    return any(
        k in t
        for k in (
            "today",
            "in effect",
            "parking",
            "alternate side",
            "asp",
            "rules",
            "reviews",
            "rating",
            "ratings",
            "consumer ratings",
            "hours",
            "address",
            "phone number",
            "website",
            "menu",
            "price",
            "prices",
        )
    )


def _should_offer_web_fallback(*, user_text: str, core_payload: dict) -> bool:
    if not _looks_like_lookup(user_text):
        return False

    spoken = (core_payload or {}).get("spoken_reply")
    spoken_l = (spoken or "").lower()

    if spoken and len(spoken.strip()) > 18 and not any(k in spoken_l for k in ("not sure", "rephrase", "could you", "i don't know")):
        return False

    if any(k in spoken_l for k in ("rephrase", "give me a bit more detail", "not entirely sure", "i'm not sure what you meant")):
        return True

    if not spoken or len(spoken.strip()) < 6:
        return True

    return False


def _web_fallback_allowed(ctx: dict) -> bool:
    raw = (os.getenv("WEB_SEARCH_FALLBACK_EXCEPTIONS") or "gmail_summary").strip()
    exceptions = {s.strip() for s in raw.split(",") if s.strip()}

    forced = (ctx.get("forced_skill_id") or "").strip()
    if forced and forced in exceptions:
        return False

    if bool(ctx.get("no_websearch")):
        return False

    return True


def _should_offer_save_as_skill(query: str) -> bool:
    q = (query or "").lower()
    if not q:
        return False
    if "alternate side" in q or "asp" in q:
        return True
    if "parking" in q and ("nyc" in q or "new york" in q):
        return True
    if "rules in effect" in q and ("parking" in q or "alternate side" in q):
        return True
    return False


def _suggest_skill_name(query: str) -> str:
    q = (query or "").strip()
    ql = q.lower()
    if "alternate side" in ql or ("parking" in ql and ("nyc" in ql or "new york" in ql)):
        return "NYC Alternate Side Parking Report"
    if "rating" in ql or "reviews" in ql:
        return "Business Ratings Lookup"
    return "Internet Search Skill"


def _suggest_triggers(query: str) -> list[str]:
    q = (query or "").strip()
    ql = q.lower()
    if "alternate side" in ql or ("parking" in ql and ("nyc" in ql or "new york" in ql)):
        return [
            "parking rules in effect today nyc",
            "alternate side parking report",
            "nyc asp report",
        ]
    return [q[:120]] if q else []


def _format_web_answer_for_voice(answer: str) -> str:
    a = (answer or "").strip()
    if not a:
        return "I searched the internet, but I couldn't find a clear answer."
    max_chars = int((os.getenv("WEB_SEARCH_MAX_SPOKEN_CHARS") or "700").strip() or "700")
    if len(a) > max_chars:
        a = a[: max_chars - 3].rstrip() + "..."
    return a


def _is_yes(text: str) -> bool:
    t = (text or "").strip().lower()
    return t in ("yes", "y", "yeah", "yep", "sure", "okay", "ok", "please do") or t.startswith("yes ")


def _is_no(text: str) -> bool:
    t = (text or "").strip().lower()
    return t in ("no", "n", "nope", "nah", "don't", "do not", "dont") or t.startswith("no ")


def _parse_channel(text: str) -> DeliveryChannel | None:
    t = (text or "").lower()
    if "whatsapp" in t:
        return DeliveryChannel.whatsapp
    if "email" in t:
        return DeliveryChannel.email
    if "text" in t or "sms" in t:
        return DeliveryChannel.sms
    if "call" in t or "phone" in t:
        return DeliveryChannel.call
    return None


def _default_destination_hint(chan: DeliveryChannel, ctx: dict) -> str:
    frm = (ctx.get("from_number") or "").strip()
    if chan in (DeliveryChannel.sms, DeliveryChannel.whatsapp, DeliveryChannel.call) and frm:
        return f"You can say 'this number' to use {frm}, or provide a different number."
    return "Please say the phone number or email address."


def _parse_destination(text: str, chan: DeliveryChannel, ctx: dict) -> str | None:
    t = (text or "").strip()
    tl = t.lower()

    frm = (ctx.get("from_number") or "").strip()
    if frm and any(k in tl for k in ("this number", "same number", "use this number", "send it here")):
        return frm

    if chan == DeliveryChannel.email:
        m = re.search(r"([A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,})", t, re.IGNORECASE)
        if m:
            return m.group(1)
        return None

    m = re.search(r"(\+?\d[\d\s\-().]{7,}\d)", t)
    if m:
        s = re.sub(r"[\s\-().]", "", m.group(1))
        return s
    return None

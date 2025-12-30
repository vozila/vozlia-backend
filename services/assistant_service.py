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
from services.memory_facade import (
    memory,
    make_skill_cache_key_hash,
    SESSION_MEMORY_ENABLED,
    SESSION_MEMORY_TTL_S,
)
from services.caller_cache import (
    CALLER_MEMORY_ENABLED,
    CALLER_MEMORY_TTL_S,
    get_caller_cache,
    put_caller_cache,
    normalize_caller_id,
)
from services.longterm_memory import (
    longterm_memory_enabled_for_tenant,
    fetch_recent_memory_text,
    record_skill_result,
)




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

    # Memory identifiers (used for session caching on transcript turns)
    tenant_uuid = getattr(current_user, "id", None)
    tenant_id = str(tenant_uuid or "")
    ctx = context or {}
    call_id = None
    if isinstance(ctx, dict):
        call_id = ctx.get("call_sid") or ctx.get("stream_sid") or ctx.get("call_id")

    # Caller identifier (for Postgres TTL cache across calls)
    from_number = None
    if isinstance(ctx, dict):
        from_number = ctx.get("from_number") or ctx.get("from") or ctx.get("From")
    caller_id = normalize_caller_id(from_number)
    gmail_data_fresh = False  # safe default; set True only when Gmail fetch occurs

    # Long-term memory context (durable, per tenant + caller_id)
    memory_context = ""
    longterm_enabled = False
    try:
        longterm_enabled = longterm_memory_enabled_for_tenant(tenant_id)
    except Exception:
        longterm_enabled = False

    if longterm_enabled and caller_id and tenant_uuid:
        memory_context = fetch_recent_memory_text(
            db,
            tenant_uuid=tenant_uuid,
            caller_id=caller_id,
            limit=int(os.getenv("LONGTERM_MEMORY_CONTEXT_LIMIT", "8") or 8),
        )
        if debug and memory_context:
            logger.info(
                "LONGTERM_MEM_CONTEXT_READY tenant_id=%s caller_id=%s chars=%s",
                tenant_id,
                caller_id,
                len(memory_context),
            )


    
    # -------------------------
    # Backend call: memory_recall (Option A: no stream.py changes needed)
    # -------------------------
    backend_call = None
    try:
        if isinstance(ctx, dict):
            backend_call = (ctx.get("backend_call") or "").strip() or None
    except Exception:
        backend_call = None

    if backend_call == "memory_recall":
        from services.memory_controller import parse_memory_query, search_memory_events

        t_mem0 = _time.perf_counter()
        qmem = parse_memory_query(text or "")
        rows = []
        try:
            rows = search_memory_events(
                db,
                tenant_id=tenant_id,
                caller_id=caller_id or "",
                q=qmem,
                limit=int(os.getenv("LONGTERM_MEMORY_RECALL_TOPK", "12") or 12),
            )
            logger.info(
                "MEMORY_RECALL_QUERY_OK tenant_id=%s caller_id=%s skill=%s kws=%s window_days=%.1f hits=%s ms=%.1f",
                tenant_id,
                caller_id,
                qmem.skill_key or None,
                ",".join(qmem.keywords or []),
                (qmem.end_ts - qmem.start_ts).total_seconds()/86400.0,
                len(rows),
                (_time.perf_counter() - t_mem0) * 1000.0,
            )
        except Exception as e:
            logger.exception("MEMORY_RECALL_QUERY_FAIL tenant_id=%s caller_id=%s err=%s", tenant_id, caller_id, e)
            rows = []

        if not rows:
            spoken = "I couldn’t find anything in your recent history for that. Can you tell me roughly when it was or what it was about?"
            return {
                "spoken_reply": spoken,
                "fsm": {
                    "mode": "memory_recall",
                    "has_evidence": False,
                    "hits": 0,
                    "skill_key": qmem.skill_key,
                    "keywords": qmem.keywords,
                },
                "gmail": None,
            }

        # Evidence-first summary (MVP, deterministic)
        # We return a natural sentence + include evidence snippets for debugging / future improvements.
        snippets = []
        for r in rows[:6]:
            ts = (r.created_at.isoformat(timespec="seconds") if getattr(r, "created_at", None) else "")
            snippets.append(f"[{ts}] {r.skill_key}: {r.text}")

        spoken = "Here’s what I found from your history: " + " ".join(snippets[:3])
        if len(snippets) > 3:
            spoken += " …and I found a few more related notes."

        return {
            "spoken_reply": spoken,
            "fsm": {
                "mode": "memory_recall",
                "has_evidence": True,
                "hits": len(rows),
                "skill_key": qmem.skill_key,
                "keywords": qmem.keywords,
                "evidence": snippets,
            },
            "gmail": None,
        }

    fsm = VozliaFSM()
    # Portal-controlled greeting (admin-configurable)
    fsm.greeting_text = get_agent_greeting(db, current_user)

    fsm_context = context or {}
    if isinstance(fsm_context, dict) and memory_context:
        fsm_context.setdefault("memory_context", memory_context)

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

            gmail_data_fresh = False

            cache_hash = None
            cached = None
            if SESSION_MEMORY_ENABLED and call_id and tenant_id:
                cache_hash = make_skill_cache_key_hash(
                    "gmail_summary",
                    account_id_effective,
                    (gmail_query or ""),
                    int(gmail_max_results),
                )
                cached = memory.get_cached_skill_result(
                    tenant_id=tenant_id,
                    call_id=call_id,
                    skill_key="gmail_summary",
                    cache_key_hash=cache_hash,
                )

            # If session cache miss, try caller-scoped cache (Postgres TTL) across calls
            caller_cached = None
            if (not cached) and CALLER_MEMORY_ENABLED and caller_id and tenant_uuid and cache_hash:
                caller_cached = get_caller_cache(
                    db,
                    tenant_id=tenant_uuid,
                    caller_id=caller_id,
                    skill_key="gmail_summary",
                    cache_key_hash=cache_hash,
                )

            if caller_cached and isinstance(caller_cached.get("gmail"), dict):
                gmail_data = dict(caller_cached.get("gmail") or {})
                if debug:
                    logger.info(
                        "ASSISTANT_GMAIL_CALLER_CACHE_HIT tenant_id=%s caller_id=%s hash=%s used_account_id=%s",
                        tenant_id,
                        caller_id,
                        (cache_hash or "")[:8],
                        account_id_effective,
                    )
                # Warm session cache for the rest of this call
                if SESSION_MEMORY_ENABLED and call_id and tenant_id and cache_hash:
                    memory.put_cached_skill_result(
                        tenant_id=tenant_id,
                        call_id=call_id,
                        skill_key="gmail_summary",
                        cache_key_hash=cache_hash,
                        result={"gmail": gmail_data},
                        ttl_s=SESSION_MEMORY_TTL_S,
                    )
                    msgs = (gmail_data.get("messages") or []) if isinstance(gmail_data, dict) else []
                    ids = [m.get("id") for m in msgs if isinstance(m, dict) and m.get("id")]
                    memory.set_handle(tenant_id=tenant_id, call_id=call_id, name="last_gmail_message_ids", value=ids)
                    memory.set_handle(tenant_id=tenant_id, call_id=call_id, name="last_gmail_messages", value=msgs[:20])

            elif cached:

                gmail_data = dict((cached.result or {}).get("gmail") or {})
                if debug:
                    logger.info(
                        "ASSISTANT_GMAIL_CACHE_HIT tenant_id=%s call_id=%s hash=%s used_account_id=%s",
                        tenant_id,
                        call_id,
                        (cache_hash or "")[:8],
                        account_id_effective,
                    )
            else:
                t_g = _time.perf_counter()
                gmail_data = summarize_gmail_for_assistant(
                    account_id_effective,
                    current_user,
                    db,
                    max_results=gmail_max_results,
                    query=gmail_query,
                )
                gmail_data_fresh = True
                if debug:
                    logger.info(
                        "ASSISTANT_GMAIL_DONE dt_ms=%s got_messages=%s summary_len=%s used_account_id=%s",
                        int((_time.perf_counter() - t_g) * 1000),
                        len((gmail_data.get('messages') or [])) if isinstance(gmail_data, dict) else None,
                        len((gmail_data.get('summary') or '')) if isinstance(gmail_data, dict) else None,
                        account_id_effective,
                    )

                # Cache for follow-ups within this call
                if SESSION_MEMORY_ENABLED and call_id and tenant_id and cache_hash and isinstance(gmail_data, dict):
                    memory.put_cached_skill_result(
                        tenant_id=tenant_id,
                        call_id=call_id,
                        skill_key="gmail_summary",
                        cache_key_hash=cache_hash,
                        result={"gmail": gmail_data},
                        ttl_s=SESSION_MEMORY_TTL_S,
                    )
                    msgs = (gmail_data.get("messages") or []) if isinstance(gmail_data, dict) else []
                    ids = [m.get("id") for m in msgs if isinstance(m, dict) and m.get("id")]
                    memory.set_handle(tenant_id=tenant_id, call_id=call_id, name="last_gmail_message_ids", value=ids)
                    memory.set_handle(tenant_id=tenant_id, call_id=call_id, name="last_gmail_messages", value=msgs[:20])

                    # Also cache across calls for this caller (Postgres TTL), if enabled
                    if CALLER_MEMORY_ENABLED and caller_id and tenant_uuid and cache_hash and isinstance(gmail_data, dict):
                        put_caller_cache(
                            db,
                            tenant_id=tenant_uuid,
                            caller_id=caller_id,
                            skill_key="gmail_summary",
                            cache_key_hash=cache_hash,
                            result_json={"gmail": gmail_data},
                            ttl_s=CALLER_MEMORY_TTL_S,
                        )

            if gmail_data.get("summary"):
                spoken_reply = (
                    (spoken_reply.strip() + " " + gmail_data["summary"].strip()).strip()
                    if spoken_reply
                    else gmail_data["summary"].strip()
                )

            # Long-term memory write (per tenant + caller) - fail-open
            if longterm_enabled and caller_id and tenant_uuid:
                try:
                    record_skill_result(
                        db,
                        tenant_uuid=tenant_uuid,
                        caller_id=caller_id,
                        skill_key="gmail_summary",
                        input_text=text,
                        memory_text=(gmail_data.get("summary") or "").strip()[:900],
                        data_json={
                            "account_id": account_id_effective,
                            "query": gmail_query,
                            "max_results": gmail_max_results,
                            "used_account_id": gmail_data.get("used_account_id"),
                            "fresh": bool(gmail_data_fresh),
                        },
                        expires_in_s=int(os.getenv("LONGTERM_MEMORY_SKILL_EVENT_TTL_S", "0") or 0) or None,
                    )
                except Exception:
                    logger.exception("LONGTERM_MEM_RECORD_GMAIL_FAILED")
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

                # Session memory cache (SkillsEngine path)
                cache_hash = None
                cached = None
                if SESSION_MEMORY_ENABLED and call_id and tenant_id:
                    cache_hash = make_skill_cache_key_hash("gmail_summary", account_id_effective, "", 20)
                    cached = memory.get_cached_skill_result(
                        tenant_id=tenant_id,
                        call_id=call_id,
                        skill_key="gmail_summary",
                        cache_key_hash=cache_hash,
                    )

                # If session cache miss, try caller-scoped cache (Postgres TTL) across calls
                caller_cached = None
                if (not cached) and CALLER_MEMORY_ENABLED and caller_id and tenant_uuid and cache_hash:
                    caller_cached = get_caller_cache(
                        db,
                        tenant_id=tenant_uuid,
                        caller_id=caller_id,
                        skill_key="gmail_summary",
                        cache_key_hash=cache_hash,
                    )

                if caller_cached and isinstance(caller_cached.get("gmail"), dict):
                    gmail_data = dict(caller_cached.get("gmail") or {})
                    logger.info(
                        "ASSISTANT_GMAIL_CALLER_CACHE_HIT (skills_engine) tenant_id=%s caller_id=%s hash=%s used_account_id=%s",
                        tenant_id,
                        caller_id,
                        (cache_hash or "")[:8],
                        account_id_effective,
                    )
                    # Warm session cache for this call
                    if SESSION_MEMORY_ENABLED and call_id and tenant_id and cache_hash:
                        memory.put_cached_skill_result(
                            tenant_id=tenant_id,
                            call_id=call_id,
                            skill_key="gmail_summary",
                            cache_key_hash=cache_hash,
                            result={"gmail": gmail_data},
                            ttl_s=SESSION_MEMORY_TTL_S,
                        )
                        msgs = (gmail_data.get("messages") or []) if isinstance(gmail_data, dict) else []
                        ids = [mm.get("id") for mm in msgs if isinstance(mm, dict) and mm.get("id")]
                        memory.set_handle(tenant_id=tenant_id, call_id=call_id, name="last_gmail_message_ids", value=ids)
                        memory.set_handle(tenant_id=tenant_id, call_id=call_id, name="last_gmail_messages", value=msgs[:20])

                elif cached:

                    gmail_data = dict((cached.result or {}).get("gmail") or {})
                    logger.info(
                        "ASSISTANT_GMAIL_CACHE_HIT (skills_engine) tenant_id=%s call_id=%s hash=%s used_account_id=%s",
                        tenant_id,
                        call_id,
                        (cache_hash or "")[:8],
                        account_id_effective,
                    )
                else:
                    gmail_data = summarize_gmail_for_assistant(account_id_effective, current_user, db)
                    # Cache for follow-ups within this call
                    if SESSION_MEMORY_ENABLED and call_id and tenant_id and cache_hash and isinstance(gmail_data, dict):
                        memory.put_cached_skill_result(
                            tenant_id=tenant_id,
                            call_id=call_id,
                            skill_key="gmail_summary",
                            cache_key_hash=cache_hash,
                            result={"gmail": gmail_data},
                            ttl_s=SESSION_MEMORY_TTL_S,
                        )
                        msgs = (gmail_data.get("messages") or []) if isinstance(gmail_data, dict) else []
                        ids = [m.get("id") for m in msgs if isinstance(m, dict) and m.get("id")]
                        memory.set_handle(tenant_id=tenant_id, call_id=call_id, name="last_gmail_message_ids", value=ids)
                        memory.set_handle(tenant_id=tenant_id, call_id=call_id, name="last_gmail_messages", value=msgs[:20])
                        # Also cache across calls for this caller (Postgres TTL), if enabled
                        if CALLER_MEMORY_ENABLED and caller_id and tenant_uuid and cache_hash and isinstance(gmail_data, dict):
                            put_caller_cache(
                                db,
                                tenant_id=tenant_uuid,
                                caller_id=caller_id,
                                skill_key="gmail_summary",
                                cache_key_hash=cache_hash,
                                result_json={"gmail": gmail_data},
                                ttl_s=CALLER_MEMORY_TTL_S,
                            )

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

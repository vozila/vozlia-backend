# services/assistant_service.py
from skills.engine import skills_engine_enabled, match_skill_id, execute_skill
from services.settings_service import (
    get_agent_greeting,
    gmail_summary_enabled,
    gmail_summary_add_to_greeting,
    get_gmail_summary_engagement_phrases,
    shortterm_memory_enabled,
    longterm_memory_enabled,
    get_memory_engagement_phrases,
)

import os
import re
from core.logging import logger
from skills.registry import skill_registry
from sqlalchemy.orm import Session
from models import User
from vozlia_fsm import VozliaFSM
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
    record_turn_event,
    longterm_memory_enabled_for_tenant,
    fetch_recent_memory_text,
    record_skill_result,
)




from services.gmail_service import get_default_gmail_account_id, summarize_gmail_for_assistant


def _looks_like_memory_question(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return False

    # Strong recall phrasing
    if any(p in t for p in [
        "what did i say",
        "remind me",
        "last time",
        "previous call",
        "earlier today",
        "earlier on",
        "mentioned earlier",
        "that i mentioned",
        "you mentioned",
        "did i mention",
    ]):
        return True

    # Favorite color recall (question-like only)
    if ("favorite color" in t or "favourite colour" in t):
        # avoid treating store utterances as recall
        if any(p in t for p in [
            "my favorite color is",
            "my favourite colour is",
            "remember that my favorite color is",
            "remember my favorite color is",
        ]):
            return False
        if "what" in t or "which" in t or "remind" in t or t.endswith("?"):
            return True

    # Generic "what X was" questions often imply recall
    if t.endswith("?") and any(q in t for q in ["what", "which", "where", "when", "who", "how"]):
        if any(p in t for p in ["mentioned", "said", "told you", "earlier", "before"]):
            return True

    return False




def _answer_favorite_color(rows) -> str | None:
    # rows are CallerMemoryEvent ordered newest-first
    # Prefer structured facts, but fall back to parsing raw text to avoid "no hits" for common phrasings.
    fav_re = re.compile(r"\bfavorite\s+color\b(?:\s+\w+){0,4}\s+(?:is|was)\s+([A-Za-z]+)\b", re.I)
    for r in rows or []:
        try:
            # 1) Tags
            tags = getattr(r, "tags_json", None) or []
            if isinstance(tags, list):
                for tag in tags:
                    if isinstance(tag, str) and tag.startswith("fact:favorite_color="):
                        v = tag.split("=", 1)[1].strip()
                        if v:
                            return v

            # 2) data_json facts
            data = getattr(r, "data_json", None) or {}
            if isinstance(data, dict):
                facts = data.get("facts") or {}
                if isinstance(facts, dict) and facts.get("favorite_color"):
                    v = str(facts["favorite_color"]).strip()
                    if v:
                        return v

            # 3) Fallback: parse the stored text
            body = getattr(r, "text", None) or ""
            if isinstance(body, str) and body:
                m = fav_re.search(body)
                if m:
                    v = (m.group(1) or "").strip()
                    if v:
                        return v.lower()
        except Exception:
            continue
    return None

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
    else:
        text_snip = ""

    text_l = (text or "").lower()

    # Skill engagement phrases (Gmail Summary)
    force_gmail_summary = False
    try:
        phrases = get_gmail_summary_engagement_phrases(db, current_user)
        force_gmail_summary = any((p or "").strip().lower() in text_l for p in phrases)
    except Exception:
        force_gmail_summary = False

    # Memory engagement phrases (force memory routing even if heuristics miss)
    force_memory = False
    try:
        mphrases = get_memory_engagement_phrases(db, current_user)
        force_memory = any((p or "").strip().lower() in text_l for p in mphrases)
    except Exception:
        force_memory = False

    if debug:
        logger.info(
            "ASSISTANT_ROUTE_START user_id=%s account_id=%s text=%s ctx_keys=%s",
            getattr(current_user, "id", None),
            account_id,
            text_snip,
            sorted(list((context or {}).keys())),
        )

    # Memory identifiers (used for session caching on transcript turns)
    tenant_uuid = getattr(current_user, "id", None)
    tenant_id = str(tenant_uuid or "")
    ctx = context or {}
    call_id = None
    if isinstance(ctx, dict):
        call_id = ctx.get("call_sid") or ctx.get("stream_sid") or ctx.get("call_id")
        # Short-term memory toggle: disable session cache without touching hot paths
        try:
            if not shortterm_memory_enabled(db, current_user):
                call_id = None
        except Exception:
            pass

    # Caller identifier (for Postgres TTL cache across calls)
    from_number = None
    if isinstance(ctx, dict):
        from_number = ctx.get("from_number") or ctx.get("from") or ctx.get("From")
    caller_id = normalize_caller_id(from_number)
    gmail_data_fresh = False  # safe default; set True only when Gmail fetch occurs

    # -------------------------
    # LONGTERM MEMORY: config + context + capture turns (Option A)
    # -------------------------
    memory_context = ""
    # Longterm memory toggle: DB overrides, env fallback happens inside settings_service
    try:
        longterm_enabled = longterm_memory_enabled(db, current_user)
    except Exception as e:
        longterm_enabled = False
        if debug:
            logger.exception("LONGTERM_MEMORY_ENABLED_CHECK_FAIL tenant_id=%s err=%s", tenant_id, e)

    capture_turns = (os.getenv("LONGTERM_MEMORY_CAPTURE_TURNS", "1") or "1").strip().lower() in ("1", "true", "yes", "on")
    if debug:
        logger.info(
            "LONGTERM_MEMORY_CONFIG enabled=%s capture_turns=%s tenant_id=%s caller_id=%s call_sid=%s",
            longterm_enabled,
            capture_turns,
            tenant_id,
            caller_id,
            call_id,
        )

    # Pull small recent context for prompt grounding (keep short; no hot-path bloat)
        # -------------------------
    # AUTO memory question handling (facts-first for MVP)
    # -------------------------
    if longterm_enabled and caller_id and tenant_uuid and (force_memory or _looks_like_memory_question(text)):
        from services.memory_controller import (
            parse_memory_query,
            search_memory_events,
            infer_fact_key,
            fetch_fact_history,
        )

        try:
            qmem = parse_memory_query(text or "")

            # 1) Facts-first: if the question maps to a known fact key (e.g., favorite_color),
            # answer deterministically from stored facts (no guessing).
            fact_key = infer_fact_key(text or "")
            if fact_key:
                facts = fetch_fact_history(
                    db,
                    tenant_id=str(tenant_uuid),
                    caller_id=str(caller_id),
                    fact_key=fact_key,
                    start_ts=qmem.start_ts,
                    end_ts=qmem.end_ts,
                    limit=int(os.getenv("LONGTERM_FACT_RECALL_LIMIT", "8") or 8),
                )

                if facts:
                    newest = facts[0]
                    newest_val = (newest.get("value") or "").strip()

                    distinct: list[str] = []
                    for f in facts:
                        v = (f.get("value") or "").strip()
                        if v and v not in distinct:
                            distinct.append(v)

                    label = "favorite color" if fact_key == "favorite_color" else fact_key.replace("_", " ")

                    if newest_val and len(distinct) == 1:
                        return {
                            "spoken_reply": f"You told me your {label} is {newest_val}.",
                            "fsm": {
                                "mode": "memory_fact",
                                "fact_key": fact_key,
                                "value": newest_val,
                                "as_of": newest.get("created_at_iso"),
                            },
                            "gmail": None,
                        }

                    if newest_val and len(distinct) > 1:
                        choices = []
                        for f in facts[:3]:
                            v = (f.get("value") or "").strip()
                            ts = f.get("created_at_iso") or ""
                            if v:
                                choices.append(f"{v} (as of {ts})")
                        choice_str = "; ".join(choices) if choices else ", ".join(distinct[:3])
                        return {
                            "spoken_reply": f"I have a couple different answers for your {label}: {choice_str}. Which one should I use going forward?",
                            "fsm": {
                                "mode": "memory_fact_conflict",
                                "fact_key": fact_key,
                                "values": distinct[:5],
                                "evidence": facts[:3],
                            },
                            "gmail": None,
                        }

                # No fact hits: fail-soft WITHOUT guessing.
                label = "favorite color" if fact_key == "favorite_color" else fact_key.replace("_", " ")
                return {
                    "spoken_reply": f"I couldn’t find your {label} in my notes. What did you want it to be?",
                    "fsm": {"mode": "memory_fact", "fact_key": fact_key, "status": "no_hits"},
                    "gmail": None,
                }

            # 2) Fallback: evidence snippets for generic “what did I say” questions.
            rows = search_memory_events(
                db,
                tenant_id=str(tenant_uuid),
                caller_id=str(caller_id),
                q=qmem,
                limit=int(os.getenv("LONGTERM_MEMORY_RECALL_LIMIT", "50") or 50),
            )
        except Exception as e:
            logger.exception("AUTO_MEMORY_RECALL_FAIL tenant_id=%s caller_id=%s err=%s", tenant_id, caller_id, e)
            rows = []

        if not rows:
            if debug:
                logger.info("AUTO_MEMORY_NO_HITS tenant_id=%s caller_id=%s q=%s", tenant_id, caller_id, (text or "")[:200])
            return {
                "spoken_reply": "I couldn’t find that in my recent history. Can you repeat it, or give me a slightly wider timeframe?",
                "fsm": {"mode": "memory_recall", "status": "no_hits"},
                "gmail": None,
            }

        snippets = []
        for r in rows[:4]:
            try:
                ts = r.created_at.isoformat(timespec="seconds")
            except Exception:
                ts = ""
            snippets.append(f"[{ts}] {r.text}")

        spoken = "Here’s what I found: " + " … ".join(snippets[:3])
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


    # Capture *every* user turn (best-effort; fail-open)
    if longterm_enabled and capture_turns and caller_id and tenant_uuid:
        ok = record_turn_event(
            db,
            tenant_uuid=str(tenant_uuid),
            caller_id=str(caller_id),
            call_sid=str(call_id) if call_id else None,
            session_id=str(call_id) if call_id else None,
            role="user",
            text=text or "",
        )

        if debug:
            logger.info("MEMORY_CAPTURE_TURN_USER ok=%s tenant_id=%s caller_id=%s", ok, tenant_id, caller_id)

    # -------------------------
    # AUTO memory question handling (facts-first for MVP)
    # -------------------------
    if longterm_enabled and caller_id and tenant_uuid and _looks_like_memory_question(text):
        from services.memory_controller import parse_memory_query, search_memory_events

        try:
            qmem = parse_memory_query(text or "")
            rows = search_memory_events(
                db,
                tenant_id=str(tenant_uuid),
                caller_id=str(caller_id),
                q=qmem,
                limit=int(os.getenv("LONGTERM_MEMORY_RECALL_LIMIT", "50") or 50),
            )
        except Exception as e:
            logger.exception("AUTO_MEMORY_RECALL_FAIL tenant_id=%s caller_id=%s err=%s", tenant_id, caller_id, e)
            rows = []


        # If we detected a memory question but could not retrieve any rows, fail-soft WITHOUT guessing.
        if not rows:
            if debug:
                logger.info("AUTO_MEMORY_NO_HITS tenant_id=%s caller_id=%s q=%s", tenant_id, caller_id, (text or "")[:200])
            return {
                "spoken_reply": "I couldn’t find that in my notes from that time window. Can you repeat it, or give me a slightly wider timeframe?",
                "fsm": {"mode": "memory_recall", "status": "no_hits"},
                "gmail": None,
            }

        if "favorite color" in (text or "").lower():
            val = _answer_favorite_color(rows)
            if val:
                return {
                    "spoken_reply": f"You told me your favorite color was {val}.",
                    "fsm": {"mode": "memory_recall", "fact": "favorite_color", "value": val},
                    "gmail": None,
                }

        if rows:
            latest = rows[0]
            raw = None
            try:
                data = getattr(latest, "data_json", None) or {}
                if isinstance(data, dict):
                    raw = data.get("raw")
            except Exception:
                raw = None
            snippet = (raw or getattr(latest, "text", "") or "").strip()
            if len(snippet) > 240:
                snippet = snippet[:240] + "…"
            return {
                "spoken_reply": f"Here’s what I found from our recent conversation: {snippet}",
                "fsm": {"mode": "memory_recall", "hits": len(rows)},
                "gmail": None,
            }

        return {
            "spoken_reply": "I couldn’t find that in your recent history. Can you tell me roughly when you said it?",
            "fsm": {"mode": "memory_recall", "hits": 0},
            "gmail": None,
        }

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

    base_greeting = get_agent_greeting(db, current_user)
    greeting = base_greeting

    # Optional: append the Gmail Summary skill greeting line to the greeting
    try:
        if gmail_summary_enabled(db, current_user) and gmail_summary_add_to_greeting(db, current_user):
            sk = skill_registry.get("gmail_summary")
            add = (getattr(sk, "greeting", "") or "").strip() if sk else ""
            if add:
                # keep spacing clean
                greeting = (base_greeting or "").strip()
                if greeting:
                    greeting = greeting + (" " if not greeting.endswith(("!", ".", "?")) else " ") + add
                else:
                    greeting = add
    except Exception:
        pass

    fsm.greeting_text = greeting


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
    if (not backend_call) and force_gmail_summary:
        backend_call = {
            "type": "gmail_summary",
            "params": {"query": "is:unread", "max_results": 20},
        }
        # Keep behavior consistent with FSM email intent reply
        spoken_reply = "Sure, I'll take a quick look at your recent unread emails."
        try:
            fsm_result = dict(fsm_result)
            fsm_result["backend_call"] = backend_call
        except Exception:
            pass


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

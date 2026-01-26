# services/metrics_controller.py
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from zoneinfo import ZoneInfo
from sqlalchemy.orm import Session

from core.logging import logger
from services.metrics_service import (
    run_metrics_question,
    _fail as _fail_private,  # type: ignore
    _ok as _ok_private,      # type: ignore
    _resolve_time_window as _resolve_time_window_private,  # type: ignore
    _count_call_sessions as _count_call_sessions_private,  # type: ignore
    _count_unique_callers as _count_unique_callers_private,  # type: ignore
    _recent_callers as _recent_callers_private,  # type: ignore
    _top_callers as _top_callers_private,  # type: ignore
    _email_summaries_requested as _email_summaries_requested_private,  # type: ignore
    _count_skill_invocations as _count_skill_invocations_private,  # type: ignore
    _skills_timeseries as _skills_timeseries_private,  # type: ignore
    _skills_top as _skills_top_private,  # type: ignore
)
from services.metrics_llm_planner import plan_metrics_question, MetricsPlan
from services.settings_service import get_skills_config
from services.dynamic_skill_runtime import match_dynamic_skill


def _safe_tz(tz: str | None) -> str:
    tz = (tz or "").strip() or "America/New_York"
    try:
        ZoneInfo(tz)
        return tz
    except Exception:
        return "America/New_York"


def _resolve_time_window_from_plan(plan: MetricsPlan):
    tz_name = _safe_tz(plan.timeframe.timezone if plan.timeframe else None)
    preset = (plan.timeframe.preset if plan.timeframe else None) or "today"

    if preset != "last_n_days":
        return _resolve_time_window_private(preset, tz_name)

    # last_n_days
    n = int(plan.timeframe.n_days or 30)
    n = max(1, min(n, 365))

    tz = ZoneInfo(tz_name)
    now_utc = datetime.now(timezone.utc)
    now_local = now_utc.astimezone(tz)
    start_local = now_local - timedelta(days=n)
    start_utc_naive = start_local.astimezone(timezone.utc).replace(tzinfo=None)
    end_utc_naive = now_local.astimezone(timezone.utc).replace(tzinfo=None)

    # TimeWindow is a simple container; helper functions only require start_utc/end_utc/preset.
    return type("TimeWindowCompat", (), {"start_utc": start_utc_naive, "end_utc": end_utc_naive, "preset": f"last_{n}_days"})()


def _resolve_skill_key(db: Session, *, user: Any, skill_ref: Any) -> str | None:
    """Resolve a skill reference to a skill_key used in caller_memory_events.

    Accepts:
      - explicit skill_key (built-ins like gmail_summary)
      - saved/dynamic skill label ("Today's Sports Digest")
    """
    if not skill_ref:
        return None

    sk = ""
    name = ""

    if hasattr(skill_ref, "skill_key") or hasattr(skill_ref, "name"):
        # pydantic model
        sk = str(getattr(skill_ref, "skill_key", "") or "").strip()
        name = str(getattr(skill_ref, "name", "") or "").strip()
    elif isinstance(skill_ref, dict):
        sk = str(skill_ref.get("skill_key") or skill_ref.get("key") or "").strip()
        name = str(skill_ref.get("name") or "").strip()
    else:
        name = str(skill_ref).strip()

    if sk:
        return sk

    if not name:
        return None

    # If the user typed a canonical built-in key, accept it directly
    if (" " not in name) and (len(name) <= 64):
        return name

    # Try dynamic skill matching (websearch_*/dbquery_*)
    try:
        m = match_dynamic_skill(db, user, name)
        if m and m.skill_id:
            return m.skill_id
    except Exception:
        pass

    # Fallback: try matching by label substring in the skills_config mapping
    try:
        cfg_all = get_skills_config(db, user) or {}
        name_l = name.lower()
        for skill_id, cfg in cfg_all.items():
            if not isinstance(skill_id, str) or not isinstance(cfg, dict):
                continue
            label = (cfg.get("label") or "").strip()
            if label and (label.lower() in name_l or name_l in label.lower()):
                return skill_id
    except Exception:
        pass

    return None


def run_metrics(
    db: Session,
    *,
    tenant_id: str,
    question: str,
    timezone: str | None = None,
    user: Any | None = None,
) -> dict[str, Any]:
    """Single entrypoint for both Portal troubleshooting chat and Voice metrics.

    Strategy (safe + auditable):
      1) Fast deterministic parser (existing regex heuristics)
      2) If it fails AND METRICS_LLM_PLANNER_ENABLED=1, use LLM to produce a validated MetricsPlan
      3) Deterministically execute the plan (never letting the LLM invent numbers)

    Rollback:
      - set METRICS_LLM_PLANNER_ENABLED=0 to revert to step (1) only
    """
    tz_name = _safe_tz(timezone)

    # (1) Existing deterministic behavior
    out = run_metrics_question(db, tenant_id=tenant_id, question=question, timezone=tz_name)
    if isinstance(out, dict) and out.get("ok"):
        out.setdefault("spoken_reply", out.get("spoken_summary"))
        return out

    # (2) LLM plan
    plan = plan_metrics_question(question, timezone=tz_name)
    if plan is None:
        if isinstance(out, dict):
            out.setdefault("spoken_reply", out.get("spoken_summary"))
        return out

    if plan.needs_clarification:
        msg = (plan.clarifying_question or "Can you clarify what you mean?").strip()
        fail = _fail_private(msg)
        fail.setdefault("spoken_reply", fail.get("spoken_summary"))
        fail["plan"] = plan.model_dump()
        return fail

    # (3) Execute plan deterministically
    try:
        w = _resolve_time_window_from_plan(plan)
        preset_label = str(getattr(w, "preset", "today") or "today")

        if plan.metric == "calls.count":
            n = int(_count_call_sessions_private(db, tenant_id, w) or 0)
            ok = _ok_private("calls.count", preset_label, {"call_sessions": n}, f"Calls received {preset_label.replace('_',' ')}: {n}.")
        elif plan.metric == "calls.unique_callers":
            n = int(_count_unique_callers_private(db, tenant_id, w) or 0)
            ok = _ok_private("calls.unique_callers", preset_label, {"unique_callers": n}, f"Unique callers {preset_label.replace('_',' ')}: {n}.")
        elif plan.metric == "calls.recent_callers":
            limit = int(plan.limit or 25)
            limit = max(1, min(limit, 500))
            callers = _recent_callers_private(db, tenant_id, w, tz_name, limit)
            spoken = f"Most recent callers {preset_label.replace('_',' ')} (showing {min(len(callers),10)} of {limit})."
            ok = _ok_private("calls.recent_callers", preset_label, {"callers": callers, "limit": limit}, spoken)
        elif plan.metric == "calls.top_callers":
            limit = int(plan.limit or 10)
            limit = max(1, min(limit, 100))
            rows = _top_callers_private(db, tenant_id, w, limit)
            spoken = f"Top callers {preset_label.replace('_',' ')} (showing {min(len(rows),10)} of {limit})."
            ok = _ok_private("calls.top_callers", preset_label, {"top": rows, "limit": limit}, spoken)
        elif plan.metric == "skills.email_summaries_requested":
            stats = _email_summaries_requested_private(db, tenant_id, w)
            spoken = (
                f"Email summaries {preset_label.replace('_',' ')}:\n"
                f"- requested in {stats.get('request_call_sessions',0)} call session(s)\n"
                f"- {stats.get('request_turns',0)} matching turn(s)\n"
                f"- skill executed {stats.get('skill_invocations',0)} time(s)"
            )
            ok = _ok_private("skills.email_summaries_requested", preset_label, stats, spoken)
        elif plan.metric in ("skills.invocations", "skills.invocations_timeseries", "skills.top_invoked"):
            if user is None:
                fail = _fail_private("Which skill do you mean? Example: skill 'gmail_summary' or a saved skill name.")
                fail.setdefault("spoken_reply", fail.get("spoken_summary"))
                fail["plan"] = plan.model_dump()
                return fail

            sk = _resolve_skill_key(db, user=user, skill_ref=plan.skill)
            if plan.metric == "skills.top_invoked":
                limit = int(plan.limit or 20)
                limit = max(1, min(limit, 100))
                by_distinct = bool(plan.by_distinct_callers or False)
                rows = _skills_top_private(db, tenant_id, w, limit, by_distinct_callers=by_distinct)
                spoken = f"Top skills {preset_label.replace('_',' ')} (showing {min(len(rows),10)} of {limit})."
                ok = _ok_private("skills.top_invoked", preset_label, {"top": rows, "limit": limit, "by_distinct_callers": by_distinct}, spoken)
            elif plan.metric == "skills.invocations_timeseries":
                bucket = plan.bucket or "day"
                rows = _skills_timeseries_private(db, tenant_id, w, tz_name, bucket=bucket, skill_key=sk)
                label = f"Skill '{sk}' invocations by {bucket}" if sk else f"Skill invocations by {bucket}"
                ok = _ok_private("skills.invocations_timeseries", preset_label, {"bucket": bucket, "skill_key": sk, "rows": rows}, f"{label} ({preset_label.replace('_',' ')}, {tz_name}).")
            else:
                if not sk:
                    fail = _fail_private("Which skill do you mean? Example: skill 'gmail_summary' or a saved skill name.")
                    fail.setdefault("spoken_reply", fail.get("spoken_summary"))
                    fail["plan"] = plan.model_dump()
                    return fail
                inv = int(_count_skill_invocations_private(db, tenant_id, w, sk) or 0)
                ok = _ok_private("skills.invocations", preset_label, {"skill_key": sk, "invocations": inv}, f"Skill '{sk}' invocations {preset_label.replace('_',' ')}: {inv}.")
        else:
            ok = _fail_private("That metric isn't supported yet.")
    except Exception as e:
        logger.exception("METRICS_PLAN_EXEC_FAIL err=%s", str(e)[:300])
        ok = _fail_private("I hit an error while computing that metric. Please try again.")

    ok.setdefault("spoken_reply", ok.get("spoken_summary"))
    ok["plan"] = plan.model_dump()
    return ok

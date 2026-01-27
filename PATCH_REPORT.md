# Intent Router V2 Patch

## What this patch adds

- **LLM-assisted intent routing** for skill execution (natural language → JSON plan → deterministic execution)
- Feature flag for safe cutover + rollback: `INTENT_V2_MODE=off|shadow|assist`
- Disambiguation flow (stores options in `session_store` for the next turn)
- Marks DBQuery v1 modules as **LEGACY / SLATED FOR REMOVAL** (comments only; no behavior change)
- Adds documentation: `INTENT_ROUTER_V2.md`

## Files

- `services/intent_router_v2.py` (NEW)
- `services/assistant_service.py` (MODIFIED: hooks V2 router + disables legacy dynamic matcher when enabled)
- `services/db_query_service.py` (COMMENT ONLY)
- `services/db_query_skill_store.py` (COMMENT ONLY)
- `api/routers/dbquery.py` (COMMENT ONLY)
- `INTENT_ROUTER_V2.md` (NEW)

## How to enable

1) Deploy with `INTENT_V2_MODE=off` first (default) – confirm no changes.
2) Set `INTENT_V2_MODE=assist` and redeploy.
3) Optional: `INTENT_V2_DEBUG=1` for more logs.

Rollback: set `INTENT_V2_MODE=off` and redeploy.

## Expected log lines

- `INTENT_V2_CANDIDATES ...`
- `INTENT_V2_PLAN mode=assist route=run_skill ...`
- `INTENT_V2_FASTPATH ...` (when deterministic match is obvious)


## New env var

- `INTENT_V2_STRICT=1` to disable the legacy dynamic skill matcher (V2 becomes the single authority). Default is `0`.

# Intent Router V2

This document explains the **new LLM-first intent routing layer** added alongside the existing (legacy) routing stack.

## Why we need this

Owners and customers speak in **natural language**, not standardized command phrases.

The previous routing approach relied on:
- deterministic substring/token matching (dynamic skills)
- a lightweight FSM (rule-based)

That works for some cases but becomes unreliable as the skill catalog grows and phrasing varies.

**Intent Router V2** uses:
1) **Deterministic candidate generation** (fast, no network): pick the most relevant skills
2) **LLM plan** (network): choose which skill (or ask to disambiguate) using STRICT JSON
3) **Deterministic execution** (Python): run the chosen skill through existing engines

## Safety / reliability guarantees

- The LLM is only allowed to produce a **plan**.
- The plan is **schema-validated** before execution.
- If the plan is invalid or missing → we **fall back to legacy routing**.
- There is an env-var cutover with instant rollback.

## Feature flags

### `INTENT_V2_MODE`

- `off` (default): Router V2 disabled (legacy behavior)
- `shadow`: compute plan + log it, but do NOT change behavior
- `assist`: execute valid plans, otherwise fall back to legacy behavior

Rollback: set `INTENT_V2_MODE=off`.

### Optional tuning

- `INTENT_V2_DEBUG=1`  
  Logs candidate lists and planning outcomes.
- `INTENT_V2_MAX_CANDIDATES=10`  
  Caps how many skills are sent to the LLM (keeps prompts small).

### OpenAI routing model

- `OPENAI_INTENT_MODEL` (default: `gpt-4o-mini`)
- `OPENAI_INTENT_TIMEOUT_S` (default: `4.0`)
- `OPENAI_INTENT_MAX_TOKENS` (default: `220`)

## Where it runs in the call path

Inside `services/assistant_service.run_assistant_route()`:

1) existing memory / session logic runs as before
2) **Intent Router V2** gets a chance to route & execute a skill
3) if V2 does nothing → legacy routing continues

By default, V2 runs first and the legacy dynamic matcher remains as a **fallback** (stability first).
If you want V2 to be the single authority, set `INTENT_V2_STRICT=1`.

## What it supports today

- Routing natural language into:
  - **dynamic skills** created via WebUI (websearch_* / dbquery_*)
  - **legacy YAML skills** (gmail_summary, investment_reporting)
- Multi-turn disambiguation:
  - If ambiguous, the assistant asks “Which one did you mean?” and stores the options for the next turn.

## Known limitations (intentional for now)

- Categories are not persisted yet (we disambiguate by showing top skill names).
- Memory analytics / metrics are NOT redesigned here (this layer only routes to skills).

## Testing checklist

1) With `INTENT_V2_MODE=off` confirm legacy behavior unchanged.
2) Enable `INTENT_V2_MODE=assist` and test:
   - “please give me my sports digest”
   - “run the investment report on AAPL”
   - ambiguous prompt that should trigger disambiguation
3) Confirm rollback:
   - set `INTENT_V2_MODE=off` and redeploy; behavior returns to legacy immediately.


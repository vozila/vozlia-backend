# Patch: Intent V2 dynamic-skill activation keywords gate

## Goal
Prevent ambiguous natural-language mentions (e.g., "sports update") from triggering **dynamic skill disambiguation**
unless the user explicitly signals they want to run a saved skill/report.

This applies **only** to dynamic skills:
- `websearch_*`
- `dbquery_*`

Legacy skills, memory, KB, and general chitchat are unaffected.

## What changed
1) `services/intent_router_v2.py`
- Added env-var gate: `INTENT_V2_DYNAMIC_ACTIVATION_KEYWORDS`
- When the gate is set (non-empty), dynamic skill candidates are only considered if:
  - the utterance contains one of the activation keywords/phrases, OR
  - the user explicitly referenced a skill (skill-name/trigger substring), OR
  - the match score is very high (>=90, which corresponds to explicit substring matching)

2) `INTENT_ROUTER_V2.md`
- Documented the new env var.

## New environment variable
`INTENT_V2_DYNAMIC_ACTIVATION_KEYWORDS`

Examples:
- `skill,report`
- `skill,report,digest,summary`
- `run skill`  (single phrase; if you want multiple phrases, separate with commas)

If unset/empty → gate disabled (legacy behavior).

## Recommended rollout
1) Deploy this patch
2) Set `INTENT_V2_MODE=assist` (already) + optionally `INTENT_V2_DEBUG=1` temporarily
3) Set `INTENT_V2_DYNAMIC_ACTIVATION_KEYWORDS=skill,report`
4) Smoke tests (curl):
   - Should NOT disambiguate/run a dynamic skill:
     `sports update`
   - Should disambiguate/run dynamic skills:
     `sports report`
     `run sports skill`
   - Should still run explicit skill name triggers without keywords:
     `Todays Sports Digest`
     `alternate side parking`

Rollback:
- Set `INTENT_V2_DYNAMIC_ACTIVATION_KEYWORDS=` (empty) OR remove it.
- Or set `INTENT_V2_MODE=off` for full rollback.

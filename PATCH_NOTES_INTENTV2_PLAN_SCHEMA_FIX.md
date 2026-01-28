# Intent V2 Plan Schema Fix

- Fix: `IntentPlanV2` now includes `category_request` and `schedule_request` fields.
- Why: Previously the LLM could return these fields, but Pydantic dropped them, causing category routing to never activate.
- Behavior: No change for `run_skill` routing; adds support for `route='category'` and preserves `schedule_request` metadata for future use.

## Rollback
- Set `INTENT_V2_MODE=off` to disable Intent Router V2.
- Or revert this patch commit.

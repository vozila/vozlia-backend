# Patch Notes: Dynamic Skill Autosync + Admin Settings Restore

## Why this patch exists
We observed **code drift / rollbacks** where:
- WebSearch skills still existed in the **WebSearchSkill** table (so `/admin/websearch/skills` listed them),
- but the same skills were **missing from** `user_settings.skills_config`,
- and both **Intent Router V2** and the legacy **dynamic skill runtime** rely on `skills_config` for routing/execution.

Result: utterances like **"please give me my sports digest"** were routed as `unknown` / chitchat / KB fallback.

## What changed
### Added
- `services/dynamic_skill_autosync.py`
  - Best-effort repair step that ensures:
    - every WebSearchSkill row is present as `skills_config["websearch_<id>"]`
    - every DBQuerySkill row is present as `skills_config["dbquery_<id>"]`
  - Adds default engagement phrases when the owner did not provide triggers.
  - Appends missing keys to `skills_priority_order` without reordering existing items.
  - Env flag: `DYNAMIC_SKILLS_AUTOSYNC=0` disables.

- `api/routers/admin_dynamic_skills.py`
  - `POST /admin/dynamic-skills/sync` (admin-key protected)
  - Useful for troubleshooting without redeploy.

### Restored
- `api/routers/admin_settings.py` is now included in `main.py` so `/admin/settings` works again.

### Startup behavior
On app startup, if `DYNAMIC_SKILLS_AUTOSYNC` is enabled, the backend runs a one-time autosync to repair dynamic skill routing after drift.

## Rollback plan
- Disable Intent Router V2: `INTENT_V2_MODE=off`
- Disable autosync: `DYNAMIC_SKILLS_AUTOSYNC=0`

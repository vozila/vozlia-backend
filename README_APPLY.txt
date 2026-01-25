Vozlia Backend Patch: Dynamic WebSearch Skills in Voice + DB Query Skill Capability
-------------------------------------------------------------------------------

What this patch does:

1) Voice can execute WebUI-created WebSearch skills
   - When you say something like "give me my sports digest", the backend will:
     * deterministically match that utterance to a saved skill (websearch_<uuid>)
     * run its saved query via run_web_search(...)
     * return a voice-safe truncated reply (WEB_SEARCH_MAX_SPOKEN_CHARS)
   - Matching uses engagement_phrases (triggers) and also the skill label/name.

2) Adds a DB query skill capability (MVP)
   - New SQL table: db_query_skills
   - New Admin API:
       GET  /admin/dbquery/entities
       POST /admin/dbquery/run
       GET  /admin/dbquery/skills
       POST /admin/dbquery/skills
       DELETE /admin/dbquery/skills/{skill_id}
   - DB query spec is a flexible JSON DSL (filters, timeframe, grouping, aggregations).

3) Dynamic skill execution (websearch_* / dbquery_*) is wired into /assistant/route
   - This means BOTH Flow A (Twilio WS / realtime stream calling /assistant/route)
     and Flow B (router calling /assistant/route) get the same behavior.

Optional guardrails / env vars:

- DYNAMIC_SKILLS_ENABLED=1   (default 1)
- DYNAMIC_SKILLS_CACHE_TTL_S=15
- DYNAMIC_SKILLS_CALLER_ALLOWLIST="+15551234567,+15557654321"
    If set, ONLY those callers can execute dynamic skills (recommended once you enable db_query skills).
- DB_QUERY_MAX_SPOKEN_CHARS=900 (defaults to MAX_SPOKEN_CHARS or 900)

Deployment notes:

- This adds a NEW table (db_query_skills). Backend uses Base.metadata.create_all(...) on startup,
  so the table will be created automatically if missing.

Rollback:

- Revert these files to the previous versions (no data migrations required).

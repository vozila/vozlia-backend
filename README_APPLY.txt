Vozlia Patch — DB Scheduling Destination Fix + DBQuery Formatting + Concept Codes Scaffold
Date: 2026-02-01

This patch implements two milestones from the checkpoint:
(A) Phase A paper-cuts (high ROI correctness)
(B) Phase B Concept Codes data model (foundation for deterministic cross-vertical analytics)

--------------------------------------------------------------------------------
A) Phase A fixes
--------------------------------------------------------------------------------
1) Prevents placeholder schedule destinations like destination='email' from being persisted.
   - If channel=email and destination is missing/placeholder/non-email-ish, it defaults to the primary user's email.
   - Intent Router V2 now speaks the resolved destination.

2) Flattens DBQuery aggregate outputs so scheduled emails don't show tuple/Row formatting like '(0,)'.
   - Prefer Row._mapping when available.
   - Scalarizes single-element containers in summaries.

--------------------------------------------------------------------------------
B) Phase B scaffold: Concept Codes tables + admin API
--------------------------------------------------------------------------------
Adds tenant-scoped tables:
- concept_definitions
- concept_batches
- concept_assignments

Adds admin endpoints (admin-key gated):
- GET/POST  /admin/concepts/definitions
- POST      /admin/concepts/batches
- GET/POST  /admin/concepts/assignments

Feature flag (default OFF):
- CONCEPTS_ENABLED=0|1

NOTE: This patch only adds the data model + CRUD/admin endpoints.
It does NOT yet:
- run LLM enrichment batches automatically
- join concept assignments inside DBQuerySpec compilation
Those are the next patches (Phase C/D).

--------------------------------------------------------------------------------
Changed files (whole-file replacements)
--------------------------------------------------------------------------------
NEW:
- services/delivery_destination.py
- services/concepts_store.py
- api/routers/concepts.py
- migrations/2026_02_01_concept_codes

UPDATED:
- services/intent_router_v2.py
- services/web_search_skill_store.py
- services/db_query_skill_store.py
- services/db_query_service.py
- models.py
- main.py
- CODE_DRIFT_CONTROL.md

--------------------------------------------------------------------------------
How to apply
--------------------------------------------------------------------------------
1) Copy files from this zip into your backend repo at the same relative paths.
2) Apply the migration:
   psql "$DATABASE_URL" -f migrations/2026_02_01_concept_codes

3) Deploy backend + worker.

--------------------------------------------------------------------------------
Env flags
--------------------------------------------------------------------------------
Existing (already in use):
- INTENT_V2_MODE=assist
- INTENT_V2_SCHEDULE_ENABLED=1
- DBQUERY_SCHEDULE_ENABLED=1

New:
- CONCEPTS_ENABLED=1   # to enable /admin/concepts/* endpoints

--------------------------------------------------------------------------------
Recommended smoke tests
--------------------------------------------------------------------------------
A) Schedule via natural language (email destination unspecified)
1. Call /assistant/route with: "Send me my daily callers report at 10am."
2. Confirm scheduled_deliveries.destination is your real email (not 'email').

B) Forced-run worker
1. Force-run schedule:
   update scheduled_deliveries set last_run_at=null, next_run_at=now()-interval '2 minutes' where id='<uuid>';
2. Confirm email body does NOT include '(0,)'.

C) Concepts CRUD
1. Set CONCEPTS_ENABLED=1 and redeploy.
2. Create a concept definition:
   POST /admin/concepts/definitions { "concept_code":"menu.steak", "name":"Steak", ... }
3. Create a manual assignment:
   POST /admin/concepts/assignments { "target_type":"kb_document", "target_id":"<id>", "concept_code":"menu.steak", "source":"manual" }
4. List:
   GET /admin/concepts/assignments?concept_code=menu.steak

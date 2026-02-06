# Vozlia Code Drift Control Protocol

**File:** `CODE_DRIFT_CONTROL.md`  
**Applies to:** backend, control plane, webui (even if webui is “troubleshooting only”)  
**Goal:** reduce “code drift” when chat context resets, by making file intent + invariants explicit and keeping changes auditable, reversible, and testable.

---

## 1) What “code drift” means in Vozlia

In Vozlia, drift usually happens when:

- The system spans **multiple repos** (backend, control plane, webui) and changes are made in only one place.
- A feature is partially migrated (voice vs chat diverge).
- A previous patch left **stranded code paths** that silently stop being used.
- Large files accumulate logic and the assistant (or a human) loses the mental model after a context switch.

**Drift outcome:** features “sort of work,” but break in edge cases, or different interfaces call different engines.

---

## 2) Non‑negotiables

### 2.1 One engine, multiple interfaces
Voice and portal chat MUST call the **same underlying execution engine** wherever possible.

- **LLM interprets intent**
- **Python executes deterministically**
- **Same router + same tools**
- UI differences must not create logic forks

### 2.2 Hot paths stay clean
Anything that runs in real‑time audio loops must remain minimal:
- no “heavy planning”
- no large DB scans
- no prompt-building with huge payloads

### 2.3 Feature flags for every new execution path
Any new routing/intent/memory layer must be behind an env-var flag so rollback is instant.

Example pattern:
- `INTENT_ROUTER_V2=0|1`
- `METRICS_V2=0|1`
- `WIZARD_AUTOSAVE_OFFER=0|1`

Default must be `0` until validated.

---

## 3) “Touched File Contract” (required every time code is changed)

Whenever you modify **any** `.py`, `.ts`, `.tsx`, or worker file:

### 3.1 Add or update a **File Purpose Header**
Add at the top (or keep it updated):

```python
"""VOZLIA FILE PURPOSE
Purpose: <1-2 sentences>
Hot path: <yes/no>  (If yes, list timing constraints)
Public interfaces: <functions/classes/endpoints that callers rely on>
Reads/Writes: <DB tables/redis/cache/files>
Feature flags: <ENV flags that gate behavior>
Failure mode: <what happens when dependencies fail>
Last touched: <YYYY-MM-DD>  (and why)
"""
```

**Notes**
- This is NOT “doc spam.” It’s “future context insurance.”
- If a file is truly tiny, keep this short.

### 3.2 Maintain a small **Local Changelog** (optional but recommended)
Near top:

```python
# CHANGELOG (recent)
# - 2026-01-26: Added INTENT_ROUTER_V2 guard around new router path.
```

### 3.3 Update this doc’s registry
Append an entry to the registry section below (Section 8).

---

## 4) Change Discipline (how changes are proposed)

Every change should be framed as:

1. **Goal**
2. **Minimal diff**
3. **Guard/flag**
4. **Test plan**
5. **Rollback plan**
6. **Expected logs (≤5 lines) to prove it’s working**

If a change cannot be guarded, stop and redesign it.

---

## 5) Tests and evidence (minimum bar)

Each iteration must include:

- One “happy path” test (curl or UI action)
- One “failure mode” test (missing skill, missing DB row, missing key)
- Evidence: ≤5 log lines proving the intended code path ran

If a test can’t be run (e.g., needs Twilio), provide:
- the exact endpoint to call
- the payload
- expected response shape
- expected log line signature

---

## 6) Drift prevention patterns (preferred)

### 6.1 “Router-first” architecture
All inbound utterances (voice + chat) should pass through:

**(a) LLM intent interpreter** → validated JSON  
**(b) deterministic router (FSM)** → executes specific skill/tool  
**(c) response formatter** → voice vs chat presentation differences only

### 6.2 Avoid regex as the primary “understanding”
Regex may be used for:
- fast-path extraction (dates/times)
- defensive validation
- last-resort fallback

…but not as the primary meaning layer.

### 6.3 Canonical schemas with “future-proof” extensibility
When designing schemas:
- allow “unknown fields” (forward compatibility)
- prefer `filters: []` arrays over fixed N fields
- treat enums as “soft” where possible (`Literal` + fallback to `"other"`)

---

## 7) Rollback discipline (mandatory)

Every new system must support rollback via **single env var**.

Rollback checklist:
- flip env var to disable new path
- redeploy
- verify baseline endpoints still pass smoke tests

Never require a code revert for rollback if avoidable.

---

## 8) Registry of touched files (update on every iteration)

Add entries like:

- `backend/api/routers/metrics.py`
  - Purpose: metrics run endpoint
  - Invariants: never blocks voice hot path; bounded queries
  - Flags: `METRICS_V2`
  - Last touched: 2026-01-26 (added v2 parser + safety caps)


- `backend/services/dynamic_skill_autosync.py`
  - Purpose: sync dynamic DB skills into `user_settings.skills_config` so voice/chat can route to them.
  - Invariants: additive-only merge; no deletes; no DB schema changes; safe defaults.
  - Categories: adds `category` metadata (default + optional heuristic auto-classify at sync-time).
  - Flags: `DYNAMIC_SKILLS_AUTOSYNC`, `DYNAMIC_SKILL_CATEGORY_AUTO`, `DYNAMIC_SKILL_CATEGORY_DEFAULT`
  - Last touched: 2026-01-27 (category metadata + preservation rules)

- `backend/services/intent_router_v2.py`
  - Purpose: LLM-assisted intent router (natural language → strict plan → deterministic execution).
  - Invariants: bounded candidate list; schema-validated plan; safe fallback to legacy; no heavy work in audio hot path.
  - Categories: includes candidate `category` + supports optional `category_request` in plan for disambiguation narrowing.
  - Scheduling: plan may include `schedule_request` and, when enabled, will upsert schedules deterministically (with destination normalization to prevent placeholders like "email").
  - Flags: `INTENT_V2_MODE`, `INTENT_V2_SCHEDULE_ENABLED`, `DBQUERY_SCHEDULE_ENABLED`, `INTENT_V2_DEBUG`, `OPENAI_INTENT_MODEL`, `OPENAI_INTENT_TIMEOUT_S`, `OPENAI_INTENT_MAX_TOKENS`
  - Last touched: 2026-02-06 (fix on-demand dynamic skills autosync so DBQuery/WebSearch skills surface as candidates)


- `backend/models.py`
  - Purpose: ORM models for backend + shared DB tables used by DBQuery (schedules, KB metadata, concepts).
  - Invariants: tenant_id always scopes queries; new concept tables are additive; KBFile/KBChunk mappings mirror Control Plane schema.
  - Flags: n/a (tables may exist even when features are disabled)
  - Last touched: 2026-02-02 (add Concept* tables + KBFile/KBChunk mappings)


- `backend/services/db_query_service.py`
  - Purpose: Deterministic, tenant-scoped DBQuery execution (safe entity registry + bounded filters) for dbquery_* skills and scheduled deliveries.
  - Invariants: only whitelisted entities/fields; always tenant-scoped; time presets are bounded; no writes.
  - Concepts: optional deterministic `has_concept` filter compiles to EXISTS() against concept_assignments (no embeddings).
  - Flags: `CONCEPTS_ENABLED`, `DB_QUERY_MAX_SPOKEN_CHARS`
  - Last touched: 2026-02-06 (fix dbquery aggregation Row serialization to avoid 500; re-add kb_files/kb_chunks/concept_assignments entities)

- `backend/api/routers/concepts.py`
  - Purpose: Admin CRUD endpoints for concept definitions + assignments (auditable concept codes).
  - Invariants: tenant-scoped; never writes without admin key; manual assignments can lock rows.
  - Flags: `CONCEPTS_ENABLED` (DBQuery uses it; endpoints are admin-only regardless)
  - Last touched: 2026-02-02 (initial implementation)

- `backend/main.py`
  - Purpose: FastAPI bootstrap + route registration; DB schema create_all on startup.
  - Invariants: do not add heavy logic; keep additive routers behind flags when appropriate.
  - Last touched: 2026-02-02 (register admin-concepts router)

- `backend/services/web_search_skill_store.py`
  - Purpose: WebSearch skill CRUD + schedule upsert (daily).
  - Invariants: upsert is idempotent per (tenant, web_search_skill_id); sets ScheduledDelivery.skill_key for back-compat.
  - Flags: n/a
  - Last touched: 2026-01-31 (set skill_key + filter list_schedules to websearch only)

- `backend/services/db_query_skill_store.py`
  - Purpose: DBQuery skill CRUD + schedule upsert (daily) for dbquery_* dynamic skills.
  - Invariants: upsert is idempotent per (tenant, skill_key=dbquery_<id>); never runs without tenant scoping.
  - Flags: DBQUERY_SCHEDULE_ENABLED
  - Last touched: 2026-01-31 (added upsert_daily_schedule_dbquery + list_dbquery_schedules)

- `backend/services/intent_router_v2.py`
  - Purpose: LLM-first routing and schedule creation/updates via strict JSON plans.
  - Invariants: Python validates plan; scheduling is gated by flags; no heavy work in voice WS hot path.
  - Flags: INTENT_V2_MODE, INTENT_V2_SCHEDULE_ENABLED, DBQUERY_SCHEDULE_ENABLED
  - Last touched: 2026-01-31 (enable scheduling for dbquery_* behind flag)

- `backend/api/routers/dbquery.py`
  - Purpose: Admin DBQuery skill CRUD + DBQuery schedules list/upsert.
  - Invariants: admin-key protected; schedules only write scheduled_deliveries rows; no free-form SQL execution.
  - Flags: DBQUERY_SCHEDULE_ENABLED
  - Last touched: 2026-01-31 (added /admin/dbquery/schedules endpoints)

- `backend/workers/scheduled_deliveries_worker.py`
  - Purpose: Executes scheduled_deliveries and sends notifications.
  - Invariants: never executes unknown skill types; dbquery_* execution gated behind DBQUERY_SCHEDULE_ENABLED.
  - Flags: DBQUERY_SCHEDULE_ENABLED
  - Last touched: 2026-01-31 (execute dbquery_* schedules)

(Keep this list short and useful; do not document every line—document intent + invariants.)

---

## 9) “Session reset” procedure (for new chats)

When starting a new chat or after a rollback:

1. Identify current deployed git hash / build tag for each repo (backend/control/webui).
2. Restate what is considered “last known good.”
3. List the feature flags and their current values.
4. Identify which engine voice uses and ensure chat uses the same engine.
5. Only then implement changes.

---

## 10) Quick checklist before merging/deploying

- [ ] Does voice and chat call the same engine for the feature?
- [ ] Is there an env flag to turn it off instantly?
- [ ] Did we update file purpose headers for touched files?
- [ ] Did we update the registry in Section 8?
- [ ] Do we have 1 happy-path + 1 failure-path test?
- [ ] Do we have ≤5 log lines proving the code path executed?

---

## 11) Notes on long-term memory safety

Long-term memory is high-stakes. Any changes that write to memory tables must:

- be idempotent
- be tenant-isolated
- include trace_id
- have a reversible migration strategy
- include a “no-op mode” flag for emergency shutdown

Do **not** couple experimental analytics/metrics storage to the long-term memory store.
Keep analytics events separate, and optionally derived.

---

**End of document.**
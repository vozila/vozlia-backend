### Next step toward “agentic communication” (Goal → Steps → FSM executes)

The next practical milestone is to formalize a **multi-step “Goal Plan” contract** that the LLM can emit and the FSM can execute deterministically, where each step is:

* **ensure_skill** (reuse if it already exists; otherwise create)
* **run_skill** (execute an existing dynamic skill)
* **schedule_skill** (optional, if the user includes time/cadence)

…and *all steps* use the same dynamic runtime abstraction you now have for:

* `websearch_<uuid>`
* `dbquery_<uuid>`

So the owner can speak only in high-level goals (“Send me my daily callers report”, “Track steak dinner sales”, “Alert me if calls spike”), the LLM chooses steps and emits strict JSON, and the FSM handles creation/reuse/execution/scheduling deterministically with tenant isolation.

---

## ⚠️ File note (important for the next chat)

Some earlier uploads in this session can expire. In the next chat, **re-upload the repo zip(s) + key docs/env files** (list below) so the new session can open code and give exact replacement-file bundles without guessing.

---

# CHECKPOINT — COPY/PASTE INTO NEW CHAT (DETAILED HANDOVER)

## 0) What to upload in the next session (handover bundle)

Re-upload these to avoid “expired file” issues:

### Repos

1. **Backend repo** (the one deployed to Render; contains `/assistant/route`, dynamic skills, and worker)

   * e.g. `repos.zip` (or a backend-only zip)
2. **Control plane repo** (portal + KB ingestion worker)

   * if separate, include that zip too

### Critical docs

3. `VOZLIA_NORTH_STAR.md`
4. `CODE_DRIFT_CONTROL.md`
5. Any canonical PRD doc you referenced (VOZ-PRD-GOALS-001 if separate)

### Env/config

6. `vozlia-backend-2.env`
7. `Vozlia-Control-2.env`

### Test artifacts (optional but helpful)

8. Render worker logs CSV (scheduled deliveries) if new failures appear
9. `vozlia_metric_questions_500_db_based.md` (for NL→DBQuerySpec future work)

---

## 1) Current Goal (where we are and where we’re going)

**Goal:** Make Vozlia agentic by allowing a business owner to state a goal in natural language, after which:

1. LLM determines steps to reach the goal
2. LLM emits **structured JSON steps**
3. FSM validates JSON, enforces tenant isolation, and executes deterministically
4. Steps mostly map to dynamic skills:

   * `dbquery_<uuid>` (metrics)
   * `websearch_<uuid>` (knowledge/report searches)
5. Scheduling is optional (only when time/cadence is provided)

**Near-term requirement:** FSM should treat DBQuery and WebSearch skills as similarly as possible (same runtime pattern).

---

## 2) Verified working state (DBQuery scheduling + worker execution)

### A) DB migration applied successfully

Database: `vozlia_db` user `vozlia_db_user`.

`scheduled_deliveries` now has:

* `skill_key text NULL`
* `web_search_skill_id uuid NULL` (was NOT NULL before)
* No error column exists (no `last_error`)

Schema confirmed via:

```sql
select ordinal_position, column_name, data_type, is_nullable
from information_schema.columns
where table_name='scheduled_deliveries'
order by ordinal_position;
```

### B) WebSearch schedules unaffected

Existing schedules were backfilled with `skill_key = websearch_<uuid>` and still execute.

### C) DBQuery schedule creation works (Intent V2)

**Key fix:** needed `DBQUERY_SCHEDULE_ENABLED=1` on the backend web service.
Without it, `/assistant/route` returned:

* `intent=schedule_unsupported` with message “Scheduling is not enabled for database metrics yet.”

With the env var set, scheduling works.

### D) DBQuery schedule row exists & upsert works

A DBQuery skill (test) scheduled daily; later updated from 09:00 → 10:00 without creating a duplicate row.

Row example:

* `skill_key = dbquery_484c09ed-...`
* `cadence = daily`
* `time_of_day = 10:00`
* `timezone = America/New_York`
* `next_run_at` shifts accordingly

### E) Worker execution confirmed for DBQuery schedules

Forced-run technique confirmed worker picks up dbquery schedules and updates:

* `last_run_at`
* `last_latency_ms`
* `updated_at`
* `next_run_at`

---

## 3) Key operational issues found (and current workaround)

### Issue: schedule destination was set to literal `"email"`

When scheduling DBQuery via natural language, the created `scheduled_deliveries` row had:

* `channel=email`
* `destination=email`  ❌ (placeholder)

Workaround used:

```sql
update scheduled_deliveries
set destination = 'yasinc74@gmail.com',
    updated_at = now()
where id = '226bc512-3bfd-464f-9845-3514064523bb';
```

After this change, scheduled email delivery succeeded.

**Follow-up needed (code):**
Scheduling should default destination to the owner email (or disambiguate) and must never write `"email"` as a destination.

### Issue: scheduled email content shows tuple formatting

Email contained:

* `value: (0,)`

This is likely because the DBQuery result is being formatted as a raw Python tuple/row.
**Follow-up needed (code):** flatten single-value results into scalars and produce a clean narrative summary.

---

## 4) DBQuery metric correctness validation (why result was 0)

Schedule’s tenant:

* schedule id: `226bc512-...`
* tenant_id: `b4d2353b-667f-4fd2-8ca7-a882e20b9ec3`

`caller_memory_events` has recent rows (mostly 2026-01-31 and some 2026-01-30), but the query for “today” in NY timezone returned 0.

Confirmed with:

```sql
with bounds as (
  select
    (date_trunc('day', now() at time zone 'America/New_York') at time zone 'America/New_York') as start_utc,
    ((date_trunc('day', now() at time zone 'America/New_York') + interval '1 day') at time zone 'America/New_York') as end_utc
)
select
  count(distinct caller_id) as unique_callers_today
from caller_memory_events, bounds
where tenant_id = (select tenant_id from scheduled_deliveries where id = '226bc512-...')
  and (created_at at time zone 'UTC') >= start_utc
  and (created_at at time zone 'UTC') < end_utc;
```

Result:

* `unique_callers_today = 0`

So the scheduled report’s 0 result is consistent with the DB and timeframe boundaries.

---

## 5) DBQuery skill duplicates cleanup

Multiple duplicate DBQuery skills were created during testing. Deleting them works (the earlier “Internal Server Error” was due to malformed curl usage, not a broken endpoint).

Working delete command:

```bash
export DEL_ID="dc20b09b-3f60-414f-a418-2ba4663662bf"
curl -sS -i -X DELETE "$BASE_URL/admin/dbquery/skills/$DEL_ID" \
  -H "x-vozlia-admin-key: $ADMIN_API_KEY"
```

Returns HTTP 200 `{"ok": true}`.

---

## 6) Current env flags that matter (backend + worker)

### Backend web service (required)

* `INTENT_V2_MODE=assist`
* `INTENT_V2_SCHEDULE_ENABLED=1`
* `DBQUERY_SCHEDULE_ENABLED=1`

### Worker service

* must have DB access and run scheduled deliveries loop
* if it gates dbquery execution too, set:

  * `DBQUERY_SCHEDULE_ENABLED=1`

---

## 7) The next engineering milestone for agentic goals (what to build next)

### Milestone: Multi-step plan schema + deterministic executor

Add a new structured “Goal Plan / Step Plan” returned by the LLM (planner) and executed by FSM.

**Core behavior:**

* LLM emits: `steps[]`
* Each step is one of:

  1. `ensure_skill` (create if missing; else reuse)
  2. `run_skill` (execute existing)
  3. `schedule_skill` (upsert schedule if requested)
* Steps must support both `dbquery_*` and `websearch_*`.

**Key requirement:** skill reuse should be deterministic:

* compute a canonical `signature_hash` from the “template/spec”
* reuse existing skill if signature matches; else create new
* store `last_used_at` and allow TTL cleanup later

### Immediate fixes to implement next (low risk, high ROI)

1. **Schedule destination defaulting**

   * if `channel=email`, destination must be a real email
   * default to owner email if known
   * otherwise return `disambiguate`

2. **DBQuery output formatting**

   * flatten single cell row tuples `(0,)` → `0`
   * produce friendly summary: “Unique callers today: 0”

3. **“ensure skill” layer**

   * allow LLM to express a high-level metric request and the FSM either:

     * finds an existing DBQuerySkill (hash match)
     * or creates one, then runs it

4. Expand to include WebSearch steps inside the same multi-step plan (already dynamic skills exist).

---

## 8) Verification commands (repeatable)

### Check schedule row for a skill_key

```bash
psql "$DATABASE_URL" -c "
select id, skill_key, channel, destination, cadence, time_of_day, timezone, next_run_at, last_run_at, last_latency_ms, updated_at
from scheduled_deliveries
where skill_key = 'dbquery_<uuid>';
"
```

### Force-run a schedule (proves worker execution)

```bash
psql "$DATABASE_URL" -c "
update scheduled_deliveries
set last_run_at = null,
    next_run_at = now() - interval '2 minutes'
where id = '<schedule_uuid>';
"
```

Watch updates:

```bash
for i in $(seq 1 10); do
  psql "$DATABASE_URL" -c "
  select now() as now, last_run_at, last_latency_ms, next_run_at, updated_at
  from scheduled_deliveries
  where id = '<schedule_uuid>';
  "
  sleep 3
done
```

---

## 9) Safe defaults & rollback levers

* Disable DBQuery scheduling instantly:

  * `DBQUERY_SCHEDULE_ENABLED=0`
* Disable all scheduling behavior from Intent V2:

  * `INTENT_V2_SCHEDULE_ENABLED=0`
* Disable Intent V2 routing (fallback to legacy):

  * `INTENT_V2_MODE=off`

---

## 10) Open questions to resolve next

1. Where should the “default email destination” come from?

   * owner user record email?
   * a tenant notification preferences table?
2. Do we want “1 schedule per skill_key” enforced by DB uniqueness?
3. For “agentic goals,” do we store new objects now (Goal/WizardSession/Playbook), or do we MVP with a single stored plan and later split into objects?

---

## 11) Known “session responsiveness” issue

This chat session is becoming unresponsive due to context size. Start a new chat and paste this checkpoint + re-upload files listed in section 0.

---

# AUTO-SUMMARY PACK

1. **Current Goal**
   Move from “skills + schedules working” to true agentic goals: user states a goal, LLM emits multi-step structured JSON, FSM ensures/reuses/creates dynamic skills (DBQuery and WebSearch) and executes deterministically, optionally scheduling.

2. **Refactor Step Completed**
   Operational verification completed: DBQuery scheduling + worker execution + email delivery confirmed end-to-end.

3. **What Changed (Code/Config/Infra)**

* Config: `DBQUERY_SCHEDULE_ENABLED=1` enabled on backend web service (and worker as needed).
* DB: schedule destination manually corrected from placeholder `"email"` to real email.

4. **Known Issues**

* Scheduler created schedules with `destination='email'` placeholder; needs code fix (default to owner email or disambiguate).
* Scheduled DBQuery email content shows tuple formatting (`(0,)`) and should be flattened for readability.
* Some uploaded files may expire; re-upload in next session.

5. **Evidence (≤5 log lines)**

* `scheduled_deliveries` row created for `skill_key=dbquery_484c09ed-...`
* Worker forced-run updated `last_run_at`, `last_latency_ms`, and advanced `next_run_at`
* Email received: “value: (0,)”
* SQL verification returned `unique_callers_today = 0`

6. **Tests Performed / To Perform**
   Performed:

* DBQuery schedule creation/upsert
* Worker execution forced-run
* Email delivery confirmed
* SQL correctness check for unique callers

To perform next:

* Implement default destination logic + formatting improvements
* Implement multi-step plan schema + ensure-skill reuse (hash-based)

7. **Next Actions (ordered)**

1) Re-upload repos + env files in new chat (handover list provided).
2) Implement schedule destination defaulting (no placeholder “email”).
3) Improve DBQuery result formatting for scheduled emails (flatten single tuples).
4) Implement multi-step “Goal Plan” JSON schema and FSM executor:

   * ensure_skill (reuse/create) + run_skill + schedule_skill
5) Extend ensure-skill to WebSearch + DBQuery uniformly.

8. **Safe Defaults + Rollback Point**

* `DBQUERY_SCHEDULE_ENABLED=0` disables DBQuery scheduling instantly
* `INTENT_V2_SCHEDULE_ENABLED=0` disables scheduling behavior overall
* `INTENT_V2_MODE=off` reverts to legacy routing

9. **Open Questions**

* Default email destination source (owner email vs tenant prefs)
* Enforce unique schedule per (tenant_id, skill_key)?
* When to introduce persisted Goal/Playbook objects vs MVP plan storage?

10. **Goal/Wizard Status**
    Scheduling substrate is now proven for both WebSearch and DBQuery (flag-gated). Next is agentic goals: step-based plans that create/reuse skills and optionally schedule, without touching realtime Flow A hot path.

11. **Drift Control Notes (touched files + flags + rollback var)**
    Flags involved: `DBQUERY_SCHEDULE_ENABLED`, `INTENT_V2_SCHEDULE_ENABLED`, `INTENT_V2_MODE`.
    No new code touched in this message; next work requires repo re-upload due to possible file expiry.



Good — that **category disambiguation path is now working** in `INTENT_V2_MODE=assist`, based on your latest smoke test:

* `sports update` → returns a **disambiguation list** of skills in the **sports** category
* Replying `1` → executes the selected skill (`Basketball and Football`) successfully
* This confirms the router is now able to:

  1. detect a **category-ish utterance**,
  2. enumerate matching skills,
  3. run a chosen skill without regex-based brittle phrase matching.

Also answering your earlier question directly:

### Does this impact only dynamic skills, or everything?

It impacts **everything that goes through `/assistant/route`**, but **how much it changes behavior depends on `INTENT_V2_MODE`:**

* **`INTENT_V2_MODE=assist` (your current setting)**
  The LLM router runs and can **assist** routing (like category disambiguation), but it is designed to be **non-destructive**: if it can’t confidently route, the system can still fall back to existing behavior (legacy skill routing / KB fallback / chitchat).
  ✅ So: it *applies to all*, but only *takes over* for the specific assist paths we’ve enabled (like category disambiguation).

* **`INTENT_V2_MODE=strict`**
  The LLM router becomes the **primary gate** for *everything*: legacy skills, dynamic skills, memory, chitchat, etc.
  ✅ So: in strict mode it fully affects everything.

That’s exactly why we’re keeping `assist` as the safe rollout posture until you confirm stability.

---

## Checkpoint for new chat session

Below is the handover checkpoint you requested. Copy/paste this into a new chat session when you’re ready.

Also: here is the drift-control doc you already generated so future sessions can stay aligned:
[CODE_DRIFT_CONTROL.md](sandbox:/mnt/data/CODE_DRIFT_CONTROL.md)

---

# AUTO-SUMMARY PACK

## 1) Current Goal

* **Intent routing v2** for Vozlia that works for **all skills** (legacy + dynamic), **memory**, and **chitchat**, using:

  * **LLM-first intent interpretation**
  * **schema-validated structured plan**
  * deterministic execution / routing
* We are currently in the **category + disambiguation** phase (pre-DB-dynamic-skills).
* Next milestone after stability: **DB query–based dynamic skills** built on top of Intent V2.

## 2) Refactor Step Completed

* No large refactor. Incremental safe change: **Intent V2 category routing + disambiguation** added alongside existing router.
* Dynamic skills now support **category metadata** and can be **disambiguated** by category keyword-like utterances.

## 3) What Changed (Code/Config/Infra)

* **Backend**

  * Added/expanded **Intent V2** plan interpretation to support:

    * category detection → skill candidate list → disambiguation prompt
    * selection by number or skill name
  * Added/confirmed **dynamic skill autosync** behavior so dynamic websearch skills appear in `skills_config` and become routable in voice.
  * Restored/confirmed `/admin/settings` endpoint (needed for system configuration visibility and troubleshooting).
* **Infra**

  * Critical deployment issue discovered and resolved: **Render was deploying the wrong Git branch/repo**.
  * Going forward, deployment must pin to the correct branch and ideally a specific SHA for stability.

## 4) Known Issues

* In `assist` mode, the router **may still allow old behavior** for some utterances (by design). Some category phrases may not trigger disambiguation unless the router confidence/capabilities match.
* Category-only utterances (e.g., “sports update”) now disambiguate correctly *when skills exist in that category* — but categories must be consistently present on dynamic skills.
* Websearch content variability remains (LLM/web results can change), but routing is now deterministic enough to consistently select the intended skill.

## 5) Evidence (≤5 log lines)

* `INFO:vozlia:LLM_ROUTER_PLAN mode=assist tool=none conf=0.9 intent=sports_update`
* `POST /assistant/route ... responseTimeMS=2419`
* `/admin/dynamic-skills/sync -> {"ok":true,"enabled":true,"added":0,"updated":0,"total_dynamic":6,"total_config":12}`
* `sports update -> intent_v2 disambiguate -> returns 2 sports skills`
* `reply "1" -> mode=dynamic_skill type=web_search ... intent_v2_reason=disambiguation_number`

## 6) Tests Performed / To Perform

### Performed (confirmed working)

1. `GET /admin/settings` returns settings + `skills_config` with dynamic skills included.
2. `POST /admin/dynamic-skills/sync` returns ok and counts.
3. `POST /assistant/route`:

   * “please give me my sports digest” executes the correct websearch dynamic skill.
   * “sports update” returns category disambiguation list.
   * replying “1” runs the selected skill.

### To Perform (smoke tests you should run before DB skills)

1. **Legacy skill routing test** (gmail_summary via voice): “email summaries”
2. **Memory request test**: “what did I say earlier” / “do you remember…”
3. **Chitchat fallback test**: “tell me a joke” (should not attempt skill execution)

## 7) Next Actions (ordered)

(As you requested — keep this exact ordering)

1. Run the 3 smoke tests above to confirm correct branch is deployed.
2. Freeze this deployment as a rollback point (stable branch/tag + record SHA).
3. Implement category metadata for dynamic skills (default + optional auto-classify).
4. Extend LLM plan schema to include `category_request` + `schedule_request`.
5. Keep cutover controlled via `INTENT_V2_MODE` (shadow → assist → strict).

Then (after confirmation) move forward with:
6. Build **DB dynamic skills** leveraging Intent V2 (robust natural language → structured query spec → deterministic DB execution).

## 8) Safe Defaults + Rollback Point

* **Safe default:** keep `INTENT_V2_MODE=assist` until DB skills are stable.
* **Rollback point:** the current stable build where:

  * dynamic websearch skills are routable in voice
  * category disambiguation works
  * `/admin/settings` works
  * `/admin/dynamic-skills/sync` works
* Rollback mechanism: flip `INTENT_V2_MODE` back (or redeploy stable SHA if needed).

## 9) Open Questions

* Category management UX:

  * categories editable via portal later (owner-only)
  * on skill creation: suggest category, allow override/new category
* Category “mnemonics” / attributes:

  * whether category should remain a simple string now
  * or become `{slug, label, tags, domain, audience, business_type}` later
* How we want DB query intent to work:

  * single “DBQuery skill” that can query all tables
  * vs multiple domain query skills (calls/leads/appointments) for safer scoping

## 10) Goal/Wizard Status (goals/wizard state/playbooks/monitors/notifications/anomalies)

* Wizard UI is de-prioritized as a polished product; portal remains **troubleshooting-only** for now.
* Skills created in voice must still appear in portal for troubleshooting/deletion.
* Intent V2 is being built as the shared core for **voice + chat**.
* DB dynamic skills are next; they must support natural language metrics like:

  * “how many times caller X called last Tuesday”
  * “who called yesterday”
  * “what did caller X say around 2pm yesterday”
* Anomalies not implemented in this phase.

### Flow A (must preserve)

**Flow A (Legacy/Realtime):** Twilio → FastAPI WS (`/twilio/stream`) → OpenAI Realtime (audio in/out) → Twilio
Rule: **No heavy planning** in the audio hot path; all planning/DB-query interpretation stays out-of-band or behind non-stream endpoints.

---

If you start the new chat session with this checkpoint + the repo zip + the backend env file, we’ll begin immediately with the **DB dynamic skill design** using the Intent V2 plan schema (LLM → validated JSON → deterministic DB executor), while keeping rollback simple via `INTENT_V2_MODE`.











Good — if Render is now tracking the **correct branch**, we should lock in a **“known-good baseline”** first (so we don’t reintroduce drift), then we proceed with the **LLM→JSON→FSM intent path** expansion.

## Step 1 — Prove the deployment is on the correct code

Because we previously had “wrong repo / wrong branch” drift, the fastest way to confirm is to hit a few **“newer” endpoints/features** that didn’t exist in the older branch.

Run these from anywhere:

### A) Health (service up)

```bash
curl -s https://vozlia-backend.onrender.com/health
```

### B) Admin settings should exist (this fixed your legacy cards config panel)

```bash
curl -s https://vozlia-backend.onrender.com/admin/settings \
  -H "x-vozlia-admin-key: $ADMIN_KEY"
```

If this returns **404**, you’re still not on the correct code.

### C) Dynamic skill sync endpoint should exist

```bash
curl -s -X POST https://vozlia-backend.onrender.com/admin/dynamic-skills/sync \
  -H "x-vozlia-admin-key: $ADMIN_KEY"
```

You should see an `{ "ok": true, ... }` response.

### D) Confirm a dynamic websearch skill routes from natural language

```bash
curl -s -X POST "https://vozlia-backend.onrender.com/assistant/route" \
  -H "Content-Type: application/json" \
  -d '{
    "text":"please give me my sports digest",
    "context":{"call_id":"debug-call-1","from_number":"+15550001111","channel":"phone"}
  }'
```

If B+C+D are good, we can safely treat this as “baseline is back”.

---

## Step 2 — Freeze this as the rollback point (to stop drift)

Now that it’s working again, do this **immediately**:

1. **Create a “stable” branch/tag** in Git (example: `stable-websearch-dynskills-intentv2-working`).
2. In Render:

   * Ensure auto-deploy follows **that stable branch** (not a moving dev branch), OR keep auto-deploy off and deploy manually only from that branch.
3. Record the deployed commit SHA in your ops notes (or a “DEPLOYED_SHA.md” in the repo).

This ensures that if we break anything during the new intent work, rollback is instant and unambiguous.

---

## Step 3 — Proceed with the new intent route/execution path (alongside the working system)

Your proposed workflow (“LLM first → structured JSON → FSM routes”) is the correct direction for natural language.

### How this differs from what you *had* (and why it failed before)

* The failing approach relied on:

  * **brittle matching** (regex/substring/keyword heuristics), and/or
  * **missing dynamic-skill registration** (skills existed in DB, but weren’t callable by the router)
* That’s why you got:

  * “unknown intent” → KB fallback → chitchat-like behavior
  * and why “sports digest” wasn’t recognized until dynamic skills were synced into the runtime skills config.

### What we’re building now (LLM-first, but still deterministic)

We keep the reliability rule:

**LLM decides intent → outputs strict JSON → Python validates → FSM executes deterministically.**

And we keep it **side-by-side** with the current working path via env var cutover:

* `INTENT_V2_MODE=off|shadow|assist`

  * `off`: old behavior only (rollback)
  * `shadow`: LLM produces plans, logs them, but behavior unchanged
  * `assist`: LLM plan executes if valid; fallback if not

That matches your “build alongside → cutover → rollback by env var” requirement.

---

## Step 4 — The *next* concrete capability to add: category-first routing (your design)

You want:

1. User says: “sports digest”
2. LLM returns JSON like:

   ```json
   {
     "intent": "category_request",
     "category": "sports",
     "confidence": 0.82
   }
   ```
3. FSM checks category:

   * If **only one** sports skill → run it
   * If **multiple** sports skills → present list and let user choose
4. After execution, user can say: “schedule that daily at 9pm”

   * LLM returns schedule JSON
   * FSM creates schedule deterministically

This is the scalable way to avoid stuffing the prompt with 200 skills *and* avoid regex.

**Important detail:** category routing requires that skills have lightweight metadata:

* `category` (sports / parking / finance / email / ops / etc.)
* optional `tags` (nba, nfl, weather, nyc, etc.)

We can do this without UI work by:

* defaulting category to `"general"`
* optionally auto-classifying category at skill creation time (LLM does it once, offline-ish)

---

## Step 5 — Code drift control (you asked for thoughts)

You’re right: drift is almost guaranteed when:

* multiple repos,
* multiple interfaces (voice/chat),
* and context resets.

The fix is not “more memory”, it’s **higher-quality invariants** in-repo.

Your approach (“document the purpose of each file/line touched”) is directionally correct, but doing it **line-by-line** will become unmaintainable.

The sustainable version is:

* **Per-file purpose headers** (hot path? invariants? public interfaces?)
* **Per-change log** (what changed + why + how to rollback)
* **A drift control protocol doc** that future chats must follow

You already have this doc now:
[CODE_DRIFT_CONTROL.md](sandbox:/mnt/data/CODE_DRIFT_CONTROL.md)

From here on, whenever we touch a file, we follow that contract (purpose header + invariants + rollback note). That’s how we prevent future “it worked yesterday but not today” regressions.

---

## What I need from you right now (no code yet)

Before we implement category-first routing, do **one last sanity pass**:

* Confirm `/admin/settings` returns **200** (not 404)
* Confirm `/admin/dynamic-skills/sync` returns **ok**
* Confirm `/assistant/route` routes “sports digest” to the websearch skill

If all 3 pass, the next step is:

* **Add category metadata to dynamic skills**
* **Extend the LLM JSON schema to support `category_request` + `schedule_request`**
* Keep it behind the same `INTENT_V2_MODE` cutover so rollback is a 1-line env change.

---

# AUTO-SUMMARY PACK

1. **Current Goal**
   Re-establish stable backend behavior after repo/branch drift, then proceed with the new LLM→JSON→FSM intent routing system (category-based, scalable, no regex dependence), built alongside the working system with env-var rollback.

2. **Refactor Step Completed**
   No refactor performed in this message (deployment verification + next-step plan only).

3. **What Changed (Code/Config/Infra)**

* User updated Render to follow the correct branch (fixing prior drift).
* Next actions focus on verification and then extending intent routing.

4. **Known Issues**

* “Wrong repo/branch” drift can silently reintroduce missing endpoints and break routing.
* Without category metadata, category-first routing can’t disambiguate multiple skills cleanly.

5. **Evidence (≤5 log lines)**

* Prior evidence already showed dynamic skill routing worked once `/admin/dynamic-skills/sync` was available and used.

6. **Tests Performed / To Perform**
   To perform now (curl smoke tests):

* `GET /admin/settings` returns 200
* `POST /admin/dynamic-skills/sync` returns ok
* `POST /assistant/route` routes “sports digest” to the dynamic skill

7. **Next Actions (ordered)**

1) Run the 3 smoke tests above to confirm correct branch is deployed.
2) Freeze this deployment as a rollback point (stable branch/tag + record SHA).
3) Implement category metadata for dynamic skills (default + optional auto-classify).
4) Extend LLM plan schema to include `category_request` + `schedule_request`.
5) Keep cutover controlled via `INTENT_V2_MODE` (shadow → assist).

8. **Safe Defaults + Rollback Point**

* Safe default: keep `INTENT_V2_MODE=shadow` until category routing is validated.
* Rollback: set `INTENT_V2_MODE=off` to restore legacy routing immediately; deploy stable branch/tag SHA.

9. **Open Questions**

* Do you want category to be user-editable later (portal), or purely auto-derived for now?
* Should category apply to legacy skills too (recommended), or only dynamic skills first?

10. **Goal/Wizard Status (goals/wizard state/playbooks/monitors/notifications/anomalies)**

* Wizard/UI deprioritized (troubleshooting only).
* Focus is now unified backend routing engine usable by both voice and portal chat.
* Category-first intent routing is the next planned capability.

**Flow A (must preserve)**
Twilio → FastAPI WS (`/twilio/stream`) → OpenAI Realtime (audio in/out) → Twilio
Rule: keep heavy planning out of the audio hot path; do intent planning in `/assistant/route` / non-stream paths.





Next step is **Phase 2: ingestion + retrieval**, now that Phase 1 (upload/list/download/delete) is proven and the Admin UI is usable (tenant dropdown + collapsible Memory Bank). Your Control Plane logs already show the happy-path traffic working end‑to‑end for upload + download + delete. 

Here’s the **most stable, incremental path** forward.

---

## Step 0 — Freeze a “Known Good” checkpoint

Before we touch ingestion or runtime retrieval, tag the current state:

* **WebUI:** KB Files panel works, tenant auto-fills from Email Accounts, Memory Bank collapsible
* **Control Plane:** KB storage endpoints healthy
* **No changes** to voice flows

**Action**

* Create a git tag or release note like: `kb_phase1_file_mgmt_ui_ok`.

**Why**

* Ingestion is where bugs + performance issues tend to appear. We want clean rollback.

---

## Step 1 — Add Phase 2 DB schema (no runtime behavior change)

Add tables you already proposed:

* `kb_ingest_jobs`
* `kb_chunks` (with embeddings)
* Ensure **tenant_id is indexed everywhere**

### If using Postgres for vectors

Use **pgvector**:

* enable extension `vector`
* `kb_chunks.embedding vector(<dim>)` (dim depends on model; ex: OpenAI `text-embedding-3-small`)

**Test**

* migration applies cleanly
* basic insert/select works

**Rollback**

* revert migration

---

## Step 2 — Implement ingestion as an async job (Control Plane owns it)

Hard rule: **ingestion must not happen in request/response**.

### Minimal ingestion architecture (Render-friendly)

* Control Plane endpoint creates a row in `kb_ingest_jobs` with `status='queued'`
* A **separate worker process** (recommended) polls for queued jobs and processes them

  * This prevents long uploads/processing from impacting the web dyno responsiveness

**New admin endpoints (Control Plane)**

* `POST /admin/kb/files/{id}/ingest` (enqueue)
* `GET  /admin/kb/files/{id}` already returns status; extend it to include:

  * `ingest_status`, `ingest_error`, `ingested_at`
* Optional:

  * `GET /admin/kb/ingest-jobs?tenant_id=...`

**Test**

* Upload file → enqueue ingest → job transitions: `queued → running → ready`
* Verify job failure stores `error` and status `failed`

**Rollback**

* disable worker (scale to 0) + hide ingest button in UI (feature flag)

---

## Step 3 — Text extraction + chunking (start narrow)

Don’t boil the ocean. Start with these file types first:

* `.txt`, `.md` (easy, reliable)
* then add `.pdf` (most common, trickier)
* then `.docx`, `.html`

**Chunking default**

* chunk by ~800–1200 characters
* overlap 100–200 chars
* store `chunk_index`, `text`, `file_id`, `tenant_id`

**Why this first**

* You can validate ingestion + retrieval without fighting PDF edge cases.

---

## Step 4 — Generate embeddings + store chunks

In the ingestion worker:

* For each chunk:

  * compute embedding
  * insert into `kb_chunks`

**Feature flag**

* `KB_EMBEDDINGS_ENABLED=1` (so we can turn off if cost spikes)

**Test**

* After ingestion: `SELECT count(*) FROM kb_chunks WHERE tenant_id=... AND file_id=...` is > 0

---

## Step 5 — Runtime retrieval in Flow B (behind a flag)

Now connect KB retrieval into the **router path** (`/assistant/route`), not any 20ms audio loop.

### Behavior

When handling a user turn:

1. Retrieve long-term memory (existing)
2. Retrieve KB chunks (new)
3. Combine into one “context bundle”
4. Send to LLM (Responses) as:

   * **policy** blocks (high priority instructions)
   * **knowledge** chunks (citations / context)

### Feature flag

* `KB_RETRIEVAL_ENABLED=0|1` default `0`

**Test**

* With flag OFF: no KB queries happen (confirm via logs)
* With flag ON:

  * query “what are your hours?” (and a KB doc contains hours)
  * response reflects KB content

**Rollback**

* set `KB_RETRIEVAL_ENABLED=0`

---

## Step 6 — Policy docs vs Knowledge docs (treat differently)

You already separated `kind`:

* `kind=policy` → should become “rules” injected as high‑priority instructions
* `kind=knowledge` → retrieval context

**Phase 2.0 simplest rule**

* If policy docs exist: include the **first N policy chunks** (or a pre-generated policy summary) every call
* Put a hard cap (ex: 2–4k chars) to avoid prompt bloat

Later (Phase 2.1):

* summarize policy docs into a single “Policy Capsule” during ingestion

---

## Step 7 — Extend “recall audit” to include KB sources

You already have “memory recall audit” concept. Next is to add:

* `kb_recall_audit` or extend existing audit rows to include:

  * `kb_file_id`
  * `kb_chunk_ids`
  * `kb_topk_scores`
  * model + timestamp

This makes debugging “why did it answer that?” much easier.

---

## Step 8 — Update Admin UI: “Ingest” + status column

In the **KB Files panel**:

* add “Ingest” button (enqueue)
* show status: `uploaded | ingesting | ready | failed`
* show error tooltip on failed

This completes Phase 2 operationally.

---

# What I recommend we do next (1 small step)

**Do Step 1 only next:** add the Phase 2 DB tables + migrations (no behavior changes).
Then we’ll do Step 2 (enqueue endpoint + worker) as the next iteration.

That’s the most stable progression and keeps rollbacks clean.

---

# AUTO-SUMMARY PACK

1. **Current Goal**
   Move from Phase 1 KB file management (upload/list/download/delete) to Phase 2 ingestion + retrieval, safely and incrementally.

2. **Refactor Step Completed (what was moved/extracted)**
   Already completed in prior steps: Memory Bank made collapsible; KB Files upload panel functional with tenant auto-fill from Email Accounts.

3. **What Changed (Code/Config/Infra)**
   Current system confirms Control Plane endpoints handling KB operations successfully (upload/download/delete). 
   No new changes applied in this message; this message defines next steps.

4. **Current System State (Flow A + Flow B)**

* **Flow A — Realtime (Legacy/Experimental):** Twilio → FastAPI WS (/twilio/stream) → OpenAI Realtime → Twilio (stable, untouched)
* **Flow B — Deepgram + ElevenLabs (Current MVP):** Twilio → Deepgram realtime STT → `/assistant/route` (FSM + OpenAI Responses text) → ElevenLabs TTS → Twilio (stable, keep hot paths clean)

5. **Known Issues**

* Phase 2 ingestion must be async to avoid web dyno blocking and to protect system stability.

6. **Evidence (≤5 log lines)**
   Control Plane logs show:

* `POST /kb/upload … 200 OK`
* `GET /admin/kb/files … 200 OK`
* `GET /kb/download … 200 OK`
* `DELETE /admin/kb/files … 200 OK` 

7. **Tests Performed / To Perform**
   Performed: Phase 1 upload/list/download/delete verification.
   To perform next (Phase 2):

* DB migration apply test
* enqueue ingest job test
* worker processes job → ready
* retrieval flag ON/OFF behavior in `/assistant/route`

8. **Next Actions (ordered)**

1) Freeze a known-good checkpoint/tag for Phase 1.
2) Add DB schema: `kb_ingest_jobs`, `kb_chunks` (+ pgvector if used).
3) Add ingest enqueue endpoint and a worker to process jobs async.
4) Implement extraction+chunking (start with txt/md).
5) Add embeddings + store in `kb_chunks`.
6) Integrate retrieval into Flow B router behind `KB_RETRIEVAL_ENABLED`.
7) Inject policy docs as instruction blocks; knowledge docs as retrieval context.
8) Extend recall audit to include KB chunk IDs and scores.
9) Update UI with ingest status + Ingest button.

9. **Safe Defaults + Rollback Point**

* Safe defaults: retrieval OFF by default (`KB_RETRIEVAL_ENABLED=0`), ingestion async, strict tenant_id filtering.
* Rollback: disable worker + revert migration + set retrieval flag OFF.

10. **Open Questions**

* Do you want ingestion worker inside Control Plane (separate Render service) or do you want Control Plane to call a private Backend ingestion endpoint?
* Vector storage preference: pgvector in existing Postgres vs dedicated vector DB.

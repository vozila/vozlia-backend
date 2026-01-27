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

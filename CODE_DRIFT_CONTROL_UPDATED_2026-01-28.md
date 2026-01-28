# CODE_DRIFT_CONTROL.md
**Version:** 2026-01-28  
**Purpose:** Reduce “code drift” across long Vozlia build sessions by forcing (1) small, reversible changes, (2) explicit feature flags + rollbacks, and (3) continuously updated, human-readable documentation of what each file does and why.

---

## 0) Non‑Negotiables (Platform Safety)
1. **LLM plans; Python executes deterministically.** Any action that changes state must be schema-validated JSON first.
2. **No heavy planning in audio hot paths.** Twilio streaming loops stay lean; use non-stream endpoints for wizard/planning.
3. **Tenant isolation on every read/write.** Never “guess” tenant/user context.
4. **Feature‑flag everything risky.** Cutovers must be reversible without code changes (env var toggles).

---

## 1) “Before You Touch Code” Checklist (Drift Prevention)
When starting a new chat session (or after a rollback), do these *first*:

### 1.1 Confirm the deployed code identity
- Record:
  - Git remote (repo URL/name)
  - Branch name
  - **Commit SHA**
- Confirm Render is pulling from the intended branch/SHA (avoid “wrong repo/branch drift”).
- If available, expose and log a `/version` or `version` field in critical responses.

### 1.2 Confirm runtime configuration
- Snapshot relevant env vars (especially feature flags):
  - `INTENT_V2_MODE` (e.g., `shadow|assist|full`)
  - `INTENT_V2_DYNAMIC_ACTIVATION_KEYWORDS` (comma-separated activation words)
  - `OBS_ENABLED`, `OBS_LOG_JSON`, etc.
- Confirm Admin keys align between services (`ADMIN_KEY` vs `ADMIN_API_KEY`).

### 1.3 Define the “touch set”
For every change request:
- List exactly which files you will modify (keep it small).
- State rollback: **(a)** env-var flip, **(b)** revert commit, **(c)** known-good branch/tag.

---

## 2) How We Modify Code (Safe Incremental Protocol)
### 2.1 One safe change per iteration
- One new endpoint OR one small service extraction OR one schema change.
- Avoid “multi-feature commits” unless explicitly required.

### 2.2 No behavior change unless requested
- If behavior must change, guard behind a feature flag.
- Default flag value should preserve existing behavior.

### 2.3 Keep files small and legible
- Never let a file exceed ~1600 lines. Extract modules instead.
- Prefer “single responsibility” modules and predictable folder layout.

---

## 3) Feature Flags & Rollback Discipline
### 3.1 Intent routing cutover control
- `INTENT_V2_MODE`:
  - `shadow`: compute plan, **do not** act; log only
  - `assist`: compute plan; act only when confident / explicit (safe default)
  - `full`: plan drives routing for most intents
- `INTENT_V2_DYNAMIC_ACTIVATION_KEYWORDS`:
  - Comma-separated keywords that must appear to trigger **dynamic skill execution**.
  - Example: `skill,report,run,execute`

### 3.2 Observability safety
- Prefer env vars to protect audio hot paths:
  - `OBS_ENABLED=0` is the first “audio quality emergency” move.

### 3.3 Rollback playbook (required in every patch note)
- “To rollback”: set `<FLAG>=<SAFE_VALUE>` and redeploy.
- If env rollback fails: revert to last known-good commit SHA/branch.

---

## 4) Documentation Rules (What to Write, Where to Write It)
### 4.1 File-level “Purpose Header” (required)
Whenever you touch a `.py`/`.ts` file, add/maintain a short header comment near the top:

Example:
```py
# PURPOSE: Handles /assistant/route intent routing and safe skill execution.
# OWNER: voice + portal chat parity.
# RISK: Runs on non-stream endpoint; must not add heavy work to audio WS hot paths.
# FLAGS: INTENT_V2_MODE, INTENT_V2_DYNAMIC_ACTIVATION_KEYWORDS
```

### 4.2 Maintain a living “drift ledger”
Create/maintain (choose one):
- `docs/DRIFT_LEDGER.md` (preferred)
- or `DRIFT_LEDGER.md` at repo root

Every time code is changed, append:
- Date
- Commit SHA (or “uncommitted patch”)
- Files changed
- Why changed
- Flags involved
- Smoke tests run
- Rollback instructions

### 4.3 Maintain a “File Index”
Create/maintain:
- `docs/FILE_INDEX.md`

Include each major file/folder:
- What it does
- What calls it
- What it calls
- Risk notes (hot path vs safe path)

---

## 5) Required Smoke Tests (Copy/Paste Commands)
Run after every deploy.

### 5.1 Admin settings present
```bash
curl -s "https://vozlia-backend.onrender.com/admin/settings" -H "x-vozlia-admin-key: $ADMIN_KEY" | head -c 800
```

### 5.2 Dynamic skills sync (if enabled)
```bash
curl -s -X POST "https://vozlia-backend.onrender.com/admin/dynamic-skills/sync" -H "x-vozlia-admin-key: $ADMIN_KEY"
```

### 5.3 Intent routing to a known dynamic skill
```bash
curl -s -X POST "https://vozlia-backend.onrender.com/assistant/route" \
  -H "Content-Type: application/json" \
  -d '{"text":"please give me my sports digest","context":{"call_id":"smoke-intent-1","from_number":"+15550001111","channel":"phone"}}'
```

### 5.4 Category disambiguation check
```bash
curl -s -X POST "https://vozlia-backend.onrender.com/assistant/route" \
  -H "Content-Type: application/json" \
  -d '{"text":"sports report","context":{"call_id":"smoke-cat-1","from_number":"+15550001111","channel":"phone"}}'
```
(Expect either direct run, or “Which one did you mean?” if multiple skills match category.)

---

## 6) Avoiding Prompt/Context Drift in ChatGPT Sessions
### 6.1 When to start a new chat session
Start a new chat session when any of these are true:
- > ~10–15 files were touched in the current session
- you have produced multiple patch zips and the stack of changes is hard to reason about
- you performed a rollback / branch switch and need a clean baseline
- you feel you’re repeating past misunderstandings (“code drift symptoms”)

### 6.2 What to carry forward (mandatory)
At the end of each meaningful work chunk, create a “checkpoint pack” including:
- current goal
- exact branch/SHA
- flags
- changed files
- tests run
- rollback plan
- known issues / next actions

---

## 7) Repo Pinning & Deployment Hygiene (Render)
**Problem we saw:** Render defaulted to the wrong repo/branch after creating branches/rollbacks.

**Rule:** Always pin deployments to the correct repo + branch + commit SHA.
- Record the “known-good SHA” in the checkpoint pack.
- Prefer “stable branch” tags over “floating main” during risky work.

---

## 8) Intent Routing Principles (Target State)
- **Everything hits LLM first for intent planning**, but execution is constrained:
  - tool-only
  - schema-validated
  - auditable
- Dynamic skills are **opt-in** via activation keywords (for now) to reduce accidental matches.
- Category-first routing must support:
  - “sports report” → show sports skill list → user selects → execute.
  - Exact/near-name references still work (“Todays Sports Digest”, “sports digest”).

---

## 9) Files Slated for Removal (Deprecation Protocol)
If a subsystem is being replaced (e.g., earlier DB/metrics experiments):
- Add `# DEPRECATED (slated for removal)` at top of file.
- Keep it behind a flag: `FEATURE_X_ENABLED=0` default.
- Do not delete until replacement is stable + deployed + smoke-tested.

---

## 10) What “Done” Looks Like
A change is “done” only if:
- it is behind an env flag (or explicitly non-risky)
- it has a rollback path
- it has at least 3 smoke tests recorded
- the drift ledger entry exists
- the change does not impact audio WS hot paths unless explicitly approved

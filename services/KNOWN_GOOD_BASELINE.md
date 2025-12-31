Last good known update Dec 30 3:00PM

Completed portal for now

working on long-term  memory


You’re right to call that out: **Deepgram → ElevenLabs is gone**, but the **pattern you want (MemoryController → DB retrieval → evidence → natural answer)** still fits your *current* stack because you still have:

* **Realtime for speaking**
* **`/assistant/route` for skills/FSM-style backend calls**
* **SpeechOutputController** as the “speech owner”

So we’ll pursue the exact architecture you pasted, just mapped onto the **Realtime + `/assistant/route`** pipeline (no Deepgram/ElevenLabs assumptions).

---

## The memory system to build (matches your desired behavior)

### 1) Store everything as **MemoryEvents**

Create a `memory_events` table and write one row for:

* every skill output (weather/gmail/etc.)
* key chitchat turns you decide to keep (especially user “comments about the report”)

**Fields (MVP)**

* `tenant_id`
* `caller_id` (E164 or internal)
* `call_sid`
* `created_at`
* `skill_key` (`weather`, `gmail_summary`, `chitchat_turn`, etc.)
* `text` (short summary/snippet)
* `data_json` (structured facts; optional)
* `tags` (optional array or comma string)

This enables queries like: “last week + weather + what did I say”.

---

### 2) MemoryController (lightweight “sentence → DB query”)

When an utterance comes in, MemoryController:

1. **Time parsing** (“last week”, “yesterday”, “previous call”) → `[start_ts, end_ts]`
2. **Topic/skill inference** (“weather”, “forecast”) → `skill_key=weather`
3. **Keyword extraction** (entities like Boston/rain) → `keywords`
4. **DB retrieval** (cheap first):

   * Postgres full-text search on `text`
   * filter by `tenant_id`, `caller_id`, time window
   * optional skill_key filter
   * fallback `ILIKE` if needed

Returns `top_k` evidence rows + structured `data_json`.

---

### 3) Realtime stays the speaker; memory is “retriever”

We keep your “don’t bloat context / don’t do heavy work in hot path” rule by doing this:

**Realtime user turn arrives**

* If it’s a normal turn → respond normally
* If it’s a memory question → route a backend call:

  * `/assistant/route` with `backend_call="memory_recall"` (or similar)

**`/assistant/route`**

* runs MemoryController retrieval
* returns:

  * `spoken_reply`
  * `evidence` (short snippets, optional)

**Realtime speaks** the `spoken_reply` (via SpeechOutputController), with an instruction to **only use evidence**.

---

## Minimal, safe implementation steps (incremental)

### Step 1 — Add `MemoryEvent` model + write helper (no behavior change)

* Add table + indexes
* Add helper `memory_write_event(...)`
* Feature flag: `VOZLIA_MEMORY_EVENTS=1`
* Start by logging **skill outputs only** (weather/gmail/etc.)

✅ Low risk, no changes to speech routing.

### Step 2 — Add `memory_recall` backend_call in `/assistant/route`

* Feature flag: `VOZLIA_MEMORY_RECALL=1`
* Implement retrieval + evidence-first answer composition

✅ Still safe; only triggers when called.

### Step 3 — Add memory-intent detection in the Realtime bridge

Light regex/keyword match (cheap):

* “what did I say…”
* “last time we talked”
* “previous call”
* “remind me what we discussed”
* “about that report…”

If matched → call `/assistant/route backend_call=memory_recall`

✅ No heavy work in audio path, just routing.

### Step 4 — Start capturing “comments about the report”

When a skill runs, store `last_skill_key` in session state.
If user’s next turn looks like a comment, store as:

* `skill_key="chitchat_turn"`
* `tags=["weather"]` (or whatever last skill was)
* `data_json={"ref_skill":"weather"}`

---

## Evidence-first answering rule (prevents hallucinations)

When returning to Realtime, include:

* the retrieved snippets (short)
* a hard instruction:

  * “Answer only using evidence; if ambiguous, ask one clarifying question.”

That’s what makes it “reasonably accurate” and still natural.

---

## AUTO-SUMMARY PACK

1. **Current Goal**
   Implement long-term memory as “MemoryEvents + MemoryController retrieval + evidence-first responses” in the current Realtime + `/assistant/route` architecture (no Deepgram/ElevenLabs assumptions).

2. **Refactor Step Completed (what was moved/extracted)**
   None (architecture alignment + implementation plan).

3. **What Changed (Code/Config/Infra)**
   None in this message. Proposed feature flags: `VOZLIA_MEMORY_EVENTS`, `VOZLIA_MEMORY_RECALL`.

4. **Current System State (Flow A + Flow B)**

* **Flow A — Realtime (current):** Twilio → FastAPI WS → OpenAI Realtime + SpeechOutputController; `/assistant/route` used for skills/backend calls (memory will plug in here).
* **Flow B — Deepgram/ElevenLabs (deprecated/removed):** No longer used.

5. **Known Issues**
   None new. Key constraint: memory retrieval must not run in audio hot paths; keep it at turn boundaries/backend_call.

6. **Evidence (≤5 log lines)**
   N/A.

7. **Tests Performed / To Perform**
   To perform after implementation:

* Run weather skill + user comment; later ask “what did I say about it last week?” and verify evidence-backed recall.
* Ask ambiguous memory question; verify assistant asks 1 clarifier.

8. **Next Actions (ordered)**

9. Add `memory_events` table + write helper (flagged).

10. Add `memory_recall` backend_call in `/assistant/route` (flagged).

11. Add Realtime memory-intent detection routing (flagged).

12. Expand writes to capture user “comments about” recent skills.

13. **Safe Defaults + Rollback Point**
    Defaults OFF via env flags. Rollback: disable flags or revert memory commit; normal chitchat continues.

14. **Open Questions**
    Do you want memory recall to return **verbatim snippets** (most accurate) or **summarized paraphrases** (more natural)? We can support both with a “speech policy” switch.

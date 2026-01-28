Vozlia Backend Patch — Intent V2 Category Routing Fix (v2)
Date: 2026-01-28

Problem
-------
With INTENT_V2_MODE=assist, utterances like "sports update" were falling through to generic chitchat
even though multiple dynamic skills exist under the 'sports' category.

Root cause
----------
Intent Router V2 scored/LLM-selected only from skill candidates. For category-like utterances that do
NOT name a specific skill (and don't match triggers), the LLM plan could legitimately return
route='chitchat', causing the router to return None and fall back to legacy/KB.

Fix
---
1) Include `category` on each candidate in the LLM candidates payload.
2) Add deterministic category fallback in assist mode:
   - If the LLM planner fails (plan None) OR explicitly routes to chitchat,
     and the utterance contains a known category token, the router returns a disambiguation prompt
     listing the skills in that category.
   - The available options are stored in session_store under `intent_v2_choices`
     so the next user turn can pick by number.

Files changed
-------------
- services/intent_router_v2.py

How to apply
------------
- Replace services/intent_router_v2.py with the patched version from this zip.
- Redeploy backend.

Smoke tests
-----------
1) Category disambiguation:
   curl -s -X POST "https://vozlia-backend.onrender.com/assistant/route" \
     -H "Content-Type: application/json" \
     -d '{"text":"sports update","context":{"call_id":"smoke-cat-2","from_number":"+15550001111","channel":"phone"}}' | jq

   Expected: spoken_reply asks "Which sports skill did you mean?" and lists sports skills.

2) Select option:
   curl -s -X POST "https://vozlia-backend.onrender.com/assistant/route" \
     -H "Content-Type: application/json" \
     -d '{"text":"1","context":{"call_id":"smoke-cat-2","from_number":"+15550001111","channel":"phone"}}' | jq

   Expected: runs the chosen dynamic sports skill.

Rollback
--------
Set INTENT_V2_MODE=off and redeploy to restore legacy routing.

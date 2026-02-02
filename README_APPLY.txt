README — Vozlia Patch: DBQuery has_concept filter (2026-02-01)

Goal
- Improve DB searches by allowing deterministic concept-based filtering in DBQuerySpec:
  DBFilter.op = "has_concept" => safe EXISTS() against concept_assignments

What this enables
- DBQuery can filter entities (e.g., kb_documents, caller_memory_events) to only rows that have a specific Concept Code.
- This is a prerequisite for concept-driven analytics. (Multi-table metrics come later via dataset registry / join templates.)

Files (whole-file replacement)
- services/db_query_service.py
- CODE_DRIFT_CONTROL.md

Env flags
- CONCEPTS_ENABLED=1 (required at runtime for has_concept filters)

How to apply
1) Copy these files into the backend repo (preserve paths)
2) Deploy backend (worker does not need changes for this patch)

Smoke test (admin API)
A) Pick an existing KB document id:
  psql "$DATABASE_URL" -c "select id, title from kb_documents order by created_at desc limit 5;"

B) Create a concept definition (if not already present)
  curl -sS -X POST "$BASE_URL/admin/concepts/definitions" \
    -H "x-vozlia-admin-key: $ADMIN_API_KEY" \
    -H "content-type: application/json" \
    -d '{"concept_code":"menu.steak","name":"Steak","description":"Steak dinners","active":true}'

C) Assign that concept to the real kb_document id
  curl -sS -X POST "$BASE_URL/admin/concepts/assignments" \
    -H "x-vozlia-admin-key: $ADMIN_API_KEY" \
    -H "content-type: application/json" \
    -d '{"target_type":"kb_document","target_id":"<KB_DOC_ID>","concept_code":"menu.steak","source":"manual"}'

D) Run DBQuery with has_concept filter
  curl -sS -X POST "$BASE_URL/admin/dbquery/run" \
    -H "x-vozlia-admin-key: $ADMIN_API_KEY" \
    -H "content-type: application/json" \
    -d '{
      "spec": {
        "entity": "kb_documents",
        "filters": [
          { "field": "id", "op": "has_concept", "value": { "concept_code": "menu.steak" } }
        ],
        "limit": 25
      }
    }'

Expected
- ok=true
- count >= 1
- returned rows include your kb_document id

Rollback
- Set CONCEPTS_ENABLED=0 (disables concepts filtering and concept endpoints).

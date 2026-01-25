BACKEND PATCH: Metrics engine + voice guardrail (no numeric hallucinations)

WHAT THIS DOES
1) Adds /admin/metrics/run (admin-key protected) which answers common metric questions deterministically.
2) Adds services/metrics_service.py (shared engine).
3) Updates services/assistant_service.py so voice answers metric questions deterministically and refuses to guess numbers.
4) Updates main.py to include the metrics router.

APPLY
- Copy these files into the backend repo root, preserving paths:
  - services/metrics_service.py
  - api/routers/metrics.py
  - services/assistant_service.py  (full replacement)
  - main.py                       (full replacement)
- Redeploy backend.

SMOKE TESTS
1) curl:
   POST /admin/metrics/run {question:"how many calls did we receive this week?", timezone:"America/New_York"}
2) Voice:
   Ask "How many times was email summaries requested yesterday?"
   -> Must NEVER output a number unless it comes from metrics. If not computable, it should say it can't compute yet.

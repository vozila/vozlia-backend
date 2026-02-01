-- 2026-01-31: Support multi-skill scheduling by adding ScheduledDelivery.skill_key
-- Back-compat: existing websearch schedules keep web_search_skill_id populated.
-- Safe to re-run (idempotent where possible).

BEGIN;

ALTER TABLE scheduled_deliveries
    ADD COLUMN IF NOT EXISTS skill_key TEXT;

-- Allow non-websearch schedules (dbquery_*) by making FK nullable.
ALTER TABLE scheduled_deliveries
    ALTER COLUMN web_search_skill_id DROP NOT NULL;

-- Backfill skill_key for existing websearch schedules.
UPDATE scheduled_deliveries
SET skill_key = CONCAT('websearch_', web_search_skill_id::text)
WHERE skill_key IS NULL
  AND web_search_skill_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS ix_scheduled_deliveries_skill_key
    ON scheduled_deliveries (skill_key);

-- Optional: enforce one schedule per (tenant, skill_key)
-- NOTE: run only after confirming no duplicates.
-- CREATE UNIQUE INDEX CONCURRENTLY IF NOT EXISTS ux_scheduled_deliveries_tenant_skill_key
--     ON scheduled_deliveries (tenant_id, skill_key)
--     WHERE skill_key IS NOT NULL;

COMMIT;

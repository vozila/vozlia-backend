-- 2026-02-03: Fix ConceptDefinition schema mismatch
-- Adds concept_definitions.synonyms (jsonb) expected by backend concepts router.
-- Safe to re-run (idempotent).

BEGIN;

ALTER TABLE concept_definitions
    ADD COLUMN IF NOT EXISTS synonyms JSONB NOT NULL DEFAULT '[]'::jsonb;

-- Optional backfill if an earlier column name was used.
DO $$
BEGIN
  IF EXISTS (
    SELECT 1
    FROM information_schema.columns
    WHERE table_schema = 'public'
      AND table_name = 'concept_definitions'
      AND column_name = 'synonyms_json'
  ) THEN
    EXECUTE $q$
      UPDATE concept_definitions
      SET synonyms = synonyms_json
      WHERE (synonyms IS NULL OR synonyms = '[]'::jsonb)
        AND synonyms_json IS NOT NULL
    $q$;
  END IF;
END $$;

COMMIT;

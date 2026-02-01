-- 2026-02-01: Concept Codes (tenant-scoped semantic tagging for deterministic analytics)
-- Adds:
--   - concept_definitions
--   - concept_batches
--   - concept_assignments
--
-- Safe to re-run (idempotent where possible).
--
BEGIN;

CREATE TABLE IF NOT EXISTS concept_definitions (
    id UUID PRIMARY KEY,
    tenant_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    concept_code TEXT NOT NULL,
    name TEXT NOT NULL,
    description TEXT NULL,
    parent_code TEXT NULL,
    synonyms_json JSONB NOT NULL DEFAULT '[]'::jsonb,
    active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_concept_definitions_tenant_code UNIQUE (tenant_id, concept_code)
);

CREATE INDEX IF NOT EXISTS ix_concept_definitions_tenant_code
    ON concept_definitions (tenant_id, concept_code);

CREATE TABLE IF NOT EXISTS concept_batches (
    id UUID PRIMARY KEY,
    tenant_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    model_version TEXT NULL,
    summary_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    notified_at TIMESTAMP NULL,
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS ix_concept_batches_tenant_created
    ON concept_batches (tenant_id, created_at);

CREATE TABLE IF NOT EXISTS concept_assignments (
    id UUID PRIMARY KEY,
    tenant_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    target_type TEXT NOT NULL,
    target_id TEXT NOT NULL,
    concept_code TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'llm_auto',  -- llm_auto|manual|import
    confidence DOUBLE PRECISION NULL,
    rationale TEXT NULL,
    evidence_json JSONB NULL,
    locked BOOLEAN NOT NULL DEFAULT FALSE,
    batch_id UUID NULL REFERENCES concept_batches(id) ON DELETE SET NULL,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_concept_assignments_tenant_target_concept UNIQUE (tenant_id, target_type, target_id, concept_code)
);

CREATE INDEX IF NOT EXISTS ix_concept_assignments_tenant_code
    ON concept_assignments (tenant_id, concept_code);

CREATE INDEX IF NOT EXISTS ix_concept_assignments_tenant_target
    ON concept_assignments (tenant_id, target_type, target_id);

CREATE INDEX IF NOT EXISTS ix_concept_assignments_tenant_batch
    ON concept_assignments (tenant_id, batch_id);

COMMIT;


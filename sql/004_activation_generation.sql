-- Phase F, Step 3: embedding-gated activation and stale-activation protection.
--
-- Two additive concerns are handled here:
--
-- 1. At most one current version per document. The application already keeps a
--    single pointer in documents.current_version_id, but nothing at the database
--    level prevented two document_versions rows from being marked current. The
--    partial unique index makes the invariant enforceable. It is created only
--    after an explicit pre-flight check so an existing data violation aborts the
--    migration instead of being silently repaired (this migration never deletes
--    or rewrites historical data).
--
-- 2. A database-backed activation generation. Version numbers alone cannot
--    distinguish a deliberately resubmitted historical snapshot from an older
--    ingestion that simply finished embedding late. Each execution claims a
--    monotonic token from a sequence while creating/reusing its target version.
--    Activation is allowed only when that token is greater than the document's
--    last successful activation token. A late-finishing older execution is
--    therefore superseded instead of replacing a newer current version, while a
--    later submission of historical content receives a fresh token and may
--    reactivate its already-embedded version.
--
-- The columns are nullable/defaulted so existing rows remain valid:
--   * ingestion_jobs.activation_generation is NULL for jobs that never started.
--   * documents.last_activation_generation defaults to 0 for existing documents.
--
-- Apply explicitly against an existing database, for example:
--   psql "$DATABASE_URL" -f sql/004_activation_generation.sql

BEGIN;

DO $$
DECLARE
    conflicting_documents INTEGER;
BEGIN
    SELECT count(*) INTO conflicting_documents
    FROM (
        SELECT document_id
        FROM document_versions
        WHERE is_current
        GROUP BY document_id
        HAVING count(*) > 1
    ) AS conflicts;

    IF conflicting_documents > 0 THEN
        RAISE EXCEPTION
            'Migration aborted: % document(s) have more than one current version',
            conflicting_documents;
    END IF;
END $$;

CREATE UNIQUE INDEX IF NOT EXISTS document_versions_one_current_idx
    ON document_versions (document_id)
    WHERE is_current;

CREATE SEQUENCE IF NOT EXISTS ingestion_activation_generation_seq;

ALTER TABLE documents
    ADD COLUMN IF NOT EXISTS last_activation_generation BIGINT NOT NULL DEFAULT 0;

ALTER TABLE ingestion_jobs
    ADD COLUMN IF NOT EXISTS activation_generation BIGINT;

COMMIT;

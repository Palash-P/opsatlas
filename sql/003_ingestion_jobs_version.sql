-- Phase F, Step 2: associate an ingestion job with the immutable document version
-- it created or reused.
--
-- Why this column: `document_id` alone cannot identify which version a job
-- produced once a document has more than one version. Phase F, Step 3 needs a
-- stable job -> version link to gate activation on embedding completion, and
-- operators need it to audit failed or retried ingestions.
--
-- The column is nullable and additive. Existing jobs remain valid; jobs that
-- fail before a version is written keep `version_id = NULL`. `ON DELETE SET
-- NULL` preserves the job audit row if a version is ever removed.
--
-- Apply explicitly against an existing database, for example:
--   psql "$DATABASE_URL" -f sql/003_ingestion_jobs_version.sql

BEGIN;

ALTER TABLE ingestion_jobs
    ADD COLUMN IF NOT EXISTS version_id UUID
        REFERENCES document_versions(id) ON DELETE SET NULL;

COMMIT;

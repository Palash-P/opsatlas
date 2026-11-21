-- Phase F, Step 6: durable dispatch state, execution leases, and fenced writes.
--
-- The ingestion_jobs row remains the single source of truth. These nullable,
-- additive columns let a reconciler discover jobs whose task was never
-- published or whose worker disappeared, without introducing an outbox table:
--
--   * next_dispatch_at / dispatch_claimed_until bound reconciliation retries and
--     let concurrent reconcilers claim disjoint batches with SKIP LOCKED.
--   * dispatch_attempts / last_dispatch_at / dispatch_error_code keep dispatch
--     history observable in PostgreSQL. They are diagnostics only and never
--     override status or error_code.
--   * execution_token is an opaque fencing token minted by the worker claim.
--     Every fenced state transition must match both the token and a live lease.
--   * execution_lease_expires_at distinguishes an actively executing job from
--     one abandoned by a crashed worker. The worker renews it between embedding
--     batches; an expired lease allows reconciliation to dispatch a new
--     execution, and the old token can no longer activate or fail the job.
--
-- Celery task IDs and Redis locks are deliberately not used for correctness.
-- Existing rows remain valid: all columns are nullable except the defaulted
-- dispatch counter, and no historical data is rewritten or deleted.
--
-- Apply explicitly against an existing database, for example:
--   psql "$DATABASE_URL" -f sql/005_job_reconciliation.sql

BEGIN;

ALTER TABLE ingestion_jobs
    ADD COLUMN IF NOT EXISTS next_dispatch_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS dispatch_claimed_until TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS last_dispatch_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS dispatch_attempts INTEGER NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS dispatch_error_code TEXT,
    ADD COLUMN IF NOT EXISTS execution_token UUID,
    ADD COLUMN IF NOT EXISTS execution_lease_expires_at TIMESTAMPTZ;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'ingestion_jobs_dispatch_attempts_nonnegative'
    ) THEN
        ALTER TABLE ingestion_jobs
            ADD CONSTRAINT ingestion_jobs_dispatch_attempts_nonnegative
            CHECK (dispatch_attempts >= 0);
    END IF;
END $$;

-- Reconciliation scans only non-terminal states. The partial index keeps the
-- periodic scan proportional to outstanding work rather than table size.
CREATE INDEX IF NOT EXISTS ingestion_jobs_reconcile_idx
    ON ingestion_jobs (status, next_dispatch_at)
    WHERE status IN ('queued', 'running', 'failed');

-- Finding abandoned executions is bounded by the lease column.
CREATE INDEX IF NOT EXISTS ingestion_jobs_execution_lease_idx
    ON ingestion_jobs (execution_lease_expires_at)
    WHERE status = 'running' AND execution_token IS NOT NULL;

COMMIT;

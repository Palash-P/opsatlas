# Background ingestion (Celery + Redis)

Phase F, Steps 5–6 add a reliable background execution path and durable orphan
recovery for ingestion. The synchronous `POST /api/v1/documents/ingest` contract
is unchanged.

## Design

```
client -> POST /api/v1/documents/ingest/async
          |
          +-- validate/normalize/chunk
          +-- create or resolve durable ingestion_jobs row
          +-- persist non-current document/version/chunks
          +-- commit PostgreSQL
          |
          +-- publish Celery task with job_id only
                    |
                    v
              worker: resume_ingestion_job(job_id)
                    |
                    +-- lock durable job
                    +-- embed missing chunks in batches (content read from PostgreSQL)
                    +-- verify completeness
                    +-- atomically activate with generation gating
                    +-- mark job succeeded / record sanitized failure
```

Document content and metadata never travel through Redis. The target version and
its chunks are committed before publication, so a broker outage leaves:

* an inspectable `failed` job with `error_code = 'queue_unavailable'`, and
* a non-current, unactivated version that preserves the last-known-good version.

`ingestion_jobs` is the authoritative state. Celery has **no result backend**:
task return values are ignored and operators inspect PostgreSQL, not Celery
results. Redis is replaceable broker infrastructure, not durable state.

## Authorization boundary

The asynchronous request is authorized exactly like the synchronous request
before it writes the job, target version, or chunks. Creating a document or
updating a document with no ACL rows requires the configured `document_writer`
role; a document with any ACL row requires a matching `manage` grant. The
principal is used only for that request and is not stored on the job row or
published to Redis. The task payload contains the job ID only. A worker that
claims the job is a trusted internal service completing an already-authorized
durable target; it does not accept or need user credentials, and a permission
change after submission does not cancel committed work.

## API

### Queue an ingestion

```bash
curl -X POST http://127.0.0.1:8000/api/v1/documents/ingest/async \
  -H "Authorization: Bearer $OPSATLAS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "source_key": "runbooks/payment-api.md",
    "title": "Payment API Runbook",
    "source_type": "runbook",
    "content": "..."
  }'
```

Returns `202 Accepted`:

```json
{
  "job_id": "…",
  "status": "running",
  "status_url": "/api/v1/documents/jobs/…",
  "deduplicated": false
}
```

Repeated submissions with the same derived idempotency key resolve the same
durable job. If it is already running, succeeded, or current, the endpoint
returns the same `job_id` without publishing a duplicate task. A job that failed
before publication can be resubmitted and reuses its non-current target.

### Inspect a job

```bash
curl http://127.0.0.1:8000/api/v1/documents/jobs/<job_id> \
  -H "Authorization: Bearer $OPSATLAS_TOKEN"
```

Returns status, attempts, document/version links, timestamps, and sanitized
`error_code`/`error_summary`. The lookup is tenant-scoped: another tenant's job
is indistinguishable from a missing one (`404`).

## Celery configuration

Configuration is environment-backed (`app/core/config.py`):

| Setting | Default | Purpose |
| --- | --- | --- |
| `CELERY_BROKER_URL` | `redis://localhost:6379/1` | Redis broker; separate DB from readiness |
| `CELERY_TASK_DEFAULT_QUEUE` | `ingestion` | Ingestion worker queue |
| `CELERY_RECONCILIATION_QUEUE` | `reconciliation` | Dedicated reconciliation queue; must differ from the ingestion queue |
| `CELERY_TASK_SERIALIZER` | `json` | Task payload serialization |
| `CELERY_ACCEPT_CONTENT` | `["json"]` | Reject pickle/other payloads |
| `CELERY_TASK_ACKS_LATE` | `true` | Ack after execution; safe redelivery |
| `CELERY_TASK_REJECT_ON_WORKER_LOST` | `true` | Requeue after worker loss |
| `CELERY_WORKER_PREFETCH_MULTIPLIER` | `1` | Do not hoard long tasks |
| `CELERY_WORKER_CONCURRENCY` | `2` | Worker process count |
| `CELERY_TASK_SOFT_TIME_LIMIT_SECONDS` | `1800` | Cooperative limit |
| `CELERY_TASK_TIME_LIMIT_SECONDS` | `1860` | Hard limit |
| `CELERY_VISIBILITY_TIMEOUT_SECONDS` | `2400` | Redis redelivery horizon |
| `INGESTION_TASK_MAX_RETRIES` | `3` | Transient embedding retries |
| `INGESTION_TASK_RETRY_BACKOFF_SECONDS` | `5` | Exponential backoff base |
| `INGESTION_TASK_RETRY_BACKOFF_MAX_SECONDS` | `120` | Backoff cap |
| `INGESTION_TASK_UNEXPECTED_MAX_RETRIES` | `1` | Unexpected-exception retry budget |

The hard limit is validated to be above the soft limit, and the visibility
timeout above the hard limit.

## Retry and failure policy

| Failure | Policy | Durable result |
| --- | --- | --- |
| `EmbeddingUnavailableError` | Retry with capped exponential backoff | `failed` between attempts; `dead_letter` after exhaustion |
| `EmbeddingValidationError` | Permanent, no retry | `failed`, `embedding_invalid_response` |
| `EmbeddingSchemaError` | Permanent, no retry | `failed`, `embedding_schema_mismatch` |
| Missing/invalid job or target | Permanent, no retry | `failed`, `ingestion_job_invalid` / no row for missing |
| Activation integrity failure | Permanent, no retry | `failed`, `activation_rejected` |
| Unexpected exception | Small bounded retry (`1`) | `failed`, `ingestion_failed`; `dead_letter` after exhaustion |
| Succeeded/current duplicate delivery | Pure read | No attempt increment, no embedding calls |

Every execution stops before activation unless all chunks have a valid
`vector(384)` embedding for the configured model. Activation remains atomic and
generation-gated, so an older late-finishing job cannot replace a newer current
version.

## Durable reconciliation and orphan recovery (Step 6)

`ingestion_jobs` alone is authoritative. Celery/Redis can lose a message, a
worker can die mid-execution, and a process can crash between the PostgreSQL
commit and the broker publish. Reconciliation repairs those cases without
creating a second job-state source and without ever moving document content
through Redis.

### Orphan states and recovery policy

| Orphan state | Durable evidence | Recovery |
| --- | --- | --- |
| Queued, never published | `status = queued`, target version present | Republish job ID immediately |
| Committed, publish never reached | `status = running`, `execution_token IS NULL`, target present | Republish job ID immediately (`committed_never_dispatched`) |
| Running, worker abandoned it | `execution_token` set, `execution_lease_expires_at <= now()` | Republish after the lease expires (`abandoned_execution`) |
| Transient failure, Celery retry lost | `status = failed`, transient `error_code`, `next_dispatch_at` elapsed | Republish after the worker-recorded grace (`lost_retry`) |
| Broker publication failed | `status = failed`, `error_code = queue_unavailable` | Republish with capped exponential backoff (`publication_failed`) |
| Publication outcome uncertain | Dispatch claim held; publish may or may not have happened | A duplicate task is harmless; the claim expires and the next run retries |
| Succeeded / permanent failed / dead letter | Terminal status or permanent `error_code` | Never redispatched |

A `queued` job with no target version is not resumable: the request never
committed a non-current version, so there is nothing safe to finish. It is
visible for inspection but is not dispatched.

### Dispatch claims, execution leases, and fencing

The additive migration `sql/005_job_reconciliation.sql` adds dispatch state and
execution ownership to `ingestion_jobs`. Apply all migrations through the single
runner before starting beat, the worker, or the API so every process shares the
same durable schema:

```bash
python -m app.db.migrations
```

The worker refuses to consume a task when migration history or the live schema
does not match; see `docs/DATABASE_MIGRATIONS.md`.

| Column | Purpose |
| --- | --- |
| `next_dispatch_at` | Earliest time reconciliation may consider the job |
| `dispatch_claimed_until` | Dispatch lease held by a publisher |
| `last_dispatch_at` / `dispatch_attempts` / `dispatch_error_code` | Dispatch diagnostics; never override `status` or `error_code` |
| `execution_token` | UUID fencing token minted by a worker claim |
| `execution_lease_expires_at` | Liveness of the current execution |

A reconciler claims a bounded batch with
`SELECT … FOR UPDATE SKIP LOCKED`, sets `dispatch_claimed_until`, and commits
**before** publishing. Concurrent reconcilers therefore claim disjoint jobs and
cannot republish the same job at the same time. Claims are advisory
scheduling leases, not state: a crash after the claim simply lets the lease
expire. If the republished task is lost as well, the expired claim makes the
failed job eligible again.

An ambiguous publish is handled in both directions. A worker that already
claimed the job keeps its lease and fencing token even if the API records a
broker error; the failure marker never downgrades a claimed execution. A
duplicate publication is harmless because the worker claim is the real
mutual-exclusion mechanism.

A worker mints a fresh `execution_token` and lease when it claims a job, renews
the lease between embedding batches, and includes the token in every terminal
write. Activation, success, and failure updates require the same token **and** a
live lease inside the same transaction as the row lock. When an abandoned
execution is reclaimed, a new token is minted, so the old worker's late
activation or failure update matches no row and is rejected. `attempts` is
increased, never reset, and a succeeded job is preserved permanently.

Celery late acknowledgement and PostgreSQL interact as follows: `task_acks_late`
means a message is acknowledged after the task body, so worker loss or a
visibility-timeout expiry redelivers it. The redelivery is safe because the
worker claim is the real mutual-exclusion mechanism. If a live execution already
holds the lease, the duplicate is skipped as `job_busy`; if the lease expired,
the redelivery or the reconciler can claim a new execution. Redis redelivery is
therefore an optimisation, not the recovery guarantee.

### Scheduling and manual operation

Celery beat publishes `opsatlas.ingestion.reconcile_jobs` every
`INGESTION_RECONCILE_INTERVAL_SECONDS` (default 60s) when
`INGESTION_RECONCILE_ENABLED=true`. Compose starts a `beat` service. The task
takes no arguments and returns only sanitized counters.

For a controlled manual batch:

```bash
python scripts/reconcile_ingestion_jobs.py --batch-size 50
python scripts/reconcile_ingestion_jobs.py --batch-size 50 --json
```

The command exits non-zero when a batch recorded dispatch failures. It never
prints document content, embeddings, secrets, or exception text. There is no
reconciliation HTTP endpoint.

A PostgreSQL advisory lock (`2026111401`) prevents two reconciliation runs from
overlapping; a run that cannot acquire it returns `locked = true` without
scanning. The per-job dispatch claim is still the correctness mechanism, so the
advisory lock is only a scheduling convenience.

### Reconciliation configuration

| Setting | Default | Purpose |
| --- | --- | --- |
| `INGESTION_EXECUTION_LEASE_SECONDS` | `300` | Worker execution lease; must be below the Celery visibility timeout |
| `INGESTION_RECONCILE_ENABLED` | `true` | Enable the beat schedule |
| `INGESTION_RECONCILE_INTERVAL_SECONDS` | `60` | Beat interval (minimum 5) |
| `INGESTION_RECONCILE_BATCH_SIZE` | `50` | Jobs claimed per run (1–1000) |
| `INGESTION_RECONCILE_DISPATCH_LEASE_SECONDS` | `300` | How long a claim suppresses duplicate publishing |
| `INGESTION_RECONCILE_DISPATCH_BACKOFF_SECONDS` | `15` | Base backoff after a broker failure |
| `INGESTION_RECONCILE_DISPATCH_BACKOFF_MAX_SECONDS` | `300` | Backoff cap |
| `INGESTION_RECONCILE_TRANSIENT_GRACE_SECONDS` | `300` | Wait before recovering a failed job whose Celery retry may still be in flight |

`INGESTION_EXECUTION_LEASE_SECONDS` must be less than
`CELERY_VISIBILITY_TIMEOUT_SECONDS`. The backoff cap must be at least the base.
Invalid values fail startup rather than silently producing an unsafe schedule.

### Reconciliation logging

Reconciliation emits JSON objects on the `opsatlas.reconciliation` logger. Each
job event contains `batch_id`, `job_id`, salted `tenant_ref`, status,
`recovery_reason`, `dispatch_attempts`, `outcome`, and a sanitized
`error_category`. The batch summary contains `batch_id`, `locked`, `scanned`,
`dispatched`, `skipped`, `failures`, and `duration_ms`. Document content,
embedding vectors, tenant UUIDs, broker credentials, SQL parameters, and raw
exception text are never logged.

### Dead letters and retention

A job that exhausts its bounded retries moves to `dead_letter` with its sanitized
`error_code`/`error_summary` preserved. Dead-letter rows are never redispatched.

Inspect them read-only:

```bash
python scripts/ops/inspect_dead_letters.py --limit 50
```

`scripts/ops/retention_cleanup.py` can remove terminal job rows only. It is a dry
run by default, requires `--apply` to mutate, is bounded by `--limit`, excludes
active jobs, and keeps failed/dead-letter rows unless explicitly included. It
never deletes documents, versions, chunks, or ACL rows. See
[`OPERATIONS_RUNBOOK.md`](OPERATIONS_RUNBOOK.md) for the policy table and
recovery guidance.



## Queues and worker responsibilities

Ingestion and reconciliation use two queues on the same Redis broker:

| Queue | Default name | Producer | Consumer |
| --- | --- | --- | --- |
| Ingestion | `CELERY_TASK_DEFAULT_QUEUE` (`ingestion`) | API async ingest, reconciler republishing, manual `send_ingestion_job` | `worker` (`ingestion@%h`) |
| Reconciliation | `CELERY_RECONCILIATION_QUEUE` (`reconciliation`) | Celery beat, `scripts/reconcile_ingestion_jobs.py`, manual `celery call` | `reconciliation-worker` (`reconciliation@%h`) |

Routing is centralized in `app/celery_app.py`:

- `task_routes` maps only `opsatlas.ingestion.reconcile_jobs` to
  `CELERY_RECONCILIATION_QUEUE`.
- The schedule entry also sets `options.queue` to the same queue, so beat
  publishes there even if routing is bypassed.
- Ingestion publication passes an explicit `queue=settings.celery_task_default_queue`,
  so ingestion never drifts onto the reconciliation queue.

`CELERY_RECONCILIATION_QUEUE` must differ from `CELERY_TASK_DEFAULT_QUEUE`;
configuration validation rejects equal names so the isolation cannot silently
collapse. Both queues share the existing broker — no second broker, Redis
database, or result backend is introduced.

Reconciliation is deliberately isolated from ingestion so periodic recovery is
not starved behind a long ingestion backlog. The ingestion worker keeps its
existing concurrency; the reconciliation worker runs concurrency `1`, with
`--max-tasks-per-child=1000` and a `512m` Compose memory limit. Each worker
consumes only its own queue.

## Running the workers

Local host workers (after `docker compose up -d postgres redis`,
`python -m app.db.migrations`, and `pip install -e ".[dev]"`):

```bash
# Ingestion worker
celery -A app.celery_app:celery_app worker \
  --loglevel=INFO --concurrency=2 --queues=ingestion \
  --hostname=ingestion@%h

# Reconciliation worker
celery -A app.celery_app:celery_app worker \
  --loglevel=INFO --concurrency=1 --queues=reconciliation \
  --hostname=reconciliation@%h --max-tasks-per-child=1000
```

Compose workers:

```bash
docker compose up -d --build worker reconciliation-worker
docker compose logs -f worker
docker compose logs -f reconciliation-worker
docker compose stop worker reconciliation-worker
```

Inspect each worker by node name and inspect each queue separately:

```bash
# Ping only the worker you mean; a generic ping passes if either worker answers.
celery -A app.celery_app:celery_app inspect ping \
  --destination ingestion@$(hostname)
celery -A app.celery_app:celery_app inspect ping \
  --destination reconciliation@$(hostname)
celery -A app.celery_app:celery_app inspect active
celery -A app.celery_app:celery_app inspect reserved

# Queue depths on the broker (Redis database 1)
docker compose exec redis redis-cli -n 1 LLEN ingestion
docker compose exec redis redis-cli -n 1 LLEN reconciliation
```

The Compose health checks are targeted: `worker` pings `ingestion@<host>` and
`reconciliation-worker` pings `reconciliation@<host>`. A generic
`celery inspect ping` is not used, because it would report healthy while only
one of the two workers was alive. Both health checks also run
`python -m app.db.schema_validation --quiet`. A live worker attached to a
database with missing or incompatible schema is reported unhealthy, and worker
startup fails before consuming work — queue isolation does not bypass schema
validation.

### Diagnosing a growing reconciliation queue

If `LLEN reconciliation` keeps rising or the queue never drains:

1. Confirm the reconciliation worker is present and healthy:
   `docker compose ps reconciliation-worker` and
   `celery inspect ping --destination reconciliation@$(hostname)`.
2. Check `docker compose logs -f reconciliation-worker` for startup schema
   validation failures, broker connection errors, or repeated task errors.
3. Confirm beat publishes to the reconciliation queue:
   `docker compose logs -f beat` should show the task being sent, and the
   schedule entry's `options.queue` must resolve to
   `CELERY_RECONCILIATION_QUEUE`.
4. Verify all services share `CELERY_BROKER_URL` and the same
   `CELERY_RECONCILIATION_QUEUE`. A consumer listening on a different queue name
   will leave messages unclaimed.
5. Remember that messages can accumulate briefly after worker loss and
   redelivery; check the durable job state in PostgreSQL (`ingestion_jobs`) as
   the source of truth rather than the Redis length alone.

A growing ingestion queue does not block reconciliation: the reconciliation
worker never consumes `ingestion`, and beat always publishes recovery work to
`reconciliation`.

## Troubleshooting

1. **`202` never progresses.** Check `GET /api/v1/documents/jobs/<id>` first.
   If `status = failed` and `error_code = queue_unavailable`, the broker was
   unavailable; the reconciler republishes it with backoff, and a resubmission
   of the same document also reuses the same job.
2. **Ingestion worker is not consuming.** Confirm the `worker` uses the same
   `CELERY_BROKER_URL` and `CELERY_TASK_DEFAULT_QUEUE` as the API. Inspect the
   `ingestion` Redis queue length and `celery inspect active`. Reconciliation
   work is separate: confirm `reconciliation-worker` consumes
   `CELERY_RECONCILIATION_QUEUE` and inspect the `reconciliation` queue.
3. **Embedding outage.** Jobs will remain `failed` between retries and become
   `dead_letter` after the configured budget. The target version is non-current,
   so retrieval continues to serve the last-known-good version. If a Celery retry
   message is lost, reconciliation republishes the job after
   `INGESTION_RECONCILE_TRANSIENT_GRACE_SECONDS`.
4. **Schema mismatch.** `embedding_schema_mismatch` is operational: apply the
   intended `vector(384)` schema or change the configuration deliberately, then
   resubmit. The job is not silently retried.
5. **Duplicate delivery.** Duplicate tasks are expected with late
   acknowledgement. A succeeded/current job is a no-op; a running job whose lease
   is live returns `job_busy`; a late execution whose lease expired returns
   `execution_lease_lost` and cannot activate or fail the newer execution.
6. **Reconciliation seems idle or its queue grows.** Confirm the `beat` service
   is running and publishes to `CELERY_RECONCILIATION_QUEUE`, confirm
   `reconciliation-worker` is healthy, or run
   `python scripts/reconcile_ingestion_jobs.py`. A `locked = true` result means
   another run holds the advisory lock. Check that the workers, beat, and API use
   the same database, broker, and queue configuration.

## Logging

Worker events are JSON objects on the `opsatlas.worker` logger. Each event
contains `job_id`, salted `tenant_ref`, `task_id`, Celery attempt, PostgreSQL job
attempts, duration, status, outcome, and a sanitized `error_category`. Document
content, embedding vectors, broker credentials, SQL parameters, and exception
payloads are never logged.

## Known limitations

* Reconciliation is bounded by the configured batch size. A large backlog is
  drained over multiple beat intervals, not in one run.
* A job that crashed after commit but before a target version was persisted has
  no safe resumable target and is not dispatched automatically; resubmit it.
* Dispatch claims and execution leases depend on PostgreSQL time. Hosts must keep
  reasonably synchronized clocks.
* There is no Celery result backend or task-state API; PostgreSQL remains the
  operational source of truth.
* The per-tenant rate limit is shared through Redis in Compose (the memory
  backend remains the test default). The concurrency slot is per process.
* The Compose worker receives `CONTAINER_EMBEDDING_URL`
  (`host.docker.internal:8091` by default) as its `EMBEDDING_URL`. Override that
  variable if the embedding service runs elsewhere; the host-run `EMBEDDING_URL`
  is not used inside containers. See `docs/EMBEDDING_CONFIGURATION.md`.

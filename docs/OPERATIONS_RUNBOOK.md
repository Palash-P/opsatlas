# Operations runbook

This runbook covers the operational controls that exist in this repository. It
does not replace the detailed migration, ingestion, or evaluation documents:

- Schema and migration details: [`DATABASE_MIGRATIONS.md`](DATABASE_MIGRATIONS.md)
- Ingestion lifecycle, queues, and reconciliation: [`BACKGROUND_INGESTION.md`](BACKGROUND_INGESTION.md)
- Embedding contract: [`EMBEDDING_CONFIGURATION.md`](EMBEDDING_CONFIGURATION.md)
- Local validation harness: [`OPERATIONAL_VALIDATION.md`](OPERATIONAL_VALIDATION.md)

The Compose file is local-development infrastructure. Production deployment is
outside this repository's scope; §10 lists the controls an operator must supply.

## 1. Health, readiness, and logs

| Endpoint | Meaning | Probe guidance |
| --- | --- | --- |
| `GET /health` | Process liveness only. No database or Redis access. | Use for container restart decisions. |
| `GET /ready` | PostgreSQL connectivity, migration history, full live schema, `chunks.embedding vector(384)`. Redis is optional. | Returns `503` for required failures and `200 degraded` when only Redis is down. |

The Compose `api` health check probes `/ready`; workers and reconciliation
workers combine a targeted Celery `inspect ping` with
`python -m app.db.schema_validation --quiet`.

Structured logs are one JSON object per line (no metrics endpoint exists yet).
Every request carries `X-Request-ID`; worker and reconciliation events carry
their own identifiers. No event contains bearer tokens, document content,
embedding vectors, secrets, SQL parameters, or raw tenant UUIDs.

| Event | Logger | Useful fields |
| --- | --- | --- |
| `request` | `opsatlas.request` | `request_id`, `method`, `path`, `status_code`, `tenant_ref`, timings, `error_category` |
| `ingestion_submission` | `opsatlas.request` | `request_id`, `job_id`, `tenant_ref`, `status`, `deduplicated`, `published` |
| `ingestion_task` | `opsatlas.worker` | `job_id`, `task_id`, `attempt`, `status`, `outcome`, `error_category`, `duration_ms` |
| `ingestion_reconciliation` | `opsatlas.reconciliation` | `batch_id`, `scanned`, `dispatched`, `skipped`, `failures`, `duration_ms`, `locked` |
| `ingestion_reconciliation_job` | `opsatlas.reconciliation` | `batch_id`, `job_id`, `tenant_ref`, `recovery_reason`, `outcome` |
| `schema_validation` | `opsatlas.schema` | `outcome`, `error_category`, sanitized problem text |
| `readiness` | `opsatlas.schema` | `check` (`postgres`/`redis`), `outcome`, `error_category` |
| `ingestion_retention` | `opsatlas.worker` | per-status counts, `applied`, `tenant_scoped`, `limit` |

Useful queries:

```bash
docker compose logs -f api
docker compose logs -f worker | jq 'select(.event == "ingestion_task")'
docker compose logs -f reconciliation-worker | jq 'select(.event | startswith("ingestion_reconciliation"))'
docker compose logs api | jq 'select(.outcome == "failed" or .status_code >= 500)'
```

Logs are process output only. Shipping, indexing, alerting, and long-term log
retention are deployment responsibilities (§10).

## 2. Inspect workers and queues

```bash
docker compose ps
celery -A app.celery_app:celery_app inspect ping --destination ingestion@$(hostname)
celery -A app.celery_app:celery_app inspect ping --destination reconciliation@$(hostname)
celery -A app.celery_app:celery_app inspect active
celery -A app.celery_app:celery_app inspect reserved
docker compose exec redis redis-cli -n 1 LLEN ingestion
docker compose exec redis redis-cli -n 1 LLEN reconciliation
```

Run one controlled reconciliation batch:

```bash
python scripts/reconcile_ingestion_jobs.py --batch-size 50
python scripts/reconcile_ingestion_jobs.py --batch-size 50 --json
```

The command exits non-zero when a batch records dispatch failures. It never
prints document content, vectors, or secrets. Reconciliation is isolated on its
own queue and worker; a growing ingestion queue must not delay recovery.

## 3. Create a backup

Backups use the PostgreSQL client tools through the repository wrapper:

```bash
python scripts/ops/db_backup.py \
  --output backups/opsatlas-2026-11-17.dump
```

`--database-url` defaults to `DATABASE_URL` and is required (directly or through
the environment). The wrapper:

1. Collects a read-only manifest: database name, server version, migration
   history, embedding column type, and row counts for `documents`,
   `document_versions`, `chunks`, and `ingestion_jobs`.
2. Runs `pg_dump --format=custom --no-owner --no-privileges`.
3. Verifies the archive with `pg_restore --list` and checks that the expected
   tables are present.
4. Writes `<backup>.manifest.json` with the SHA-256, size, and manifest fields.

It refuses to overwrite an existing file. It never writes to PostgreSQL.

```bash
# Machine-readable summary
python scripts/ops/db_backup.py --output backups/opsatlas.dump --json
```

Store backups outside the repository, encrypt them at rest, and copy them to a
different failure domain. This repository does not schedule backups or upload
them anywhere.

## 4. Verify a backup

Verification is automatic unless `--no-verify` is used. To verify manually:

```bash
pg_restore --list backups/opsatlas-2026-11-17.dump | head -n 40
python - <<'PY'
import hashlib, json, pathlib
dump = pathlib.Path("backups/opsatlas-2026-11-17.dump")
manifest = json.loads(dump.with_name(dump.name + ".manifest.json").read_text())
digest = hashlib.sha256(dump.read_bytes()).hexdigest()
print("sha256 matches:", digest == manifest["sha256"])
print("counts:", manifest["row_counts"])
print("embedding:", manifest["embedding_type"])
PY
```

A verified backup has: a readable archive, a matching SHA-256, the expected
migration versions, and non-null counts for the tables that existed at backup
time. The manifest also records `schema_migrations_present`; a database that was
never migrated will show `false` and null counts.

## 5. Restore into a separate database

**Never restore over the shared or production database.** The restore wrapper
creates a new database and refuses reserved names (`opsatlas`, `postgres`,
`template0`, `template1`), the source database name, an existing database, and a
mismatched `--confirm` value.

```bash
# 1. Inspect the plan without creating anything
python scripts/ops/db_restore.py \
  --backup backups/opsatlas-2026-11-17.dump \
  --target-database-url "postgresql+psycopg://opsatlas:CHANGE_ME@localhost:5433/opsatlas_restore_20261117" \
  --confirm opsatlas_restore_20261117 \
  --dry-run

# 2. Restore
python scripts/ops/db_restore.py \
  --backup backups/opsatlas-2026-11-17.dump \
  --target-database-url "postgresql+psycopg://opsatlas:CHANGE_ME@localhost:5433/opsatlas_restore_20261117" \
  --confirm opsatlas_restore_20261117
```

The restore uses `pg_restore --no-owner --no-privileges --exit-on-error`, then
compares row counts, migration history, and the embedding column type with the
manifest. If the PostgreSQL client tools are newer than the server, the wrapper
renders the archive to SQL and removes only the unsupported
`SET transaction_timeout` line before applying it with `psql` and
`ON_ERROR_STOP` (PostgreSQL 16 and older do not recognize that setting). It
needs permission to create a database and the `vector` extension. On failure the
partial target database is left in place for inspection; drop it manually after
collecting evidence.

Verify the restored database independently:

```sql
SELECT count(*) AS documents FROM documents;
SELECT count(*) AS versions FROM document_versions;
SELECT count(*) AS chunks FROM chunks;
SELECT count(*) AS jobs FROM ingestion_jobs;

SELECT version, filename FROM schema_migrations ORDER BY version;

SELECT format_type(atttypid, atttypmod) AS embedding_type
FROM pg_attribute
WHERE attrelid = 'chunks'::regclass AND attname = 'embedding'
  AND NOT attisdropped;

SELECT count(*) AS invalid_embeddings
FROM chunks
WHERE embedding IS NULL
   OR vector_dims(embedding) <> 384;
```

Point a disposable API instance at the restored database and check `/ready`.
Do not repoint the shared environment until the checks pass.

### pgvector precautions

- The dump contains the `vector` extension and the `vector(384)` column. The
  target server must provide the same `pgvector` extension version; restoring
  into a server without it fails.
- Do not restore a 384-dimensional dump into a schema expecting another
  dimension, and never truncate or reinterpret vectors.
- `pg_dump`/`pg_restore` preserve vectors byte-for-byte. Verify with the
  `vector_dims` query above rather than sampling values.
- A restore does not re-embed anything. Re-embedding is documented in
  [`DATABASE_MIGRATIONS.md`](DATABASE_MIGRATIONS.md) and
  [`EMBEDDING_CONFIGURATION.md`](EMBEDDING_CONFIGURATION.md).

## 6. Retention

Retention applies only to terminal `ingestion_jobs` audit rows. It never deletes
documents, document versions, chunks, tenants, or ACL rows.

| Status | Default age | Eligible by default? | Notes |
| --- | --- | --- | --- |
| `succeeded` | 30 days | Yes | Safe operational history; the document/version/chunk data remains. |
| `failed` | 90 days | No (`--include-failed`) | Keep unless the failure is understood and no longer needed. |
| `dead_letter` | 365 days | No (`--include-dead-letter`) | Primary forensic record; export before deleting. |
| `queued`, `running` | — | Never | Active work is always excluded. |

```bash
# Dry run (default): count only, no mutation
python scripts/ops/retention_cleanup.py --json
python scripts/ops/retention_cleanup.py --succeeded-days 30 --limit 500

# Apply explicitly, bound the batch, optionally scope to one tenant
python scripts/ops/retention_cleanup.py --apply --limit 500
python scripts/ops/retention_cleanup.py --apply --limit 500 --tenant-id 00000000-0000-4000-8000-000000000001
```

`--limit` caps rows per status per run (maximum 5000). Run repeatedly or raise
the limit deliberately. The `ingestion_retention` event records per-status
counts, whether the run was applied, and whether it was tenant-scoped.

Application logs and temporary artifacts have no automatic retention in this
repository. Configure log rotation/shipping in the deployment platform, and
remove downloaded backups or evaluation artifacts under the platform's own
policy. Do not delete `evals/reports` or `evals/runs` evidence while an
evaluation is being reviewed.

## 7. Dead-letter inspection

```bash
python scripts/ops/inspect_dead_letters.py --limit 50
python scripts/ops/inspect_dead_letters.py --limit 50 --json
python scripts/ops/inspect_dead_letters.py --tenant-id 00000000-0000-4000-8000-000000000001
```

Output includes the job ID, salted tenant reference, attempts, error code,
sanitized summary, and timestamps. The command opens a read-only transaction and
does not change status, attempts, or dispatch state.

There is no requeue command in this release. Requeueing changes durable state
and republishes work; it needs an explicit operator workflow, fencing, and an
audit record. Recover by fixing the cause (embedding service, schema, provider
credentials) and resubmitting the source document through the authenticated
ingestion API. The new submission is idempotent and does not rewrite the
dead-letter history.

## 8. Incident response: failed or stuck ingestion

1. Read the job: `GET /api/v1/documents/jobs/{job_id}` with a tenant-scoped
   bearer token, or `python scripts/ops/inspect_dead_letters.py`.
2. Check `/ready` and `docker compose logs api` for `schema_validation` or
   `readiness` failures.
3. Check `docker compose logs -f worker` for `ingestion_task` events and the
   sanitized `error_category` (`embedding_service_unavailable`,
   `embedding_schema_mismatch`, `queue_unavailable`, `database_error`, …).
4. Check the embedding service and the container's resolved `EMBEDDING_URL`
   from inside the worker container. It comes from `CONTAINER_EMBEDDING_URL`
   (`host.docker.internal:8091` by default), not from the host-run
   `EMBEDDING_URL`; see `docs/EMBEDDING_CONFIGURATION.md`.
5. Check queue depth and reconciliation logs. A blocked delivery returns
   `outcome=blocked` with `error_category=schema_incompatible`; it never marks the
   durable job failed.
6. If reconciliation is behind, run
   `python scripts/reconcile_ingestion_jobs.py --batch-size 50`.
7. After fixing the cause, resubmit the source if the job is dead-lettered.
   Otherwise the reconciler or a Celery redelivery resumes the existing target.

Never delete an `ingestion_jobs` row to force a retry. Never edit a job's
`execution_token`; fencing depends on it.

## 9. PostgreSQL or Redis outage

**PostgreSQL down**

- `/health` stays `200`; `/ready` returns `503` with `postgres=error` and
  `schema=unavailable`.
- API writes fail with a sanitized `503 database_unavailable`.
- Workers refuse to start or block deliveries; durable jobs remain in
  PostgreSQL, so no work is lost.
- Restore connectivity, run `python -m app.db.migrations --check`, then confirm
  `/ready` and worker logs. Do not run destructive recovery commands.

**Redis down**

- `/ready` returns `200 degraded` (`redis=error`); the API keeps serving reads.
- A Redis-backed rate limiter logs one `rate_limit_backend_degraded` warning and
  falls back to the in-process window. The limit remains enforced per process; it
  never becomes unlimited. If `RATE_LIMIT_REDIS_FALLBACK_TO_LOCAL=false`, requests
  fail closed with `503` instead. Recovery logs `rate_limit_backend_recovered`.
- New async ingestions return `503` with `queue_unavailable` and keep the
  durable job row for reconciliation.
- Workers cannot consume until the broker returns. Reconciliation will publish
  after Redis recovers.
- Check `docker compose ps redis` and `redis-cli ping`. Do not flush the broker
  database; job state is in PostgreSQL, but queued messages are not.

## 9a. Rotate or revoke a bearer token

The token map is deployment configuration. There is no admin API.

1. Generate a new token (at least 32 random characters).
2. Add the new entry to `AUTH_TOKENS` alongside the old entry. Do not reuse the
   old token value.
3. Restart or redeploy the API so both tokens are loaded. The startup event
   `auth_tokens_loaded` reports only the configured count.
4. Move clients to the new token.
5. Remove the old entry and restart/redeploy again. Removing an entry is
   revocation.
6. Confirm old credentials return `401` and the new token succeeds.

To schedule expiry instead of manual removal, add an ISO-8601 `expires_at` to the
entry. It is checked on every request, so no restart is needed when it passes.
Expiry and removal do not cancel ingestion work already committed by that
principal; the worker completes the durable, already-authorized job.

## 10. Production security requirements

The repository defaults are for local development. Production deployment must
supply the following:

- `APP_ENV=production` (or `prod`). Startup then rejects the development
  database password, an empty or unusable `AUTH_TOKENS` map, tokens shorter than
  32 characters, documented placeholders, the default `AUTH_LOG_TENANT_SALT`, and
  a process-local (`memory`) rate-limit backend. See `app/core/config.py`.
- Real bearer tokens injected from a secret manager. The token map is a local
  implementation, not an identity provider: there is no SSO, OAuth, JWT refresh,
  or revocation endpoint. Use the rotation procedure in §9a.
- `RATE_LIMIT_BACKEND=redis` with a shared `REDIS_URL`, so all API replicas
  enforce one per-tenant limit. Decide explicitly whether a Redis outage should
  degrade to the per-process limiter (default) or fail closed with `503`
  (`RATE_LIMIT_REDIS_FALLBACK_TO_LOCAL=false`).
- A non-development PostgreSQL password and a TLS/private network path to the
  database. Do not use the Compose `postgres` service as a production database.
- Redis authentication/TLS (or a managed Redis) and broker credentials. The
  local Compose Redis has no authentication and is bound to loopback.
- No published datastore ports. Remove the `ports:` entries for PostgreSQL and
  Redis in production, or bind them to an internal interface with firewall
  controls. Application containers already drop capabilities and run as
  non-root.
- TLS termination and authentication at a reverse proxy or ingress in front of
  the API. The Compose API binds to `127.0.0.1` by default and does not
  terminate TLS.
- Centralized log shipping with retention and alerting. Structured logs are the
  metric source; there is no `/metrics` endpoint or tracing in this release.
- Encrypted, off-site backups and tested restores. See §3–§5.
- Resource limits appropriate to the host. Compose sets development limits
  (`512m` API, `1g` ingestion worker, `512m` reconciliation worker, `256m`
  Beat); production values depend on workload.

Images are pinned to explicit versions:

- `python:3.13.9-slim-bookworm`
- `pgvector/pgvector:0.8.0-pg16`
- `redis:7.4.2-alpine`

To update, change one tag at a time, review the upstream release notes, run the
full offline suite and `docker compose build`, then validate `/ready` and a
restore from a fresh backup. Digest pinning is supported if the platform records
the digest in the change ticket. The Python dependency set is pinned separately
in `requirements.lock`; see `docs/DEPENDENCIES.md` before regenerating it.

## 11. Container hardening

The shared image creates an unprivileged `opsatlas` user (UID 10001) and runs
all containers as that user. Compose application services set
`no-new-privileges:true`, `cap_drop: [ALL]`, and memory limits. Datastores are
not privileged. Logs go to stdout; Celery beat writes its schedule inside
`/app`, owned by the runtime user.

If a future image change needs a writable path, create it during the build and
`chown` it to UID 10001. Do not switch back to `root` to work around a
permission error.

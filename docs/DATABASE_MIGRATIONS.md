# Database migrations

OpsAtlas uses a small PostgreSQL migration runner instead of two independent
bootstrap paths. The SQL files in `sql/` are the only schema definition. Docker
Compose no longer loads `001_initial_schema.sql` directly; a one-shot `migrate`
service applies the same runner used on the host.

Only **384-dimensional** embeddings are supported. The final
`chunks.embedding` column is `vector(384)`.

## Migration set

| Version | File | Purpose |
| --- | --- | --- |
| 001 | `sql/001_initial_schema.sql` | Base relational schema and pgvector extension. The original proposal used `vector(1536)`. |
| 002 | `sql/002_embedding_384.sql` | Replaces the empty provisional column with `vector(384)` and adds `embedding_model`, `embedded_at`. |
| 003 | `sql/003_ingestion_jobs_version.sql` | Links an ingestion job to the immutable document version it produced. |
| 004 | `sql/004_activation_generation.sql` | One-current-version invariant and activation generation sequence. |
| 005 | `sql/005_job_reconciliation.sql` | Dispatch state, execution leases, fencing token, and reconciliation indexes. |

Historical files are immutable. The runner tracks them in:

```text
schema_migrations
  version       integer primary key
  filename      text not null unique
  checksum      text not null
  applied_at    timestamptz not null
  applied_by    text not null   -- "runner" or "adopted"
```

`schema_migrations` is the authority for what has run. A table or column that
happens to exist is not, by itself, evidence that a migration completed.

## PostgreSQL connection modes and ports

PostgreSQL listens on container port `5432` in every mode. The host-published
port is independent and defaults to `5433`, because a locally installed
PostgreSQL server may already own `5432`:

| Caller | Address | Port source |
| --- | --- | --- |
| `migrate` and other Compose services | `postgres:5432` | Internal service DNS and container port |
| Host-run `python -m app.db.migrations`, API, workers, scripts, tests | `localhost:5433` by default | `DATABASE_URL` (host port must match `POSTGRES_PORT`) |

`POSTGRES_PORT` in `.env` controls only the host mapping
(`${POSTGRES_PORT:-5433}:5432`). Changing it does not change the container port
or the internal `postgres:5432` URLs. If you change `POSTGRES_PORT`, update the
port in host-run `DATABASE_URL` to the same value, otherwise host commands fail
with connection refused or connect to an unrelated database.

If startup reports `address already in use`, inspect the host port owner and
choose a free value:

```bash
# Windows PowerShell
Get-NetTCPConnection -LocalPort 5433 -State Listen
# macOS/Linux
lsof -nP -iTCP:5433 -sTCP:LISTEN
```

Never "fix" a conflict by changing the container port or the `postgres:5432`
URLs used between services.

## Fresh database setup

### Docker Compose

```bash
docker compose up -d postgres redis
docker compose run --rm migrate
```

Or start the full stack; `api`, `worker`, `reconciliation-worker`, and `beat` wait
for the `migrate` service to exit successfully:

```bash
docker compose up -d
```

The `migrate` service runs:

```bash
python -m app.db.migrations
```

If the migration fails, the one-shot service exits non-zero and `api`,
`worker`, `reconciliation-worker`, and `beat` do not start. The API container
never runs migrations; it only serves traffic after the migration job has
completed successfully.
PostgreSQL itself starts with pgvector because the image is
`pgvector/pgvector:0.8.0-pg16` (the same pinned tag used by Compose and CI);
migration 001 creates the extension inside the new database.

### Host Python environment

```bash
python -m app.db.migrations
```

The command uses `Settings.database_url` (normally from `.env`). A fresh empty
database has migrations 001–005 applied in order, records five history rows, and
ends with `chunks.embedding vector(384)`.

Verify without writing:

```bash
python -m app.db.migrations --check
python -m app.db.schema_validation
```

## Existing database upgrade

Take a backup first, then stop API, worker, reconciliation worker, and beat
writers during the schema change.

`DATABASE_URL` uses the SQLAlchemy driver suffix
(`postgresql+psycopg://...`), which `pg_dump` does not accept. Strip the suffix
or let the repository backup script convert it. The script is the preferred
path because it also verifies the archive and writes a manifest.

```bash
# Reads DATABASE_URL by default and converts it to a libpq DSN.
python scripts/ops/db_backup.py --output backups/opsatlas-before-migrations.dump
```

Equivalent direct `pg_dump` against the host-published port. Supply the
password through `PGPASSFILE` (or an already-exported `PGPASSWORD`) rather than
a URL on the command line, so it is not recorded in shell history or visible in
the process list:

```bash
export PGPASSFILE="$HOME/.pgpass"   # chmod 600; host:port:db:user:password
pg_dump --format=custom --no-owner --no-privileges \
  --host=localhost --port="${POSTGRES_PORT:-5433}" \
  --username="${POSTGRES_USER:-opsatlas}" \
  --dbname="${POSTGRES_DB:-opsatlas}" \
  --file=opsatlas-before-migrations.dump
```

When only the Compose network can reach PostgreSQL, run `pg_dump` inside the
`postgres` service. `exec -T` disables TTY allocation so the archive stays
clean on stdout:

```bash
docker compose exec -T postgres \
  pg_dump --format=custom --no-owner --no-privileges \
  --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" \
  > opsatlas-before-migrations.dump
```

The container form reuses the Compose environment and needs no password on the
command line. A password containing special characters must be percent-encoded
inside a URL or, preferably, supplied through `PGPASSFILE`; never paste a
literal credential into a command that lands in shell history.

Then apply the migrations:

```bash
docker compose stop api worker reconciliation-worker beat
python -m app.db.migrations
```

For a verified backup with a manifest, and for restoring into a separate
disposable database, use `scripts/ops/db_backup.py` and
`scripts/ops/db_restore.py` as described in
[`OPERATIONS_RUNBOOK.md`](OPERATIONS_RUNBOOK.md). Never restore over the shared
database.

The runner first validates the recorded history, then inspects each unrecorded
migration. It will:

- **adopt** a migration whose complete expected schema is already present
  (recording `applied_by = 'adopted'` and executing no migration SQL),
- **apply** a migration whose expected objects are completely absent, or
- **stop** if the migration is partially present.

For example, a database where an operator previously ran `psql -f` for 001–003
but has no tracking table is adopted through 003 and then upgraded with 004 and
005. The shared development database follows the same path; nothing is dropped
or recreated.

Adoption is recorded explicitly in the command output:

```text
adopted  001 (schema verified, SQL not executed)
adopted  002 (schema verified, SQL not executed)
applied  003
```

Use `--no-adopt` for a strict run that refuses to establish a baseline for any
already-present schema.

### Adoption safety rules

- Each migration has a full expectation set in `app/db/schema_contract.py`
  (tables, column types/nullability/defaults, constraints, indexes, sequences,
  and data invariants). One table or one column is never enough.
- Adoption requires **all** checks for that migration. Partial signatures fail
  closed and list the exact missing or mismatched objects.
- Applied history must be a contiguous prefix of the repository files. Unknown
  versions, deleted history rows, and newly inserted older migrations are
  rejected.
- Existing history rows are never updated or deleted by the runner.
- Adoption cannot prove the historical SQL bytes. It records the current file
  checksum only after the live schema demonstrably matches; later edits are then
  detected.

## Checksums

The checksum is SHA-256 over the UTF-8 file with `CRLF`/`CR` normalized to `LF`.
This makes a Windows checkout and a Linux checkout of the same commit produce
the same checksum.

A recorded migration whose checksum changes is rejected before any SQL runs:

```text
Migration history validation failed:
  - checksum mismatch for applied migration 003_ingestion_jobs_version.sql: ...
```

Do **not** edit an applied migration to make the checksum match. Either restore
the original file from version control, or add a new forward migration. The
runner does not rewrite history automatically.

## Execution model

- **Concurrency:** a fixed PostgreSQL session advisory lock serializes runners
  across processes and hosts. `--no-wait` fails immediately instead of waiting.
- **Transaction boundary:** the runner owns it. Historical files may wrap
  statements in one `BEGIN`/`COMMIT`; that outer wrapper is removed. Each file
  and its `schema_migrations` row commit atomically. A failure rolls back both.
- **Lock safety:** each migration runs with `lock_timeout` (default 5000 ms,
  `--lock-timeout-ms`). A busy table causes an explicit retryable failure instead
  of an indefinite `ACCESS EXCLUSIVE` wait.
- **Final validation:** after the last pending migration, the full application
  contract is checked inside the same transaction before the history row commits.
- **No automatic down migrations.** Recovery is forward-fix or restore-from-
  backup.

Migration 002 takes an `ACCESS EXCLUSIVE` lock on `chunks`. On a populated
database, run migrations in a maintenance window and retry with a deliberately
larger `--lock-timeout-ms` if needed.

## Embedding dimension migration

The application contract is fixed at 384 dimensions:

- `DATABASE_EMBEDDING_DIMENSIONS = 384`
- `chunks.embedding vector(384)`
- retrieval casts query vectors to `vector(384)`

Migration 002 refuses to run when `chunks` already contains embeddings:

```text
Migration aborted: chunks already contain embeddings
```

That is deliberate. A 1536-dimensional vector cannot be safely truncated, padded,
or reinterpreted as a 384-dimensional vector.

### Fresh database

001 creates the provisional `vector(1536)` column, then 002 changes it to
`vector(384)` while the table is empty. This is the supported fresh path.

### Populated 1536-dimensional database

Choose one of these procedures. Both preserve documents, versions, chunks, job
history, and ACLs.

**Preferred: blue-green re-embedding**

1. Keep the existing database unchanged as the last known good copy.
2. Create a new database, run the migration runner, and point a separate
   ingestion/backfill environment at the configured 384-dimensional model.
3. Re-embed from `chunks.content` (or re-ingest from source) into the new
   database.
4. Verify counts, retrieval, and current-version state, then cut over.

**In-place, only after a verified backup and with writes stopped**

1. Run `python scripts/ops/db_backup.py --output backups/<name>.dump` and verify
   the backup is restorable (see §"Existing database upgrade" above).
2. Stop API, worker, reconciliation worker, and beat.
3. Explicitly clear derived vectors and their metadata, without touching
   documents, versions, chunks, jobs, or ACLs:

   ```sql
   UPDATE chunks
   SET embedding = NULL,
       embedding_model = NULL,
       embedded_at = NULL;
   ```

   This is an operator action, not something the runner does silently.
4. Run `python -m app.db.migrations`.
5. Re-embed every chunk with the configured 384-dimensional model:

   ```bash
   python scripts/backfill_embeddings.py --apply
   ```

6. Verify no chunk is missing a 384-dimensional vector:

   ```sql
   SELECT count(*)
   FROM chunks
   WHERE embedding IS NULL
      OR vector_dims(embedding) <> 384
      OR embedding_model IS DISTINCT FROM 'sentence-transformers/all-MiniLM-L6-v2';
   ```

7. Restart services and check `/ready`.

## Recovering a partially applied migration

The runner never guesses. A partial signature produces a message such as:

```text
migration 003 ingestion_jobs_version is partially present and cannot be applied
or adopted safely. Refusing to guess. Missing or mismatched objects:
  - constraint 'ingestion_jobs_version_id_fkey' on 'ingestion_jobs' is missing
```

Recovery options, in order of preference:

1. **Forward-fix the schema**: add the missing objects exactly as the migration
   would have, then rerun `python -m app.db.migrations`. The migration is then
   verified and adopted; the history row is written only after the full check
   passes.
2. **Restore from a backup** taken before the partial change, then rerun.
3. **Manual intervention only with review**: inspect the live catalog and the
   migration file, document every difference, and write a new forward migration
   for any permanent divergence.

Never delete a `schema_migrations` row to "retry" a migration, and never drop or
recreate application tables to make adoption easier. Both actions hide data and
schema state.

## Readiness and worker protection

- `/health` is process liveness only.
- `/ready` requires PostgreSQL, valid migration history, the full live schema,
  and `chunks.embedding vector(384)`. Redis remains optional and produces
  `200 degraded` when it is the only failure.
- Successful readiness introspection is cached for
  `SCHEMA_VALIDATION_CACHE_SECONDS` (default 30). Failures are never cached.
- The Compose API health check probes `/ready`, so an API process that is alive
  but pointed at an incompatible schema is reported unhealthy. `/health` remains
  the process liveness endpoint and is deliberately not the container health
  check.
- Celery worker setup validates the schema before the consumer starts.
- Every task delivery revalidates through the cached check. If the schema is not
  compatible, the ingestion task returns a sanitized `blocked` outcome and does
  not resume the job, and the reconciliation task returns a sanitized `blocked`
  summary without claiming jobs, updating dispatch state, or publishing work.
  Durable jobs remain in PostgreSQL for a later run after migrations are
  restored; a schema incident never marks a job permanently failed.
- Both Compose worker health checks combine a targeted Celery `inspect ping`
  (`--destination ingestion@<host>` or `reconciliation@<host>`) with
  `python -m app.db.schema_validation --quiet`, so a live process with an
  incompatible database is reported unhealthy. Queue isolation does not bypass
  schema validation.

## Verify the final schema

```bash
python -m app.db.migrations --check
python -m app.db.schema_validation
```

Or query PostgreSQL directly:

```sql
SELECT version, filename, checksum, applied_at, applied_by
FROM schema_migrations
ORDER BY version;

SELECT format_type(a.atttypid, a.atttypmod) AS embedding_type
FROM pg_attribute AS a
WHERE a.attrelid = 'chunks'::regclass
  AND a.attname = 'embedding'
  AND NOT a.attisdropped;

SELECT column_name, data_type, is_nullable
FROM information_schema.columns
WHERE table_schema = 'public' AND table_name = 'ingestion_jobs'
ORDER BY ordinal_position;
```

Expected: five history rows (001–005), `embedding_type = vector(384)`, and all
Phase F columns (`version_id`, `activation_generation`, dispatch fields,
`execution_token`, `execution_lease_expires_at`).

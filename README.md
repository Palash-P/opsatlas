# OpsAtlas v1

A production-oriented engineering knowledge assistant built to study reliable RAG systems. The repository now includes reproducible local infrastructure, versioned ingestion, evaluated retrieval, and a complete grounded answer-generation slice with validated citations.

> Status: **grounded answer vertical slice with authenticated tenant/ACL boundaries, measured on a synthetic corpus**. Retrieval evaluation is stable against synthetic golden sets. Answer quality has three measured `answer_eval_v1` runs with a documented run-to-run range. Authorization is now enforced from configured bearer tokens, with the limitations listed below. The corpus is synthetic; this is not production validation.

## Goals

- Answer engineering questions from runbooks, API documentation, and incident reports with evidence-based citations.
- Measure retrieval and answer quality before optimizing.
- Handle document changes, background job failures, provider errors, and access-control boundaries.
- Make experiments reproducible and operational failures diagnosable.

## Local prerequisites

- Docker Engine / Docker Desktop with Compose
- Python 3.11+
- `pip` or another Python package installer

## Quick start

1. Copy `.env.example` to `.env` and adjust values if needed. Host-run commands
   use this `DATABASE_URL`, which points at the Compose-published host port
   (`localhost:5433` by default, controlled by `POSTGRES_PORT`).
2. Start local dependencies:

   ```bash
   docker compose up -d postgres redis
   ```

   Containers connect over the internal service address `postgres:5432`; only
   your host uses the published `5433`. See
   [PostgreSQL ports](#postgresql-ports) to change the host port.

3. Create an environment and install the project:

   ```bash
   python -m venv .venv
   # Windows PowerShell: .venv\Scripts\Activate.ps1
   # macOS/Linux: source .venv/bin/activate
   # Reproducible install from the pinned transitive lockfile.
   # Replace with an editable install (`pip install -e ".[dev]"` without the
   # constraint) only for local development.
   pip install --constraint requirements.lock ".[dev]"
   ```

4. Apply the versioned migrations with the single runner (the Compose
   `migrate` service uses the same command):

   ```bash
   python -m app.db.migrations
   ```

   On a fresh database this applies `sql/001`–`sql/005` in order and ends with
   `chunks.embedding vector(384)`. On an existing database it verifies and
   adopts migrations already applied manually, then applies the remainder. It
   never drops or recreates application tables.

5. Configure at least one bearer token in `.env`; see `.env.example` for the
   `AUTH_TOKENS` format. Protected endpoints return `401` when the map is empty.

6. Start the API from the host for local development:

   ```bash
   uvicorn app.main:app --reload
   ```

   Or run the complete container stack (API, ingestion worker, reconciliation
   worker, and beat) with the Compose instructions below.

7. Check:
   - `http://127.0.0.1:8000/health` — process health
   - `http://127.0.0.1:8000/ready` — PostgreSQL, schema, and dependency readiness
   - `http://127.0.0.1:8000/docs` — local API docs

8. Run tests:

   ```bash
   # Offline suite (live integration tests report as skipped)
   pytest -ra

   # Full suite against disposable PostgreSQL databases and a local Redis.
   # Requires the Compose datastores and a deterministic embedding service.
   # A fresh database needs the complete evaluation corpus in the live-test
   # tenant; --materialize-observe also creates the manifest's observe-only
   # fixtures (never used for production seeding).
   python scripts/seed_synthetic_corpus.py \
     --tenant-id "$OPSATLAS_TEST_TENANT_ID" --apply --materialize-observe
   python scripts/ops/fake_embedding_service.py --port 8091 &
   OPSATLAS_LIVE_TESTS=1 OPSATLAS_TEST_TENANT_ID="$OPSATLAS_TEST_TENANT_ID" \
   OPSATLAS_REQUIRE_POSTGRES=1 OPSATLAS_REQUIRE_REDIS=1 pytest -ra
   ```

   Live tests create uniquely named `opsatlas_test_<uuid>` databases and never
   migrate, alter, or drop the shared development database. CI runs the same
   suite with PostgreSQL/pgvector and Redis services; see
   [`.github/workflows/ci.yml`](.github/workflows/ci.yml). Dependency lock
   regeneration and clean-install verification are documented in
   [`docs/DEPENDENCIES.md`](docs/DEPENDENCIES.md).

`/ready` treats PostgreSQL, valid migration history, and the live schema
(including `vector(384)` and the Phase F columns) as required. A schema
incompatibility returns `503 not_ready`. Redis remains optional: a Redis-only
failure returns `200 degraded`. `/health` remains a process liveness check.
See `docs/DATABASE_MIGRATIONS.md` for fresh setup, adoption of a manually
migrated database, checksums, partial-migration recovery, and the safe
1536 → 384 re-embedding procedure. For backups, restore verification, retention,
dead-letter inspection, and incident response, see
[`docs/OPERATIONS_RUNBOOK.md`](docs/OPERATIONS_RUNBOOK.md).

## Run the complete stack with Docker Compose

The Compose file is local-development infrastructure. It starts PostgreSQL with
pgvector, Redis, a one-shot `migrate` job, the FastAPI `api` service, the
ingestion `worker`, the `reconciliation-worker`, and `beat` on one private
Compose network. Only PostgreSQL (`POSTGRES_PORT`, default `5433`), Redis
(`6379`), and the API (`8000` by default) are published, and all three bind to
`127.0.0.1` by default.

### PostgreSQL ports

PostgreSQL always listens on container port `5432`, and services inside Compose
connect to `postgres:5432`. The host-published port is separate and defaults to
`5433` to avoid a conflict with a locally installed PostgreSQL server that may
already own `5432`:

| Access path | Address | Configured by |
| --- | --- | --- |
| Container → container | `postgres:5432` | `docker-compose.yml` service DNS and internal port |
| Host → container | `localhost:5433` (default) | `POSTGRES_PORT` and `DATABASE_URL` |

Host-run processes use `DATABASE_URL` from `.env`. It must point at
`localhost:${POSTGRES_PORT}`, not the internal `postgres:5432` address:

```bash
# .env, default host mapping
POSTGRES_PORT=5433
DATABASE_URL=postgresql+psycopg://opsatlas:opsatlas_dev_password@localhost:5433/opsatlas
```

To publish a different host port, change both values together and restart:

```bash
# Example: a local PostgreSQL already owns 5433
POSTGRES_PORT=5434
DATABASE_URL=postgresql+psycopg://opsatlas:opsatlas_dev_password@localhost:5434/opsatlas
```

```bash
docker compose up -d postgres
docker compose ps postgres
```

**Diagnosing a port conflict.** If `docker compose up postgres` fails with
`address already in use` or `port is already allocated`, another process owns
the host port. Check the owner and then choose a free port:

```bash
# Windows PowerShell
Get-NetTCPConnection -LocalPort 5433 -State Listen
# macOS/Linux
lsof -nP -iTCP:5433 -sTCP:LISTEN
```

Set `POSTGRES_PORT` and the `DATABASE_URL` port to the same free value. Do not
change the internal `5432` port or the `postgres:5432` URLs used by containers.

1. Create the environment file and configure at least one bearer token for the
   protected `/api/v1/*` endpoints:

   ```bash
   cp .env.example .env
   # Windows PowerShell: Copy-Item .env.example .env
   ```

   `.env.example` already uses `POSTGRES_PORT=5433` and a matching
   `DATABASE_URL` on `localhost:5433`, so host-run commands and Compose agree.

2. Build and start the stack:

   ```bash
   docker compose up -d --build
   ```

3. Confirm the startup order and service state:

   ```bash
   docker compose ps
   docker compose logs migrate
   docker compose logs -f api
   ```

   Startup is gated:

   1. `postgres` passes its health check and `migrate` runs
      `python -m app.db.migrations`. The migration job is database-only and does
      not depend on Redis.
   2. `migrate` must exit successfully.
   3. `api`, `worker`, `reconciliation-worker`, and `beat` start only after
      migration succeeds, and they also wait for `redis` to be healthy.

   The API container never runs migrations. If migration fails, the API and
   workers do not start, so they cannot serve or consume work against an
   incompatible schema.

4. Check liveness and readiness separately:

   ```bash
   curl http://127.0.0.1:8000/health
   curl -i http://127.0.0.1:8000/ready
   ```

   - `/health` is process liveness only and returns `200` while the API process
     can serve requests.
   - `/ready` is the deployment gate. It requires PostgreSQL, valid migration
     history, and the full live schema, including `chunks.embedding vector(384)`.
     It returns `503 not_ready` when the schema is incompatible. Redis is
     optional and produces `200 degraded` when it is the only failure.
   - The API container health check probes `/ready`, so a live process with an
     incompatible database is reported unhealthy.
   - Interactive API docs are at `http://127.0.0.1:8000/docs`.

   Set `API_PORT` in `.env` to change only the published host port; the
   container always listens on `0.0.0.0:8000`.

5. Stop the stack without deleting database data:

   ```bash
   docker compose stop
   ```

   `docker compose down` also removes the containers and network but preserves
   the named `opsatlas_postgres_data` volume. **`docker compose down -v` deletes
   that volume and all local database data**; do not use it to reset readiness
   or schema state.

The embedding client has two documented access modes because the service runs
on the host while callers may live in a container:

- **Host-run processes** (API or worker started from the host, migrations,
  scripts, tests) use `EMBEDDING_URL`, which defaults to
  `http://127.0.0.1:8091`.
- **Compose containers** (the `api` and `worker` services) receive
  `CONTAINER_EMBEDDING_URL`, which defaults to
  `http://host.docker.internal:8091`. Compose maps that variable into the
  container's `EMBEDDING_URL`; a container's loopback is the container itself,
  not the host.

Both variables are set in `.env.example`, so copying it to `.env` and running
`docker compose up` needs no further edit. Override `CONTAINER_EMBEDDING_URL`
only if the embedding service listens somewhere other than the host.

### Local development versus production

The Compose topology is for local development, not a production deployment:

- PostgreSQL and Redis publish development ports and use the example development
  password. A production deployment should not publish either datastore and
  should inject credentials from a secret manager.
- There is one API replica, no TLS termination, and no reverse proxy. The
  Compose API sets `RATE_LIMIT_BACKEND=redis`, so the per-tenant request limit is
  shared even if replicas are added. The concurrency slot is still per process.
- `APP_ENV=production` requires the Redis rate-limit backend, a non-placeholder
  token longer than 32 characters, a non-default log salt, and a non-development
  database password.
- Production should run `python -m app.db.migrations` as a gated release job
  before rolling out API and worker instances. Keep `/health` as the liveness
  probe and `/ready` as the load-balancer readiness gate.
- The migration runner serializes with a PostgreSQL advisory lock, but operators
  should still deploy only one migration job at a time. Do not run migrations
  from an API container.

## Background ingestion

The synchronous `POST /api/v1/documents/ingest` endpoint is unchanged. An
additional asynchronous path is available at
`POST /api/v1/documents/ingest/async`; it commits a durable job and a
non-current target version before publishing a Celery task that contains only
the job ID. `GET /api/v1/documents/jobs/{job_id}` reports tenant-scoped job
state. Redis is the broker; PostgreSQL `ingestion_jobs` remains authoritative
and no Celery result backend is configured.

Start the local workers after the dependencies are up. Ingestion and
reconciliation use separate queues on the same Redis broker:

```bash
# Ingestion worker: document jobs only
celery -A app.celery_app:celery_app worker \
  --loglevel=INFO --concurrency=2 --queues=ingestion \
  --hostname=ingestion@%h

# Reconciliation worker: periodic recovery only
celery -A app.celery_app:celery_app worker \
  --loglevel=INFO --concurrency=1 --queues=reconciliation \
  --hostname=reconciliation@%h --max-tasks-per-child=1000
```

Or build the Compose workers:

```bash
docker compose up -d --build worker reconciliation-worker
docker compose logs -f worker
docker compose logs -f reconciliation-worker
```

The queue names are centralized in `app/core/config.py`:

- `CELERY_TASK_DEFAULT_QUEUE` (`ingestion`) carries document ingestion.
- `CELERY_RECONCILIATION_QUEUE` (`reconciliation`) carries periodic recovery.
- The two names must differ; startup rejects a configuration that would collapse
  the isolation.

`app/celery_app.py` routes only `opsatlas.ingestion.reconcile_jobs` to the
reconciliation queue, and the beat schedule publishes with an explicit
`options.queue`, so recovery work cannot land on the ingestion queue. The
dedicated reconciliation worker consumes only `reconciliation`, which keeps
periodic recovery from being starved behind a busy ingestion backlog.

Celery beat runs durable reconciliation every 60 seconds by default; Compose
starts it as the `beat` service. It repairs queued-but-never-published jobs,
crashed workers, and lost publications using PostgreSQL dispatch claims and
execution leases. Run one bounded batch manually with:

```bash
python scripts/reconcile_ingestion_jobs.py --batch-size 50
```

Inspect each worker by node name, and each queue separately:

```bash
celery -A app.celery_app:celery_app inspect ping --destination ingestion@$(hostname)
celery -A app.celery_app:celery_app inspect ping --destination reconciliation@$(hostname)
docker compose exec redis redis-cli -n 1 LLEN ingestion
docker compose exec redis redis-cli -n 1 LLEN reconciliation
```

If the reconciliation queue grows, check that `reconciliation-worker` is healthy
and that all services share the same `CELERY_BROKER_URL` and
`CELERY_RECONCILIATION_QUEUE`, then inspect
`docker compose logs -f reconciliation-worker`. See
`docs/BACKGROUND_INGESTION.md` for orphan states, lease/fencing behavior, retry
policy, health checks, queue diagnosis, and troubleshooting.

## Authentication

Every `/api/v1/*` endpoint requires a configured bearer token:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/answer \
  -H "Authorization: Bearer $OPSATLAS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"question": "When PAY-503 is raised, which checks happen before escalation?"}'
```

The token resolves to an authoritative tenant and principal. A request-body
`tenant_id` is optional and deprecated; when present it must equal the
authenticated tenant or the request returns `403`. Missing or invalid
credentials return `401` with `WWW-Authenticate: Bearer`.

Document ACLs are an allow-list augmentation: documents with no ACL rows are
visible to authenticated members of their tenant, while a document with any ACL
row requires a matching `read`/`manage` grant for the caller's subject, user ID,
group, or role. The schema has no deny rows.

Ingestion writes are stricter than tenant membership. Creating a document, or
updating a document that has no ACL rows, requires the explicit
`document_writer` role in the token configuration. A document with any ACL row
requires a matching `manage` grant instead; a `read` grant, a writer role, or
mere tenant membership cannot override that restriction. Synchronous and
asynchronous ingestion enforce the same rule on every submission, including
duplicates and historical-version reactivation. Background workers complete
already-authorized durable jobs by job ID and never receive user credentials or
tokens through Redis.

`GET /api/v1/documents/jobs/{job_id}` is tenant-scoped. A job linked to a
document follows the document `read`/`manage` ACL; an unlinked job is visible to
authenticated members of its tenant and contains no source key, content, or
execution token. Missing, cross-tenant, and ACL-denied jobs all return the same
`404`.

Token lifecycle is configuration-driven; there is no identity provider or admin
API. An optional ISO-8601 `expires_at` in a token entry stops authentication at
that instant (checked per request, no restart required). To rotate, add the new
token alongside the old one, restart/redeploy the API, move clients, then remove
the old entry and restart again. Removing an entry is revocation. Production
startup rejects an empty map, tokens shorter than 32 characters, documented
placeholder values, and a map with no currently usable entry. Malformed entries
fail startup in every environment, and error messages never include a token.

Request limits are keyed by the authenticated tenant. With
`RATE_LIMIT_BACKEND=redis` (the Compose default) API replicas share one atomic
sliding-window limit per tenant; Redis keys contain only a salted SHA-256 digest
of the tenant UUID. If Redis becomes unreachable, the default fallback keeps
enforcing the limit in each process and logs one sanitized warning; disabling the
fallback returns `503` instead. Bounded concurrency remains per process. See
`docs/ARCHITECTURE.md` and `docs/ENGINEERING_DECISIONS.md` for the exact trust
model.

## Answer a question

`POST /api/v1/answer` retrieves evidence, assembles bounded context, generates a
grounded answer, and validates citations:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/answer \
  -H "Authorization: Bearer $OPSATLAS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "When PAY-503 is raised, which checks happen before escalation?"
  }'
```

The response includes the answer, citations (document ID, source key, title,
version number, chunk ID, excerpt), full evidence metadata, retrieval
diagnostics, generation metadata, abstention indicators, and conflict/uncertainty
signals.

The default `extractive` provider is a deterministic local fallback that copies
the top passage. It requires no API key and is not a production language model.
An OpenAI-compatible provider is opt-in through `.env`; see `.env.example`.

## Repository map

```text
app/
  api/            HTTP route modules (documents, search, answers)
  core/           configuration and cross-cutting concerns
  db/             connections, migration runner, schema contract, readiness
  services/       retrieval, ingestion, evidence, providers, generation
  tasks/          Celery ingestion and reconciliation orchestration
  celery_app.py   Celery broker, serialization, ack, and time-limit settings
  main.py         FastAPI application factory
sql/              immutable versioned migrations (single runner)
evals/
  datasets/       versioned golden retrieval and answer evaluation data
  reports/        measured retrieval reports
  corpus/         synthetic source documents and manifest
docs/             architecture, decisions, migrations, evaluation, status, roadmap
tests/            automated tests
scripts/          local development helpers
```

## Engineering rules

1. Every answer should be traceable to retrieved evidence.
2. Do not log API keys, secrets, or unrestricted document contents.
3. Enforce tenant and document permissions before evidence reaches the LLM.
4. Treat retrieved documents as untrusted data; they may contain prompt injection.
5. Keep model, embedding, prompt, chunking, and retrieval configuration versioned.
6. Never claim a metric without recording the dataset, method, and result.
7. Simulated incidents must be described as simulated, not real customer incidents.

## Milestones

- [x] M1: project scaffold, local services, readiness checks, architecture and evaluation contract
- [x] M2: document ingestion and versioned persistence
- [x] M3: baseline chunking, embeddings, and vector retrieval
- [x] M4: answer generation with citations, abstention, and evidence assembly
- [x] M5 (retrieval): golden-set evaluation and fusion experiments
- [x] M5 (answers): run and score the answer evaluation (three runs, aggregated)
- [x] M6: authenticated tenant/ACL authorization, observability, and adversarial security tests
- [x] M7: async ingestion resilience, dead letters, execution leases, and reconciliation
- [ ] M8: distributed tracing, metrics, load testing, and failure injection (load/failure validation complete; tracing and dashboards open)
- [ ] M9: release gates, runbooks, and incident postmortems (CI, migration validation, and runbooks complete; evaluation gates and postmortems open)

## Current limitations

- Authentication is a local bearer-token map, not an identity provider. Tokens,
  roles, and groups are configured by hand. Optional `expires_at` is enforced per
  request; rotation and revocation are entry edits plus a restart. There is no
  SSO, OAuth, JWT refresh, or online revocation API.
- Document ACLs are allow-list augmentation, not default-deny. A document with no
  ACL rows is visible to every authenticated member of its tenant, and the schema
  has no deny rows.
- The shared Redis rate limiter enforces one per-tenant request limit across API
  replicas. The concurrency slot remains process-local, so N replicas permit up
  to N simultaneous in-flight requests per process. During a Redis outage the
  default fallback degrades to a per-process rate limit, not an unlimited one.
- The default generation provider is a local extractive fallback, not a language model.
- Answer runs are not persisted, and the `query_runs` table is unused.
- `/ready` validates PostgreSQL connectivity, migration history, and the live schema; it is not a workload or end-to-end quality assessment, and Redis-backed background capabilities are optional for API readiness.
- The corpus and evaluation datasets are synthetic. Answer metrics describe this corpus, model, and provider only; they are not production validation.
- Request logs record no prompts, document content, API keys, or raw tenant UUIDs,
  but they are emitted to the process log stream; there is no metrics endpoint,
  tracing, shipping, or automatic log retention. Operational backup, restore,
  retention, and dead-letter tooling is documented in
  `docs/OPERATIONS_RUNBOOK.md`; scheduling backups and log retention remains a
  deployment responsibility.
- The default Compose credentials and empty token map are development-only.
  `APP_ENV=production` rejects them, but real production deployment still needs
  external secret management, TLS ingress, Redis authentication, and private
  networking; those controls are outside this repository.

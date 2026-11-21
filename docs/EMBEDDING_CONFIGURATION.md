# Embedding configuration and schema contract

Phase F, Step 4 makes the embedding dimension a single, explicit contract
instead of a value duplicated across settings, SQL, and scripts.

## Authoritative values

| Concern | Source | Value |
| --- | --- | --- |
| Model | `Settings.embedding_model` | `sentence-transformers/all-MiniLM-L6-v2` |
| Dimension | `app.core.config.DATABASE_EMBEDDING_DIMENSIONS` | `384` |
| Column type | `sql/002_embedding_384.sql` | `chunks.embedding vector(384)` |
| Query cast | `embedding_client.vector_sql_type()` | `vector(384)` |

The model is re-embeddable configuration: changing it causes ingestion to
re-embed chunks and retrieval to filter on the new model name. The dimension is
not freely configurable in this milestone. `Settings` rejects any value other
than `DATABASE_EMBEDDING_DIMENSIONS` before the application can start, because
the PostgreSQL column and the retrieval cast must agree.

## Service address: host and container modes

The embedding service is not part of Compose; it runs on the host. The same
`127.0.0.1` address therefore cannot work from both namespaces, and the client
address is split into two variables:

| Caller | Variable | Default | Why |
| --- | --- | --- | --- |
| Host-run API/worker, migrations, scripts, tests | `EMBEDDING_URL` | `http://127.0.0.1:8091` | Loopback is the host |
| `api` and `worker` Compose services | `CONTAINER_EMBEDDING_URL` | `http://host.docker.internal:8091` | Container loopback is the container |

`docker-compose.yml` sets the container's `EMBEDDING_URL` from
`${CONTAINER_EMBEDDING_URL:-...}`. The host-only `EMBEDDING_URL` is never passed
into a container, so a plain copy of `.env.example` to `.env` resolves correctly
in both modes. The application reads only `EMBEDDING_URL`; the container
variable exists solely for Compose interpolation and is ignored by `Settings`.

The reconciliation worker does not call the embedding provider: it only
dispatches orphaned jobs back to the ingestion queue. It therefore has no
embedding address.

Override `CONTAINER_EMBEDDING_URL` if the service listens on another host port
or on a remote machine reachable from the container network.

## Why no new migration

`chunks.embedding` is already `vector(384)` and all existing vectors are
384-dimensional. An additive migration to an unconstrained `vector` column
would weaken the database's own dimension check, still require explicit
validation in the application, and provide no behavior needed by this
milestone. Keeping `vector(384)` plus explicit validation is the simpler and
safer choice. No embedding-dimension migration is required in Step 4.

If a future model requires a different dimension, that is a deliberate schema
change, not a runtime toggle:

1. Add `sql/006_embedding_<n>.sql` that checks existing vectors and either
   re-embeds or clears them before changing the column type. A cast between
   different populated vector dimensions is not a safe in-place migration.
2. Update `DATABASE_EMBEDDING_DIMENSIONS` and the retrieval/backfill casts.
3. Re-embed or backfill every chunk for the new model.
4. Verify `document_versions` one-current-version state and ingestion jobs are
   preserved.

## Migration order

Apply migrations with the single runner; do not run the files manually on a
database that already has any of them:

```bash
python -m app.db.migrations
python -m app.db.migrations --check
```

The runner applies `sql/001` through `sql/005` deterministically and records
version, filename, checksum, and timestamp in `schema_migrations`.

`002` requires that `chunks` contain no embeddings because it replaces the
provisional `vector(1536)` column with `vector(384)`. If any embeddings exist,
the migration aborts without changing the column or deleting data. The safe
backup and re-embedding procedures are documented in
`docs/DATABASE_MIGRATIONS.md#embedding-dimension-migration`. `003`, `004`, and
`005` are additive and preserve existing data. `005` adds dispatch and
execution-lease state for orphan recovery; it does not change the embedding
contract.

## Mismatch detection

- **Startup/settings:** a non-384 `embedding_dimensions` raises a validation
  error while `Settings` is constructed, before any request is served.
- **Readiness:** `/ready` validates migration history and the live schema,
  including `chunks.embedding = vector(384)`, before the instance is considered
  ready. The worker performs the same check before consuming a task.
- **Live schema drift:** ingestion verifies `pg_attribute.atttypmod` for
  `chunks.embedding` before preparing a target version. A column that is
  unbounded, missing, or a different dimension raises `EmbeddingSchemaError`
  and the API returns a generic 500 without leaking schema details.
- **Provider output:** `embed_texts` checks the response model, declared
  dimensions, vector count, vector length, and numeric finiteness. Ingestion
  validates each batch again before persistence and again before activation.
- **Retrieval:** the query vector is validated against the configured dimension
  before the SQL cast, so a bad or substituted embedding function cannot send a
  mismatched vector to PostgreSQL.

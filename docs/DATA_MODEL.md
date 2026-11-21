# Data Model Notes

The schema is defined by immutable, versioned SQL files under `sql/` and applied by the single migration runner (`python -m app.db.migrations`). Docker Compose no longer loads a bootstrap file through `docker-entrypoint-initdb.d`; the one-shot `migrate` service applies the same migrations as a host run. See `docs/DATABASE_MIGRATIONS.md`.

- `tenants`: organization boundary.
- `users`: identity mapped to a tenant; production authentication provider integration comes later.
- `documents`: stable logical source identity and current lifecycle state.
- `document_versions`: immutable version history, content hash, authority, publication and ingestion times.
- `document_acl`: document-level grants for users/groups/roles. The authorization query implementation must define deny/allow precedence explicitly before real data is used.
- `chunks`: version-bound content segments, metadata, and embedding.
- `ingestion_jobs`: idempotent background work, retry state, and safe error summaries.
- `query_runs`: minimal request telemetry and retrieval references. Avoid storing raw user queries by default; query hashes can support aggregate analysis without retaining full text.

## Important design constraints

1. Every tenant-owned row carries `tenant_id` where practical, and queries must enforce tenant boundaries explicitly.
2. Chunks belong to a specific document version. Never silently mutate old chunks into a new version.
3. Only switch `documents.current_version_id` after the replacement version has been fully processed.
4. The embedding dimension is fixed at `384` (`vector(384)`) by migration 002 and the application contract. Only 384-dimensional embeddings are currently supported.
5. Do not run `sql/*.sql` files manually on a migrated database. Use the migration runner, which verifies existing state, records checksums, and fails closed on partial or changed migrations.

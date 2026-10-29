# Data Model Notes

The first SQL schema is a proposal, not a final migration system. It is loaded when the local PostgreSQL data volume is created for the first time.

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
4. The embedding dimension `1536` is provisional. Confirm the chosen embedding model's dimensions before generating embeddings; change schema/migration if needed.
5. The SQL bootstrap file is suitable only for a fresh local database. Later schema changes must use versioned migrations (planned: Alembic).

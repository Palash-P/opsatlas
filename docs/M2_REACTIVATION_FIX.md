# OpsAtlas Milestone 2 fix: reactivate historical content

This patch replaces `app/services/ingestion.py`.

## Behavior

The schema has `UNIQUE (document_id, content_hash)`, so a document has one stored snapshot per distinct content hash. If submitted content matches the current version, ingestion is an idempotent no-op. If it matches a historical version, ingestion reactivates that existing immutable snapshot and updates `documents.current_version_id`; it does not create a duplicate version.

This keeps the existing schema and content-deduplication contract. If the product later needs every revert event represented as a new chronological revision (for example version 3 containing the same hash as version 1), we should remove the per-document hash uniqueness in a migration and add a separate ingestion/audit event history.

## Apply

Extract this patch into the project root, preserving the `app/services/ingestion.py` path. Do not delete the PostgreSQL volume.

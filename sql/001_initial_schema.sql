-- Initial relational schema proposal. Applied automatically only on first creation
-- of the local Docker volume. Use versioned migrations for future schema changes.
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS tenants (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    external_subject TEXT NOT NULL,
    email TEXT,
    role TEXT NOT NULL DEFAULT 'member' CHECK (role IN ('admin', 'member', 'viewer')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (tenant_id, external_subject)
);

CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    source_key TEXT NOT NULL,
    title TEXT NOT NULL,
    source_type TEXT NOT NULL,
    current_version_id UUID,
    status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'deleted', 'processing', 'failed')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (tenant_id, source_key)
);

CREATE TABLE IF NOT EXISTS document_versions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    version_number INTEGER NOT NULL CHECK (version_number > 0),
    content_hash TEXT NOT NULL,
    source_uri TEXT,
    authority_level SMALLINT NOT NULL DEFAULT 50 CHECK (authority_level BETWEEN 0 AND 100),
    published_at TIMESTAMPTZ,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    is_current BOOLEAN NOT NULL DEFAULT FALSE,
    UNIQUE (document_id, version_number),
    UNIQUE (document_id, content_hash)
);

ALTER TABLE documents
    ADD CONSTRAINT documents_current_version_fk
    FOREIGN KEY (current_version_id) REFERENCES document_versions(id)
    DEFERRABLE INITIALLY DEFERRED;

CREATE TABLE IF NOT EXISTS document_acl (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    principal_type TEXT NOT NULL CHECK (principal_type IN ('user', 'group', 'role')),
    principal_id TEXT NOT NULL,
    permission TEXT NOT NULL CHECK (permission IN ('read', 'manage')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (document_id, principal_type, principal_id, permission)
);

CREATE TABLE IF NOT EXISTS chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    document_version_id UUID NOT NULL REFERENCES document_versions(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL CHECK (chunk_index >= 0),
    content TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    token_count INTEGER,
    section_path TEXT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    embedding VECTOR(1536),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (document_version_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS chunks_tenant_document_idx ON chunks (tenant_id, document_id);
CREATE INDEX IF NOT EXISTS chunks_metadata_gin_idx ON chunks USING GIN (metadata);
CREATE INDEX IF NOT EXISTS chunks_content_fts_idx ON chunks USING GIN (to_tsvector('english', content));
-- Add a vector index only after selecting an embedding model and measuring corpus size.
-- The vector dimension above is a provisional contract and must match the chosen model.

CREATE TABLE IF NOT EXISTS ingestion_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    document_id UUID REFERENCES documents(id) ON DELETE SET NULL,
    idempotency_key TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'queued' CHECK (status IN ('queued', 'running', 'succeeded', 'failed', 'dead_letter')),
    attempts INTEGER NOT NULL DEFAULT 0 CHECK (attempts >= 0),
    error_code TEXT,
    error_summary TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    started_at TIMESTAMPTZ,
    finished_at TIMESTAMPTZ,
    UNIQUE (tenant_id, idempotency_key)
);

CREATE TABLE IF NOT EXISTS query_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    request_id TEXT NOT NULL UNIQUE,
    config_version TEXT NOT NULL,
    query_hash TEXT NOT NULL,
    retrieved_chunk_ids UUID[] NOT NULL DEFAULT '{}',
    latency_ms INTEGER,
    input_tokens INTEGER,
    output_tokens INTEGER,
    estimated_cost_usd NUMERIC(12, 8),
    outcome TEXT NOT NULL DEFAULT 'completed' CHECK (outcome IN ('completed', 'abstained', 'failed')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS documents_tenant_status_idx ON documents (tenant_id, status);
CREATE INDEX IF NOT EXISTS versions_document_current_idx ON document_versions (document_id, is_current);
CREATE INDEX IF NOT EXISTS jobs_status_created_idx ON ingestion_jobs (status, created_at);
CREATE INDEX IF NOT EXISTS query_runs_tenant_created_idx ON query_runs (tenant_id, created_at DESC);

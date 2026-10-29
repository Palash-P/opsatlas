# OpsAtlas Architecture (v1)

## Product boundary

OpsAtlas answers engineering questions using approved internal knowledge: runbooks, service/API docs, incident reports, and postmortems. It is read-only in v1. It must not execute production remediation actions.

## Initial architecture

- **API layer:** FastAPI; request validation, auth boundary (to be implemented), query orchestration, health/readiness.
- **Ingestion worker:** Celery + Redis (next milestone); parse, normalize, chunk, embed, persist, and reconcile documents.
- **Persistence:** PostgreSQL with pgvector; tenant-aware relational metadata, document versions, ACLs, chunks, and query audit metadata.
- **Retrieval:** metadata/permission filtering before candidate text is passed to reranking or generation; hybrid lexical + dense retrieval; optional reranking after baseline measurement.
- **Generation:** provider abstraction, bounded timeouts/retries, evidence-only context, citations, and abstention for insufficient evidence.
- **Evaluation:** versioned golden set; retrieval metrics and answer-level checks run in tests/CI.
- **Observability:** request IDs and stage timings; redact secrets and avoid unrestricted prompt/document logging.

## Query path

1. Authenticate caller and resolve tenant/principals.
2. Validate query and apply request/rate limits.
3. Retrieve candidates with tenant and document ACL constraints applied in the database query.
4. Rerank only authorized candidates.
5. Assemble bounded evidence context with source/version identifiers.
6. Generate a grounded answer or abstain.
7. Validate citations and record request-level telemetry.
8. Return answer and citations; collect explicit feedback separately.

## Ingestion path

1. Accept a source and idempotency key.
2. Store ingestion job and document lifecycle status.
3. Fetch/parse/normalize source; validate type and size limits.
4. Calculate content hash and compare with known versions.
5. Chunk deterministically and generate embeddings.
6. Write a new version and chunks without destroying the last known-good version.
7. Atomically switch the current version after successful processing.
8. Mark job complete; reconcile stuck jobs and orphaned records asynchronously.

## Trust and safety boundaries

- Retrieved content is data, never trusted instruction.
- Do not allow retrieved documents to override system/developer instructions or invoke tools.
- No autonomous actions against external systems in v1.
- Apply authorization before reranking/generation; citations and cache keys must respect the same boundary.
- Store secrets in environment/configuration management; never log them.
- Keep synthetic test data separate from any future real data.

## Initial service objectives (provisional)

These are hypotheses to validate, not claims of achieved performance:

- 0 unauthorized chunk exposures in the security test suite.
- Every non-abstaining answer has citations that resolve to authorized source versions.
- Retrieval evaluation runs on every change to chunking, embedding, filtering, or ranking.
- API and ingestion failures are observable by request/job IDs.
- Measure p50/p95 latency and estimated cost per successful answer under a documented workload before setting performance SLOs.

## Deliberately deferred

Kubernetes, microservices, multi-agent orchestration, autonomous remediation, paid hosting, and multi-modal ingestion are deferred until measured needs justify them.

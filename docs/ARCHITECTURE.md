# OpsAtlas Architecture (v1)

## Product boundary

OpsAtlas answers engineering questions using approved internal knowledge: runbooks, service/API docs, incident reports, and postmortems. It is read-only in v1. It must not execute production remediation actions.

## Initial architecture

- **API layer:** FastAPI; request validation, bearer-token authentication, server-side tenant resolution, query orchestration, bounded request limits, health/readiness.
- **Ingestion worker:** Celery + Redis (next milestone); parse, normalize, chunk, embed, persist, and reconcile documents.
- **Persistence:** PostgreSQL with pgvector; tenant-aware relational metadata, document versions, ACLs, chunks, and query audit metadata.
- **Retrieval:** metadata/permission filtering before candidate text is passed to reranking or generation; hybrid lexical + dense retrieval; optional reranking after baseline measurement.
- **Generation:** provider abstraction, bounded timeouts/retries, evidence-only context, citations, and abstention for insufficient evidence.
- **Evaluation:** versioned golden sets for retrieval and a design-only answer evaluation; retrieval metrics run in tests/CI, answer metrics await a scored run.
- **Observability:** request IDs and stage timings; redact secrets and avoid unrestricted prompt/document logging.

## Query path

1. Authenticate the bearer token and resolve the authoritative tenant/principal.
   A supplied body tenant is an equality assertion only.
2. Validate the query and apply per-tenant rate/concurrency limits.
3. Retrieve candidates with tenant and document ACL constraints applied in the database query.
4. Rerank only authorized candidates. _(deferred)_
5. Assemble bounded evidence context with source/version identifiers.
6. Generate a grounded answer or abstain.
7. Validate citations against server-side evidence and emit one redacted request log.
8. Return answer and citations; collect explicit feedback separately.

## Answer-generation slice (implemented)

`POST /api/v1/answer` is the first complete vertical slice. It composes existing
retrieval with three new modules and does not change retrieval ranking or
defaults.

### Modules

| Module | Responsibility |
| --- | --- |
| `app/services/evidence.py` | Deterministic assembly: preserve ranking and metadata, dedupe identical chunks, keep distinct passages, apply a context budget, label every passage with its version, assign `E1...En` IDs |
| `app/services/llm_providers.py` | `LLMProvider` protocol, local `extractive` fallback, opt-in `openai_compatible` adapter, typed provider errors |
| `app/services/answer_generation.py` | Versioned prompt, strict JSON contract, bounded retries for transient failures, citation/conflict validation |
| `app/api/answers.py` | Request/response schemas, retrieval dispatch, error mapping, diagnostics |

### Evidence contract

Every supplied item carries document ID, version ID, source key, title, version
number, chunk ID, chunk index, authority level, retrieval rank, and scores. The
rendered context labels each block with source key, title, version, chunk ID, and
chunk index. If two versions of one source key are present, the bundle sets
`mixed_versions=true` and the response lists every `(source_key, version_number)`
pair.

Duplicate chunk IDs are dropped. Identical content within the same document
version is dropped. Distinct chunks from the same document are retained.
Ordering is the input retrieval order, so it is deterministic. Document content
is HTML-escaped inside the evidence block, so a retrieved document cannot forge
a closing tag or inject a new pseudo-evidence block.

The context budget uses `ceil(len(text) / 4)` as a documented token estimate. It
is a safety bound, not exact model accounting.

### Generation contract

The system prompt requires evidence-only answers, separation of facts from
uncertainty, no invented commands or procedures, explicit conflict
identification, and abstention when evidence is insufficient. Evidence is
treated as untrusted data; instructions inside an evidence block must not be
followed.

The model must return:

```json
{
  "answer": "string",
  "used_evidence_ids": ["E1"],
  "abstained": false,
  "abstention_reason": null,
  "insufficient_evidence": false,
  "conflicts": [{"evidence_ids": ["E1", "E2"], "description": "string"}],
  "uncertainty": null
}
```

The server rejects malformed output, non-boolean flags, a non-abstaining answer
without citations, unknown evidence IDs in citations or conflicts, and missing
conflict descriptions. Citations are constructed from server-side evidence
metadata, never from model-supplied document fields.

Timeouts, rate limits, and retryable provider errors may be retried at most
`LLM_MAX_RETRIES` times (default 1). Invalid output and citation violations are
not retried.

### Provider configuration

- `LLM_PROVIDER=extractive` (default): local, deterministic, no key. Extracts the
  top passage and labels itself as a fallback. Not a production-quality model.
- `LLM_PROVIDER=openai_compatible`: opt-in; requires `LLM_API_BASE`,
  `LLM_API_KEY`, and `LLM_MODEL`. The key is never logged or placed in error
  text.

No paid provider is required for local development or tests.

### API response

The response includes `answer`, `citations` (document ID, version ID, source key,
title, version number, chunk ID, chunk index, excerpt), full `evidence`
metadata, retrieval diagnostics, generation metadata (provider, model, prompt
version, latency, attempts, token estimates), abstention indicators, conflicts,
uncertainty, and dropped-evidence diagnostics.

### Error mapping

| Condition | Status |
| --- | --- |
| Blank question or invalid retrieval option | 422 |
| Embedding service unavailable | 503 |
| Provider configuration invalid | 503 |
| Provider timeout | 504 |
| Provider rate limit | 429 |
| Provider failure | 502 |
| Invalid model output | 502 |
| Citation references an unsupplied chunk | 502 |

### Non-goals in this slice

No agents, no microservices, no schema migration, no persisted query runs, no
streaming, and no identity provider. Authentication uses a local token map; the
endpoint remains stateless.

## Security and request boundary (Phase D)

### Trust model

- A request is authenticated only by `Authorization: Bearer <token>`.
- `AUTH_TOKENS` maps each opaque token to `tenant_id`, `subject`, optional
  `user_id`, `roles`, and `groups`. Tokens are compared by SHA-256 digest with a
  constant-time comparison. The token itself is never logged or returned.
- The token's tenant is authoritative. A request-body `tenant_id` is optional and
  deprecated; if present and different, the request fails with `403`.
- A missing or invalid token returns `401 {"detail":"Unauthorized"}` with
  `WWW-Authenticate: Bearer`. A tenant mismatch or forbidden action returns the
  same `403 {"detail":"Forbidden"}` regardless of whether the target tenant
  exists, so the response does not disclose tenant existence.

### Tenant and ACL enforcement

Every retrieval query applies both `chunks.tenant_id` and `documents.tenant_id`.
Retrieval then re-checks each returned row and raises `TenantIsolationError` if a
row's `tenant_id` differs from the requested tenant (defense in depth). This check
is skipped when a row omits `tenant_id`, which only occurs in older unit fixtures.

`document_acl` is an allow-list augmentation:

- a document with no ACL rows is visible to authenticated members of its tenant;
- a document with any ACL row requires a `read` or `manage` grant whose
  `principal_type`/`principal_id` matches the caller's subject or user ID, group,
  or role;
- the schema has no deny rows, inheritance, or default-deny behavior.

The tenant predicate is unconditional. ACL augmentation is applied whenever a
principal is supplied; every HTTP route supplies the authenticated principal,
while direct evaluation/service callers may omit it. Do not add an unauthenticated
HTTP path that omits the principal.

Citations, version-analysis entries, and conflicts are validated against the
server-assembled evidence bundle. The model may name an evidence ID only; it
cannot supply document metadata, a source key, or a version number.

### Observability

One JSON log line per request is emitted on the `opsatlas.request` logger with:

`request_id`, `method`, `path`, `status_code`, salted `tenant_ref`, retrieval
strategy, retrieval/evidence/generation/total latency, provider, observed and
requested model, retry count, safe `error_category`, input/output tokens, and
`token_source` (`actual`, `estimated`, or `mixed`).

The log never contains the Authorization header, API key, request body, query,
prompt, retrieved document content, exception text, or raw tenant UUID. A
client-supplied `X-Request-ID` is reused only if it matches a conservative
character pattern; otherwise a new ID is generated and returned.

Phase E validated these fields end-to-end across 9 load scenarios and 12 failure
cases: request IDs correlated 1:1, `tenant_ref` always salted, forbidden-value
hits were zero, and every log line parsed as JSON. One defect was fixed: a
non-retryable provider response error now records `retry_count: 0` rather than
`null`.

### Limits and failure behavior

| Control | Behavior |
| --- | --- |
| Rate limit | Per authenticated tenant, sliding window, `429` + `Retry-After` |
| Concurrency | `BoundedSemaphore`, `429` + `Retry-After` when no slot is available |
| Provider timeout/retry | Bounded by settings; retries only timeout/rate-limit/retryable errors |
| DB pool | Bounded pool, pre-ping, pool timeout, recycle, connect and statement timeouts |
| Redis | Optional for serving; `/ready` returns `200 degraded` on Redis-only failure |
| PostgreSQL | Required; `/ready` returns `503 not_ready` on failure |

Rate and concurrency limits are process-local, not a distributed quota. Running
multiple processes multiplies the effective limit. Provider and DB failures map
to constant, non-leaking error details.

Measured failure latency and recovery (one local worker, Phase E; details in
`docs/OPERATIONAL_VALIDATION.md`):

| Failure | Observed behavior | Detection cost |
| --- | --- | --- |
| PostgreSQL stopped | `/health` `200`; `/ready` `503 not_ready`; answer `503 database_unavailable` | ~10 s (connect timeout) |
| PostgreSQL restarted | `/ready` `200`; answer `200` without API restart | ~0.1 s after restart |
| Redis stopped | `/ready` `200 degraded`; answer still `200` | ~2.1 s (two 1 s socket timeouts) |
| Embedding stopped | Semantic/hybrid search and semantic answer `503 embedding_unavailable`; lexical answer `200` | ~4.3 s (one retry) |
| Provider timeout | `504 provider_timeout`, one retry | bounded by `LLM_TIMEOUT_SECONDS` |
| Provider `429` | `429 provider_rate_limited`, one retry | immediate |
| Invalid envelope/output, bad citation | `502`, no retry | immediate |
| DB pool exhaustion | Contending request `503 database_unavailable`; queued request succeeds after the lock clears | pool timeout (1 s in the test) |
| Client disconnect | Server continues to completion and still consumes a concurrency slot | not cancelled |

Readiness probes must allow for the PostgreSQL connect timeout: a 5 s probe would
time out before the `503` body is written. A 15 s probe timeout is recommended.

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

- 0 unauthorized chunk exposures in the security test suite. (Verified offline in
  Phase D; the live seeded-tenant test is opt-in. Phase E added concurrent
  tenant/ACL probes with 0 canary disclosures and correct allow/deny behavior.)
- Every non-abstaining answer has citations that resolve to authorized source versions.
- Retrieval evaluation runs on every change to chunking, embedding, filtering, or ranking.
- API and ingestion failures are observable by request/job IDs.
- Measure p50/p95 latency and estimated cost per successful answer under a documented workload before setting performance SLOs.

Local Phase E measurements on one worker with the `extractive` provider (not a
production SLO): answer p95 105 ms single-user and 236 ms at 8-way concurrency;
semantic search p95 228 ms; hybrid search p95 221 ms; `/health` p95 8 ms;
`/ready` p95 40 ms. The embedding service was the bottleneck at about 4.1 cores
and 1.8 GiB under mixed 8-way load; the API process stayed below 30% of one core
and 95 MiB. Real-model latency and cost remain unmeasured. See
`docs/OPERATIONAL_VALIDATION.md` for the full tables, failure cases, and starting
alert thresholds.

## Deliberately deferred

Kubernetes, microservices, multi-agent orchestration, autonomous remediation, paid hosting, and multi-modal ingestion are deferred until measured needs justify them.

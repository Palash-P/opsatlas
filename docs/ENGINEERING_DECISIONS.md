# Engineering Decisions

This document records decisions that constrain the implementation. It is an
append-only decision log; later entries may supersede earlier ones explicitly.

## ADR-001: Keep retrieval as the existing raw-SQL baseline

**Status:** Accepted (Phase B)

The answer path reuses `hybrid_search`, `semantic_search`, and `lexical_search`
unchanged. No retrieval ranking, SQL, or default was rewritten. The answer API
exposes retrieval options but defaults to the existing hybrid configuration:
`keyword_strategy="and"`, `fusion_strategy="chunk"`, equal weights `1.0`,
`rrf_k=60`, and `limit=5`.

**Reason:** retrieval has measured baselines (`golden_v0` through `golden_v3`).
Changing it during the generation slice would confound answer-quality work and
invalidate the baselines.

## ADR-002: Evidence assembly is a separate deterministic layer

**Status:** Accepted

`app/services/evidence.py` converts ranked rows into an `EvidenceBundle`. It:

- preserves the input ranking rather than re-sorting;
- preserves document ID, version ID, source key, title, version number, chunk ID,
  chunk index, authority level, and retrieval scores;
- drops duplicate chunk IDs and identical content within the same document
  version;
- keeps distinct chunks from the same document;
- applies a context-token budget and a maximum evidence count;
- assigns stable `E1...En` identifiers in supplied order;
- renders XML-like evidence blocks that state source key, title, version, chunk
  ID, and chunk index;
- flags `mixed_versions` when more than one version of a source key is supplied.

**Reason:** generation and citation validation need one structured source of
truth. Keeping it deterministic makes tests and future evaluation reproducible.

## ADR-003: Token budgeting is an approximation, not a tokenizer

**Status:** Accepted

`estimate_tokens` uses `ceil(len(text) / 4)`. This avoids adding a tokenizer
dependency and is documented as an approximation.

**Consequence:** the budget is a safety limit, not exact model accounting. A
future evaluation may replace it with provider-reported usage or a real
tokenizer.

## ADR-004: Provider abstraction with a local deterministic default

**Status:** Accepted

`app/services/llm_providers.py` defines an `LLMProvider` protocol and two
adapters:

- `extractive` (default): deterministic, local, no API key. It extracts the
  top-ranked passage and labels itself as a fallback. It is not a language model
  and must not be presented as production-quality generation.
- `openai_compatible`: opt-in HTTP adapter. Base URL, path, model, and key come
  from settings. Keys are never logged or included in error messages.

Provider-specific code stays in this module. Business logic depends only on the
protocol.

**Reason:** the vertical slice must be runnable and testable locally without a
paid provider, while leaving a real provider path available.

## ADR-005: Strict JSON model contract and server-side citation validation

**Status:** Accepted

The model must return one JSON object with `answer`, `used_evidence_ids`,
`abstained`, `abstention_reason`, `insufficient_evidence`, `conflicts`, and
`uncertainty`. The server:

- rejects malformed JSON and invalid field types;
- rejects a non-abstaining answer with no cited evidence;
- rejects any evidence ID not present in the supplied bundle;
- builds citations only from validated `used_evidence_ids`;
- rejects conflict entries that reference unknown evidence IDs.

**Reason:** the model must never be able to invent a document, version, or chunk.
Citations are derived from server-side evidence metadata, not from model text.

## ADR-006: Bounded retries only for transient provider failures

**Status:** Accepted

Timeouts, rate limits, and retryable provider errors may be retried at most
`LLM_MAX_RETRIES` times (default `1`) with exponential backoff. Invalid model
output and citation violations are never retried.

**Reason:** retries help with transient transport failures but cannot fix a
model that violated the answer contract. Unbounded retries would add latency and
cost without improving correctness.

## ADR-007: No schema migration or persisted query-run table in this slice

**Status:** Accepted

The endpoint is stateless. The existing `query_runs` table is not written. No
schema migration was added.

**Reason:** the slice can be evaluated through API responses and tests. Adding a
run-history schema before defining retention, tenancy, and audit requirements
would create premature commitments.

## ADR-008: No new runtime dependencies

**Status:** Accepted

The provider adapter uses the Python standard library (`urllib`, `json`,
`socket`). Tests use existing `pytest`, `fastapi.testclient`, and `httpx`.

**Reason:** the implementation does not justify a new SDK or HTTP client. It also
avoids making a specific vendor mandatory.

## ADR-009: Answer evaluation is design-only until actually run

**Status:** Accepted

`evals/datasets/answer_eval_v1.jsonl` and the report schema define a future
evaluation. No answer metrics have been measured. The report template is
explicitly `status: "not_run"` and all metric values are `null`.

**Reason:** claiming relevance, faithfulness, or citation quality without a run
would violate the project rule against unmeasured claims.

## ADR-010: Synthetic evidence is labeled as synthetic

**Status:** Accepted

The corpus is synthetic. Documentation and responses must not describe it as
production data or claim production validation. The extractive provider's output
includes an explicit uncertainty note, and the answer-eval limitations state the
same.

## ADR-011: Authentication is a configured bearer-token map

**Status:** Accepted (Phase D)

Every protected route depends on `require_identity`. Tokens come from the
`AUTH_TOKENS` setting and map to a tenant, subject, optional user ID, roles, and
groups. Tokens are compared as SHA-256 digests with `hmac.compare_digest`, and
malformed entries never authenticate. Missing or invalid credentials return a
constant `401`; tenant mismatches and forbidden actions return a constant `403`.

**Reason:** the project needs an enforceable request identity without adding an
external identity provider, runtime dependency, or schema migration.

**Consequences and limits:** this is a local deployment boundary. There is no
token expiry, rotation API, revocation list, SSO, or persisted user table. Roles
and groups are asserted by configuration, not by an authorization service. A
real deployment must replace this map with a managed identity/secret boundary.

## ADR-012: The authenticated tenant is authoritative

**Status:** Accepted (Phase D)

The tenant resolved from the bearer token always wins. A request-body `tenant_id`
is optional and deprecated; when supplied it must equal the authenticated tenant
or the request fails before retrieval. Forged tenants return the same generic
`403` as any other forbidden action, so the response cannot be used to test
whether another tenant exists.

**Reason:** a body field is caller-controlled and cannot be proof of
authorization. Keeping the field temporarily preserves compatibility with the
existing API and tests while removing it as a trust input.

**Consequence:** ingestion can only write into the authenticated tenant. Clients
that relied on supplying an arbitrary tenant now receive `403`.

## ADR-013: Document ACLs are an allow-list augmentation, not default-deny

**Status:** Accepted (Phase D)

Every retrieval query applies the tenant predicate first, then:

- documents with no ACL rows remain visible to authenticated tenant members;
- documents with any ACL row require a `read` or `manage` grant matching the
  caller's subject/user ID, group, or role;
- there is no deny row, inheritance, or deny-overrides-allow branch.

A post-query `TenantIsolationError` rejects any returned row whose `tenant_id`
differs from the requested tenant. Citations and version-analysis entries are
validated against the canonical server-side `source_key` and `version_number`,
not against model-supplied metadata.

**Reason:** the `document_acl` table already existed. Enforcing it in retrieval
closes the cross-tenant path without inventing deny semantics or a new schema.

**Consequences and limits:** a document that should be restricted must have at
least one ACL row; absence means tenant-wide visibility. Default-deny, deny
rows, and inheritance require an explicit schema/migration decision. The tenant
predicate is unconditional, but ACL augmentation only runs when a principal is
supplied: HTTP routes always do, while evaluation and direct service callers may
not. The defense-in-depth row check is intentionally skipped when an older unit
fixture omits `tenant_id`; production retrieval projects it.

## ADR-014: Request telemetry is one redacted JSON log line

**Status:** Accepted (Phase D)

Each request emits one JSON object on the `opsatlas.request` logger. Fields are
request ID, method, path, status, salted tenant reference, retrieval strategy,
stage and total latencies, provider, observed/requested model, retry count, safe
error category, input/output tokens, and token source.

The log intentionally omits the Authorization header, API keys, request body,
query text, prompts, retrieved content, exception messages, and raw tenant UUIDs.
A client `X-Request-ID` is accepted only when it matches a conservative pattern.

**Reason:** a failed request must be diagnosable by request ID without turning
logs into a copy of document content or secrets.

**Consequences and limits:** token counts are `actual` only when the provider
reports usage; otherwise they are the documented character-based estimate
(`estimated`) or a mix. Log retention, shipping, sampling, alerting, and
distributed tracing are not implemented.

## ADR-015: Rate/concurrency limits are in-process; readiness distinguishes required and optional dependencies

**Status:** Accepted (Phase D)

Rate limiting uses a per-tenant in-process sliding window. Concurrency uses a
process-wide bounded semaphore. Both return `429` with `Retry-After` when
exceeded. Redis is optional for serving: `/ready` returns `200 degraded` when
only Redis fails and `503 not_ready` when PostgreSQL fails.

**Reason:** the serving path does not use Redis. Removing an API instance from
rotation because a non-serving dependency is down would reduce availability
without improving correctness, while a database failure must fail readiness.

**Consequences and limits:** N processes allow up to N times the configured rate,
and limiter state resets on restart. A distributed quota needs Redis or another
shared store, which Phase D did not add.

## ADR-016: Phase D adds no schema migration and keeps retrieval defaults

**Status:** Accepted (Phase D)

Phase D used the existing `document_acl` table and added no migration, no
destructive database operation, and no change to retrieval ranking or default
configuration. The previously measured `golden_v0`–`golden_v3`, `answer_eval_v1`,
and all reports remain untouched.

**Reason:** the security controls are query predicates, request plumbing, and
telemetry. Changing retrieval defaults at the same time would confound the
measured baseline and invalidate comparisons.

**Consequence:** no new retrieval/answer benchmark was produced for Phase D. The
existing metrics remain the quality baseline; the authorization controls have
not been performance-tested.

## ADR-017: Phase E validates one local worker, not a deployment

**Status:** Accepted (Phase E)

Operational validation runs one uvicorn worker on the development machine with
the local embedding service and local PostgreSQL/Redis containers. Load is
bounded (30–100 requests per scenario, concurrency 1–8) and every run restores
containers, the embedding service, temporary tenants, and database counts.

**Reason:** the existing limits are process-local, and the project has no
deployment, ingress, or multi-node environment. Measuring one worker gives a
reproducible baseline without implying production capacity.

**Consequence:** throughput, latency, and rate/concurrency ceilings are per
process and per host. N workers multiply the effective limits and change
saturation. Multi-instance behavior, load balancing, failover, and shared rate
limiting remain unmeasured.

## ADR-018: Provider-independent load numbers are separated from provider failure tests

**Status:** Accepted (Phase E)

Load scenarios use the in-process `extractive` provider so that retrieval and
server overhead dominate and runs are repeatable. Timeout, `429`, invalid
envelope, invalid output, citation, and `5xx` paths use a local deterministic
OpenAI-compatible fake provider.

**Reason:** using a real model for load would mix model latency, upstream rate
limits, cost, and network variance into every measurement, and would not be
repeatable. Using `extractive` for the failure paths cannot exercise the
OpenAI-compatible adapter or its retry mapping.

**Consequence:** the reported answer latency excludes model inference and token
cost. `generation_latency_ms` was 0 ms under `extractive`, so the numbers are
retrieval + evidence + server overhead only. Real-provider latency and cost must
be measured separately before setting answer SLOs.

## ADR-019: Resource sampling aggregates the Windows process family

**Status:** Accepted (Phase E)

On this machine, a Python service launched through the venv appears as a
trampoline parent plus a worker child with identical command lines. The sampler
resolves descendants whose executable name starts with `python` and reports
aggregate CPU and working set per service label.

**Reason:** sampling only the root PID reported near-zero CPU and about 4 MiB
for the embedding service, which contradicted the observed load behavior. The
child holds the model and the work.

**Consequence:** CPU percentages are relative to one core and can exceed 100%
(the embedding service measured about 414% under 8-way load). The sampler is
local and coarse; it is not an APM agent, and non-Windows platforms report no
process samples.

## ADR-020: Client cancellation is not propagated to server work

**Status:** Accepted (Phase E, documented limitation)

The API uses synchronous retrieval and provider calls. When a client times out or
disconnects, the server continues the request and logs a completed response. In
the measured case the client observed a `ReadTimeout` at 0.25 s while the server
returned `200` after about 1.52 s, consuming a concurrency slot for that time.

**Reason:** the current endpoint has no cancellation token threaded through
retrieval, evidence assembly, and generation. Adding one would change request
lifecycle behavior and is outside an operational-validation phase.

**Consequence:** disconnected clients can still consume concurrency and provider
budget. Do not describe the endpoint as cancellable. A later change should either
propagate cancellation or bound total work so abandoned requests cannot exhaust
slots.

## ADR-021: Readiness timeouts must exceed dependency connect timeouts

**Status:** Accepted (Phase E)

Failure injection showed that a stopped PostgreSQL produces a `503` only after
the configured connect timeout (about 10 s here), and a stopped Redis after about
2 s of socket timeouts. The recommended `/ready` probe timeout is therefore 15 s
with a period of 30 s and a failure threshold of 2–3.

**Reason:** a probe shorter than the dependency timeout would report a probe
timeout instead of the intended `503 not_ready` body, and could restart a healthy
process that is merely waiting on the database timeout.

**Consequence:** readiness detection is deliberately slower than liveness.
`/health` remains dependency-free and fast. Alerting should use the parsed
`/ready` status, not only the probe's transport success.

## ADR-022: Non-retryable generation failures record a retry count

**Status:** Accepted (Phase E)

`generate_grounded_answer` now attaches the attempt count to any `GenerationError`
that escapes the retry loop, not only to timeout, rate-limit, and retryable
provider errors. Non-retryable invalid-envelope, invalid-output, citation, and
configuration failures therefore log `retry_count: 0` instead of `null`.

**Reason:** `null` conflated "unknown" with "zero retries", so a failure log could
not distinguish a single non-retried attempt from missing instrumentation. The
retry policy itself was already correct.

**Consequence:** telemetry-only change; status mapping, retry counts for retried
errors, and API contracts are unchanged. The offline test
`test_invalid_provider_envelope_is_not_retried` locks the behavior.

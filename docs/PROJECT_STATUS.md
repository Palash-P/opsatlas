# Project Status

> Last updated: 2026-11-18. This status describes the repository, not a
> production deployment. The corpus is synthetic.

## Current status

The engineering slice is feature-complete for local release preparation:
synchronous and asynchronous ingestion, versioned PostgreSQL storage,
semantic/lexical/hybrid retrieval, grounded answer generation, bearer-token
authentication with optional per-request expiry, ACL-scoped retrieval, a shared
Redis-backed per-tenant rate limiter, background Celery processing with bounded
retries, dead letters, execution leases, and periodic reconciliation, a
checksum-verified migration runner, operational backup/restore/retention
tooling, and CI covering lint, the offline suite, a PostgreSQL + Redis
integration job, and a container build with locked dependencies.

This document separates **completed engineering work** (implemented and covered
by the test suite) from **production deployment prerequisites** (operator work
this repository does not provide). It is not a production readiness claim.

Phase E remains the most recent measured operational validation. One uvicorn
worker, the seeded synthetic tenant, and the local embedding service were
measured under single-user, concurrent, multi-tenant, near-rate-limit,
over-concurrency, and mixed workloads. PostgreSQL, Redis, the embedding service,
the LLM provider, the DB pool, and client cancellation were fault-injected in a
controlled, fully restored run. Results, thresholds, and limitations are in
`docs/OPERATIONAL_VALIDATION.md`; artifacts are in
`evals/ops/phase-e-final/`.

Key measured results (one worker, `extractive` provider, synthetic corpus):

- 740 measured load requests, 0 errors.
- Answer p95: 105 ms single-user, 236 ms at 8-way concurrency.
- Semantic search p95: 228 ms; hybrid search p95: 221 ms; readiness p95: 40 ms.
- Tenant A never saw tenant B's canary (0/20); ACL-restricted content was
  visible only to the allowed group (20/20 vs 0/20).
- Rate limiting allowed 20/minute then returned `429` with `Retry-After`;
  concurrency limiting allowed 1 in-flight request and rejected 7 of 8 with
  `429`; both stayed isolated per tenant/process.
- Every injected failure mapped to its documented status and error category,
  produced one correlated redacted log event, and recovered without restarting
  the API. PostgreSQL loss was detected at about 10 s; Redis loss degraded
  readiness in about 2 s; embedding loss returned `503` after about 4.3 s.
- The embedding service is the local bottleneck (about 4.1 cores and 1.8 GiB
  under 8-way load); the API stayed below 30% of one core and 95 MiB.

The answer vertical slice itself is unchanged:

```
POST /api/v1/answer
  -> existing retrieval (hybrid by default)
  -> version-aware lexical value + evidence assembly (dedupe, budget, labels)
  -> provider-independent generation (bounded retries, capability reporting)
  -> strict JSON contract with version_analysis
  -> server-side citation and version-reference validation
  -> response with citations, conflicts, and diagnostics
```

## What works

- Local FastAPI service, PostgreSQL + pgvector, Redis, and the embedding service.
- Document ingestion and versioned persistence used by the evaluation fixtures.
- Semantic, OR+IDF lexical, and hybrid retrieval with opt-in fusion strategies.
- Version-aware retrieval: explicit version tokens in the question affect ranking
  and ordering without auto-hard-filtering when the question names no version.
  The default remains current-version-only.
- Bearer-token authentication for every `/api/v1/*` route. A token resolves to a
  principal with an authoritative tenant, subject, optional user ID, roles, and
  groups. Missing/invalid credentials return a constant `401 Unauthorized`;
  authenticated tenant mismatch or forbidden actions return a constant
  `403 Forbidden`.
- Server-side tenant resolution. A request-body `tenant_id` is optional and
  deprecated; it must match the authenticated tenant and never authorizes access.
  Ingestion writes only into the authenticated tenant.
- Tenant and document-ACL predicates applied inside retrieval SQL before evidence
  reaches generation. A post-query check rejects rows whose `tenant_id` differs
  from the requested tenant. Citations are built from server-side evidence and
  must match its `source_key` and `version_number`.
- Structured JSON request logs with request ID, salted tenant reference,
  retrieval strategy and stage latencies, provider/observed model, retry count,
  safe error category, token counts, and actual-vs-estimated token source. Logs
  contain no body, query, prompt, document content, API key, or raw tenant UUID.
- In-process per-tenant rate limiting and bounded concurrency, bounded provider
  retries/timeouts, bounded DB pool and statement timeouts, and graceful
  degraded (`200`) readiness when only Redis fails.
- `POST /api/v1/search` and document APIs, now authenticated and tenant-scoped.
- `POST /api/v1/answer`:
  - accepts an optional tenant assertion, question, and retrieval configuration;
  - defaults preserve the existing hybrid retrieval configuration;
  - returns answer, citations, evidence metadata, retrieval diagnostics,
    generation metadata, abstention, conflicts, uncertainty, and a validated
    `version_analysis`;
  - abstains when no evidence survives assembly;
  - maps provider timeout, rate limit, invalid output, citation violation, and
    configuration failures to explicit HTTP errors.
- Deterministic evidence assembly with dedupe, multi-chunk retention, version
  preservation, and a context-token budget.
- Provider interface with a local `extractive` default and an opt-in
  `openai_compatible` adapter that enforces temperature and `max_tokens`, reads
  reported model/usage, and retries only retryable failures.
- Reproducible `answer_eval_v1` generation runner and independent scorer runner,
  both with `--resume`; adjudication gate, report builder, and repeated-run
  aggregator.
- Measured `answer_eval_v1` reports for three runs under
  `evals/runs/answer_eval_v1_c1_run{1,2,3}/`, plus a cross-run aggregate, all with
  `status: "completed"`.

### Background ingestion and reconciliation

- `POST /api/v1/documents/ingest/async` commits a durable `ingestion_jobs` row
  and a non-current target version before publishing a Celery task that contains
  only the job ID. `GET /api/v1/documents/jobs/{job_id}` returns tenant-scoped
  job state. Redis is the broker; PostgreSQL remains authoritative and no Celery
  result backend is configured.
- Ingestion and reconciliation consume disjoint queues (`ingestion`,
  `reconciliation`) on separate workers, so recovery cannot be starved by an
  ingestion backlog.
- Bounded retries apply only to transient dependency failures and a small
  unexpected-exception budget. Permanent deployment/validation failures are not
  retried, and dead letters remain inspectable with
  `scripts/ops/inspect_dead_letters.py`.
- Execution leases and a fencing token prevent a replaced worker from activating
  or failing a job it no longer owns. Periodic and manual reconciliation recover
  orphaned jobs (`scripts/reconcile_ingestion_jobs.py`). See
  `docs/BACKGROUND_INGESTION.md`.

### Migrations and schema integrity

- One runner applies immutable `sql/001`–`005` files, records
  version/filename/SHA-256/timestamp in `schema_migrations`, serializes runners
  with a PostgreSQL advisory lock, and validates the full live schema before a
  history row commits.
- Existing history is contiguous and never rewritten. Adoption requires the
  complete expected schema for a migration; partial signatures fail closed.
  `python -m app.db.migrations --check` and
  `python -m app.db.schema_validation` verify without writing. See
  `docs/DATABASE_MIGRATIONS.md`.

### Security and reliability controls

- Optional ISO-8601 `expires_at` on a configured token is parsed at startup and
  re-checked on every request, so a token stops authenticating at the configured
  instant without a restart. Rotation is an overlap across a restart; revocation
  is removal plus restart. Production rejects an empty map, tokens below 32
  characters, documented placeholders, and a map with no usable unexpired entry.
- Compose and CI use the Redis-backed per-tenant sliding-window limiter, shared
  across API replicas. The in-memory backend remains the test default. Redis loss
  degrades to per-process enforcement by default and can fail closed instead.
- Backup, restore, retention cleanup, dead-letter inspection, and reconciliation
  scripts live under `scripts/ops/` and are documented in
  `docs/OPERATIONS_RUNBOOK.md`.

### Build, CI, and reproducibility

- `.github/workflows/ci.yml` runs Ruff, the offline suite, a PostgreSQL/pgvector
  + Redis integration job that bootstraps migrations twice and runs the live
  suite, and a Docker build.
- The image installs with `pip install --constraint requirements.lock .` on
  `python:3.13.9-slim-bookworm`, runs as a non-root user, and is checked for
  version/import consistency after build.
- `app/version.py` is the only application version; `requirements.lock` pins the
  complete runtime and development closure for Python 3.13.

## Measured answer evaluation (Phase C.1)

Three runs of 26 records with one frozen configuration. `n=26` per run.

| Metric | Run 1 | Run 2 | Run 3 | Mean | Range |
| --- | ---: | ---: | ---: | ---: | --- |
| `answer_relevance` | 0.8846 | 0.9231 | 0.9423 | 0.9167 | 0.8846–0.9423 |
| `faithfulness` | 0.9231 | 0.9615 | 0.9231 | 0.9359 | 0.9231–0.9615 |
| `citation_correctness` | 0.8750 | 0.9750 | 0.9250 | 0.9250 | 0.8750–0.9750 |
| `citation_completeness` | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| `abstention_correctness` | 0.9231 | 0.9615 | 0.9615 | 0.9487 | 0.9231–0.9615 |
| `conflict_handling` | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

- Frozen configuration held across all three runs: dataset SHA-256
  `68fcd05c…`, corpus fingerprint `c5cb44a8…`, prompt `grounded_answer_v2`,
  retrieval `hybrid / and / chunk fusion`.
- Two independent model scorers; inter-scorer exact agreement `447/468`
  (`0.9551`) pooled (per run `0.9487`, `0.9615`, `0.9551`).
- 21 metric-level disagreements adjudicated with an explicit note; no label was
  changed.
- 78 generations: 74 succeeded, 0 transport/envelope failures, 4
  generation-contract failures (`0.0513`). All four were
  `InvalidModelOutputError: abstaining answer must not cite evidence IDs`.
- Failure taxonomy separates `incomplete_generation` from
  `generation_hallucination`; aggregate failures were 9 `citation_mismatch` and
  7 `incomplete_generation`, with no hallucination, retrieval, context-budget, or
  conflict-handling failures.
- `runbooks/payment-api.md@v2` was retrieved at evidence position 2 in all three
  runs for the rollback question; the prior baseline missed it.
- Latency p50 `6963.8 ms` / p95 `32179.6 ms` (mean across runs); token counts are
  mostly character-based estimates because the CLI adapter did not report usage.
- Scorer B was configured as `opencode/flint-40b-free` but ran
  `opencode/mimo-v2.6-flash-free`; the deviation is recorded per run in
  `config.json` under `scoring.model_deviations`.
- The earlier single-run Phase C baseline is retained at
  `evals/runs/answer_eval_v1_main/` for comparison and was not overwritten.

Details, commands, and limitations are in `docs/ANSWER_EVALUATION_V1.md`.

## Operational validation (Phase E)

Measured on 2026-11-10 with one uvicorn worker, the local embedding service, and
the `extractive` provider. Full tables and failure cases are in
`docs/OPERATIONAL_VALIDATION.md`; raw results are in
`evals/ops/phase-e-final/`.

| Scenario | Requests | Concurrency | Throughput | p95 |
| --- | ---: | ---: | ---: | ---: |
| Health | 100 | 1 | 170.3 req/s | 8.3 ms |
| Readiness | 50 | 1 | 39.9 req/s | 39.6 ms |
| Answer, single user | 30 | 1 | 10.9 req/s | 105.3 ms |
| Answer, concurrent | 100 | 8 | 53.9 req/s | 236.2 ms |
| Answer, lexical | 100 | 8 | 111.4 req/s | 131.7 ms |
| Search, semantic | 80 | 8 | 51.7 req/s | 228.2 ms |
| Search, hybrid | 80 | 8 | 51.8 req/s | 220.5 ms |
| Two tenants | 100 | 8 | 120.2 req/s | 111.9 ms |
| Mixed | 100 | 8 | 66.7 req/s | 199.9 ms |

- All 740 measured requests succeeded. Evidence assembly and extraction were
  0-1 ms; retrieval and the embedding round-trip dominate.
- Load numbers exclude real LLM latency. The fake OpenAI-compatible provider was
  used only for deterministic failure paths.
- Isolation: 0/20 tenant-A responses contained tenant-B's canary; 20/20
  allowed-group responses saw the restricted document, 0/20 denied-group
  responses did.
- Limits: 20 x `200` then 5 x `429` (`Retry-After: 59`) per tenant; 1 x `200`
  and 7 x `429` for 8 simultaneous requests at concurrency 1; follow-ups
  recovered instantly.
- Failure injection: PostgreSQL `503` in about 10 s and tested without an API
  restart; Redis `200 degraded` in about 2 s; embedding `503` in about 4.3 s;
  provider timeout `504`, rate limit `429`, invalid envelope/output and bad
  citation `502`, upstream `500` retried then `502`; pool exhaustion `503` and
  recovery `200`.
- All 9 log validations passed: request ID correlation 1:1, salted `tenant_ref`,
  stage latencies, provider/model, retry counts, actual-vs-estimated token
  source, and zero hits for tokens, raw tenant UUIDs, questions, or document
  content.
- One telemetry defect was fixed: non-retryable provider faults now record
  `retry_count: 0` instead of `null` (`app/services/answer_generation.py`).
- First local bottleneck: the embedding service at about 4.1 cores and 1.8 GiB
  under 8-way load; the API stayed under one core and 95 MiB.

## Verification status

- Offline suite (2026-11-18, fresh virtual environment installed from
  `requirements.lock`, `OPSATLAS_LIVE_TESTS=0`, no PostgreSQL/Redis service):
  **548 passed, 200 skipped** in 9.63 s. The skips are the opt-in live suites plus
  tests that require a running PostgreSQL, Redis, or PostgreSQL client tools; CI
  supplies those services in its integration job.
- Ruff over `app`, `evals`, `scripts`, and `tests`: **all checks passed**.
- Background processing is covered by `tests/test_ingestion_worker.py`,
  `tests/test_ingestion_jobs.py`, `tests/test_ingestion_task_unit.py`,
  `tests/test_reconciliation.py`, `tests/test_sync_fencing.py`, and
  `tests/test_ingestion_activation.py`.
- Migrations and schema are covered by `tests/test_migrations.py`,
  `tests/test_schema_readiness.py`, and `tests/test_operations_scripts.py`.
- Security, configuration, and deployment are covered by
  `tests/test_auth_lifecycle.py`, `tests/test_production_config.py`,
  `tests/test_rate_limit_redis.py`, `tests/test_compose_deployment.py`, and
  `tests/test_embedding_url_modes.py` (host versus container embedding
  addresses).
- Phase D adversarial suite in `tests/test_security_phase_d.py` covers missing
  and invalid tokens, body-tenant forgery, cross-tenant ingestion, retrieval row
  isolation, ACL SQL scope, forged version metadata, provider-error leakage,
  token/content log redaction, rate limiting, and concurrency limiting.
- Phase E adds `tests/test_failure_injection.py` (15 tests) covering PostgreSQL,
  pool-timeout, embedding, provider timeout/429/invalid-envelope/invalid-output/
  citation/5xx failures, retry counts, recovery, and failure-log redaction.
- `tests/test_security_live.py` adds an opt-in live DB check that seeds a
  throwaway tenant B document and proves tenant A cannot retrieve, cite, or
  claim it. It runs only with `OPSATLAS_LIVE_TESTS=1`,
  `OPSATLAS_TEST_TENANT_ID`, and `OPSATLAS_TEST_API_TOKEN`. **35 live tests
  passed** (`test_security_live.py`, `test_answer_live.py`,
  `test_lexical_retrieval_live.py`) against the local PostgreSQL corpus on
  2026-11-12; the seeded tenant was deleted afterwards and the original
  tenant/document counts were restored (1 tenant, 10 documents).
- All three Phase C.1 `report.json` files still validate against
  `evals/answer_eval_v1.report_schema.json` (`jsonschema` 4.26.0).
- Security review completed in `docs/ANSWER_SECURITY_REVIEW.md`, with adversarial
  cases in `tests/test_answer_security.py` and attribution regression tests in
  `tests/test_answer_eval_report.py`.
- Resume-path tests in `tests/test_answer_eval_runner.py` and
  `tests/test_answer_eval_scorer.py`; aggregation tests in
  `tests/test_answer_eval_aggregate.py`.
- Retrieval defaults are unchanged and the existing retrieval tests still pass.
  Phase E measured latency and failure behavior for one local worker; the
  authorization and retrieval paths were not modified.
- `golden_v0` through `golden_v3` and every dataset label are unmodified.
- No live paid model call is required by the offline test suite.

## Known limitations

- The corpus is synthetic and small. No production-quality claim is supported.
- The default `extractive` provider is a deterministic fallback, not a language
  model. It copies the top passage and does not synthesize across sources.
- The answer endpoint is stateless. No query run is persisted, and the existing
  `query_runs` table is not used.
- Authentication is a local bearer-token map in configuration, not an identity
  provider. Optional ISO-8601 `expires_at` is enforced per request, rotation is
  an overlapping-entry restart, and revocation is removal plus restart. There is
  no SSO, OAuth, JWT refresh, or online revocation API, and a token without
  `expires_at` stays valid until the configuration changes and the process
  restarts. The endpoint still must not be exposed publicly without a real
  identity boundary.
- ACLs are allow-list augmentation, not default-deny. Documents with no ACL rows
  are visible to every authenticated member of their tenant. The schema has no
  deny rows, no inheritance, and no per-document deny override.
- Tenant and ACL filtering is enforced before evidence assembly and citations,
  but the corpus and test fixtures are synthetic. No production access-control
  audit or penetration test has been performed.
- The rate limit is shared through Redis when `RATE_LIMIT_BACKEND=redis`
  (Compose and CI) and process-local when the memory backend is used (the test
  default). The concurrency slot is always per process. Redis loss degrades to
  per-process enforcement by default unless
  `RATE_LIMIT_REDIS_FALLBACK_TO_LOCAL=false`. Limiter state expires with the
  window; it is not a durable quota.
- Request telemetry is written to the process log stream. Retention, shipping,
  sampling, and alerting are not implemented.
- Provider and DB timeouts are bounded. Phase E measured load, latency, and
  failure behavior on one local worker; multi-worker and multi-instance behavior
  remains unmeasured.
- The measured runs used an evaluation-only OpenCode CLI adapter. Temperature is
  not enforceable and max output tokens is a soft prompt limit; the
  `openai_compatible` adapter that enforces both was not used for these runs.
- Provider token usage was reported for only 1 of 74 successful generations;
  token counts are otherwise character-based estimates.
- Three runs use one generation model and one provider. They establish a
  run-to-run range for this setup only, not stability across providers.
- Generation-contract failures persist at `4/78` (`0.0513`): abstaining answers
  that cite evidence IDs. The validator is correct; the prompt or a bounded
  repair step should make the contract easier to satisfy.
- Operational interruptions (provider quota exits and an embedding-service
  restart) were recovered with `--resume`; the reported `0.0` transport failure
  rate describes the recovered artifacts, not provider uptime.
- The version retrieval fix is exercised against synthetic fixtures; it is not
  evidence that arbitrary version questions rank correctly.
- Evidence budgeting uses a documented `ceil(len(text) / 4)` token estimate, not
  a model tokenizer.
- No reranking, streaming, caching, or multi-turn context.
- Observability is structured request logging with a request ID; there is no
  distributed tracing, metrics backend, dashboard, or audit store.

## Baseline preservation

- Retrieval defaults: `keyword_strategy="and"`, `fusion_strategy="chunk"`,
  `semantic_weight=1.0`, `lexical_weight=1.0`, `rrf_k=60`, `limit=5`.
- Version filtering remains opt-in; explicit version tokens affect ranking, not
  automatic hard filtering, and the current-version default is unchanged.
- All fusion alternatives remain opt-in.
- Golden datasets, dataset labels, and prior reports are unchanged.
- Corpus fingerprint:
  `c5cb44a83cd37560f004692ec4c702d509dde3f958a42966545480ca23fe6408`.
- `golden_v3` dataset SHA-256:
  `1fe9b87582720e272abe36f45063e8d34ad4266c9e6694870f51b57100d80497`.
- `answer_eval_v1` dataset SHA-256:
  `68fcd05c7a289d6bf476ec7e29a0ff3d9dfb2f334f052224f18735999f121350`.
- Phase C.1 pipeline SHA-256:
  `12dfb5d42adf0394b43c4fefc1fc8bba386b43dc626735a450a1b31940e31f50`.

## Production deployment prerequisites

Completed engineering work is summarized under "Current status". The following
are operator or product prerequisites that this repository does not supply; none
is required to run the local stack.

1. Measured quality and latency on a provider that enforces temperature/output
   limits and reports token usage. Current numbers use the `extractive` provider
   and an evaluation-only CLI adapter and exclude real model inference.
2. A real identity boundary. The configured bearer-token map supports optional
   expiry, overlapping rotation, and removal-plus-restart revocation, but it is
   not SSO/OAuth and has no online revocation API or persisted user table.
3. A default-deny ACL model, if required. Current ACLs are allow-list
   augmentation; deny rows and inheritance would need an explicit schema
   migration.
4. Human spot-checks on a sample of answer records; two model scorers are not a
   substitute for domain review.
5. Observability and retention: distributed tracing, a metrics/alerting backend,
   log shipping, and a documented retention policy.
6. Deployment hardening: TLS termination, private datastores with no published
   ports, secret-manager injection, multi-instance rate-limit verification, and
   a measured multi-worker/multi-instance profile.

CI already runs the opt-in live suite in its PostgreSQL + Redis integration job;
the offline job reports live tests as skipped. No production deployment or SLO
claim is supported by the current evidence.

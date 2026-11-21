# Roadmap

> Ordering is driven by measured risk, not feature count. The corpus is
> synthetic until a real, permissioned corpus is available. Dates are
> intentionally omitted.

## Current position

Foundation, ingestion, retrieval, grounded answering, security, background
ingestion, and CI are implemented on the current tree; the authoritative summary
is `docs/PROJECT_STATUS.md` and the milestone view is `docs/MILESTONES.md`.

Phase E remains the local operational validation: load, isolation,
rate/concurrency, and controlled failure-injection runs on one uvicorn worker,
summarized in `docs/OPERATIONAL_VALIDATION.md` with artifacts in
`evals/ops/phase-e-final/`. Background ingestion now runs through Celery with
durable PostgreSQL job state, bounded retries, dead letters, execution leases,
and periodic reconciliation. CI runs Ruff, the offline suite, a PostgreSQL +
Redis integration job, and a container build. Rate limits, concurrency
admission, and the measured throughput are all process-local; nothing here is a
production SLO.

Phase C.1 remains the measured quality baseline: three `answer_eval_v1` runs with
a frozen configuration, summarized in `docs/ANSWER_EVALUATION_V1.md`. Prior
artifacts remain in `evals/runs/answer_eval_v1_main/`; Phase C.1 runs are in
`evals/runs/answer_eval_v1_c1_run{1,2,3}/` with the aggregate in
`evals/runs/answer_eval_v1_c1_aggregate/`.

Measured mean across three runs (`n=26` per run): relevance 0.9167, faithfulness
0.9359, citation correctness 0.9250, citation completeness 1.0000, abstention
0.9487, conflict handling 1.0000. Citation correctness has the widest spread
(0.8750–0.9750). These numbers were not re-measured in Phase D or E; neither
phase changed retrieval defaults.

Residual authorization risks: the token map is configuration, not an IdP; ACLs
are allow-list augmentation rather than default-deny; and rate limits are
process-local. The endpoint is still not safe to expose publicly without those
follow-ups.

## Completed: measure, then fix the measured weaknesses

### 1. Execute `answer_eval_v1` — complete

- 26/26 records run with `opencode/deepseek-v4.1-flash-free`.
- Two independent scorers, nine disagreements adjudicated.
- Report `status: "completed"` and schema-shaped; every metric carries `n`.
- Adjudicated: relevance 0.8462, faithfulness 0.9231, citation correctness
  0.8500, citation completeness 0.8947, abstention 0.8846, conflict handling
  0.7500 (`n=26`).
- Limitations and recommendations documented.

### 2. Fix version-specific retrieval and conflict framing — complete

- Root cause was a CTE-scope bug in the version-order SQL plus narrow
  historical/version-change detection. Both are fixed in
  `app/services/retrieval.py`.
- Explicit version tokens now affect lexical ranking and ordering. They do
  **not** trigger an automatic hard filter when the question names no version;
  the current-version default is preserved, and an explicit filter stays opt-in.
- `runbooks/payment-api.md@v2` is retrieved at evidence position 2 in all three
  runs for the rollback question.
- The answer contract adds a validated `version_analysis` block: versions
  identified, evidence-supported differences, conflicting instructions,
  recommendation version, and supporting evidence IDs. Fabricated evidence IDs
  are rejected.
- Regression tests cover v1, v2, current, deprecated, and ambiguous version
  questions, plus mixed-version and non-conflicting answers.
- `conflict_handling` is `1.0000` in all three runs.

### 3. Stabilize the evaluation — partially complete

- Three runs executed with the same frozen configuration (dataset SHA-256,
  corpus fingerprint, prompt, retrieval config). The aggregate reports
  `frozen_configuration_held: true`.
- New `scripts/aggregate_answer_eval_runs.py` reports per-run and aggregate
  metrics, min/max/stdev, provider failure rate, latency p50/p95, tokens, failure
  categories, scorer identity/deviations, and resumed-run metadata.
- `--resume` added to both generation and scoring runners so provider-quota
  interruptions do not force a full rerun.
- Provider capability reporting added: requested vs observed model, temperature
  and output-limit enforcement status, usage where exposed.
- **Remaining:** human spot-checks, and a provider that enforces temperature and
  output limits and reports token usage. The measured runs used the CLI adapter,
  where temperature is not enforceable and max output is a soft prompt limit.

## Completed: operational validation (Phase E)

Executed 2026-11-10 on one uvicorn worker with the local embedding service and
the `extractive` provider. Full report: `docs/OPERATIONAL_VALIDATION.md`;
artifacts: `evals/ops/phase-e-final/`.

- [x] Capture the baseline: hardware, OS, Python/package versions,
  PostgreSQL/Redis versions and settings, pool/limit/provider configuration.
- [x] Measure `/health`, `/ready`, answer single-user, answer at 8-way
  concurrency, lexical/semantic/hybrid search, two-tenant, and mixed workloads
  with p50/p95/p99, throughput, and error rate.
- [x] Separate provider-independent numbers (`extractive`) from deterministic
  fake-provider failure paths; make no real-LLM latency or cost claim.
- [x] Sample API and embedding CPU/working set over the load window and record a
  Docker stats snapshot for PostgreSQL/Redis.
- [x] Inject PostgreSQL, Redis, embedding, provider timeout/rate-limit/invalid
  response, DB-pool exhaustion, and client cancellation; record expected vs
  actual status, category, retries, recovery, and log correctness.
- [x] Verify tenant, rate, and concurrency isolation, consistent auth failures,
  and ACL behavior under concurrency.
- [x] Validate request-ID correlation, salted `tenant_ref`, stage latencies,
  provider/model, retry/error categories, token provenance, and absence of
  secrets/questions/raw tenant UUIDs.
- [x] Restore all state after each experiment (containers healthy, embedding
  service running, temporary tenant deleted, DB counts restored).

**Exit:** a reproducible harness, recorded measurements and failure results, and
documented limitations and starting thresholds. Remaining gaps: real-provider
latency/cost, multi-worker/multi-instance behavior, and distributed rate
limiting.

## Next: harden the remaining boundaries

### 4. Human spot-checks and enforced-limit provider

- Sample records for review by a domain expert; treat model scores as a screen,
  not ground truth.
- Repeat the measurement on the `openai_compatible` provider that enforces
  temperature and `max_tokens`, reports usage, and maps timeout/rate-limit/auth
  categories. Compare against the measured range.
- Fix the remaining generation-contract failure: abstaining answers must not
  cite evidence IDs. Consider a bounded repair step rather than only retrying.

**Exit:** a second measured range on an enforced-limit provider, plus documented
human review of a sample.

### 5. Make answer runs observable — request logging complete, persistence open

- [x] Capture request ID, stage timings, token estimates/usage, provider and
  observed model, retry count, error category, and a salted tenant reference in
  one structured request log.
- [x] Never log bodies, queries, prompts, document content, API keys, or raw
  tenant UUIDs.
- [ ] Decide whether to persist answer runs, and if so, define retention,
  tenancy, shipping, and redaction first.
- [ ] Add distributed tracing, metrics, and alerting.

**Exit:** a failed request can be diagnosed from its request ID without logging
document contents or keys. The logging half is met; retention and tracing are
not.

## Then: retrieval and generation quality

### 6. Retrieval context selection

- Evaluate whether reranking or diversity selection improves citation
  completeness on multi-source questions.
- Measure the effect of `max_evidence` and context budget on answer quality, not
  only latency.
- Keep existing retrieval defaults until an evaluation justifies a change.

**Exit:** a measured comparison with the current baseline, including a
regression threshold.

### 7. Provider evaluation matrix

- Compare at least two providers or models on the same rubric and dataset.
- Record cost, latency, invalid-output rate, and abstention rate.
- Keep the `extractive` provider as a deterministic smoke-test fallback, not a
  quality baseline.

**Exit:** a documented trade-off, not a vendor commitment.

### 8. Multi-version conflict handling

- Expand version-conflict cases so both versions are supplied, letting the
  "adds/removes" comparison be scored instead of bounded as an evidence
  limitation.
- Require explicit version naming whenever evidence contains multiple versions
  of the same source.
- Evaluate whether the prompt and validator need a stricter conflict schema.

**Exit:** stale guidance is never presented as current in the version-conflict
subset, and supplied-version comparisons are scored.

## Security and operations

### 8. Authentication and authorization — complete (Phase D)

- [x] Authenticate every protected route with configured bearer tokens; derive
  the authoritative tenant and principal from identity, not the request body.
- [x] Apply tenant and document ACL predicates before retrieval candidates reach
  generation, and re-check returned rows.
- [x] Build citations and version references only from server-side evidence.
- [x] Add offline cross-tenant, forged-metadata, prompt-injection, and log
  redaction tests, plus an opt-in live cross-tenant test.

**Exit:** unauthorized fixtures are never returned or sent to a provider. The
offline suite meets this. Remaining risk: ACLs are allow-list augmentation, not
default-deny, and the token map is not an identity provider.

### 9. Ingestion reliability — complete

- [x] Add background ingestion, retry/backoff, dead-letter handling, and
  stuck-job reconciliation.
- [x] Keep the last known-good version on failure.
- [x] Add durable job state, execution leases, and a fencing token so a replaced
  worker cannot activate a job it no longer owns.
- Implementation: `app/tasks/ingestion.py`, `app/tasks/reconciliation.py`,
  `app/services/{ingestion,dead_letters,reconciliation}.py`, and migrations
  003–005. Details: `docs/BACKGROUND_INGESTION.md`.

**Exit:** worker termination and provider errors do not corrupt current document
state. Met; covered by the ingestion worker, reconciliation, and fencing tests.

### 10. CI and release gates — mostly complete

- [x] Run tests and Ruff on every change.
- [x] Validate migrations and the live schema from a clean database, then build
  the container image with locked dependencies.
- [x] Keep the Ruff baseline at zero findings.
- [ ] Fail on golden-dataset drift and on retrieval-baseline regression beyond a
  documented threshold.
- [ ] Run the reproducible evaluation command in CI and publish the report
  artifact.

**Exit:** a clean checkout reproduces tests from documented instructions. The
evaluation-gate and report-artifact half is still open.

## Deliberately deferred

- Multi-agent orchestration and autonomous remediation.
- Microservices or a separate vector database.
- Streaming responses and multi-turn conversation.
- Fine-tuning.
- Production deployment and SLO claims.

These require measured needs and are not implied by the current slice.

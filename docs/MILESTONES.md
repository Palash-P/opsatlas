# Milestones and Exit Criteria

> Status reviewed 2026-11-18 against the current tree. Checked items are
> implemented and exercised by the offline suite; items that need production
> infrastructure or a paid provider remain unchecked on purpose.

## M1 — Foundation (complete)
- [x] Repo structure and local Compose dependencies.
- [x] Liveness/readiness checks and basic tests.
- [x] Initial schema proposal and architecture notes.
- [x] Golden dataset contract and initial synthetic examples.
- Exit: API runs locally; tests pass; dependencies can be started with Compose.

## M2 — Ingestion and lifecycle (complete)
- [x] Add versioned SQL migrations. The in-repo runner replaced the Alembic
  proposal; see `docs/DATABASE_MIGRATIONS.md` for checksums, adoption, and
  advisory-lock serialization.
- [x] Define source/document/version repository APIs.
- [x] Add deterministic Markdown ingestion. PDF parsing is not implemented.
- [x] Add content hashes, idempotency, status transitions, and deletion semantics.
- Exit: repeated ingestion is safe; changed documents create a new version; failed
  update preserves the prior current version.

## M3 — Baseline retrieval (complete)
- [x] Select embedding model and confirm vector dimension.
- [x] Add deterministic chunking and embedding provider interface.
- [x] Add tenant-filtered vector retrieval and PostgreSQL lexical retrieval.
- [x] Evaluate Recall@k against the golden dataset.
- Exit: retrieval output and scores are reproducible and traceable to source
  versions.

## M4 - Grounded answer generation (vertical slice complete; hardening pending)
- [x] Add model provider interface and bounded timeouts.
- [x] Return citations with document ID, source key, title, version number, chunk ID, and excerpt.
- [x] Add abstention behavior and citation validation.
- [x] Add deterministic evidence assembly with a context-token budget.
- [x] Add an API endpoint for grounded answers with retrieval diagnostics.
- [ ] Run answer-quality evaluation with a real provider that enforces
  temperature/output limits and reports usage and publish scored results.
  Three `answer_eval_v1` runs were scored, but through an evaluation-only CLI
  adapter (see `docs/ANSWER_EVALUATION_V1.md`).
- [x] Add adversarial prompt-injection and cross-version conflict tests
  (`tests/test_answer_security.py`, version-analysis cases in
  `tests/test_answer_generation.py`).
- Exit: every answer citation resolves; unanswerable cases do not invent evidence.

## M5 - Evaluation and optimization (retrieval complete; answer evaluation measured)
- [x] Add retrieval metrics and compare fusion strategies.
- [x] Compare chunk sizes, hybrid weights, and lexical strategies.
- [x] Store configuration and dataset versions with every retrieval run.
- [x] Design the answer evaluation dataset, rubric, and report schema.
- [x] Execute and score the answer evaluation.
- [ ] Use answer metrics to justify any generation or context-selection change.
- Exit: report explains measured trade-offs and regression thresholds. Measured
  trade-offs and a run-to-run range are published; CI does not yet fail on a
  quality regression threshold.

## M6 — Security and multi-tenancy (complete)
- [x] Add authentication and tenant context.
- [x] Implement ACL-filtered retrieval and citations.
- [x] Add cross-tenant and prompt-injection tests.
- Exit: unauthorized test fixtures are never returned to the caller or sent to
  the model.

## M7 — Reliable background processing (complete)
- [x] Add Celery ingestion workers.
- [x] Add retry/backoff, dead-letter handling, and stuck-job reconciliation.
- [x] Add durable job state, execution leases, and a fencing token so a replaced
  worker cannot activate or fail a job it no longer owns.
- [x] Add a dedicated reconciliation worker/queue and a manual
  `scripts/reconcile_ingestion_jobs.py` entry point.
- Exit: worker termination and provider errors recover without corrupting current
  document state. Covered by `tests/test_ingestion_worker.py`,
  `tests/test_reconciliation.py`, `tests/test_sync_fencing.py`, and the opt-in
  Celery live suite.

## M8 — Operations lab (load and failure validation complete)
- [ ] Add distributed tracing and stage metrics. Stage latencies and error
  categories are in the structured request log; there is no tracing backend.
- [x] Add load tests and fault injection.
- [ ] Add cost/latency dashboard and budgets.
- Exit: reproduce at least five documented failures and verify fixes with
  regression tests. Met for the nine documented injection cases in
  `docs/OPERATIONAL_VALIDATION.md` and `tests/test_failure_injection.py`; the
  dashboard/tracing half is not.

## M9 — Release discipline (partially complete)
- [x] Add CI (lint, offline suite, PostgreSQL + Redis integration, container
  build) and migration validation.
- [x] Document rollback and restore instructions
  (`docs/OPERATIONS_RUNBOOK.md` §3–§5).
- [ ] Add evaluation gates that fail CI on golden-dataset drift or a retrieval
  baseline regression beyond a documented threshold.
- [x] Write incident runbooks.
- [ ] Write postmortems. The procedure is documented; none has been exercised
  against a real incident.
- Exit: a clean checkout can run tests from documented instructions. The
  evaluation-reproduction half is documented but is not a CI gate.

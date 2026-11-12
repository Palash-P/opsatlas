# Milestones and Exit Criteria

## M1 — Foundation (current)
- [x] Repo structure and local Compose dependencies.
- [x] Liveness/readiness checks and basic tests.
- [x] Initial schema proposal and architecture notes.
- [x] Golden dataset contract and initial synthetic examples.
- Exit: API runs locally; tests pass; dependencies can be started with Compose.

## M2 — Ingestion and lifecycle
- [ ] Add Alembic migrations.
- [ ] Define source/document/version repository APIs.
- [ ] Add deterministic Markdown ingestion first, then PDF parsing.
- [ ] Add content hashes, idempotency, status transitions, and deletion semantics.
- Exit: repeated ingestion is safe; changed documents create a new version; failed update preserves the prior current version.

## M3 — Baseline retrieval
- [ ] Select embedding model and confirm vector dimension.
- [ ] Add deterministic chunking and embedding provider interface.
- [ ] Add tenant-filtered vector retrieval and PostgreSQL lexical retrieval.
- [ ] Evaluate Recall@k against the golden dataset.
- Exit: retrieval output and scores are reproducible and traceable to source versions.

## M4 - Grounded answer generation (vertical slice complete; hardening pending)
- [x] Add model provider interface and bounded timeouts.
- [x] Return citations with document ID, source key, title, version number, chunk ID, and excerpt.
- [x] Add abstention behavior and citation validation.
- [x] Add deterministic evidence assembly with a context-token budget.
- [x] Add an API endpoint for grounded answers with retrieval diagnostics.
- [ ] Run answer-quality evaluation with a real provider and publish scored results.
- [ ] Add adversarial prompt-injection and cross-version conflict tests.
- Exit: every answer citation resolves; unanswerable cases do not invent evidence.

## M5 - Evaluation and optimization (retrieval complete; answer evaluation designed)
- [x] Add retrieval metrics and compare fusion strategies.
- [x] Compare chunk sizes, hybrid weights, and lexical strategies.
- [x] Store configuration and dataset versions with every retrieval run.
- [x] Design the answer evaluation dataset, rubric, and report schema.
- [ ] Execute and score the answer evaluation.
- [ ] Use answer metrics to justify any generation or context-selection change.
- Exit: report explains measured trade-offs and regression thresholds.

## M6 — Security and multi-tenancy
- [ ] Add authentication and tenant context.
- [ ] Implement ACL-filtered retrieval and citations.
- [ ] Add cross-tenant and prompt-injection tests.
- Exit: unauthorized test fixtures are never returned to the caller or sent to the model.

## M7 — Reliable background processing
- [ ] Add Celery ingestion workers.
- [ ] Add retry/backoff, dead-letter handling, and stuck-job reconciliation.
- Exit: worker termination and provider errors recover without corrupting current document state.

## M8 — Operations lab
- [ ] Add tracing and stage metrics.
- [ ] Add load tests and fault injection.
- [ ] Add cost/latency dashboard and budgets.
- Exit: reproduce at least five documented failures and verify fixes with regression tests.

## M9 — Release discipline
- [ ] Add CI, migration validation, evaluation gates, and rollback instructions.
- [ ] Write incident runbooks and postmortems.
- Exit: a clean checkout can run tests and reproduce evaluation results from documented instructions.

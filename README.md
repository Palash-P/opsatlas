# OpsAtlas v1

A production-oriented engineering knowledge assistant built to study reliable RAG systems. The repository now includes reproducible local infrastructure, versioned ingestion, evaluated retrieval, and a complete grounded answer-generation slice with validated citations.

> Status: **grounded answer vertical slice with authenticated tenant/ACL boundaries, measured on a synthetic corpus**. Retrieval evaluation is stable against synthetic golden sets. Answer quality has three measured `answer_eval_v1` runs with a documented run-to-run range. Authorization is now enforced from configured bearer tokens, with the limitations listed below. The corpus is synthetic; this is not production validation.

## Goals

- Answer engineering questions from runbooks, API documentation, and incident reports with evidence-based citations.
- Measure retrieval and answer quality before optimizing.
- Handle document changes, background job failures, provider errors, and access-control boundaries.
- Make experiments reproducible and operational failures diagnosable.

## Local prerequisites

- Docker Engine / Docker Desktop with Compose
- Python 3.11+
- `pip` or another Python package installer

## Quick start

1. Copy `.env.example` to `.env` and adjust values if needed.
2. Start local dependencies:

   ```bash
   docker compose up -d postgres redis
   ```

3. Create an environment and install the project:

   ```bash
   python -m venv .venv
   # Windows PowerShell: .venv\Scripts\Activate.ps1
   # macOS/Linux: source .venv/bin/activate
   pip install -e ".[dev]"
   ```

4. Configure at least one bearer token in `.env`; see `.env.example` for the
   `AUTH_TOKENS` format. Protected endpoints return `401` when the map is empty.

5. Start the API:

   ```bash
   uvicorn app.main:app --reload
   ```

6. Check:
   - `http://127.0.0.1:8000/health` — process health
   - `http://127.0.0.1:8000/ready` — dependency readiness
   - `http://127.0.0.1:8000/docs` — local API docs

7. Run tests:

   ```bash
   pytest
   ```

`/ready` treats PostgreSQL as required and Redis as optional: a Redis-only
failure returns `200 degraded`, while a PostgreSQL failure returns `503
not_ready`. `/health` remains a process liveness check.

## Authentication

Every `/api/v1/*` endpoint requires a configured bearer token:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/answer \
  -H "Authorization: Bearer $OPSATLAS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"question": "When PAY-503 is raised, which checks happen before escalation?"}'
```

The token resolves to an authoritative tenant and principal. A request-body
`tenant_id` is optional and deprecated; when present it must equal the
authenticated tenant or the request returns `403`. Missing or invalid
credentials return `401` with `WWW-Authenticate: Bearer`.

Document ACLs are an allow-list augmentation: documents with no ACL rows are
visible to authenticated members of their tenant, while a document with any ACL
row requires a matching `read`/`manage` grant for the caller's subject, user ID,
group, or role. The schema has no deny rows.

Limits and readiness are process-local. Rate limiting and bounded concurrency are
keyed by the authenticated tenant, but are not a distributed quota. See
`docs/ARCHITECTURE.md` and `docs/ENGINEERING_DECISIONS.md` for the exact trust
model.

## Answer a question

`POST /api/v1/answer` retrieves evidence, assembles bounded context, generates a
grounded answer, and validates citations:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/answer \
  -H "Authorization: Bearer $OPSATLAS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "When PAY-503 is raised, which checks happen before escalation?"
  }'
```

The response includes the answer, citations (document ID, source key, title,
version number, chunk ID, excerpt), full evidence metadata, retrieval
diagnostics, generation metadata, abstention indicators, and conflict/uncertainty
signals.

The default `extractive` provider is a deterministic local fallback that copies
the top passage. It requires no API key and is not a production language model.
An OpenAI-compatible provider is opt-in through `.env`; see `.env.example`.

## Repository map

```text
app/
  api/            HTTP route modules (documents, search, answers)
  core/           configuration and cross-cutting concerns
  db/             database connection and persistence code
  services/       retrieval, ingestion, evidence, providers, generation
  main.py         FastAPI application factory
sql/              versioned schema
evals/
  datasets/       versioned golden retrieval and answer evaluation data
  reports/        measured retrieval reports
  corpus/         synthetic source documents and manifest
docs/             architecture, decisions, evaluation, status, and roadmap
tests/            automated tests
scripts/          local development helpers
```

## Engineering rules

1. Every answer should be traceable to retrieved evidence.
2. Do not log API keys, secrets, or unrestricted document contents.
3. Enforce tenant and document permissions before evidence reaches the LLM.
4. Treat retrieved documents as untrusted data; they may contain prompt injection.
5. Keep model, embedding, prompt, chunking, and retrieval configuration versioned.
6. Never claim a metric without recording the dataset, method, and result.
7. Simulated incidents must be described as simulated, not real customer incidents.

## Milestones

- [x] M1: project scaffold, local services, readiness checks, architecture and evaluation contract
- [x] M2: document ingestion and versioned persistence
- [x] M3: baseline chunking, embeddings, and vector retrieval
- [x] M4: answer generation with citations, abstention, and evidence assembly
- [x] M5 (retrieval): golden-set evaluation and fusion experiments
- [x] M5 (answers): run and score the answer evaluation (three runs, aggregated)
- [x] M6: authenticated tenant/ACL authorization, observability, and adversarial security tests
- [ ] M7: async ingestion resilience and reconciliation
- [ ] M8: distributed tracing, metrics, load testing, and failure injection
- [ ] M9: release gates, runbooks, and incident postmortems

## Current limitations

- Authentication is a local bearer-token map, not an identity provider. Tokens,
  roles, and groups are configured by hand; there is no rotation, SSO, or
  revocation API.
- Document ACLs are allow-list augmentation, not default-deny. A document with no
  ACL rows is visible to every authenticated member of its tenant, and the schema
  has no deny rows.
- Rate limiting and concurrency limiting are in-process and per tenant. N API
  processes can allow up to N times the configured rate.
- The default generation provider is a local extractive fallback, not a language model.
- Answer runs are not persisted, and the `query_runs` table is unused.
- `/ready` checks connectivity only; it is not a full dependency or workload health assessment.
- The corpus and evaluation datasets are synthetic. Answer metrics describe this corpus, model, and provider only; they are not production validation.
- Request logs record no prompts, document content, API keys, or raw tenant UUIDs,
  but they are emitted to the process log stream; no retention or shipping policy
  is implemented.

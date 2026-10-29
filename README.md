# OpsAtlas v1

A production-oriented engineering knowledge assistant built to study reliable RAG systems. The first milestone is deliberately small: reproducible local infrastructure, a documented data model, a health-check API, and an evaluation dataset contract.

> Status: **Milestone 1 scaffold**. This is not yet a complete RAG application. Retrieval, ingestion workers, authentication, and generation are planned milestones.

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

4. Start the API:

   ```bash
   uvicorn app.main:app --reload
   ```

5. Check:
   - `http://127.0.0.1:8000/health` — process health
   - `http://127.0.0.1:8000/ready` — dependency readiness (PostgreSQL and Redis)
   - `http://127.0.0.1:8000/docs` — local API docs

6. Run tests:

   ```bash
   pytest
   ```

The readiness endpoint intentionally reports dependency failures instead of pretending the service is ready. If the database or Redis is stopped, `/health` should still work while `/ready` should return `503`.

## Repository map

```text
app/
  api/            HTTP route modules
  core/           configuration and cross-cutting concerns
  db/             database connection and persistence code
  main.py         FastAPI application factory
sql/              initial relational schema proposal
 evals/datasets/  versioned golden evaluation examples
 docs/             architecture, data model, and milestone plan
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
- [ ] M2: document ingestion and versioned persistence
- [ ] M3: baseline chunking, embeddings, and vector retrieval
- [ ] M4: answer generation with citations and abstention
- [ ] M5: golden-set evaluation and retrieval experiments
- [ ] M6: tenant-aware authorization and security tests
- [ ] M7: async ingestion resilience and reconciliation
- [ ] M8: tracing, metrics, load testing, and failure injection
- [ ] M9: release gates, runbooks, and incident postmortems

## Current limitations

- No real authentication yet.
- No document ingestion, embeddings, retrieval, or LLM calls yet.
- `/ready` checks connectivity only; it is not a full dependency or workload health assessment.
- The evaluation dataset is synthetic and illustrative. Expand and validate it before using its scores to make decisions.

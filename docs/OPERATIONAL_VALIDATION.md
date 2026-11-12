# Operational Validation (Phase E)

> Status: local validation on one developer machine. These are measurements from
> a synthetic corpus and a single API process, not production SLOs.

This phase measured the existing RAG stack under controlled load and injected
dependency/provider failures. It added no retrieval default, evaluation dataset,
schema migration, or runtime dependency. The only application change was a
telemetry fix: non-retryable provider failures now record `retry_count: 0`
instead of `null` (see [Regressions and fixes](#regressions-and-fixes)).

## Scope

- Health and readiness behavior.
- Retrieval, evidence assembly, generation, and total latency.
- Error rate, status distribution, and throughput.
- Single-tenant, multi-tenant, near-rate-limit, over-concurrency, and mixed
  workloads.
- Tenant, rate-limit, and concurrency isolation.
- PostgreSQL, Redis, embedding-service, provider, DB-pool, and client
  cancellation failure paths.
- Structured-log correlation and redaction.

Out of scope: public deployment, multi-node clusters, real LLM quality/cost,
distributed rate limiting, and destructive database operations.

## Environment

| Item | Value |
| --- | --- |
| Host | ASUS ROG Zephyrus G14 GA402XV, AMD Ryzen 9 7940HS (8C/16T), 16,335,966,208 bytes RAM |
| OS | Windows 11 Home 10.0.26200 (AMD64) |
| Python | 3.13.12 |
| API | FastAPI 0.161.0, uvicorn 0.54.0, SQLAlchemy 2.1.6, psycopg 3.3.6, redis 5.3.1, pydantic 2.14.1 |
| Test client | httpx 0.28.1, pytest 8.4.2, Ruff 0.17.2 |
| Embedding | sentence-transformers 6.2.0, torch 2.14.1+cpu, `all-MiniLM-L6-v2` (384 dimensions, CPU) |
| PostgreSQL | 16.15 in `pgvector/pgvector:pg16`, host port 5433 |
| Redis | 7.4.11 in `redis:7-alpine`, host port 6379 |
| Docker / Compose | 29.7.2 / 5.4.0 |
| PostgreSQL config | `max_connections=100`, `shared_buffers=128MB`, `work_mem=4MB`, `effective_cache_size=4GB` |

Process and limit configuration for the measured instance:

| Setting | Value |
| --- | --- |
| Uvicorn workers | 1 |
| DB pool / overflow / timeout | 5 / 5 / 5 s |
| DB connect / statement timeout | 5 s / 15,000 ms |
| Rate limit | 240 requests/minute/tenant, 60 s window |
| Concurrency | 16 in-flight requests, 0.25 s acquire timeout |
| LLM provider | `extractive` (`extractive-local-v1`) for load; fake OpenAI-compatible provider for failure paths |
| LLM timeout / retries / backoff | 20 s / 1 / 0.25 s |
| Embedding timeout / retries | 10 s / 1 |
| Evidence budget / max items | 4,000 tokens / 8 |

Corpus: the seeded tenant (`Local Demo Engineering`) with 10 documents and 12
versions. Isolation and ACL checks also used a temporary tenant created for the
run with two documents, then deleted. The database returned to 1 tenant and 10
documents after every run (`db_counts_before` and `db_counts_final` in
`results.json`).

## Harness

| File | Purpose |
| --- | --- |
| `scripts/ops/load_test.py` | Async bounded-concurrency HTTP load harness; p50/p95/p99, throughput, status/error distribution, canary leakage check, JSONL records |
| `scripts/ops/live_failure_injection.py` | Stops and restores PostgreSQL, Redis, and the embedding service; records expected vs actual status per case |
| `scripts/ops/fake_llm_provider.py` | Deterministic OpenAI-compatible provider with `ok`, `slow`, `timeout`, `rate_limit`, `invalid_envelope`, `invalid_output`, `bad_citation`, and `server_error` modes |
| `scripts/ops/validate_logs.py` | Correlates request IDs with structured logs and scans for forbidden values |
| `scripts/ops/resource_sampler.py` | CPU and working-set sampling for a service's Python process family, plus one Docker stats snapshot |
| `scripts/ops/run_validation.py` | Orchestrates the scenarios, isolation probes, log validation, and failure injection; restores state in `finally` |
| `tests/test_failure_injection.py` | Offline failure-path coverage for dependency, provider, pool, and logging behavior |

Safety rules enforced by the harness:

- One uvicorn worker, bounded request counts (30-100 per scenario), bounded
  concurrency (1-8), and a fixed six-question rotation.
- Temporary tenant data only; tenant A data is read-only during the run.
- Containers are stopped and restarted one at a time; readiness is re-checked
  before continuing. The embedding service is restarted from the same
  interpreter and left running.
- Bearer tokens, raw tenant UUIDs, questions, document content, and API keys are
  never written to the result artifacts. Only the salted `tenant_ref` appears in
  logs.

### Reproduction

```powershell
# One-time: containers and the local embedding service.
docker compose up -d postgres redis
.venv\Scripts\python.exe scripts\embedding_service.py

# Full Phase E run (load, isolation, limits, failure injection).
.venv\Scripts\python.exe scripts\ops\run_validation.py --out-dir evals\ops\phase-e-final

# Load-only run against an existing API.
.venv\Scripts\python.exe scripts\ops\load_test.py `
  --base-url http://127.0.0.1:8010 --scenario answer --requests 100 `
  --concurrency 8 --token a=$env:OPSATLAS_TEST_API_TOKEN `
  --output evals\ops\answer.json
```

Results from the measured run are in `evals/ops/phase-e-final/`:
`results.json`, `load-*.json` plus `.records.jsonl`, `logs-*.json`,
`api-*.log`, and `failure-injection.json`.

### Verification

After the final run, the full offline suite reported **389 passed, 35 skipped**
in 2.31 s (the skips are the opt-in live tests), and `ruff check .` reported no
findings. With `OPSATLAS_LIVE_TESTS=1` and the seeded tenant, the live files
(`test_security_live.py`, `test_answer_live.py`,
`test_lexical_retrieval_live.py`) reported **35 passed**. The live security test
deleted its temporary tenant; the database was left at 1 tenant and 10
documents.

## Load results

Measured requests exclude warm-up requests (warm-up uses a separate request-ID
prefix). `stage` latencies come from the structured request log of the same
requests, matched 1:1.

| Scenario | Requests | Concurrency | Throughput (req/s) | p50 (ms) | p95 (ms) | p99 (ms) | Retrieval p95 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `/health` | 100 | 1 | 170.28 | 5.36 | 8.28 | 10.07 | n/a |
| `/ready` | 50 | 1 | 39.85 | 24.05 | 39.57 | 42.41 | n/a |
| Answer, single user | 30 | 1 | 10.93 | 89.03 | 105.30 | 112.08 | 92.55 |
| Answer, concurrent | 100 | 8 | 53.90 | 132.95 | 236.20 | 263.94 | 169.15 |
| Answer, lexical | 100 | 8 | 111.43 | 57.87 | 131.72 | 149.97 | 56.20 |
| Search, semantic | 80 | 8 | 51.65 | 137.82 | 228.15 | 237.77 | 192.05 |
| Search, hybrid | 80 | 8 | 51.76 | 140.29 | 220.50 | 232.83 | 162.05 |
| Two tenants, lexical answer | 100 | 8 | 120.24 | 56.63 | 111.89 | 179.44 | 27.15 |
| Mixed workload | 100 | 8 | 66.74 | 124.47 | 199.91 | 244.94 | 138.30 |

Mixed workload: 25% answer, 25% semantic search, 25% hybrid search, 25%
readiness.

- Every scenario returned `0` errors (740 measured requests, all `200`).
- `answer_concurrent` is 2.2x the single-user p95 because the embedding service
  and PostgreSQL both saturate first; the API process itself stayed below 30% of
  one core.
- `multi_tenant` per-label p95: tenant A 112.80 ms, tenant B 111.09 ms. No
  bias or cross-tenant slowdown was visible at 8-way concurrency.
- Evidence assembly and extractive generation round to 0-1 ms at this corpus
  size. Retrieval and the embedding round-trip dominate.
- `/ready` is slower than `/health` by design: it opens PostgreSQL and Redis
  connections (and returns a degraded `200` when only Redis fails).

### Provider-independent vs real-LLM numbers

All load numbers above use the `extractive` provider, which runs in-process and
does not call a network model. The `generation_latency_ms` p95 was 0 ms, so the
reported answer latency is retrieval + evidence assembly + server overhead.

These measurements therefore say nothing about real LLM latency, token cost,
provider rate limits, or retry behavior. The failure-path measurements below
use a local fake OpenAI-compatible provider to exercise the adapter, timeouts,
retries, and error mapping deterministically. A real provider must be measured
separately before setting answer SLOs or cost budgets.

### Resource consumption

Sampled over the 24.1 s load window. Windows on this machine launches each
Python service through a small trampoline process, so the sampler reports the
aggregate for the service's Python process family.

| Service | Processes | CPU (s) | Average CPU (one core = 100%) | Peak working set |
| --- | ---: | ---: | ---: | ---: |
| API (`uvicorn`) | 2 | 6.88 | 28.6% | 94.9 MiB |
| Embedding service | 2 | 99.72 | 414.2% (about 4.1 cores) | 1.82 GiB |

Docker stats captured once at the start of the load window:

| Container | CPU | Memory | Network I/O |
| --- | ---: | ---: | ---: |
| PostgreSQL | 0.03% | 27.3 MiB / 7.37 GiB | 57.4 kB / 43.3 kB |
| Redis | 2.38% | 5.0 MiB / 7.37 GiB | 20.1 kB / 12.4 kB |

The embedding service is the bottleneck: it used about 4.1 cores and 1.8 GiB
under 8-way answer/search load, while the API stayed under one core. The Docker
values are a single idle snapshot, not an average over the load run.

## Isolation and limits

### Authentication

| Probe | Requests | Result |
| --- | ---: | --- |
| Invalid bearer token | 30 | 30 x `401`, identical body |
| Missing bearer token | 15 | 15 x `401`, identical body |

Every request had a unique, correlated log event with
`error_category=unauthorized` and `tenant_ref=null`.

### Tenant isolation and ACL

| Check | Requests | Result |
| --- | ---: | --- |
| Tenant A queried for tenant B's canary token | 20 | 0 responses contained the canary |
| Tenant B queried for its own canary token | 20 | 20 responses contained the canary |
| Principal in group `validation-allowed` queried restricted document | 20 | 20 saw the document |
| Principal without that group queried the same document | 20 | 0 saw the document |

All 80 requests returned `200`. The ACL check used a document with an explicit
`group:validation-allowed` read grant; the denied principal received an empty
result rather than a disclosure or an error.

### Rate limiting

A dedicated API process was started with `RATE_LIMIT_PER_MINUTE=20`:

| Burst | Result |
| --- | --- |
| Tenant A, 25 sequential requests | 20 x `200`, 5 x `429`, `Retry-After: 59` |
| Tenant B, 25 sequential requests after A was exhausted | 20 x `200`, 5 x `429`, `Retry-After: 59` |
| Tenant A after both bursts | `429`, `Retry-After: 57` |
| Tenant B after both bursts | `429`, `Retry-After: 58` |

Tenant B was not penalized by tenant A's exhausted window, and the limit applied
per tenant across different paths (`/api/v1/search`) using the authenticated
tenant, not a header or body value. This is process-local: N workers would allow
up to N times the configured rate.

### Concurrency limiting

A dedicated API process was started with `MAX_CONCURRENT_REQUESTS=1` and a zero
acquire timeout:

| Check | Result |
| --- | --- |
| 8 simultaneous slow answers | 1 x `200`, 7 x `429` (`concurrency_limited`) |
| Follow-up after the burst | `200` |

Logs recorded `error_category=concurrency_limited` for every rejected request.
Admission is per process, so the effective limit also multiplies with workers.

### Client cancellation

A client sent a slow answer request with a 0.25 s timeout. The client observed
`ReadTimeout`; the server continued and logged a completed `200` after
1,524 ms (generation 1,504 ms). A follow-up request returned `200`.

The endpoint does not currently cancel in-flight synchronous generation when the
client disconnects. Work can continue until the provider call or retry policy
finishes, so a disconnected client can still consume a concurrency slot and
provider budget. This is documented as a limitation, not a guarantee.

## Failure injection

| Case | Expected | Actual | Detection / recovery |
| --- | --- | --- | --- |
| PostgreSQL stopped: `/health` | `200` | `200` in 2.9 ms | Liveness does not touch PostgreSQL |
| PostgreSQL stopped: `/ready` | `503 not_ready` | `503` in 10,031 ms | `postgres: error`, `error_category=dependency_unavailable` |
| PostgreSQL stopped: lexical answer | `503 database_unavailable` | `503` in 10,017 ms | Constant detail; no DSN/SQL leak |
| PostgreSQL restarted | `/ready` `200`; answer `200` | `200` in 38 ms; answer `200` in 106 ms | Pool pre-ping recovered without restarting the API |
| Redis stopped: `/ready` | `200 degraded` | `200` in 2,065 ms | `redis: error`, PostgreSQL still `ok` |
| Redis stopped: lexical answer | `200` | `200` in 53 ms | Redis is not on the serving path |
| Redis restarted | `/ready` `200 ready` | `200` in 29 ms | Normal state restored |
| Embedding service stopped: search | `503 embedding_unavailable` | `503` in 4,263 ms | Embedding retries exhausted; retrieval latency 4,284 ms |
| Embedding stopped: lexical answer | `200` | `200` in 33 ms | Lexical retrieval does not call the embedder |
| Embedding restarted | Search `200` | `200` in 101 ms | Same PID/log path restored |
| Provider timeout (1 s limit, 3 s response) | `504 provider_timeout`, 1 retry | `504` in 2,086 ms, `retry_count=1` | Retried once, then mapped |
| Provider `429` | `429 provider_rate_limited`, 1 retry | `429` in 74 ms, `retry_count=1` | Retried once; retry count logged |
| Invalid provider envelope | `502 invalid_model_output`, no retry | `502` in 23 ms, `retry_count=0` | Non-retryable |
| Invalid model output | `502 invalid_model_output`, no retry | `502` in 25 ms, `retry_count=0` | Parse failure after one attempt |
| Unknown citation | `502 citation_validation_error`, no retry | `502` in 27 ms, `retry_count=0` | Server rejected `E999` |
| Provider `500` | `502 provider_error`, 1 retry | `502` in 98 ms, `retry_count=1` | Retryable 5xx |
| DB pool exhaustion | `503 database_unavailable` | One request `503` in 1,015 ms; one `200` in 2,581 ms | Dedicated `DB_POOL_SIZE=1`; the blocked query completed after the lock was released |
| Pool recovery | `200` | `200` in 18 ms | Lock released; pool reused |
| Client cancellation | Document actual behavior | Server completed `200` in 1,524 ms | No server-side cancel |

All failure cases produced exactly one correlated structured log event. The
`failure_injection_log_validation` check passed 12/12.

Notable detection costs:

- A stopped PostgreSQL took about 10 s to surface because the connection attempt
  uses the configured connect timeout. Readiness probes must allow for this.
- A stopped Redis took about 2 s (two 1 s socket timeouts).
- A stopped embedder took about 4.3 s because the client retries once and then
  reports `embedding_unavailable`.

## Observability validation

`validate_logs.py` checked every scenario against the load manifest:

| Check | Result |
| --- | --- |
| Request IDs correlated 1:1 with responses | Passed for all 9 load scenarios and all probes |
| `tenant_ref` present and salted | Passed; format `t_<hash>`, no raw UUID |
| Stage latencies present and non-null | Passed for health/ready totals, retrieval on searches, and retrieval/evidence/generation on answers |
| `provider`, `observed_model`, `requested_model` | `extractive`/`extractive-local-v1` for load; `openai_compatible`/`fake-model` for fake-provider tests |
| `retry_count` | `0` on success; `1` for timeout/rate-limit/5xx; `0` for non-retryable parse/citation failures |
| `error_category` | `unauthorized`, `rate_limited`, `concurrency_limited`, `database_unavailable`, `dependency_unavailable`, `embedding_unavailable`, `provider_timeout`, `provider_rate_limited`, `invalid_model_output`, `citation_validation_error`, `provider_error` |
| `token_source` | `estimated` for `extractive`; `actual` when the fake provider reported usage |
| Forbidden values | 0 hits for bearer tokens, `Authorization`/`Bearer`, API key, raw tenant UUIDs, canary content, and questions |
| JSON integrity | 0 non-JSON request-log lines |

Token counts are honest about provenance: the deterministic provider reports no
usage, so logs show `estimated`; the OpenAI-compatible adapter shows `actual`
when the response includes `usage`.

## Regressions and fixes

One observability defect was found and fixed during Phase E:

- **Non-retryable provider faults logged `retry_count: null`.** A structurally
  invalid response envelope raised `GenerationInvalidResponseError`, which was
  not caught by the retry loop, so `attempts` was never attached. The failure
  was not retried (correct), but the log could not distinguish "zero retries"
  from "unknown". `generate_grounded_answer` now attaches `attempts` to any
  `GenerationError` that escapes the loop, so `retry_count=0` is recorded. This
  affects telemetry only; retry policy and status mapping are unchanged. The new
  offline test `test_invalid_provider_envelope_is_not_retried` covers it.

No retrieval default, evaluation dataset, baseline, schema, or API contract was
changed.

## Known limitations

1. **Single process, single host.** The measurements use one uvicorn worker on a
   laptop. Rate and concurrency limits are process-local; N workers multiply
   both. No multi-instance saturation or failover test was run.
2. **Extractive provider only for load.** Retrieval and server overhead are
   measured; real model latency, token cost, and upstream rate limiting are not.
3. **Synthetic, small corpus.** 10 documents / 12 versions. Retrieval latency and
   embedding cost will change with corpus size, chunk count, and concurrency.
4. **Short load windows.** The load phase is about 24 s, so it cannot detect
   slow leaks, pool fragmentation, or long-window rate-limit fairness.
5. **Temporary tenant is small.** Isolation/ACL probes use 20 requests per
   principal and two documents, which is enough to catch obvious leakage but not
   a statistical proof.
6. **Cancellation is not propagated.** Disconnected clients leave server work
   running until the provider call or retry policy completes.
7. **Provider retry budget is bounded but additive.** `llm_timeout_seconds=20`
   plus one retry can exceed a 30 s ingress timeout. The worst case is roughly
   40 s plus backoff.
8. **Readiness depends on connect timeout.** A dead PostgreSQL is reported after
   about 10 s; a readiness probe with a shorter timeout would flap or time out
   before the body arrives.
9. **Container stats are a point-in-time snapshot.** API and embedding process
   metrics are sampled across the load window; PostgreSQL/Redis CPU and memory
   are one `docker stats --no-stream` sample at the start.
10. **Windows process model.** Services appear as a trampoline plus a worker; the
    sampler aggregates the family. Other platforms report no process samples.
11. **No log pipeline.** Validation reads local log files. Production needs
    collection, retention, and alerting on the fields validated here.
12. **No distributed quota or shared limiter.** Redis is used for readiness only,
    not for rate-limit state.

## Recommended production thresholds

These are starting guards derived from the measurements, not measured SLOs.
Re-measure with a real provider, a production-sized corpus, and production
worker counts before publishing them as targets.

| Area | Recommendation |
| --- | --- |
| Answer latency | Alert when p95 > 1,000 ms per worker at 8 concurrent requests; page when p95 > 2,000 ms. Measured 236 ms with `extractive`. |
| Search latency | Alert when p95 > 750 ms; page when p95 > 1,500 ms. Measured 228 ms semantic / 221 ms hybrid. |
| Health | `/health` timeout 1 s, period 10 s. |
| Readiness | `/ready` timeout 15 s, period 30 s, failure threshold 2-3. PostgreSQL detection alone took about 10 s. |
| Server errors | Alert when `status_code >= 500` exceeds 1% of requests over 5 minutes, or when any `database_unavailable` / `provider_timeout` spike appears. |
| Rate limiting | Keep a per-tenant limit tied to provider budget. With N workers, either multiply the configured limit by N or move to a shared limiter; alert when `429` rate exceeds 5%. |
| Concurrency | Size `MAX_CONCURRENT_REQUESTS` to roughly 2x the measured comfortable in-flight count; alert when `concurrency_limited` exceeds 1%. |
| DB pool | Keep `instances * (DB_POOL_SIZE + DB_MAX_OVERFLOW) + 10` at or below `max_connections` (100 here; 8 instances x 10 + 10 = 90). Alert on checkout wait p95 > 100 ms. |
| Embedding service | Provision at least 4 cores per embedding instance and about 2 GiB of headroom; measured 4.1 cores and 1.82 GiB under 8-way load. Scale out before raising API concurrency. |
| Provider calls | Set the request deadline above `LLM_TIMEOUT_SECONDS * (LLM_MAX_RETRIES + 1)` plus backoff; keep retries to 1 until upstream rate limits and idempotency are measured. |
| Token accounting | Track the `estimated` share of `token_source`; require `actual` usage for billing or hard context-budget enforcement. |
| Telemetry | Alert on `retry_count > 0`, any unexpected `error_category`, missing request IDs, or `tenant_ref` that is not salted. |
| PostgreSQL | Alert on connection count, long-running transactions, and pool wait; the failure test showed a table lock can block retrieval for seconds without a 500. |

## Conclusion

Within the limits above, the stack behaved predictably:

- Health stayed independent of dependencies; readiness correctly separated
  required PostgreSQL from optional Redis.
- Tenant and ACL scoping held under concurrent access; no canary crossed a
  tenant boundary.
- Rate and concurrency limits rejected excess requests with `429` and
  `Retry-After`, per tenant and per process.
- Every injected dependency/provider failure mapped to a documented status and
  error category, produced one correlated redacted log event, and recovered
  without restarting the API.
- The embedding service, not the API process, is the first local bottleneck.

The remaining production blockers are the ones already implied by the limits:
one process-local limiter, no server-side cancellation, a synthetic/small
corpus, and no real-LLM measurement.

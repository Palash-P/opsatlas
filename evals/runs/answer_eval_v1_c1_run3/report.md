# Answer Evaluation v1 - Run Report

- Dataset: `answer_eval_v1`
- Status: `completed`
- Records: 26 (20 answerable, 6 unanswerable)
- Provider/model: `opencode_cli` / `opencode/deepseek-v4.1-flash-free`
- Prompt version: `grounded_answer_v2`
- Dataset SHA-256: `68fcd05c7a289d6bf476ec7e29a0ff3d9dfb2f334f052224f18735999f121350`
- Corpus fingerprint: `c5cb44a83cd37560f004692ec4c702d509dde3f958a42966545480ca23fe6408`
- Pipeline SHA-256: `12dfb5d42adf0394b43c4fefc1fc8bba386b43dc626735a450a1b31940e31f50`

## Overall metrics

| Metric | Score |
| --- | ---: |
| `answer_relevance` | 0.942 |
| `faithfulness` | 0.923 |
| `citation_correctness` | 0.925 |
| `citation_completeness` | 1.000 |
| `abstention_correctness` | 0.962 |
| `conflict_handling` | 1.000 |

`n` is the number of records in the block. Metric values average only non-null scores.

## Per-category metrics

| Category | n | Relevance | Faithfulness | Citation correctness | Citation completeness | Abstention | Conflict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `distractor` | 2 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | n/a |
| `insufficient_evidence` | 4 | 0.75 | 0.75 | n/a | n/a | 0.75 | n/a |
| `multi_chunk` | 2 | 1.00 | 1.00 | 0.75 | 1.00 | 1.00 | n/a |
| `multi_source` | 3 | 1.00 | 0.67 | 0.67 | 1.00 | 1.00 | n/a |
| `sensitive_unanswerable` | 2 | 1.00 | 1.00 | n/a | n/a | 1.00 | n/a |
| `single_source_factual` | 9 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | n/a |
| `version_conflict` | 4 | 0.88 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

## Independent scorers

| Scorer | Model | Relevance | Faithfulness | Citation correctness | Citation completeness | Abstention | Conflict | Claim coverage |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| scorer_a | `opencode/glm-5.3-flash-free` | 0.94 | 0.94 | 0.97 | 1.00 | 0.96 | 1.00 | 0.95 |
| scorer_b | `opencode/mimo-v2.6-flash-free` | 0.96 | 0.96 | 0.93 | 1.00 | 0.96 | 1.00 | 0.90 |

Inter-scorer exact agreement: 149/156 (0.9551). Disagreements are adjudicated in `disagreements.json`.

## Failure attribution

| Primary failure | Records |
| --- | ---: |
| `citation_mismatch` | 3 |
| `incomplete_generation` | 2 |

## Provider latency and usage

- Latency p50/p95/max: 6388.0 / 33193.0 / 94958.0 ms
- Provider attempts: 26; retries: 0
- Provider failure rate: 0.0
- Generation-contract failures: 1
- Token estimates (input/output): 51201 / 4661

## Per-question results

| Record | Category | Status | Abstained | Primary failure | Relevance | Faithfulness | Citation corr. | Citation comp. | Abstention | Conflict |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `answer-v1-pay503-initial-checks` | `single_source_factual` | ok | False | `none` | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | n/a |
| `answer-v1-pay400-resolution` | `single_source_factual` | ok | False | `none` | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | n/a |
| `answer-v1-pay409-reconcile` | `single_source_factual` | ok | False | `none` | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | n/a |
| `answer-v1-v3-endpoint` | `single_source_factual` | ok | False | `none` | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | n/a |
| `answer-v1-kafka-consumer-recovery` | `single_source_factual` | ok | False | `none` | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | n/a |
| `answer-v1-redis-restore-traffic` | `single_source_factual` | ok | False | `none` | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | n/a |
| `answer-v1-sev1-postmortem` | `single_source_factual` | ok | False | `none` | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | n/a |
| `answer-v1-search-rollback-verification` | `single_source_factual` | ok | False | `none` | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | n/a |
| `answer-v1-search-health-insufficient` | `single_source_factual` | ok | False | `none` | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | n/a |
| `answer-v1-pay503-and-spec` | `multi_source` | ok | False | `none` | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | n/a |
| `answer-v1-endpoint-and-idempotency` | `multi_source` | ok | False | `citation_mismatch` | 1.0 | 0.5 | 0.5 | 1.0 | 1.0 | n/a |
| `answer-v1-deploy-outage-escalation` | `multi_source` | ok | False | `citation_mismatch` | 1.0 | 0.5 | 0.5 | 1.0 | 1.0 | n/a |
| `answer-v1-postgres-diagnostic-and-mitigation` | `multi_chunk` | ok | False | `none` | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | n/a |
| `answer-v1-payment-errors-multi-code` | `multi_chunk` | ok | False | `citation_mismatch` | 1.0 | 1.0 | 0.5 | 1.0 | 1.0 | n/a |
| `answer-v1-runbook-version-difference` | `version_conflict` | ok | False | `incomplete_generation` | 0.5 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| `answer-v1-endpoint-version-conflict` | `version_conflict` | ok | False | `none` | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| `answer-v1-migration-version-guidance` | `version_conflict` | ok | False | `none` | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | n/a |
| `answer-v1-v2-rollback-instruction` | `version_conflict` | ok | False | `none` | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | n/a |
| `answer-v1-unans-postgres-major` | `insufficient_evidence` | ok | True | `none` | 1.0 | 1.0 | n/a | n/a | 1.0 | n/a |
| `answer-v1-unans-revenue-loss` | `insufficient_evidence` | ok | True | `none` | 1.0 | 1.0 | n/a | n/a | 1.0 | n/a |
| `answer-v1-unans-incident-timestamp` | `insufficient_evidence` | error | False | `incomplete_generation` | 0.0 | 0.0 | n/a | n/a | 0.0 | n/a |
| `answer-v1-unans-root-cause` | `insufficient_evidence` | ok | True | `none` | 1.0 | 1.0 | n/a | n/a | 1.0 | n/a |
| `answer-v1-sensitive-api-key` | `sensitive_unanswerable` | ok | True | `none` | 1.0 | 1.0 | n/a | n/a | 1.0 | n/a |
| `answer-v1-sensitive-customer-cards` | `sensitive_unanswerable` | ok | True | `none` | 1.0 | 1.0 | n/a | n/a | 1.0 | n/a |
| `answer-v1-distractor-kafka-vs-db-pool` | `distractor` | ok | False | `none` | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | n/a |
| `answer-v1-distractor-redis-ping` | `distractor` | ok | False | `none` | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | n/a |

## Limitations and recommendations

- The corpus is synthetic; no production validation is claimed.
- Scoring was performed by two language models using a fixed rubric, not by two human domain experts.
- Claim-level citation coverage is reported per scorer; disagreements were adjudicated at the record-metric level.
- The evaluation provider is an evaluation-only OpenCode CLI adapter.
- Provider token usage was not exposed by the CLI; token counts are character-based estimates.
- Scorer model identity is the requested model unless the CLI event stream reports a different one.
- Repeated runs characterize this provider and model only; they do not establish stability across provider versions or model updates.

Recommendations are recorded in `docs/ANSWER_EVALUATION_V1.md` after reviewing this report.

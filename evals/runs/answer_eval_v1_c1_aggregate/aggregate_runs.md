# Answer Evaluation v1 - Repeated-Run Aggregate

- Runs: 3
- Frozen configuration held: `True`

## Per-run metrics

| Run | Status | Records | Provider failure rate | Relevance | Faithfulness | Citation corr. | Citation comp. | Abstention | Conflict |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `20261105T202032Z` | completed | 26 | 0.000 | 0.885 | 0.923 | 0.875 | 1.000 | 0.923 | 1.000 |
| `20261106T112007Z` | completed | 26 | 0.000 | 0.923 | 0.962 | 0.975 | 1.000 | 0.962 | 1.000 |
| `20261107T083709Z` | completed | 26 | 0.000 | 0.942 | 0.923 | 0.925 | 1.000 | 0.962 | 1.000 |

## Cross-run metric summary

| Metric | n | Mean | Min | Max | Stdev |
| --- | ---: | ---: | ---: | ---: | ---: |
| `answer_relevance` | 3 | 0.9167 | 0.8846 | 0.9423 | 0.0294 |
| `faithfulness` | 3 | 0.9359 | 0.9231 | 0.9615 | 0.0222 |
| `citation_correctness` | 3 | 0.9250 | 0.8750 | 0.9750 | 0.0500 |
| `citation_completeness` | 3 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| `abstention_correctness` | 3 | 0.9487 | 0.9231 | 0.9615 | 0.0222 |
| `conflict_handling` | 3 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |

## Provider reliability

- Records: 78
- Errored generations: 4
- Provider transport/envelope failures: 0
- Generation-contract failures (invalid model output, not provider downtime): 4
- Overall provider failure rate: 0.0
- Overall generation-contract failure rate: 0.0513

| Run | Records | Errors | Provider failures | Contract failures | Provider rate | Attempts | Retries |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `20261105T202032Z` | 26 | 2 | 0 | 2 | 0.0 | 26 | 0 |
| `20261106T112007Z` | 26 | 1 | 0 | 1 | 0.0 | 26 | 0 |
| `20261107T083709Z` | 26 | 1 | 0 | 1 | 0.0 | 26 | 0 |

## Latency and tokens

| Measure | Mean | Min | Max | Stdev |
| --- | ---: | ---: | ---: | ---: |
| Latency p50 (ms) | 6963.8 | 5375.0 | 9128.5 | 1941.9 |
| Latency p95 (ms) | 32179.6 | 30881.4 | 33193.0 | 1181.8 |
| Latency mean (ms) | 11338.4 | 10283.5 | 11959.5 | 918.4 |
| Latency max (ms) | 66883.0 | 47964.0 | 94958.0 | 24798.9 |
| Token estimate input (total) | 151730 | | | |
| Token estimate output (total) | 13291 | | | |

## Failure categories

| Failure | Total |
| --- | ---: |
| `citation_mismatch` | 9 |
| `incomplete_generation` | 7 |

## Scorer disagreement

- Total adjudicated disagreements: 21

| Run | Disagreements |
| --- | ---: |
| `20261105T202032Z` | 8 |
| `20261106T112007Z` | 6 |
| `20261107T083709Z` | 7 |

## Run metadata

| Run | Model | Prompt | Dataset SHA-256 | Corpus fingerprint |
| --- | --- | --- | --- | --- |
| `20261105T202032Z` | `opencode/deepseek-v4.1-flash-free` | `grounded_answer_v2` | `68fcd05c7a289d6bf476ec7e29a0ff3d9dfb2f334f052224f18735999f121350` | `c5cb44a83cd37560f004692ec4c702d509dde3f958a42966545480ca23fe6408` |
| `20261106T112007Z` | `opencode/deepseek-v4.1-flash-free` | `grounded_answer_v2` | `68fcd05c7a289d6bf476ec7e29a0ff3d9dfb2f334f052224f18735999f121350` | `c5cb44a83cd37560f004692ec4c702d509dde3f958a42966545480ca23fe6408` |
| `20261107T083709Z` | `opencode/deepseek-v4.1-flash-free` | `grounded_answer_v2` | `68fcd05c7a289d6bf476ec7e29a0ff3d9dfb2f334f052224f18735999f121350` | `c5cb44a83cd37560f004692ec4c702d509dde3f958a42966545480ca23fe6408` |

## Scorer identity

| Run | Scorer A actual | Scorer A configured | Scorer B actual | Scorer B configured | Deviations |
| --- | --- | --- | --- | --- | --- |
| `20261105T202032Z` | `opencode/glm-5.3-flash-free` | `opencode/glm-5.3-flash-free` | `opencode/mimo-v2.6-flash-free` | `opencode/flint-40b-free` | scorer_b: configured opencode/flint-40b-free, used opencode/mimo-v2.6-flash-free |
| `20261106T112007Z` | `opencode/glm-5.3-flash-free` | `opencode/glm-5.3-flash-free` | `opencode/mimo-v2.6-flash-free` | `opencode/flint-40b-free` | scorer_b: configured opencode/flint-40b-free, used opencode/mimo-v2.6-flash-free |
| `20261107T083709Z` | `opencode/glm-5.3-flash-free` | `opencode/glm-5.3-flash-free` | `opencode/mimo-v2.6-flash-free` | `opencode/flint-40b-free` | scorer_b: configured opencode/flint-40b-free, used opencode/mimo-v2.6-flash-free |

## Resumed runs

- `20261107T083709Z`: reused 11 records, reran 15 records (previous run `20261107T083458Z`).

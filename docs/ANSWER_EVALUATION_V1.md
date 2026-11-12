# Answer Evaluation v1 (Measured)

> Status: **completed.** Three measured runs of all 26 records exist under
> `evals/runs/answer_eval_v1_c1_run{1,2,3}/`. Each report validates against
> `evals/answer_eval_v1.report_schema.json` and reports `status: "completed"`.
> The corpus is synthetic, so the values describe this corpus and pipeline only;
> they do not establish production quality.

## Purpose

`answer_eval_v1` evaluates the complete grounded-answer path, not retrieval
alone. It covers:

- answer relevance;
- faithfulness to supplied evidence;
- citation correctness;
- citation completeness;
- abstention behavior;
- conflict handling across document versions.

Retrieval-only metrics remain in the existing `golden_v*` reports. This dataset
does not replace them and does not modify any golden dataset.

Phase C.1 additionally measures:

- version-token retrieval and version-aware ranking;
- a version-aware answer contract that names versions and separates supported
  differences from unsupported claims;
- provider reliability and failure categories;
- repeated-run variance instead of a single point estimate.

## Files

| File | Purpose |
| --- | --- |
| `evals/datasets/answer_eval_v1.jsonl` | 26 labelled questions |
| `evals/datasets/answer_eval_v1.profile.json` | counts and validation rules |
| `evals/answer_eval_v1.report_schema.json` | JSON Schema for a run report |
| `evals/answer_eval_v1.scoring_rubric.md` | frozen scorer rubric |
| `evals/answer_eval.py` | dataset validation and failure-type constants |
| `evals/opencode_provider.py` | evaluation-only OpenCode CLI adapter |
| `scripts/run_answer_eval.py` | generation run, configuration freeze, resume |
| `scripts/score_answer_eval.py` | independent scorer runner, resume |
| `scripts/answer_eval_report.py` | adjudication gate, metrics, failure analysis |
| `scripts/aggregate_answer_eval_runs.py` | repeated-run aggregation and comparability checks |
| `evals/runs/answer_eval_v1_main/` | prior single-run baseline artifacts |
| `evals/runs/answer_eval_v1_c1_run1/` | Phase C.1 measured run 1 |
| `evals/runs/answer_eval_v1_c1_run2/` | Phase C.1 measured run 2 |
| `evals/runs/answer_eval_v1_c1_run3/` | Phase C.1 measured run 3 (resumed after a provider quota interruption) |
| `evals/runs/answer_eval_v1_c1_aggregate/` | cross-run aggregate |
| `tests/test_answer_eval_dataset.py` | dataset consistency and no-drift tests |
| `tests/test_answer_eval_report.py` | attribution and agreement regression tests |
| `tests/test_answer_eval_aggregate.py` | repeated-run aggregation regression tests |
| `tests/test_answer_eval_runner.py` | generation resume-path regression tests |
| `tests/test_answer_eval_scorer.py` | scoring resume-path regression tests |
| `tests/test_answer_security.py` | answer-path threat-model tests |

The dataset is tied to `synthetic_v1` and the corpus fingerprint
`c5cb44a83cd37560f004692ec4c702d509dde3f958a42966545480ca23fe6408`. Source
keys and versions are validated against the synthetic corpus manifest.

## Dataset composition

| Category | Count | What it probes |
| --- | --- | --- |
| `single_source_factual` | 9 | One source contains the complete answer |
| `multi_source` | 3 | Two required sources, including payment and incident pairs |
| `multi_chunk` | 2 | Multiple passages from one document must be preserved |
| `version_conflict` | 4 | Mixed versions, stale guidance, and version-specific citations |
| `insufficient_evidence` | 4 | The corpus genuinely lacks the requested fact |
| `sensitive_unanswerable` | 2 | Credentials or customer data must not be invented |
| `distractor` | 2 | Shared vocabulary where only one source owns the process |

Totals: 26 records, 20 answerable, 6 unanswerable.

### Record fields

- `id`: stable identifier.
- `question`: user question.
- `category`: one of the categories above.
- `answerable`: whether evidence exists in the corpus.
- `expected_behavior`: `answer`, `abstain`, or `identify_conflict`.
- `expected_facts`: required facts with source key and version.
- `expected_citations`: sources that should be cited, with exact supporting text
  that exists in the corpus.
- `min_required_citations`: minimum required citations for an answer.
- `conflict_expected`: whether the answer must surface a version conflict.
- `abstention_reason`: expected reason for unanswerable records, otherwise null.
- `notes`: author rationale.

## Metrics and rubric

Each record is scored on six dimensions. Scores are in `[0, 1]` or `null` when
not scored. `null` means "not evaluated", not zero.

| Metric | Definition |
| --- | --- |
| `answer_relevance` | Does the answer address the question and include expected facts without unrelated material? Unanswerable questions score 1 when the system abstains. |
| `faithfulness` | Is every factual claim supported by a supplied chunk, with no outside knowledge or invented procedure? |
| `citation_correctness` | Do citations reference chunks actually supplied, with the correct source key, version, and chunk ID? Does the excerpt support the claim? |
| `citation_completeness` | Are all required sources and versions cited? |
| `abstention_correctness` | Binary: abstain exactly when unanswerable, answer when sufficient evidence exists. Applied to every record. |
| `conflict_handling` | For conflict records and mixed-version evidence: are the conflict and versions identified, without presenting stale guidance as current? `null` where version conflict is not applicable. |

Scoring guidance is frozen in `evals/answer_eval_v1.scoring_rubric.md`. Scorers
see the rubric, labels, and one prediction; they never see each other's scores.

## Phase C.1 measured runs

### Frozen configuration

| Item | Value |
| --- | --- |
| Run IDs | `20261105T202032Z`, `20261106T112007Z`, `20261107T083709Z` |
| Code revision | `4d59a74c9c59f23fd93250d4a570691c280b0f63` (dirty) |
| Pipeline SHA-256 | `12dfb5d42adf0394b43c4fefc1fc8bba386b43dc626735a450a1b31940e31f50` |
| Dataset SHA-256 | `68fcd05c7a289d6bf476ec7e29a0ff3d9dfb2f334f052224f18735999f121350` |
| Rubric SHA-256 | `ec4ba01e8df4d46c9aa90a502424b002f5117fdb753a8e4134bfbdff2b027239` |
| Corpus fingerprint | `c5cb44a83cd37560f004692ec4c702d509dde3f958a42966545480ca23fe6408` |
| Retrieval | `hybrid`, keyword `and`, fusion `chunk`, weights `1.0/1.0`, RRF `k=60`, `limit=5` |
| Embedding | `sentence-transformers/all-MiniLM-L6-v2`, 384 dimensions |
| Identifier weight | `2.0` |
| Evidence | context budget `4000`, `max_evidence=8`, excerpt `600` chars, `ceil(len/4)` tokens |
| Generation provider | `opencode_cli` (evaluation only) |
| Generation model | `opencode/deepseek-v4.1-flash-free` |
| Prompt version | `grounded_answer_v2` |
| Temperature | `0.0` requested, **not enforceable** through the CLI adapter |
| Max output tokens | `700`, **soft prompt limit only** on the CLI adapter |
| Generation retries | `2` with `15 s` exponential backoff |
| Tenant | `96e52951-46e1-4547-9103-b82de97e6822` |
| Scorer A | `opencode/glm-5.3-flash-free` |
| Scorer B | `opencode/mimo-v2.6-flash-free` (configured as `opencode/flint-40b-free`; deviation recorded) |

The aggregate reports `frozen_configuration_held: true`: dataset hash, corpus
fingerprint, prompt version, and retrieval configuration are identical across
the three runs.

Run 3 was resumed after a provider quota interruption and an embedding-service
restart. Its `config.json` records `resume.enabled=true`,
`reused_records=11`, `rerun_records=15`. Only records with `status="ok"` were
reused; failed records were regenerated. No labels or predictions were edited by
hand.

### Commands

```powershell
# Generation run (repeat with a distinct output directory for each run)
.venv\Scripts\python.exe scripts\run_answer_eval.py `
  --tenant-id 96e52951-46e1-4547-9103-b82de97e6822 `
  --model opencode/deepseek-v4.1-flash-free `
  --max-retries 2 --retry-backoff-seconds 15 `
  --inter-record-delay-seconds 3 `
  --output-dir evals\runs\answer_eval_v1_c1_run1

# Resume an interrupted generation run without regenerating successful records
.venv\Scripts\python.exe scripts\run_answer_eval.py `
  --tenant-id 96e52951-46e1-4547-9103-b82de97e6822 `
  --model opencode/deepseek-v4.1-flash-free `
  --max-retries 2 --retry-backoff-seconds 15 `
  --inter-record-delay-seconds 3 --resume `
  --output-dir evals\runs\answer_eval_v1_c1_run3

# Two independent scorers, separate sessions
.venv\Scripts\python.exe scripts\score_answer_eval.py `
  --predictions evals\runs\answer_eval_v1_c1_run1\predictions.jsonl `
  --scorer-id scorer_a --model opencode/glm-5.3-flash-free --batch-size 3

.venv\Scripts\python.exe scripts\score_answer_eval.py `
  --predictions evals\runs\answer_eval_v1_c1_run1\predictions.jsonl `
  --scorer-id scorer_b --model opencode/mimo-v2.6-flash-free `
  --batch-size 2 --timeout-seconds 600 --max-attempts 4 `
  --retry-backoff-seconds 15 --inter-batch-delay-seconds 5

# Resume a scorer run, regenerating only failed batches
.venv\Scripts\python.exe scripts\score_answer_eval.py `
  --predictions evals\runs\answer_eval_v1_c1_run1\predictions.jsonl `
  --scorer-id scorer_b --model opencode/mimo-v2.6-flash-free `
  --batch-size 2 --timeout-seconds 600 --max-attempts 4 `
  --retry-backoff-seconds 15 --inter-batch-delay-seconds 5 --resume

# First report invocation writes adjudication.jsonl and exits; fill it and rerun
.venv\Scripts\python.exe scripts\answer_eval_report.py `
  --predictions evals\runs\answer_eval_v1_c1_run1\predictions.jsonl `
  --scores-a evals\runs\answer_eval_v1_c1_run1\scores_scorer_a.jsonl `
  --scores-b evals\runs\answer_eval_v1_c1_run1\scores_scorer_b.jsonl `
  --adjudication evals\runs\answer_eval_v1_c1_run1\adjudication.jsonl `
  --output-dir evals\runs\answer_eval_v1_c1_run1

# Cross-run aggregate
.venv\Scripts\python.exe scripts\aggregate_answer_eval_runs.py `
  --run evals\runs\answer_eval_v1_c1_run1 `
  --run evals\runs\answer_eval_v1_c1_run2 `
  --run evals\runs\answer_eval_v1_c1_run3 `
  --output-dir evals\runs\answer_eval_v1_c1_aggregate
```

### Overall metrics per run

`n = 26` for every metric. Values average non-null scores only.

| Run | Relevance | Faithfulness | Citation corr. | Citation comp. | Abstention | Conflict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Run 1 | 0.8846 | 0.9231 | 0.8750 | 1.0000 | 0.9231 | 1.0000 |
| Run 2 | 0.9231 | 0.9615 | 0.9750 | 1.0000 | 0.9615 | 1.0000 |
| Run 3 | 0.9423 | 0.9231 | 0.9250 | 1.0000 | 0.9615 | 1.0000 |

### Cross-run summary

| Metric | n | Mean | Min | Max | Stdev |
| --- | ---: | ---: | ---: | ---: | ---: |
| `answer_relevance` | 3 | 0.9167 | 0.8846 | 0.9423 | 0.0294 |
| `faithfulness` | 3 | 0.9359 | 0.9231 | 0.9615 | 0.0222 |
| `citation_correctness` | 3 | 0.9250 | 0.8750 | 0.9750 | 0.0500 |
| `citation_completeness` | 3 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| `abstention_correctness` | 3 | 0.9487 | 0.9231 | 0.9615 | 0.0222 |
| `conflict_handling` | 3 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |

Citation correctness has the widest spread (`0.8750`–`0.9750`); relevance,
faithfulness, and abstention vary by roughly one record. That is a
run-to-run range for **this** provider and model, not a general stability claim.

### Per-category metrics (mean across runs)

| Category | n | Relevance | Faithfulness | Citation corr. | Citation comp. | Abstention | Conflict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `single_source_factual` | 9 | 1.00 | 1.00 | 0.98 | 1.00 | 1.00 | n/a |
| `multi_source` | 3 | 0.94 | 0.89 | 0.72 | 1.00 | 1.00 | n/a |
| `multi_chunk` | 2 | 0.92 | 1.00 | 0.92 | 1.00 | 1.00 | n/a |
| `version_conflict` | 4 | 0.88 | 1.00 | 0.96 | 1.00 | 1.00 | 1.00 |
| `insufficient_evidence` | 4 | 0.75 | 0.75 | n/a | n/a | 0.75 | n/a |
| `sensitive_unanswerable` | 2 | 0.83 | 0.83 | n/a | n/a | 0.83 | n/a |
| `distractor` | 2 | 1.00 | 1.00 | 0.92 | 1.00 | 1.00 | n/a |

The `insufficient_evidence` and `sensitive_unanswerable` values are pulled down
by the generation-contract errors, not by invented answers: failed records score
0 on abstention. The `distractor` records now answer in all three runs; the prior
baseline lost both to provider errors.

### Version-specific retrieval and answer contract

- For `answer-v1-v2-rollback-instruction`, `runbooks/payment-api.md@v2` is
  present in the evidence in all three runs at evidence position 2. The prior
  baseline missed it entirely.
- The answer names version 2 and cites the version-2 chunk in all three runs.
- The version-aware prompt adds `version_analysis` with
  `versions_identified`, `differences`, `conflicting_instructions`,
  `recommendation_applies_to`, and `supporting_evidence_ids`.
- Version references are validated against supplied evidence IDs. A fabricated
  evidence ID is rejected; the answer is not accepted on the model's word.
- Mixed-version answers identify each supplied version separately and mark
  unsupported comparisons as unsupported. Run 3 surfaced a real wording
  conflict (`must use` in the migration guide vs `should use` in the v3 spec)
  and recorded it in `conflicting_instructions` instead of choosing silently.
- `conflict_handling` is `1.0000` in all three runs. The remaining version
  weakness is relevance: `answer-v1-runbook-version-difference` cannot state
  what changed from v1 because v1 was not supplied. That is adjudicated `0.5`
  for relevance, not a conflict-handling failure.

### Independent scorers

| Scorer | Model | Note |
| --- | --- | --- |
| A | `opencode/glm-5.3-flash-free` | Requested and used |
| B | `opencode/mimo-v2.6-flash-free` | Configured as `opencode/flint-40b-free`; deviation recorded in `config.json` under `scoring.model_deviations` |

Each scorer ran in a separate OpenCode session and received only the rubric,
labels, and one prediction. Inter-scorer exact agreement:

| Run | Exact | Records | Rate |
| --- | ---: | ---: | ---: |
| Run 1 | 148 | 156 | 0.9487 |
| Run 2 | 150 | 156 | 0.9615 |
| Run 3 | 149 | 156 | 0.9551 |
| Pooled | 447 | 468 | 0.9551 |

### Adjudication

| Run | Metric-level disagreements |
| --- | ---: |
| Run 1 | 8 |
| Run 2 | 6 |
| Run 3 | 7 |
| Total | 21 |

Every disagreement was filled with `resolved: true`, an adjudicated score, and a
note before the report builder accepted it. Consistent policies:

- A faithful answer that omits a required fact scores `0.5` relevance and is
  attributed to `incomplete_generation`, not hallucination.
- An answer that attributes a supported claim to an extra, weak citation scores
  `0.5` citation correctness even when the claim itself is supported.
- A failed generation scores `0` on abstention and `0` on faithfulness because
  there is no content to credit. It is reported as `incomplete_generation`.
- `answer-v1-runbook-version-difference` scores `0.5` relevance because the
  required v1 comparison cannot be made from the supplied evidence, but `1.0`
  conflict handling because v2 is named correctly and no stale guidance is
  presented as current.

### Failure attribution

Primary failure order is provider, retrieval/context, evidence assembly,
abstention, conflict, citation, incomplete generation, hallucination. A record
may have multiple failures; the report counts the first.

| Primary failure | Run 1 | Run 2 | Run 3 | Total |
| --- | ---: | ---: | ---: | ---: |
| `provider_failure` | 0 | 0 | 0 | 0 |
| `retrieval_failure` | 0 | 0 | 0 | 0 |
| `context_budget_omission` | 0 | 0 | 0 | 0 |
| `evidence_assembly_failure` | 0 | 0 | 0 | 0 |
| `abstention_failure` | 0 | 0 | 0 | 0 |
| `conflict_handling_failure` | 0 | 0 | 0 | 0 |
| `citation_mismatch` | 5 | 1 | 3 | 9 |
| `incomplete_generation` | 2 | 3 | 2 | 7 |
| `generation_hallucination` | 0 | 0 | 0 | 0 |

The failure taxonomy now separates `incomplete_generation` from
`generation_hallucination`:

- invalid model output or a faithful omission is `incomplete_generation`;
- a fabricated citation is `citation_mismatch`;
- missing required evidence is `retrieval_failure` or
  `context_budget_omission`;
- a transport/envelope failure is `provider_failure`.

All four generation errors across the three runs were
`InvalidModelOutputError: abstaining answer must not cite evidence IDs`. They are
generation-contract failures, not provider downtime and not hallucination. The
error records are:

- Run 1: `answer-v1-unans-root-cause`, `answer-v1-sensitive-api-key`.
- Run 2: `answer-v1-unans-root-cause`.
- Run 3: `answer-v1-unans-incident-timestamp`.

### Provider reliability

| Measure | Run 1 | Run 2 | Run 3 | Aggregate |
| --- | ---: | ---: | ---: | ---: |
| Records | 26 | 26 | 26 | 78 |
| Successful generations | 24 | 25 | 25 | 74 |
| Transport/envelope failures | 0 | 0 | 0 | 0 |
| Generation-contract failures | 2 | 1 | 1 | 4 |
| Provider failure rate | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Generation-contract failure rate | 0.0769 | 0.0385 | 0.0385 | 0.0513 |

The provider failure rate counts only `GenerationTimeoutError`,
`GenerationRateLimitError`, `GenerationProviderError`, and
`GenerationInvalidResponseError`. Invalid model output is deliberately excluded
so provider downtime is not conflated with a schema violation.

Operational interruptions are recorded separately from the final run summaries:

- Run 3's first attempt lost 15 records to provider exits and an embedding-service
  restart. The run was resumed from 11 successful predictions; the final
  predictions contain no transport failures.
- Run 1's scorer B lost 11 of 26 scores to provider exits. Scoring was resumed
  and completed with no `scoring_error`.
- The final transport failure rate of `0.0` therefore describes the recovered
  artifacts. It does **not** mean the free-tier provider was reliable during the
  experiment.

Provider behavior that is measured and recorded:

- Temperature `0.0` is requested but not enforceable through the CLI adapter.
- Max output tokens `700` is a soft prompt limit only.
- Token usage was reported for 1 of 74 successful generations; the rest are
  character-based estimates.
- The requested and observed generation model match in every run; no silent
  model substitution occurred.
- The OpenAI-compatible adapter enforces temperature and `max_tokens`, reads
  reported model and usage, maps 429/timeouts/auth/5xx categories, and retries
  only retryable failures. It is not used by these measured runs.
- The deterministic `extractive` provider remains available as a test fallback,
  not a quality baseline.

### Latency and token usage

| Measure | Run 1 | Run 2 | Run 3 | Mean |
| --- | ---: | ---: | ---: | ---: |
| Latency p50 (ms) | 9128.5 | 5375.0 | 6388.0 | 6963.8 |
| Latency p95 (ms) | 32464.5 | 30881.4 | 33193.0 | 32179.6 |
| Latency mean (ms) | 11772.3 | 10283.5 | 11959.5 | 11338.4 |
| Latency max (ms) | 47964 | 57727 | 94958 | 66883.0 |
| Token estimate input | 49271 | 51258 | 51201 | 151730 total |
| Token estimate output | 4497 | 4133 | 4661 | 13291 total |
| Provider attempts | 26 | 26 | 26 | 78 |
| Provider retries | 0 | 0 | 0 | 0 |

The p95 tail is dominated by free-tier provider latency, not retrieval or
evidence assembly. Token counts are estimates except where the provider reported
usage.

### Prior single-run baseline

The earlier Phase C baseline remains at `evals/runs/answer_eval_v1_main/`
(prompt `grounded_answer_v1`, one run):

| Metric | Baseline |
| --- | ---: |
| `answer_relevance` | 0.8462 |
| `faithfulness` | 0.9231 |
| `citation_correctness` | 0.8500 |
| `citation_completeness` | 0.8947 |
| `abstention_correctness` | 0.8846 |
| `conflict_handling` | 0.7500 |

That run lost the v2 rollback retrieval and two distractor records to provider
errors. Phase C.1 fixes the retrieval miss, answers the distractor records, and
raises conflict handling to `1.0000`. The baseline is retained for comparison and
is not overwritten.

## Reproducing the report

The generation run requires PostgreSQL, Redis, the embedding service, and the
synthetic corpus. Scoring requires the OpenCode CLI and network access to the
configured models. To rebuild a report from checked artifacts, run the
`answer_eval_report.py` command above; it validates that every record has two
scores and that every disagreement is adjudicated before writing `report.json`.

Validate generated reports against the schema:

```powershell
.venv\Scripts\python.exe -c "import json, jsonschema; schema=json.load(open('evals/answer_eval_v1.report_schema.json', encoding='utf-8')); [jsonschema.validate(json.load(open(f'evals/runs/answer_eval_v1_c1_{run}/report.json', encoding='utf-8')), schema) for run in ('run1','run2','run3')]; print('valid')"
```

The report is not committed as a source of truth for model quality. It is an
artifact of three runs against a synthetic corpus.

## Security review

`docs/ANSWER_SECURITY_REVIEW.md` records the focused threat-model review:
cross-tenant retrieval and citation leakage, prompt injection, evidence metadata
forgery, version confusion, and provider error leakage. `tests/test_answer_security.py`
adds ten adversarial cases; `tests/test_answer_eval_report.py` adds attribution
and agreement regression tests. No test proves that the endpoint is safe to
expose publicly. Phase D later added bearer-token authentication, server-resolved
tenants, ACL-scoped retrieval, and adversarial isolation tests; this evaluation
was not re-run after that change, and its quality metrics remain the Phase C.1
baseline.

## Known limitations

- The corpus is synthetic and small; results do not establish production
  quality.
- Three runs use one generation model and one provider. They establish a
  run-to-run range for this setup only.
- Scoring was performed by two language models using a fixed rubric, not by
  human domain experts.
- Claim-level citation coverage is reported per scorer; disagreements were
  adjudicated at the record-metric level, not at the claim level.
- The evaluation provider is an evaluation-only OpenCode CLI adapter, not a
  production provider. Temperature is not enforceable and max output tokens is a
  soft prompt limit.
- Provider token usage was mostly not reported; token counts are estimates.
- The final provider failure rate excludes generated-but-invalid output and
  excludes recovered interruptions. It is not an uptime measure.
- Generation-contract failures persist at `4/78` (`0.0513`). All four are
  abstention answers that cited evidence IDs.
- The version retrieval fix is tested on synthetic fixtures; it is not evidence
  that arbitrary version questions rank correctly.
- Authorization is absent; the endpoint must not be exposed publicly.

## Recommendations

1. Add human spot-checks on a sample of records. Two model scorers agree often
   enough to be useful but are not a substitute for domain review.
2. Repeat the measurement with a provider that enforces temperature and output
   limits and reports token usage, then compare against this range.
3. Fix the remaining generation-contract failure: an abstaining answer must not
   cite evidence IDs. The validator is correct; the prompt or a bounded repair
   step should make the contract easier to satisfy.
4. Expand version-conflict cases where both versions are supplied, so the
   "adds" comparison can be scored instead of bounded as an evidence limitation.
5. Implement authentication and derive tenant identity from the principal
   before any public use.
6. Add request-level observability and retention rules before persisting answer
   runs.

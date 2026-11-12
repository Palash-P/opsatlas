# Held-Out Evaluation v2

## Purpose

`golden_v1` was used to design the OR+IDF lexical strategy. Its 16 queries can
diagnose implementation behavior, but they cannot support a default-selection
decision. `golden_v2` is a separate 35-record dataset built after the lexical v1
implementation and evaluated without changing retrieval behavior.

The question this run answers is narrow:

> On a second, independently authored query set over the same synthetic corpus,
> do OR+IDF lexical retrieval and source-aware fusion outperform the existing
> semantic, AND-keyword, and chunk-level RRF paths?

No claim of generalization beyond this corpus and query set is made.

## Dataset

| Property | Value |
| --- | --- |
| Records | 35 |
| Answerable | 29 |
| Unanswerable | 6 |
| Categories | 7 answerable, 3 unanswerable |
| Corpus | `synthetic_v1`, 10 source documents, 12 versions, 14 current chunks |
| Corpus fingerprint | `c5cb44a83cd37560f004692ec4c702d509dde3f958a42966545480ca23fe6408` |

Category counts:

| Category | Count | What it probes |
| --- | --- | --- |
| `paraphrase_no_rare_term` | 5 | Questions that avoid the corpus's distinctive terms |
| `multi_source` | 5 | Two required sources, with at least eight competing sources |
| `identifier_shared` | 3 | One identifier defined in two or three current documents |
| `status_vs_identifier` | 3 | HTTP status codes versus incident codes such as `PAY-503` and `SEV-1` |
| `api_version_url` | 5 | `v1`/`v3` inside URLs and versioning guidance shared by two specs |
| `long_noisy` | 4 | Long questions with repeated terms and distractor vocabulary |
| `multi_chunk_source` | 4 | Sources that contribute two chunks to the candidate pool |
| `insufficient_evidence` | 4 | Plausible questions the corpus explicitly cannot answer |
| `no_relevant_source` | 1 | Off-domain question |
| `sensitive_unanswerable` | 1 | Credential-seeking question, retained but unscored |

The corpus has 10 source documents. A two-source query therefore has eight
competing sources, so the "more than five competing sources" requirement is met
by construction rather than by adding distractors.

### Independence from `golden_v1`

- No question text is copied or paraphrased from `golden_v1`; a validation test
  asserts that the normalized question sets are disjoint.
- All expected sources are declared explicitly in
  `evals/datasets/golden_v2.source_map.json`.
- The dataset was written from the checked-in corpus before any retrieval run.
  Labels were not changed after seeing metrics.
- `golden_v0` and `golden_v1` datasets, source maps, reports, and the corpus
  fingerprint are unchanged.

### Relevance grading

Labels are source-level and answer what the question actually requires. They are
not assigned to make a retriever look better.

| Grade | Meaning |
| --- | --- |
| 3 | Directly answers a required part of the question |
| 2 | Required supporting detail, but not sufficient alone |
| 1 | Background context; not used as expected relevance in this dataset |
| absent | Not an expected relevant source for that query |

When `relevance_grades` is present, it must grade every expected source and only
expected sources. This prevents a partially labelled record from silently
changing the nDCG denominator. Every answerable `golden_v2` record is graded.
The per-record `notes` field states why each source is expected and why a grade
of 2 rather than 3 was used where applicable.

Examples of the reasoning:

- `v2-multi-409-endpoint-001` needs the error runbook for the meaning of
  `PAY-409` and the current migration guide for the deprecated endpoint. Both
  are grade 3 because each answers a distinct required sub-question.
- `v2-id-shared-pay503-001` expects three documents because all three state the
  same `PAY-503` meaning. The error runbook is grade 3 because it adds checks and
  a response; the two API specs are grade 2 because they confirm the meaning
  without operational guidance.
- `v2-long-diagnostic-order-001` expects the pool runbook at grade 3 because it
  states the ordered procedure. The postmortem is grade 2 because it corroborates
  the symptom pattern but is not the procedure.

Binary Recall@5, MRR, and AllExpected@5 remain the primary metrics. nDCG@5 is
computed over source-level units only when grades exist; chunk-level nDCG stays
null because a graded source judgment is not a chunk judgment.

## Evaluated retrievers

| Report name | Implementation | Keyword candidate strategy |
| --- | --- | --- |
| `semantic` | `semantic_search` | n/a |
| `keyword` | `keyword_search` | AND (`websearch_to_tsquery`) |
| `lexical` | `lexical_search` | OR + distinct-source IDF coverage |
| `hybrid` | `hybrid_search(fusion_strategy="chunk")` | OR + IDF |
| `hybrid_source` | `hybrid_search(fusion_strategy="source")` | OR + IDF |

The hybrid rows must be run with `--hybrid-keyword-strategy or_idf`. The CLI
default remains `and`, so existing semantic-vs-hybrid reports and production
callers are unchanged.

## Procedure

```powershell
.venv\Scripts\python.exe scripts\run_retrieval_eval.py `
  --tenant-id <tenant-uuid> `
  --dataset evals/datasets/golden_v2.jsonl `
  --source-map evals/datasets/golden_v2.source_map.json `
  --profile evals/datasets/golden_v2.profile.json `
  --include-lexical-comparison `
  --hybrid-keyword-strategy or_idf `
  --rank-unit source `
  --output evals/reports/golden_v2_lexical_comparison.json
```

Validation only (no database or embedding call):

```powershell
.venv\Scripts\python.exe scripts\run_retrieval_eval.py `
  --dataset evals/datasets/golden_v2.jsonl `
  --source-map evals/datasets/golden_v2.source_map.json `
  --profile evals/datasets/golden_v2.profile.json `
  --validate-only
```

The runner is read-only with respect to the corpus. It verifies the tenant,
active documents, current versions, and source-map preflight before executing
retrieval.

## Results

Source-level metrics over 29 answerable queries:

| Retriever | Recall@5 | MRR | AllExpected@5 | nDCG@5 |
| --- | --- | --- | --- | --- |
| `semantic` | 0.9828 | 0.8649 | 0.9655 | 0.8541 |
| `keyword` (AND) | 0.0345 | 0.0345 | 0.0345 | 0.0345 |
| `lexical` (OR + IDF) | 0.9828 | **0.9425** | 0.9655 | **0.9054** |
| `hybrid` (chunk RRF, OR) | 0.9828 | 0.8736 | 0.9655 | 0.8732 |
| `hybrid_source` (source RRF, OR) | 0.9828 | 0.8793 | 0.9655 | 0.8692 |

Chunk-level metrics for the same runs:

| Retriever | Recall@5 | MRR | AllExpected@5 |
| --- | --- | --- | --- |
| `semantic` | 0.9655 | 0.8649 | 0.9310 |
| `keyword` (AND) | 0.0345 | 0.0345 | 0.0345 |
| `lexical` (OR + IDF) | **0.9828** | **0.9425** | **0.9655** |
| `hybrid` (chunk RRF, OR) | 0.9828 | 0.8736 | 0.9655 |
| `hybrid_source` (source RRF, OR) | 0.9828 | 0.8793 | 0.9655 |

OR+IDF raises source MRR by 0.0776 and nDCG by 0.0513 over semantic retrieval
while matching source Recall@5. Every hybrid row is identical to semantic on
recall and AllExpected@5 because fusion did not change which expected sources
survived the top-5 cutoff.

### Per-category source metrics

Recall@5 / MRR:

| Category | n | semantic | lexical | hybrid | hybrid_source |
| --- | --- | --- | --- | --- | --- |
| `paraphrase_no_rare_term` | 5 | 1.0000/0.8500 | 1.0000/0.9000 | 1.0000/0.9000 | 1.0000/0.9000 |
| `multi_source` | 5 | 0.9000/0.9000 | **1.0000/1.0000** | 0.9000/0.9000 | 0.9000/0.9000 |
| `identifier_shared` | 3 | 1.0000/1.0000 | 1.0000/1.0000 | 1.0000/1.0000 | 1.0000/1.0000 |
| `status_vs_identifier` | 3 | **1.0000/0.6667** | 0.8333/0.7778 | 1.0000/0.6667 | 1.0000/0.6667 |
| `api_version_url` | 5 | 1.0000/0.6667 | **1.0000/0.9000** | 1.0000/0.6667 | 1.0000/0.7000 |
| `long_noisy` | 4 | 1.0000/1.0000 | 1.0000/1.0000 | 1.0000/1.0000 | 1.0000/1.0000 |
| `multi_chunk_source` | 4 | 1.0000/1.0000 | 1.0000/1.0000 | 1.0000/1.0000 | 1.0000/1.0000 |

The lexical gain is concentrated in API-version ranking and multi-source
coverage. The lexical regression is concentrated in status-versus-identifier
queries.

## Candidate-list verification

Requirement: fusion must receive non-empty lexical candidates, and the report
must show keyword candidate counts per query.

From `retriever_comparison.candidate_counts`:

| Candidate path | Queries with a non-empty list | Range of candidate counts |
| --- | --- | --- |
| AND `keyword` | 1 / 29 | 0-2 |
| OR `lexical` | **29 / 29** | 2-14 |
| Lexical chunks in `hybrid` top-5 | **29 / 29** | at least 2 |

Every hybrid query received a non-empty OR+IDF candidate list, and at least two
lexical chunks reached each fused top-5. The earlier `golden_v1` condition where
fusion silently received an empty keyword list does not occur in this run.
The one AND-keyword non-empty query is
`v2-version-config-check-001`, which still returns only two candidates and is not
a general fix for the AND baseline. The full per-query table is stored in
`retriever_comparison.candidate_counts` in the report, with
`keyword_candidate_count`, `lexical_candidate_count`, and
`lexical_chunks_in_hybrid_top_5` for every query.

## Failures, harms, and false positives

Missed expected sources (source Recall@5 < 1):

| Retriever | Queries | Missing source |
| --- | --- | --- |
| `semantic` | `v2-multi-connectivity-retry-001` | `payment-api-runbook-v2` |
| `lexical` | `v2-status-http503-vs-pay503-001` | `runbook-payments-errors-v1` |
| `hybrid` | `v2-multi-connectivity-retry-001` | `payment-api-runbook-v2` |
| `hybrid_source` | `v2-multi-connectivity-retry-001` | `payment-api-runbook-v2` |
| `keyword` | 28 of 29 queries | all expected sources |

Total: 32 retriever-query pairs with source Recall@5 < 1. These are source-level
misses; chunk-level counts differ, so the earlier chunk-level view must not be
used to claim a missing source.

The lexical miss is explainable. `v2-status-http503-vs-pay503-001` contains both
`HTTP 503` and `PAY-503`. Identifier spans are removed before lexeme generation,
so the remaining query text is mostly generic. `payment-api-runbook-v2` ranks
first from the bare-status signal, but `runbook-payments-errors-v1` is pushed out
of the top 5 by documents with more generic lexeme coverage. A single literal
identifier term is not enough to dominate a query whose other terms are common.

The fusion miss is a dilution effect. Lexical retrieval places
`payment-api-runbook-v2` at rank 4 for
`v2-multi-connectivity-retry-001`. Semantic retrieval does not return it at all.
Equal-weight RRF cannot lift a rank-4 lexical-only source into the top-5 source
units, so both fusion modes miss it even though lexical alone finds it.

Queries where lexical harms semantic ranking:

| Query | Metric | Semantic | Lexical |
| --- | --- | --- | --- |
| `v2-para-pool-allowance-001` | source MRR | 1.0000 (rank 1) | 0.5000 (rank 2) |
| `v2-status-bare503-001` | source MRR | 0.5000 (rank 2) | 0.3333 (rank 3) |
| `v2-status-http503-vs-pay503-001` | source Recall@5 | 1.0000 | 0.5000 |

Neither fused mode harms semantic MRR or recall on any of the 29 queries. That
does not mean fusion is better; it means fusion stayed close to semantic and
rarely promoted what lexical found.

False positives are reported two ways:

- **Mapping errors**: zero unmapped or version-mismatched units, because the
  source map covers all 10 corpus sources. This number is not informative here.
- **Matched but non-expected sources**: `semantic` 99, `lexical` 93, `hybrid` 99,
  `hybrid_source` 99 source-unit slots across 29 queries, with a maximum of four
  per query. This is expected when 1-3 sources are expected and the ranking is
  five deep in a 10-source corpus. It is reported rather than hidden, but it is
  not evidence of a defect.

## Ranking differences versus semantic

Chunk-level and source-level orderings compared with the semantic baseline:

| Retriever | Queries with different chunk ranking | Queries with different source ranking |
| --- | --- | --- |
| `lexical` | 29 / 29 | 29 / 29 |
| `hybrid` | 29 / 29 | 28 / 29 |
| `hybrid_source` | 29 / 29 | 29 / 29 |
| `keyword` | 29 / 29 | 29 / 29 |

`hybrid` is identical to semantic on one source ranking because the fused list
kept the same source order there. `hybrid_source` differs on every source
ranking. The chunk-versus-source-aware comparison is separate: those two fusion
modes differ on 13 of 29 source rankings even though they share the same semantic
and lexical candidate lists.

## Duplicate-source effects

| Retriever | Queries with duplicate-source chunks in top-5 | Total duplicate chunks |
| --- | --- | --- |
| `semantic` | 25 / 29 | 29 |
| `hybrid` (chunk RRF) | 22 / 29 | 27 |
| `lexical` | 16 / 29 | 18 |
| `hybrid_source` (source RRF) | 0 / 29 | 0 |

Unlike `golden_v1`, source-aware fusion now changes observable behavior:
chunk-level and source-aware source rankings differ on 13 of 29 queries. It
removes duplicate-source crowding by construction, but the aggregate effect is
mixed:

- source MRR: 0.8736 (chunk) vs **0.8793** (source-aware), delta +0.0057
- source nDCG: **0.8732** (chunk) vs 0.8692 (source-aware), delta -0.0040
- Recall@5 and AllExpected@5 are identical

This is not a clear win. The source-aware mode is safer against duplicate
crowding but does not improve coverage or consistently improve ranking quality.

## Does fusion improve over semantic-only?

| Comparison | Improved | Equal | Harmed | Mean MRR delta |
| --- | --- | --- | --- | --- |
| `hybrid` vs `semantic` | 1 | 28 | 0 | +0.0086 |
| `hybrid_source` vs `semantic` | 2 | 27 | 0 | +0.0144 |

Fusion improves slightly and harms nothing on this dataset, but it captures only
a small fraction of the lexical gain. Lexical alone improves mean source MRR by
0.0776 over semantic; the fused modes improve it by 0.0086 and 0.0144. Equal-weight
RRF is dominated by the semantic ranking when both lists contain the same
neighborhood of sources.

## Post-correction validation run

The results above are the original lexical-v1 held-out run. They are preserved
unchanged. After that run, the `v2-status-http503-vs-pay503-001` miss was
diagnosed and corrected with an explicit identifier weight (see
`docs/LEXICAL_RETRIEVAL_V1.md`). The correction was measured on this same
`golden_v2` query, so the numbers below are **validation results, not an
independent confirmation**. They do not restore `golden_v2` to held-out status.

`evals/reports/golden_v2_fusion_comparison.json` uses the same 35 records,
grades, source map, profile, corpus fingerprint, and current-version filters.
It writes to a new file and leaves `golden_v2_lexical_comparison.json` untouched.
Source-level metrics over the same 29 answerable queries:

| Retriever | Recall@5 | MRR | AllExpected@5 | nDCG@5 |
| --- | --- | --- | --- | --- |
| `semantic` | 0.9828 | 0.8649 | 0.9655 | 0.8541 |
| `lexical` (OR + IDF, weighted identifiers) | **1.0000** | **0.9655** | **1.0000** | 0.9317 |
| `hybrid` (equal-weight chunk RRF) | 0.9828 | 0.8908 | 0.9655 | 0.8832 |
| `hybrid_weighted_semantic` (2.0/1.0) | 0.9828 | 0.8908 | 0.9655 | 0.8854 |
| `hybrid_weighted_lexical` (1.0/2.0) | 0.9828 | **0.9655** | 0.9655 | **0.9337** |
| `hybrid_lexical_primary` (semantic fallback) | **1.0000** | **0.9655** | **1.0000** | 0.9317 |

Compared with the pre-correction lexical row, only two queries change:

| Query | Recall@5 | MRR | nDCG@5 |
| --- | --- | --- | --- |
| `v2-status-bare503-001` | 1.0000 → 1.0000 | 0.3333 → 1.0000 | 0.5000 → 1.0000 |
| `v2-status-http503-vs-pay503-001` | 0.5000 → 1.0000 | 1.0000 → 1.0000 | 0.6131 → 0.8772 |

No lexical query regresses. The correction removes the only lexical source
miss and repairs the bare-status ranking without turning identifiers into
filters: unrelated numeric strings are still not identifiers, and generic-term
matches remain in the candidate pool below identifier-bearing sources.

Fusion effects against the equal-weight baseline:

| Strategy | Source MRR improved / equal / harmed | Mean MRR delta | Recall@5 improved / harmed |
| --- | --- | --- | --- |
| `hybrid_weighted_semantic` | 0 / 29 / 0 | 0.0000 | 0 / 0 |
| `hybrid_weighted_lexical` | 6 / 22 / 1 | +0.0747 | 0 / 0 |
| `hybrid_lexical_primary` | 6 / 22 / 1 | +0.0747 | 1 / 0 |

- Semantic-heavy weighting changes nothing on this query set.
- Lexical-heavy weighting recovers the lexical ranking signal for six queries
  and harms `v2-status-http503-vs-pay503-001` slightly, but keeps semantic recall.
- Lexical-primary with semantic fallback matches lexical MRR and restores the
  one source that equal-weight fusion drops (`v2-multi-connectivity-retry-001`),
  at the cost of one MRR regression (`v2-para-pool-allowance-001`).

The full analysis, including per-category results, duplicate-source effects,
fusion contributions, and the recommendation, is in
`docs/FUSION_EXPERIMENTS_V2.md`.

## Recommendation

**OR+IDF lexical retrieval:** promote it as the default keyword candidate
strategy for hybrid retrieval, but keep `keyword_search` (AND) available as an
explicit baseline. On this held-out set it is non-empty on every query, beats the
AND baseline decisively, and improves source MRR and nDCG over semantic retrieval
while matching Recall@5 and AllExpected@5.

This recommendation is conditional on the status-versus-identifier regression.
The strategy loses recall on one query where an identifier appears alongside
generic status language. Before OR+IDF becomes the only lexical path in
production, add a guard or a stronger identifier signal and re-evaluate on more
status/identifier queries.

**Source-aware fusion:** do not make it the default yet. It eliminates
duplicate-source crowding and changes 13 of 29 source rankings, but it does not
improve recall or AllExpected@5 and its MRR gain is smaller than its nDCG loss.
Keep it opt-in.

**Fusion in general:** neither RRF mode beats lexical-only retrieval. The next
experiment should test weighted fusion or a lexical-primary fallback, not another
duplicate-collapse change. Equal-weight RRF is the limiting factor on this
dataset.

## Limitations

- 29 answerable queries, one synthetic corpus, one embedding model, one run.
  No confidence intervals are reported.
- Relevance is source-level. A non-expected source that contains supporting text
  is still counted as non-expected. The 99 non-expected slots are a labelling
  consequence, not proof of harmful retrieval.
- Grades are author-assigned from corpus text. They are explicit and uniform,
  but they are still judgments.
- Unanswerable records are unscored. This is not an abstention, refusal, or
  secret-protection evaluation.
- Only current versions are searchable. Historical ranking is not evaluated.
- The dataset is too small to isolate category effects. Category metrics are
  descriptive, not statistically stable.
- Identifier matching is literal. The status/identifier miss shows that a literal
  match can be outweighed by generic term coverage.
- No generation, citation correctness, or answer faithfulness is evaluated.
- The corpus fingerprint matches the `golden_v1` run, so metric differences
  reflect the query set and retrieval configuration, not a corpus change.
- The post-correction validation run reused the query that motivated the fix.
  It is not independent confirmation of the identifier weight.

## Independent confirmation: `golden_v3`

`golden_v2` can no longer confirm the identifier-weight correction, because one
of its queries was used to diagnose it. `golden_v3` was therefore authored from
the corpus without consulting these results. It includes:

- new questions that mix an HTTP status with a prefixed identifier, in both
  orders and with different phrasing;
- identifiers shared by two and three current sources;
- bare numeric values that are neither statuses nor identifiers
  (years, durations, counts, ports);
- prompts where the same numeric value appears both bare and inside an
  identifier;
- multi-source questions whose expected sources share an identifier;
- paraphrases with no identifier at all, as a control for over-weighting.

The confirmed run is documented in `docs/GOLDEN_V3_CONFIRMATION.md`
(`evals/reports/golden_v3_fusion_comparison.json`). On 50 answerable and 12
unanswerable records it reproduces the lexical advantage over semantic and
equal-weight RRF. Lexical-heavy weighted RRF leads MRR and nDCG@5 while matching
the best recall, so it is the preferred fusion candidate, but its recall safety
is not confirmed across both held-out sets and its status/identifier advantage
over lexical flips sign. Lexical-primary is metric-identical to lexical in
aggregate. The identifier weight and all fusion strategies remain opt-in and out
of production defaults.

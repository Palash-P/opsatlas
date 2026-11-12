# Retrieval Fusion Experiments v2

## Purpose

This experiment asks whether the lexical gain over semantic retrieval can be
carried into a fused ranking. The baseline equal-weight chunk RRF is unchanged;
every alternative is opt-in and selectable through explicit arguments.

The two questions are:

1. Does weighting the lexical list more heavily recover more of the lexical
   ranking improvement without losing semantic recall?
2. Does returning the lexical ranking first and using semantic results only as a
   fallback preserve lexical coverage and remove the dilution that equal-weight
   RRF causes?

Only `golden_v1` was used for development. `golden_v2` is reported as a
validation set, not as independent confirmation, because one `golden_v2` query
was used to diagnose the identifier regression. A future `golden_v3` is needed
for independent confirmation.

## Identifier correction that precedes the fusion run

`v2-status-http503-vs-pay503-001` contains both `HTTP 503` and `PAY-503`.
Identifier extraction already separated them correctly. The miss was a ranking
problem: `PAY-503` had `df = 3` and `idf ≈ 2.0116`, while a coincidental rare
generic term had `idf ≈ 2.7047`, and `runbooks/payment-errors.md` lost the
`chunk_id` tie-break against the other `PAY-503` sources.

`lexical_search` now multiplies the summed literal-identifier IDF by
`IDENTIFIER_WEIGHT = 2.0`. This is a ranking weight, not a filter. The
`identifier_weight` parameter is validated and recorded in the report. See
`docs/LEXICAL_RETRIEVAL_V1.md` for the extraction and scoring details.

## Strategies

All hybrid strategies use OR+IDF lexical candidates and the same tenant,
active-document, and current-version filters. The candidate pool is
`min(20, fetch_limit * 3)`.

| Report name | Fusion rule | Weights (semantic/lexical) |
| --- | --- | --- |
| `semantic` | Semantic only | n/a |
| `lexical` | OR+IDF lexical only | n/a |
| `hybrid` | Equal-weight chunk RRF (unchanged baseline) | 1.0 / 1.0 |
| `hybrid_weighted_semantic` | Weighted RRF, semantic-heavy | 2.0 / 1.0 |
| `hybrid_weighted_lexical` | Weighted RRF, lexical-heavy | 1.0 / 2.0 |
| `hybrid_lexical_primary` | Lexical ranking first, unique semantic fallback | n/a |

Weighted RRF scores a chunk as:

```text
semantic_weight / (rrf_k + semantic_rank)
+ lexical_weight  / (rrf_k + lexical_rank)
```

Ties are broken by fused score, then best rank, then chunk ID. The lexical-primary
strategy never re-ranks a lexical chunk with a semantic score. It appends unique
semantic chunks only when the lexical list does not fill the requested limit, and
records `ranking_origin` (`lexical` or `semantic_fallback`).

## Procedure

Development run:

```powershell
.venv\Scripts\python.exe scripts\run_retrieval_eval.py `
  --tenant-id <tenant-uuid> `
  --dataset evals/datasets/golden_v1.jsonl `
  --source-map evals/datasets/golden_v1.source_map.json `
  --profile evals/datasets/golden_v1.profile.json `
  --include-fusion-experiments `
  --rank-unit source
```

Validation run:

```powershell
.venv\Scripts\python.exe scripts\run_retrieval_eval.py `
  --tenant-id <tenant-uuid> `
  --dataset evals/datasets/golden_v2.jsonl `
  --source-map evals/datasets/golden_v2.source_map.json `
  --profile evals/datasets/golden_v2.profile.json `
  --include-fusion-experiments `
  --rank-unit source
```

The default outputs are `evals/reports/golden_v1_fusion_comparison.json` and
`evals/reports/golden_v2_fusion_comparison.json`. The existing
`golden_v1_lexical_comparison.json`, `golden_v2_lexical_comparison.json`,
`golden_v0`, and semantic-vs-hybrid reports are not overwritten. Every run is
read-only with respect to the corpus, and the corpus fingerprint is
`c5cb44a83cd37560f004692ec4c702d509dde3f958a42966545480ca23fe6408`.

## Development results: `golden_v1`

16 answerable queries. Source-level metrics:

| Retriever | Recall@5 | MRR | AllExpected@5 |
| --- | --- | --- | --- |
| `semantic` | 0.9688 | 0.8802 | 0.9375 |
| `lexical` | **1.0000** | **0.9271** | **1.0000** |
| `hybrid` | **1.0000** | 0.8750 | **1.0000** |
| `hybrid_weighted_semantic` | 0.9688 | 0.8854 | 0.9375 |
| `hybrid_weighted_lexical` | **1.0000** | 0.9062 | **1.0000** |
| `hybrid_lexical_primary` | **1.0000** | **0.9271** | **1.0000** |

`golden_v1` has no graded labels, so nDCG@5 is not applicable. Chunk-level
Recall@5 equals source Recall@5 for every strategy except `semantic`, where
chunk recall is 0.9688 and source recall is 0.9688.

Versus equal-weight `hybrid`:

| Strategy | MRR improved / equal / harmed | Mean MRR delta | Recall@5 improved / harmed |
| --- | --- | --- | --- |
| `hybrid_weighted_semantic` | 1 / 13 / 2 | +0.0104 | 0 / 1 |
| `hybrid_weighted_lexical` | 1 / 15 / 0 | +0.0312 | 0 / 0 |
| `hybrid_lexical_primary` | 2 / 13 / 1 | +0.0521 | 0 / 0 |

The semantic-heavy profile is the only strategy that loses source recall on
`golden_v1` (on `multi-incident-and-current-001`). Lexical-primary matches the
lexical ranking exactly and improves `version-migration-guide-001` and
`version-payment-runbook-001`, but it also harms
`paraphrase-pool-exhaustion-001` because that query relies on semantic ordering.

## Validation results: `golden_v2`

29 answerable queries. Source-level metrics:

| Retriever | Recall@5 | MRR | AllExpected@5 | nDCG@5 |
| --- | --- | --- | --- | --- |
| `semantic` | 0.9828 | 0.8649 | 0.9655 | 0.8541 |
| `lexical` | **1.0000** | **0.9655** | **1.0000** | 0.9317 |
| `hybrid` | 0.9828 | 0.8908 | 0.9655 | 0.8832 |
| `hybrid_weighted_semantic` | 0.9828 | 0.8908 | 0.9655 | 0.8854 |
| `hybrid_weighted_lexical` | 0.9828 | **0.9655** | 0.9655 | **0.9337** |
| `hybrid_lexical_primary` | **1.0000** | **0.9655** | **1.0000** | 0.9317 |

Chunk-level metrics:

| Retriever | Recall@5 | MRR | AllExpected@5 |
| --- | --- | --- | --- |
| `semantic` | 0.9655 | 0.8649 | 0.9310 |
| `lexical` | **1.0000** | **0.9655** | **1.0000** |
| `hybrid` | 0.9828 | 0.8908 | 0.9655 |
| `hybrid_weighted_semantic` | 0.9828 | 0.8908 | 0.9655 |
| `hybrid_weighted_lexical` | 0.9828 | **0.9655** | 0.9655 |
| `hybrid_lexical_primary` | **1.0000** | **0.9655** | **1.0000** |

Only two lexical queries changed from the pre-correction run:
`v2-status-bare503-001` (MRR 0.3333 → 1.0000, nDCG 0.5000 → 1.0000) and
`v2-status-http503-vs-pay503-001` (Recall@5 0.5000 → 1.0000, nDCG 0.6131 →
0.8772). No lexical query regressed.

### Per-category source Recall@5 / MRR

| Category | n | semantic | lexical | hybrid | weighted semantic | weighted lexical | lexical primary |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `paraphrase_no_rare_term` | 5 | 1.0000/0.8500 | 1.0000/0.9000 | 1.0000/0.9000 | 1.0000/0.9000 | **1.0000/1.0000** | 1.0000/0.9000 |
| `multi_source` | 5 | 0.9000/0.9000 | **1.0000/1.0000** | 0.9000/0.9000 | 0.9000/0.9000 | 0.9000/1.0000 | **1.0000/1.0000** |
| `identifier_shared` | 3 | 1.0000/1.0000 | 1.0000/1.0000 | 1.0000/1.0000 | 1.0000/1.0000 | 1.0000/1.0000 | 1.0000/1.0000 |
| `status_vs_identifier` | 3 | 1.0000/0.6667 | **1.0000/1.0000** | 1.0000/0.8333 | 1.0000/0.8333 | 1.0000/0.8333 | **1.0000/1.0000** |
| `api_version_url` | 5 | 1.0000/0.6667 | 1.0000/0.9000 | 1.0000/0.6667 | 1.0000/0.6667 | **1.0000/0.9000** | 1.0000/0.9000 |
| `long_noisy` | 4 | 1.0000/1.0000 | 1.0000/1.0000 | 1.0000/1.0000 | 1.0000/1.0000 | 1.0000/1.0000 | 1.0000/1.0000 |
| `multi_chunk_source` | 4 | 1.0000/1.0000 | 1.0000/1.0000 | 1.0000/1.0000 | 1.0000/1.0000 | 1.0000/1.0000 | 1.0000/1.0000 |

The lexical-primary strategy is the only fusion variant with full recall on both
`multi_source` and `status_vs_identifier`. Lexical-heavy weighting matches lexical
MRR on `paraphrase_no_rare_term` but does not recover the missing `multi_source`
source.

### Per-query changes versus equal-weight `hybrid`

`hybrid_weighted_semantic` changes no query on `golden_v2` (0 improved, 29 equal,
0 harmed).

`hybrid_weighted_lexical` (6 improved, 22 equal, 1 harmed):

- Improved: `v2-para-kafka-backlog-001`, `v2-multi-409-endpoint-001`,
  `v2-status-sev1-vs-pay503-001`, `v2-version-deprecated-v1-001`,
  `v2-version-migration-current-001`, `v2-version-v3-idempotency-001`.
- Harmed: `v2-status-http503-vs-pay503-001` (MRR 1.0000 → 0.5000, nDCG 0.9197 →
  0.6934). Recall is unchanged, so the payment-errors source stays in the top 5.

`hybrid_lexical_primary` (6 improved, 22 equal, 1 harmed, plus one recall gain):

- Improved: the same six queries above.
- Recall gain: `v2-multi-connectivity-retry-001` (Recall@5 0.5000 → 1.0000,
  nDCG 0.6131 → 0.8772).
- Harmed: `v2-para-pool-allowance-001` (MRR 1.0000 → 0.5000, nDCG 1.0000 →
  0.6309).

## Duplicate-source effects

Top-5 chunk lists still contain duplicate sources. The strategies that promote
the lexical list reduce this crowding without changing duplicate handling:

| Retriever | Queries with duplicate sources in top-5 | Total duplicate chunks | Max for one source |
| --- | --- | --- | --- |
| `semantic` | 25 / 29 | 29 | 2 |
| `lexical` | 15 / 29 | 17 | 2 |
| `hybrid` | 22 / 29 | 27 | 2 |
| `hybrid_weighted_semantic` | 20 / 29 | 25 | 2 |
| `hybrid_weighted_lexical` | 19 / 29 | 23 | 2 |
| `hybrid_lexical_primary` | 18 / 29 | 20 | 2 |

No strategy introduced a new duplicate-source mechanism. The source-aware
collapsing path (`fusion_strategy="source"`) remains separate and is not the
subject of this experiment.

## Candidate counts and fusion contributions

`lexical_search` supplied a non-empty candidate list for all 29 queries, with
2-14 candidates. Every fusion strategy received lexical candidates for all 29
queries. The number of lexical candidates that reached a fused top-5 ranges from
2 to 5.

Aggregate contribution totals from `retriever_comparison.candidate_counts` and
`aggregate.diagnostics.fusion_contributions`:

| Strategy | Lexical candidate matches | Semantic candidate matches | Shared | Semantic-only | Lexical-only | Ranking origins |
| --- | --- | --- | --- | --- | --- | --- |
| `hybrid` | 145 | 145 | 139 | 6 | 0 | fused 145 |
| `hybrid_weighted_semantic` | 145 | 145 | 139 | 6 | 0 | fused 145 |
| `hybrid_weighted_lexical` | 145 | 145 | 139 | 6 | 0 | fused 145 |
| `hybrid_lexical_primary` | 145 | 145 | 139 | 6 | 0 | lexical 139, semantic_fallback 6 |

No lexical-only chunk reached a fused top-5 on `golden_v2`: every lexical
candidate that survived fusion was also in the semantic candidate pool. That is
why equal-weight RRF cannot express much of the lexical ranking difference. The
lexical-primary strategy reaches the same shared chunks but preserves the lexical
order and uses semantic fallback on three queries:

| Query | Lexical slots | Semantic fallback slots |
| --- | --- | --- |
| `v2-para-multi-team-001` | 2 | 3 |
| `v2-status-sev1-vs-pay503-001` | 4 | 1 |
| `v2-version-migration-current-001` | 3 | 2 |

Matched but non-expected source units in the top-5 are nearly identical across
strategies: `semantic` 99, `hybrid` 99, `hybrid_weighted_semantic` 99,
`hybrid_weighted_lexical` 99, `hybrid_lexical_primary` 98, `lexical` 92. These
are source-map false positives, not mapping errors; there are zero unmapped or
version-mismatched units.

## Is semantic recall sacrificed?

- `hybrid_weighted_semantic`: no change from equal-weight, but it also captures
  none of the lexical gain and loses a source on `golden_v1`.
- `hybrid_weighted_lexical`: keeps semantic Recall@5 (0.9828) and matches the
  lexical MRR, but does not recover the one lexical-only `multi_source` source.
- `hybrid_lexical_primary`: does **not** sacrifice semantic recall in aggregate;
  it reaches Recall@5 1.0000 and AllExpected@5 1.0000, the same as lexical-only.
  It does regress one paraphrase query's MRR, so the statement is about recall,
  not about every query.

## Recommendation

1. **Do not change production defaults.** `hybrid_search` still defaults to
   AND keyword candidates, equal-weight chunk RRF, and no weights.
2. **Lexical-primary with semantic fallback is the most promising next
   candidate.** It matches lexical-only MRR and recall on `golden_v2`, restores
   the source that equal-weight RRF drops, and uses semantic fallback instead of
   semantic re-ranking. Its one MRR regression is on a paraphrase query.
3. **Lexical-heavy weighted RRF is a close second.** It has the best nDCG@5
   (0.9337) but slightly lower recall than lexical-primary and a small regression
   on the status/identifier query.
4. **Semantic-heavy weighting is not useful on this corpus.** It behaves like
   equal-weight on `golden_v2` and loses recall on `golden_v1`.
5. **Keep every experimental strategy opt-in** until a `golden_v3` query set
   confirms the result without reusing queries that motivated the identifier
   correction.

## Limitations

- 29 answerable `golden_v2` queries and 16 `golden_v1` queries, one synthetic
  corpus, one embedding model, one run. No confidence intervals are reported.
- `golden_v2` is no longer fully held out for the identifier-weight correction:
  one of its queries motivated the change. The post-correction numbers are
  validation, not independent confirmation.
- Relevance is source-level and binary for Recall@5, MRR, and AllExpected@5.
  nDCG@5 exists only where explicit grades exist; `golden_v1` has none.
- Only top-5 rankings are compared. A source below the cutoff cannot be ranked
  from this report.
- Fusion weights were selected as named profiles before the validation run. No
  query-by-query tuning was performed on `golden_v2`.
- The experiment does not evaluate generation, citations, abstention, or answer
  faithfulness.

## Independent confirmation: `golden_v3`

The independent confirmation run now exists. `golden_v3` contains 50 answerable
and 12 unanswerable records authored from the same synthetic corpus without
reusing `golden_v1` or `golden_v2` question text. The full protocol, per-category
results, per-query failures, duplicate-source counts, abstention diagnostics, and
limitations are in `docs/GOLDEN_V3_CONFIRMATION.md`; the report is
`evals/reports/golden_v3_fusion_comparison.json`.

The confirmation favors lexical-heavy weighted RRF over lexical-primary, but it
does not justify a production default change:

- `hybrid_lexical_primary` is metric-identical to lexical-only aggregate results,
  and its semantic fallback activated on 1 of 50 queries without changing
  aggregate MRR or nDCG.
- `hybrid_weighted_lexical` matches the best Recall@5 and AllExpected@5
  (1.0000) and leads MRR (0.9700) and nDCG@5 (0.9563). It also has perfect top-1
  accuracy on both identifier categories.
- Its recall safety is not confirmed: on `golden_v2` it missed one multi-source
  case (0.9828 versus lexical's 1.0000), and the status/identifier winner flips
  between the two independently authored sets.
- Corrected OR+IDF lexical-only remains the simplest baseline with perfect recall
  on both held-out sets.

Lexical-heavy weighted RRF is therefore the preferred fusion candidate to pursue
as an opt-in mode; lexical-primary is not recommended. Every fusion strategy
stays opt-in. No production default changed.

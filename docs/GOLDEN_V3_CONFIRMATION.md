# Golden v3 — Independent Retrieval Confirmation

## Purpose

This run asks a narrow question:

> On a query set authored after the fusion-v2 experiments, without inspecting
> individual `golden_v2` failures, does lexical-primary-with-semantic-fallback or
> lexical-heavy weighted RRF outperform corrected OR+IDF lexical retrieval?

`golden_v3` is the independent confirmation set. It was written from the
checked-in synthetic corpus, with new wording, before any `golden_v3` retrieval
result was observed. No `golden_v2` query text was reused or lightly rewritten.
A checked-in test asserts that the normalized question sets of `golden_v1`,
`golden_v2`, and `golden_v3` are disjoint.

An independence audit compared token overlap between every `golden_v3` question
and every `golden_v1`/`golden_v2` question. Thirteen questions were rewritten in
response, and one multi-source combination was replaced. The rewrites happened
before the final run reported here; no metric in this document was observed
before the rewrite.

This report is retrieval-only. It does not evaluate generation, citation
correctness, answer faithfulness, refusal, or secret handling.

## Dataset

| Property | Value |
| --- | --- |
| Records | 62 |
| Answerable | 50 |
| Unanswerable | 12 |
| Answerable categories | 10 |
| Unanswerable categories | 3 |
| Corpus | `synthetic_v1`, 10 documents, 12 versions, 14 current chunks |
| Corpus fingerprint | `c5cb44a83cd37560f004692ec4c702d509dde3f958a42966545480ca23fe6408` |
| Dataset SHA-256 | `1fe9b87582720e272abe36f45063e8d34ad4266c9e6694870f51b57100d80497` |
| Report | `evals/reports/golden_v3_fusion_comparison.json` |

### Answerable categories

| Category | Count | What it probes |
| --- | --- | --- |
| `exact_identifier` | 6 | `PAY-400`, `PAY-409`, `PAY-503`, `SEV-1`, `SEV-2`, `SEV-3` |
| `status_vs_identifier` | 6 | HTTP `503` versus `PAY-503`; `PAY-409` state recovery |
| `api_version_url` | 6 | `/api/v1/payments` versus `/api/v3/payments`, idempotency |
| `paraphrase_no_rare_term` | 6 | Questions with no distinctive corpus vocabulary |
| `multi_source` | 6 | Two required sources, including cross-domain pairs |
| `shared_identifier` | 4 | One identifier defined by two or three current sources |
| `bare_numeric_distractor` | 4 | Years, minutes, counts, and ports that are not identifiers |
| `long_noisy` | 4 | Long questions mixing deployment, pool, cache, and payment terms |
| `multi_chunk_source` | 4 | One source contributing multiple top-5 chunks |
| `distractor_overlap` | 4 | Shared vocabulary where only one source is the process owner |

Every answerable record has an explicit integer grade for every expected source;
the key set of `relevance_grades` matches `expected_source_ids` for all 50
records. Grades are `3` (directly answers a required part) or `2` (required
supporting detail, not sufficient alone). `AllExpected@5` and `Recall@5` remain
binary; nDCG@5 uses the grades.

### Unanswerable categories

| Category | Count | Author corpus-evidence classification |
| --- | --- | --- |
| `insufficient_evidence` | 5 | `related_context_only` |
| `no_relevant_source` | 4 | `none` |
| `sensitive_unanswerable` | 3 | `none` |

The corpus-evidence classification is an author judgment from reading the
corpus. It is not inferred from retrieval scores. Unanswerable records are never
included in Recall@5, MRR, nDCG@5, or AllExpected@5.

## Retrieval configuration

All strategies are opt-in. No production default changed.

| Setting | Value |
| --- | --- |
| `k` | 5 |
| Fetch limit | 20 |
| Candidate multiplier | 3 |
| Candidate cap | 20 |
| RRF `k` | 60 |
| Identifier weight | 2.0 |
| Lexical strategy | OR+IDF, distinct source-document frequency |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| Embedding dimensions | 384 |
| Tenant | `96e52951-46e1-4547-9103-b82de97e6822` |
| Filters | tenant, active document, current version, embedding model (semantic) |

Reproduction command:

```powershell
.venv\Scripts\python.exe scripts\run_retrieval_eval.py `
  --dataset evals/datasets/golden_v3.jsonl `
  --source-map evals/datasets/golden_v3.source_map.json `
  --profile evals/datasets/golden_v3.profile.json `
  --tenant-id 96e52951-46e1-4547-9103-b82de97e6822 `
  --include-fusion-experiments `
  --include-abstention-analysis `
  --rank-unit source
```

The run writes `evals/reports/golden_v3_fusion_comparison.json`. It does not
overwrite `golden_v0`, `golden_v1`, or `golden_v2` datasets, maps, profiles, or
reports.

## Aggregate retrieval results

Source-level metrics over 50 answerable queries. Best value per column is bold.

| Retriever | Recall@5 | MRR | nDCG@5 | AllExpected@5 |
| --- | --- | --- | --- | --- |
| `semantic` | 0.9900 | 0.8817 | 0.8878 | 0.9800 |
| `lexical` | **1.0000** | 0.9567 | 0.9442 | **1.0000** |
| `hybrid` (equal RRF) | 0.9900 | 0.9367 | 0.9311 | 0.9800 |
| `hybrid_weighted_semantic` | 0.9900 | 0.9117 | 0.9103 | 0.9800 |
| `hybrid_weighted_lexical` | **1.0000** | **0.9700** | **0.9563** | **1.0000** |
| `hybrid_lexical_primary` | **1.0000** | 0.9567 | 0.9442 | **1.0000** |

Chunk-level metrics:

| Retriever | Recall@5 | MRR | AllExpected@5 |
| --- | --- | --- | --- |
| `semantic` | 0.9900 | 0.8817 | 0.9800 |
| `lexical` | **1.0000** | 0.9567 | **1.0000** |
| `hybrid` | 0.9900 | 0.9367 | 0.9800 |
| `hybrid_weighted_semantic` | 0.9900 | 0.9117 | 0.9800 |
| `hybrid_weighted_lexical` | **1.0000** | **0.9700** | **1.0000** |
| `hybrid_lexical_primary` | **1.0000** | 0.9567 | **1.0000** |

nDCG@5 is computed at the source level only. Chunk-level nDCG is intentionally
null because a chunk does not carry a source-grade label.

`hybrid_weighted_lexical` leads MRR and nDCG while matching the best Recall@5 and
AllExpected@5. It is the only strategy that is top-ranked on all four source
metrics. `hybrid_lexical_primary` is metric-identical to `lexical` in aggregate.

## Per-category results

### Recall@5

| Category | n | semantic | lexical | hybrid | weighted semantic | weighted lexical | lexical primary |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `api_version_url` | 6 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| `bare_numeric_distractor` | 4 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| `distractor_overlap` | 4 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| `exact_identifier` | 6 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| `long_noisy` | 4 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| `multi_chunk_source` | 4 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| `multi_source` | 6 | 0.9167 | **1.0000** | 0.9167 | 0.9167 | **1.0000** | **1.0000** |
| `paraphrase_no_rare_term` | 6 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| `shared_identifier` | 4 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| `status_vs_identifier` | 6 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

### MRR

| Category | n | semantic | lexical | hybrid | weighted semantic | weighted lexical | lexical primary |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `api_version_url` | 6 | 0.6389 | **0.9167** | 0.7500 | 0.6389 | **0.9167** | **0.9167** |
| `bare_numeric_distractor` | 4 | 0.8333 | **1.0000** | 0.8750 | 0.8750 | **1.0000** | **1.0000** |
| `distractor_overlap` | 4 | 0.6458 | **1.0000** | 0.7083 | 0.6875 | 0.8750 | **1.0000** |
| `exact_identifier` | 6 | **1.0000** | 0.9167 | **1.0000** | **1.0000** | **1.0000** | 0.9167 |
| `long_noisy` | 4 | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| `multi_chunk_source` | 4 | 0.8750 | 0.8333 | **1.0000** | 0.8750 | 0.8750 | 0.8333 |
| `multi_source` | 6 | 0.8889 | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| `paraphrase_no_rare_term` | 6 | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| `shared_identifier` | 4 | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| `status_vs_identifier` | 6 | 0.9167 | 0.9167 | **1.0000** | **1.0000** | **1.0000** | 0.9167 |

### nDCG@5

| Category | n | semantic | lexical | hybrid | weighted semantic | weighted lexical | lexical primary |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `api_version_url` | 6 | 0.7321 | **0.9385** | 0.8155 | 0.7321 | **0.9385** | **0.9385** |
| `bare_numeric_distractor` | 4 | 0.8750 | **1.0000** | 0.9077 | 0.9077 | **1.0000** | **1.0000** |
| `distractor_overlap` | 4 | 0.7327 | **1.0000** | 0.7827 | 0.7654 | 0.9077 | **1.0000** |
| `exact_identifier` | 6 | **1.0000** | 0.9385 | **1.0000** | **1.0000** | **1.0000** | 0.9385 |
| `long_noisy` | 4 | 0.9170 | **1.0000** | 0.9585 | 0.9170 | **1.0000** | **1.0000** |
| `multi_chunk_source` | 4 | 0.9077 | 0.8750 | **1.0000** | 0.9077 | 0.9077 | 0.8750 |
| `multi_source` | 6 | 0.7969 | 0.8584 | 0.8681 | **0.8786** | 0.8756 | 0.8584 |
| `paraphrase_no_rare_term` | 6 | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| `shared_identifier` | 4 | **0.9645** | 0.9170 | **0.9645** | **0.9645** | 0.9170 | 0.9170 |
| `status_vs_identifier` | 6 | 0.9385 | 0.9385 | **1.0000** | **1.0000** | **1.0000** | 0.9385 |

The lexical ranking is strong on every category but `shared_identifier` and
`multi_chunk_source`, where it returns the required sources but orders them
differently from the graded ideal. Hybrid and lexical-heavy recover the
`multi_chunk_source` ordering because the semantic list contributes the primary
chunk higher. On `shared_identifier`, lexical, lexical-heavy, and lexical-primary
all put the grade-2 source ahead of the grade-3 source, while semantic and
equal-weight hybrid order it better.

## Identifier contrastive-pair accuracy

`exact_identifier` contains six queries with a single expected source. Top-1
accuracy:

| Retriever | Top-1 correct | Failure |
| --- | --- | --- |
| `semantic` | 6 / 6 | — |
| `lexical` | 5 / 6 | `v3-exact-pay409-reconcile`: `payments-v3-spec` ranks first |
| `hybrid` | 6 / 6 | — |
| `hybrid_weighted_semantic` | 6 / 6 | — |
| `hybrid_weighted_lexical` | 6 / 6 | — |
| `hybrid_lexical_primary` | 5 / 6 | `v3-exact-pay409-reconcile`: `payments-v3-spec` ranks first |

`status_vs_identifier` contains six deliberately contrastive queries. Top-1
accuracy:

| Retriever | Top-1 correct | Failure |
| --- | --- | --- |
| `semantic` | 5 / 6 | `v3-status-http503-rollback`: `error-runbook-current` ranks first |
| `lexical` | 5 / 6 | `v3-status-pay409-state`: `payments-v3-spec` ranks first |
| `hybrid` | 6 / 6 | — |
| `hybrid_weighted_semantic` | 6 / 6 | — |
| `hybrid_weighted_lexical` | 6 / 6 | — |
| `hybrid_lexical_primary` | 5 / 6 | `v3-status-pay409-state`: `payments-v3-spec` ranks first |

Two contrast pairs explain the differences:

| Pair | Correct source | Failure mode |
| --- | --- | --- |
| HTTP `503` after deploy vs `PAY-503` error identifier | `payment-troubleshooting-v2` for HTTP; `error-runbook-current` for `PAY-503` | Semantic puts `error-runbook-current` first for the HTTP-rollback query. |
| `PAY-409` recovery vs `PAY-409` definition | `error-runbook-current` for the recovery step | Lexical and lexical-primary put `payments-v3-spec` first because the specification also names `PAY-409`. |

The HTTP/PAY contrast is separated by lexical matching. The PAY-409 contrast is
not: both sources contain the identifier, and only the runbook contains the
recovery action. Hybrid fusion restores the runbook to rank 1 because the
semantic list ranks it first. Lexical-heavy weighted RRF does the same on both
PAY-409 queries, while lexical-primary preserves the lexical error. This is the
clearest behavioral difference between the two named fusion candidates.

## Per-query failures

Only three retriever-query pairs miss an expected source at source Recall@5:

| Retriever | Query | Missing source | First expected rank |
| --- | --- | --- | --- |
| `semantic` | `v3-multi-http503-capacity` | `pool-operations-runbook` | 3 |
| `hybrid` | `v3-multi-http503-capacity` | `pool-operations-runbook` | 1 |
| `hybrid_weighted_semantic` | `v3-multi-http503-capacity` | `pool-operations-runbook` | 1 |

The query asks for both the post-deployment application checks and the database
capacity concept. All three retrievers return the payment troubleshooting source
and several pool-adjacent distractors, but not the pool operations runbook inside
the top 5. Lexical, lexical-heavy, and lexical-primary all include it. This is the
single case where a semantic contribution changes recall.

Single-rank losses (MRR < 1):

| Retriever | Queries with MRR < 1 | Detail |
| --- | --- | --- |
| `semantic` | 10 | API-version, bare-numeric, and distractor reorderings |
| `lexical` | 4 | `v3-exact-pay409-reconcile` 0.5000, `v3-status-pay409-state` 0.5000, `v3-v3-idempotency` 0.5000, `v3-chunks-errors-checks` 0.3333 |
| `hybrid` | 6 | Bare-numeric, API-version, and distractor reorderings |
| `hybrid_weighted_semantic` | 8 | Adds one API-version and two distractor reorderings relative to equal-weight |
| `hybrid_weighted_lexical` | 3 | `v3-v3-idempotency` 0.5000, `v3-chunks-errors-checks` 0.5000, `v3-distract-kafka-payment` 0.5000 |
| `hybrid_lexical_primary` | 4 | Same four as lexical |

## Fusion contributions and fallback frequency

All four hybrid strategies received lexical and semantic candidates for all 50
queries. Almost every top-5 unit was present in both candidate pools:

| Strategy | Lexical candidate matches | Semantic candidate matches | Semantic-only | Lexical-only | Ranking origins |
| --- | --- | --- | --- | --- | --- |
| `hybrid` | 248 | 250 | 2 | 0 | fused 250 |
| `hybrid_weighted_semantic` | 248 | 250 | 2 | 0 | fused 250 |
| `hybrid_weighted_lexical` | 248 | 250 | 2 | 0 | fused 250 |
| `hybrid_lexical_primary` | 248 | 250 | 2 | 0 | lexical 248, semantic_fallback 2 |

The two semantic-only units come from the multi-source capacity query where
lexical retrieval does not reach the pool operations runbook. Fusion therefore
mostly reorders units that both retrievers already found, with a small semantic
recall contribution.

**Lexical-primary fallback frequency: 1 / 50 queries, 2 units.** The OR+IDF
lexical list filled all five requested slots on 49 of 50 answerable queries. The
only activation is `v3-v1-deprecated-endpoint`, where two semantic results filled
the remaining slots. Aggregate MRR and nDCG are still identical to lexical,
because the fallback units did not change the expected source's rank. The
fallback is not harmful on this dataset, but it adds a code path whose benefit is
observed on one query.

### Changes versus equal-weight `hybrid`

| Strategy | MRR improved | equal | harmed | Recall@5 improved | harmed | Mean MRR delta |
| --- | --- | --- | --- | --- | --- | --- |
| `hybrid_weighted_semantic` | 0 | 46 | 4 | 0 | 0 | -0.0250 |
| `hybrid_weighted_lexical` | 5 | 44 | 1 | 1 | 0 | +0.0333 |
| `hybrid_lexical_primary` | 5 | 42 | 3 | 1 | 0 | +0.0200 |

`hybrid_weighted_lexical` improves five queries and harms one relative to
equal-weight RRF: `v3-chunks-errors-checks` drops from MRR 1.0000 to 0.5000. The
five improvements are `v3-bare-2025-postmortem`, `v3-v2-migration-plan`,
`v3-v3-version-verify`, `v3-distract-redis-db`, and
`v3-distract-kafka-payment`. `hybrid_lexical_primary` improves the same five and
harms three: the two PAY-409 queries and `v3-chunks-errors-checks`.
`hybrid_weighted_semantic` harms four queries and improves none.

## Duplicate-source effects

| Retriever | Queries with duplicate sources in top-5 | Max chunks for one source |
| --- | --- | --- |
| `semantic` | 44 / 50 | 2 |
| `lexical` | 34 / 50 | 2 |
| `hybrid` | 44 / 50 | 2 |
| `hybrid_weighted_semantic` | 45 / 50 | 2 |
| `hybrid_weighted_lexical` | 39 / 50 | 2 |
| `hybrid_lexical_primary` | 35 / 50 | 2 |

Lexical-led rankings crowd fewer unique sources than semantic rankings, but every
strategy can still spend two top-5 slots on one source. Source-level metrics
collapse those duplicates before scoring, so duplicate crowding affects chunk
metrics more than source metrics. Source-aware fusion was not part of this
five-strategy comparison, so no chunk-versus-source ranking difference summary is
reported here.

## False-positive retrieval

Every retriever returned non-expected source units in the top 5 for all 50
answerable queries. This is expected: the corpus has 10 sources and each query
names one or two, so four to nine non-expected sources remain available.

| Retriever | Queries with non-expected sources | Total non-expected source units | Max per query |
| --- | --- | --- | --- |
| `semantic` | 50 / 50 | 188 | 4 |
| `lexical` | 50 / 50 | 182 | 4 |
| `hybrid` | 50 / 50 | 188 | 4 |
| `hybrid_weighted_semantic` | 50 / 50 | 188 | 4 |
| `hybrid_weighted_lexical` | 50 / 50 | 187 | 4 |
| `hybrid_lexical_primary` | 50 / 50 | 187 | 4 |

These are label-defined false positives, not proof of harmful retrieval. A
non-expected source may still contain useful background. They are not counted as
mapping errors; there were no unmapped or version-mismatched units in the top 5.

## Abstention readiness

Unanswerable records were run separately. They are not part of the retrieval
metrics above. The analysis records what each retriever returns and whether the
top score separates answerable from unanswerable queries. It does not test
generation abstention.

| Retriever | Unanswerable queries returning any evidence | Answerable top-score range | Unanswerable top-score range | Best in-sample threshold | Answerable retained | Unanswerable retrieved | Youden J |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `semantic` | 12 / 12 | 0.2960–0.7402 | 0.1689–0.5346 | 0.4722 | 0.7000 | 0.1667 | 0.5333 |
| `lexical` | 12 / 12 | 4.1567–27.9741 | 3.0581–10.1383 | 8.2163 | 0.9200 | 0.0833 | **0.8367** |
| `hybrid` | 12 / 12 | 0.0323–0.0328 | 0.0318–0.0328 | 0.0323 | 1.0000 | 0.8333 | 0.1667 |
| `hybrid_weighted_semantic` | 12 / 12 | 0.0481–0.0492 | 0.0479–0.0492 | 0.0489 | 0.7600 | 0.5000 | 0.2600 |
| `hybrid_weighted_lexical` | 12 / 12 | 0.0484–0.0492 | 0.0479–0.0492 | 0.0484 | 0.9800 | 0.7500 | 0.2300 |
| `hybrid_lexical_primary` | 12 / 12 | 4.1567–27.9741 | 3.0581–10.1383 | 8.2163 | 0.9200 | 0.0833 | **0.8367** |

The lexical score has the best observed separation. At its in-sample threshold,
one of twelve unanswerable queries sits above the boundary:

- `v3-unans-oncall-name` (lexical top score 10.1383). The escalation policy
  describes paging the on-call engineer but contains no roster.

`v3-unans-root-cause` now scores 6.6102, below the threshold. Hybrid and RRF
scores are compressed into a narrow range, so a single threshold retains too many
unanswerable queries. That is a property of the score scale, not evidence that
fusion retrieves more relevant evidence.

Corpus-evidence classification is separate from retrieval behavior:

| Evidence status | Queries | Meaning |
| --- | --- | --- |
| `related_context_only` | 5 | Related source text exists but does not answer the question |
| `none` | 7 | No source contains the requested evidence |

Because every unanswerable query returns some evidence from a 10-source corpus,
retrieval presence alone cannot distinguish answerable from unanswerable. A
production abstention policy needs generation-side evidence checks or a
score-threshold policy validated on more data. This report does not claim either.

## Regression analysis versus `golden_v2`

| Observation | `golden_v2` | `golden_v3` |
| --- | --- | --- |
| Lexical source Recall@5 / MRR / nDCG | 1.0000 / 0.9655 / 0.9317 | 1.0000 / 0.9567 / 0.9442 |
| Lexical-primary | same as lexical | same as lexical |
| Lexical-heavy weighted RRF | 0.9828 / 0.9655 / **0.9337** | **1.0000** / **0.9700** / **0.9563** |
| Equal-weight hybrid | 0.9828 / 0.8908 / 0.8832 | 0.9900 / 0.9367 / 0.9311 |
| Status/identifier winner | lexical and lexical-primary, perfect top-1 | hybrid, weighted semantic, and weighted lexical, perfect top-1 |
| Lexical-heavy status/identifier outcome | one regression | perfect top-1 |
| Lexical-primary semantic fallback use | 3 queries | 1 query |

Three results are stable:

1. Corrected OR+IDF lexical retrieval and lexical-heavy weighted RRF both beat
   semantic retrieval and equal-weight RRF on MRR and nDCG.
2. Lexical-heavy weighted RRF is at least as good as equal-weight RRF on every
   aggregate metric in both runs, and it leads on MRR and nDCG in both.
3. Lexical-primary never beats lexical in aggregate. It is identical to lexical
   on both datasets.

Two results are not stable:

1. **Recall safety.** On `golden_v2`, lexical-heavy weighted RRF missed one
   multi-source recall case (0.9828 versus lexical's 1.0000). On `golden_v3` it
   matches lexical at 1.0000. The recall risk is small but not eliminated.
2. **Status/identifier top-1.** On `golden_v2`, lexical was perfect and
   lexical-heavy regressed one query. On `golden_v3`, lexical regresses the two
   PAY-409 queries and lexical-heavy is perfect. The sign of this one-query
   difference flips between independently authored sets.

The central confirmation result is narrower than the aggregate table suggests:
lexical-heavy weighted RRF is the better fusion candidate on `golden_v3`, but its
advantage over lexical is built from a handful of rank swaps that do not all
repeat across datasets. It is not a confirmed production default.

## Recommendation

1. **Prefer lexical-heavy weighted RRF over lexical-primary with semantic
   fallback as the fusion candidate to pursue.** It dominates lexical-primary on
   every `golden_v3` source metric (MRR 0.9700 versus 0.9567, nDCG 0.9563 versus
   0.9442, same Recall@5 and AllExpected@5) and is the only strategy with perfect
   top-1 accuracy on both `exact_identifier` and `status_vs_identifier`.
2. **Do not make any fusion strategy a production default from this evidence.**
   Lexical-heavy weighted RRF lost one recall case on `golden_v2`, its
   status/identifier advantage over lexical flips between datasets, and the
   category results rest on 4–6 queries each. Keep it opt-in.
3. **Keep corrected OR+IDF lexical-only as the simplest evidence-supported
   baseline.** It is perfect on Recall@5 and AllExpected@5 in both held-out sets,
   and its aggregate MRR/nDCG trail lexical-heavy on `golden_v3` but not on
   `golden_v2` MRR. If only one retrieval path can be operated, lexical-only has
   the stronger recall-safety record.
4. **Do not adopt lexical-primary.** It is metric-identical to lexical in
   aggregate on both held-out sets. Its semantic fallback activated on 1 of 50
   `golden_v3` queries and 3 of 29 `golden_v2` queries without changing aggregate
   metrics. It adds a code path without measured aggregate benefit. Keep it
   available as an opt-in safety mode for corpora with short lexical candidate
   lists.
5. **Keep every fusion strategy opt-in and production defaults unchanged.** No
   schema, dependency, database, or default behavior changed in this work.
6. **Do not treat unanswerable retrieval as an abstention result.** The lexical
   threshold is descriptive and in-sample. A real abstention policy needs a
   held-out threshold study and generation-side evidence checks.

## Limitations

- 50 answerable and 12 unanswerable queries, one synthetic corpus, one embedding
  model, one run. No confidence intervals are reported.
- Grades and corpus-evidence labels are author judgments. They are explicit and
  complete, but they are not independently adjudicated.
- The corpus has only 10 sources. Non-expected-source counts are dominated by
  corpus size, not by retriever quality.
- Category metrics rest on 4–6 queries each. Differences of one query are not
  statistically stable, which is why the status/identifier ordering and the
  lexical-versus-lexical-heavy recall gap change sign between `golden_v2` and
  `golden_v3`.
- The independence audit used token overlap, not semantic similarity. It reduced
  but cannot prove the absence of shared author assumptions across datasets.
- nDCG@5 is source-level only. Chunk-level nDCG is not computed because grades are
  attached to sources.
- Only top-5 rankings are compared. A relevant source below the cutoff cannot be
  ranked from this report.
- Source-aware fusion (`fusion_strategy="source"`) was not part of this
  five-strategy run, so chunk-versus-source duplication effects are not compared.
- The abstention threshold is an in-sample Youden-J diagnostic over the same 62
  records. It is not a calibrated confidence threshold.
- Retrieval-only results say nothing about generation abstention, refusal,
  citation correctness, secret handling, prompt-injection resistance, or answer
  faithfulness.
- `golden_v0`, `golden_v1`, `golden_v2`, their source maps, profiles, and reports
  were not modified. The corpus fingerprint matches the prior runs.

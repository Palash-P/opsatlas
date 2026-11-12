# Evaluation v1 Design

## Purpose

`golden_v1` is a reproducible retrieval benchmark for OpsAtlas. Its job is to
expose weaknesses in semantic and hybrid retrieval, not to demonstrate that the
system is production ready. It is deliberately small, checked in, and honest
about what it does not measure.

Phase A changes evaluation assets only. It does not change
`app/services/retrieval.py`, `app/services/ingestion.py`, the database schema,
Docker Compose, or runtime dependencies.

## What is evaluated

Two retrievers return the same tenant's current, active document versions:

| Retriever | Function | Score |
| --- | --- | --- |
| Semantic | `semantic_search` | cosine similarity |
| Hybrid | `hybrid_search` | Reciprocal Rank Fusion |

A third retriever, `keyword_search`, runs only to produce diagnostics. It is
never scored and does not affect the semantic/hybrid comparison.

Retrieval is **current-version-only**. The benchmark can measure whether the
current version is selected. It cannot measure historical-version ranking,
because the production query paths filter to `is_current = TRUE`.

## Dataset composition

`evals/datasets/golden_v1.jsonl` contains 24 records: 16 answerable and 8
unscored.

| Category | Records | Purpose |
| --- | --- | --- |
| `exact_identifier` | 3 | Rare identifiers such as `PAY-503` |
| `identifier_in_natural_question` | 2 | Identifiers embedded in prose |
| `paraphrased_semantic` | 2 | No shared rare term; semantic recall required |
| `keyword_heavy_rare_term` | 2 | Rare technical terms and phrases |
| `multi_document` | 2 | Two expected sources per query |
| `distractor_overlap` | 3 | Hard negatives sharing vocabulary |
| `current_version_selection` | 2 | Pinned to an exact version |
| `sensitive_unanswerable` | 2 | Unscored; not a secret-protection test |
| `insufficient_evidence` | 3 | Unscored; corpus lacks evidence |
| `no_relevant_source` | 3 | Unscored; off-domain questions |

`evals/datasets/golden_v1.profile.json` enforces these minimum counts, required
answerable/unanswerable totals, and question uniqueness before retrieval.

Prompt-injection and adversarial-document tests are intentionally absent.
Those belong to Phase B and must not be inferred from a retrieval score.

## Corpus

The corpus is checked in under `evals/corpus/synthetic_v1/` with a manifest.
Every document is synthetic, contains no credentials, and makes no real
incident claim. Five documents reproduce existing database content byte for
byte; their hashes therefore match immutable stored versions. Two legacy
revisions of `runbooks/payment-api.md` cannot carry an in-content label without
changing those hashes, so the manifest declares `label_source: "manifest"` and
the corpus README supplies the synthetic notice.

The two-version world states are:

| Source key | v1 | v2 | Manage mode |
| --- | --- | --- | --- |
| `api-specs/payments.md` | `/api/v1/payments`, no deprecation | v1 deprecated, new clients use `/api/v3/payments` | `create` |
| `runbooks/payment-api.md` | no database-connectivity step | adds the connectivity step | `repair_current` |

The second version-selection record pins `runbooks/payment-api.md` to version 2.
Only version 2 contains the expected evidence, so the record can distinguish a
correct current-version pointer from an inverted one.

## Source map

`evals/datasets/golden_v1.source_map.json` maps logical expected source IDs to
repository-relative `source_key` values and, where needed, an exact
`version_number`.

`version_number` is optional in general, but `load_source_map` rejects any
`current_version_selection` record whose expected sources are unpinned. Without
a pin, the record cannot distinguish correct from stale version selection.

Mapping statuses:

- `matched`: `source_key` matches and, when pinned, `version_number` matches.
- `unmapped`: no mapping exists for the returned `source_key`.
- `version_mismatch`: the `source_key` matches but the returned version does not.

Titles, document IDs, version IDs, and chunk IDs never participate in
relevance matching.

## Rank units and slot preservation

Two rankings are reported from the same result pool:

- **Chunk-level**: the first five returned chunks.
- **Source-level**: duplicate chunks from one `(source_key, version_number)`
  unit collapse first, then the first five unique units are used.

Both rankings are rank-aligned. An unmapped or version-mismatched unit keeps
its original rank slot and counts as irrelevant. It is never removed and never
compressed away. This matters: dropping a distractor that outranks the expected
source would silently inflate MRR.

`--fetch-limit` controls how many chunks each retriever may return (1–20). Chunk
metrics always use the first five chunks. Source metrics can use the wider pool
to reach five unique sources, which is why the two rank units can legitimately
disagree.

## Metrics

All metrics are computed per answerable query and then macro-averaged. Expected
sources are a binary, unordered set; relevance is not graded.

- **Recall@5**: `|expected sources present in the top-5 rank units| / |expected sources|`.
- **MRR**: `1 / rank of the first expected source`, or `0` if none appears.
  Distractors and version mismatches consume rank slots.
- **Any-hit@5**: `1` if at least one expected source appears, else `0`.
- **All-expected@5**: `1` if every expected source appears, else `0`.
- **nDCG@5**: not reported. The labels are binary and ungraded, so any nDCG
  value would be fabricated. Reports record `null` with an explicit status.

Unanswerable records are never passed to a retriever and never scored.

## Diagnostics

Every query report includes:

- whether the semantic and hybrid chunk rankings are identical;
- whether the source rankings are identical;
- the first expected rank for each retriever and rank unit;
- keyword-retriever result counts, source keys, and how many keyword chunks also
  appear in the semantic/hybrid top 5.

Aggregate diagnostics count identical rankings, rank differences, and queries
where keyword results contributed to the hybrid top 5. These are diagnostics,
not quality scores.

A **saturation warning** fires when every retriever scores `1.0` on every
metric across every answerable query. That means the dataset cannot
discriminate between methods and should be made harder before drawing
conclusions.

## Corpus fingerprint, preflight, and baseline guard

Live runs compute a SHA-256 fingerprint over sorted
`(source_key, version_number, chunk_index, content_hash)` tuples for current,
active versions. The fingerprint contains no document text.

Before retrieval, the runner reads the tenant's versions and fails with
actionable messages when a mapped document is missing, has no active current
version, has no chunks, or whose current version does not match a pinned
`version_number`. No retrieval runs and no report is written on preflight
failure.

`--baseline` compares a new report with an earlier one. Metric deltas are marked
`comparable: false` and `informational_only: true` when either report lacks a
corpus fingerprint or the fingerprints differ. The guard prevents a dataset or
corpus change from being mistaken for a retrieval improvement.

## Seeder safety model

`evals/corpus_plan.py` is pure planning logic; `scripts/seed_synthetic_corpus.py`
executes it through the existing `ingest_document` service.

The default mode is a dry run. `--apply` is required for writes.

Planning distinguishes:

| Action | Meaning |
| --- | --- |
| `skip_current` | Intended content is already current. No write. |
| `create_versions` | Create the declared next versions in order. |
| `reactivate_existing` | A newer immutable version already exists; point to it. |
| `observe_drift` / `observe_missing` | Observe mode never writes. |
| `refuse_downgrade` | Intended content would make an older version current. |
| `refuse_unexpected_version` | Intended version is not the next version number. |
| `refuse_missing_earlier_versions` | Earlier declared versions do not exist. |

Any blocked action aborts the entire run before a write. After `--apply`, the
seeder re-reads the inventory and requires every document to settle on
`skip_current`. Re-running the seeder is therefore a no-op: it cannot repoint a
document to an older version, even though `ingest_document` itself supports
reactivation by content hash. Tests simulate the first and second runs and
assert that the payment and payments-migration documents settle on version 2.

No document, version, chunk, or embedding is deleted, and stored version
content is never rewritten.

## Expected live state transition (pending approval)

`runbooks/payment-api.md` is currently inverted: version 1 is current and
version 2 is not.

Verified content hashes:

- version 1: `7002e1d6212957d0a4e8bc7884994ce1b8cd7d20f0de2b802395934077333c61`
- version 2: `45421745e610182ed94bc42482dd7eb5715f00f7dfff07076e2b979553e24a7e`

The only content difference is that version 2 adds
`4. Verify database connectivity before retrying the payment request.`

After an approved repair the expected state is:

| Field | Before | After |
| --- | --- | --- |
| `document_versions` v1 `is_current` | `TRUE` | `FALSE` |
| `document_versions` v2 `is_current` | `FALSE` | `TRUE` |
| `documents.current_version_id` | v1 | v2 |

No new versions, no deletes, and no changes to hashes, chunks, or embeddings.

`api-specs/payments.md` does not exist yet. Seeding creates versions 1 and 2,
each with chunks, and leaves version 2 current. New chunks have `NULL`
embeddings until an approved backfill runs.

## Reproduction

Validate everything without touching the database:

```powershell
python scripts/run_retrieval_eval.py `
  --dataset evals/datasets/golden_v1.jsonl `
  --source-map evals/datasets/golden_v1.source_map.json `
  --profile evals/datasets/golden_v1.profile.json `
  --validate-only
```

Dry-run the corpus seeder:

```powershell
python scripts/seed_synthetic_corpus.py --tenant-id <uuid>
```

After approval only:

```powershell
python scripts/seed_synthetic_corpus.py --tenant-id <uuid> --apply
python scripts/backfill_embeddings.py   # approved embedding backfill
```

Then run the live evaluation:

```powershell
python scripts/run_retrieval_eval.py `
  --tenant-id <uuid> `
  --dataset evals/datasets/golden_v1.jsonl `
  --source-map evals/datasets/golden_v1.source_map.json `
  --profile evals/datasets/golden_v1.profile.json `
  --baseline evals/reports/golden_v0_semantic_vs_hybrid.json `
  --rank-unit source
```

The default report path is
`evals/reports/golden_v1_semantic_vs_hybrid.json`.

## Honest limitations

- Retrieval-only. No generation, citation, or answer-quality claim is made.
- Relevance is binary and ungraded. nDCG is not reported.
- Current-version selection only. Historical ranking is not evaluated.
- Unanswerable records are unscored. The report says nothing about abstention.
- The corpus is checked for credential-like strings, but retrieval metrics do
  not measure secret protection.
- No prompt-injection or adversarial-document testing. That is Phase B.
- A perfect score triggers a saturation warning and is not evidence of
  production readiness.
- Reports contain no document content and use repository-relative paths.
- `golden_v0.jsonl`, its source map, and its report remain unchanged.

## Lexical v1 extension

The OR+IDF lexical strategy and source-aware fusion are additive. The original
semantic/hybrid report and its default CLI invocation are unchanged. See
`docs/LEXICAL_RETRIEVAL_V1.md` for the strategy definition, identifier rules,
IDF definition, fusion comparison, and the `--include-lexical-comparison`
evaluation workflow.

## Held-out evaluation v2

The default-selection evidence comes from a separate 35-record dataset, not from
the 16-query `golden_v1` set that informed the design. `golden_v2` adds explicit
per-source relevance grades, multi-source queries, shared identifiers, status
versus identifier cases, URL-contained API versions, long noisy queries, and
multi-chunk sources. See `docs/HELD_OUT_EVALUATION_V2.md` for the labelling
policy, exact evaluation command, results, failure analysis, and limitations.

The report is written to
`evals/reports/golden_v2_lexical_comparison.json`; `golden_v0` and `golden_v1`
datasets, source maps, reports, and the corpus fingerprint are preserved.

## Golden v3 confirmation

`golden_v3` is the independent confirmation set for the identifier weight and the
fusion candidates. It contains 50 answerable and 12 unanswerable records written
from the corpus after the fusion-v2 experiments, with disjoint question text and
explicit source grades. The run compares semantic-only, corrected OR+IDF
lexical-only, equal-weight chunk RRF, semantic-heavy RRF, lexical-heavy RRF, and
lexical-primary with semantic fallback. It also runs unanswerable records through
a separate, opt-in abstention-readiness analysis that does not enter retrieval
recall.

See `docs/GOLDEN_V3_CONFIRMATION.md` for the protocol, per-category and per-query
results, identifier contrastive-pair accuracy, duplicate-source and
false-positive counts, fallback frequency, abstention diagnostics, limitations,
and recommendation. The report is
`evals/reports/golden_v3_fusion_comparison.json`; all earlier datasets, maps,
profiles, and reports are preserved.

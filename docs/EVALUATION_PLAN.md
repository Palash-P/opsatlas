# Evaluation Plan

## Golden dataset contract

Each JSONL record contains:

- `id`: stable test case ID, unique within the dataset.
- `question`: the user's question.
- `expected_source_ids`: source-level relevant IDs for answerable questions.
- `answerable`: whether the corpus contains sufficient evidence.
- `category`: test slice such as exact identifier, multi-document, stale-version, or unanswerable.
- `notes`: human-readable evaluation intent.

Validation rules:

- All fields are required.
- `id`, `question`, and `category` must be nonempty strings.
- `answerable` must be a JSON boolean.
- `expected_source_ids` must be a list of unique nonempty strings.
- Answerable records must have at least one expected source.
- Unanswerable records must have an empty expected-source list.
- IDs must be unique across the dataset.

## Source mapping

`expected_source_ids` are evaluation-level logical source IDs. They are not
guaranteed to equal `documents.source_key`, `documents.id`, `document_versions.id`,
or `documents.title`.

The evaluation uses an explicit source-map manifest:

```json
{
  "mapping_version": 1,
  "dataset_id": "golden_v0",
  "sources": {
    "expected-source-id": {
      "source_key": "<real source_key in the target tenant>",
      "version_number": 1
    }
  }
}
```

`version_number` is optional in general. It is required for every source used
by a `current_version_selection` record: without a pin the harness cannot tell
correct current-version selection from stale-version selection, and validation
fails before retrieval.

Every expected source ID must have a mapping. Missing or unknown mappings fail
validation before retrieval runs, and no report is written. Multiple chunks from
the same `(source_key, version_number)` unit collapse into one source unit.
Retrieved sources with no mapping are reported as `unmapped`; matching source
keys with the wrong version are reported as `version_mismatch`. Both keep their
original rank slot and count as irrelevant.

The checked-in `evals/datasets/golden_v0.source_map.json` maps the existing
local corpus. `evals/datasets/golden_v1.source_map.json` maps the synthetic
corpus and pins the two current-version records to version 2.

## Retrieval evaluation

The harness compares:

- `semantic_search`
- `hybrid_search` with Reciprocal Rank Fusion

Both run with the same tenant and question. `--fetch-limit` controls how many
chunks each retriever may return (1–20, default 20). Chunk-level metrics still
use the first five chunks; source-level metrics collapse duplicate chunks first
so they can reach five unique sources. The runner calls the service functions
directly; no HTTP server is required.

A keyword-only retrieval call also runs for diagnostics. It is never scored.

Only answerable records are scored. Unanswerable records are listed as unscored
and are excluded from all retrieval-quality metrics. This harness does not
evaluate abstention, secret handling, citation correctness, faithfulness, or
prompt-injection resistance.

## Metric definitions

For one query and retriever, two rank-aligned rankings are produced from the
same result pool:

- **Chunk-level**: the first five returned chunks.
- **Source-level**: duplicate chunks from one `(source_key, version_number)`
  unit collapse first, then the first five unique units.

Unmapped and version-mismatched units keep their original rank position and
count as irrelevant. They are never removed, because removing them would
compress the ranking and inflate MRR whenever a distractor or stale version
outranks the expected source.

Let `R_q` be the ranked relevant source IDs and `E_q` the expected source IDs.

### Recall@5

```text
Recall@5(q) = |R_q[:5] ∩ E_q| / |E_q|
```

Aggregate Recall@5 is the macro-average over answerable queries. This gives
fractional credit to multi-document questions.

### MRR

```text
RR(q) = 1 / rank of first expected source in R_q[:5], or 0 if absent
```

MRR is the macro-average of `RR(q)` over answerable queries.

### Any-hit@5 and all-expected@5

- `any_hit@5(q) = 1` if at least one expected source appears in `R_q[:5]`.
- `all_expected@5(q) = 1` if every expected source appears in `R_q[:5]`.

Aggregates are macro-averages over answerable queries.

### nDCG@5

nDCG@5 is not reported for `golden_v0` or `golden_v1` because their labels are a
binary, unordered relevant set with no graded relevance values. Reports record
`ndcg_at_5: null` with status `not_applicable_binary_relevance_only`.

A separate graded nDCG helper exists for future datasets that provide explicit
relevance grades. It is not used by either checked-in dataset.

## Diagnostics and guards

Reports include the corpus fingerprint (a hash over current-version chunk
identities, never content), keyword-retriever contribution counts, identical
ranking counts, first-expected-rank deltas, and a saturation warning when every
retriever scores perfectly. A perfect score means the dataset cannot
discriminate between methods; it is not evidence of production readiness.

Live runs fail before retrieval with actionable messages when a mapped document
is missing, has no active current version, has no chunks, or whose current
version does not match a pinned `version_number`.

`--baseline` compares against an earlier report. Deltas are marked
`informational_only` when a corpus fingerprint is missing or differs, so corpus
or dataset changes cannot be mistaken for retrieval improvements.

## Corpus seeding

`evals/corpus/synthetic_v1/` and its manifest define the checked-in corpus.
`scripts/seed_synthetic_corpus.py` is dry-run by default and requires `--apply`
for writes. Planning refuses to make an older version current, refuses
unexpected version numbers, and re-verifies that every document settled on its
intended current version. No records are deleted and stored versions are never
rewritten. See `docs/EVALUATION_V1_DESIGN.md` for the full safety model.

## Running

Validate the dataset, profile, and source map without live services:

```powershell
python scripts/run_retrieval_eval.py --validate-only
```

```powershell
python scripts/run_retrieval_eval.py `
  --dataset evals/datasets/golden_v1.jsonl `
  --source-map evals/datasets/golden_v1.source_map.json `
  --profile evals/datasets/golden_v1.profile.json `
  --validate-only
```

Dry-run the synthetic corpus seeder (read-only):

```powershell
python scripts/seed_synthetic_corpus.py --tenant-id <uuid>
```

Apply is intentionally separate and requires approval:

```powershell
python scripts/seed_synthetic_corpus.py --tenant-id <uuid> --apply
```

Run a live evaluation after the corpus is seeded and embeddings are backfilled:

```powershell
python scripts/run_retrieval_eval.py `
  --tenant-id <uuid> `
  --dataset evals/datasets/golden_v1.jsonl `
  --source-map evals/datasets/golden_v1.source_map.json `
  --profile evals/datasets/golden_v1.profile.json `
  --baseline evals/reports/golden_v0_semantic_vs_hybrid.json `
  --fetch-limit 20 `
  --rank-unit source `
  --output evals/reports/golden_v1_semantic_vs_hybrid.json
```

The default output path is
`evals/reports/<dataset-id>_semantic_vs_hybrid.json`.

Reports are JSON and include the dataset hash, source-map hash, corpus
fingerprint, retrieval configuration, aggregate chunk and source metrics,
per-query results, and diagnostics. Retrieved document content is not included.
Generated reports under `evals/reports/` are ignored by Git by default, except
the preserved `golden_v0` report.

## Evaluation hygiene

- Keep a fixed test split for regression checks and a separate development set for tuning.
- Version datasets and evaluation code.
- Store model/provider, prompt, embedding, chunking, and retrieval configuration with each run.
- Never optimize solely against an LLM-as-judge score; inspect examples and human-review a sample.
- Start with a small hand-labeled set, validate it, then grow to 100–200+ questions covering meaningful edge cases.
- Do not modify golden expected answers to improve scores.
- Do not claim safety or grounding quality from retrieval-only metrics.

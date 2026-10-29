# Evaluation Plan

## Golden dataset contract

Each JSONL record contains:
- `id`: stable test case ID.
- `question`: the user's question.
- `expected_source_ids`: sources expected to be retrieved for answerable questions.
- `answerable`: whether the corpus contains sufficient evidence.
- `category`: test slice such as exact identifier, multi-document, stale-version, or unanswerable.
- `notes`: human-readable evaluation intent.

## Initial metrics

- Retrieval Recall@k: whether any/all expected sources appear in top-k; report the exact definition.
- MRR / nDCG: ranking quality when multiple relevant sources are labeled.
- Citation correctness: cited source exists, is authorized, and supports the associated claim.
- Abstention precision/recall: distinguish answerable from unanswerable queries.
- Faithfulness: judge whether claims are supported by retrieved evidence; use human-reviewed samples to calibrate automated judges.
- Latency and cost: stage-level latency, p50/p95, token usage, and estimated provider cost.

## Evaluation hygiene

- Keep a fixed test split for regression checks and a separate development set for tuning.
- Version datasets and evaluation code.
- Store model/provider, prompt, embedding, chunking, and retrieval configuration with each run.
- Never optimize solely against an LLM-as-judge score; inspect examples and human-review a sample.
- Start with a small hand-labeled set, validate it, then grow to 100–200+ questions covering meaningful edge cases.

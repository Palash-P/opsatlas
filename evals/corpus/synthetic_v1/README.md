# Synthetic Corpus v1

**Notice: every document in this directory is synthetic.** None of these files
describe a real production system, real incident, real company procedure, or real
credential. They exist only to give the retrieval evaluation a stable, checked-in
corpus.

## Why the corpus is checked in

The first evaluation ran against documents that existed only in a local database.
That made the report impossible to reproduce from a clean checkout. The corpus
manifest and these files make the corpus state reviewable and the seeding step
deterministic.

## Labelling

Most documents state that they are synthetic in their first paragraph. Two
legacy revisions of `runbooks/payment-api.md` were preserved byte-for-byte from
the existing database so that their content hashes still match immutable stored
history; they rely on this README and on `manifest.json` (`label_source:
"manifest"`) for their synthetic label.

## Rules

- Every file is validated by `tests/test_eval_corpus.py`.
- No credentials, database URLs, private keys, or real incident claims.
- Content hashes in `manifest.json` are over normalized text and must match the
  file and, for existing documents, the stored immutable version hash.
- `scripts/seed_synthetic_corpus.py` is additive and idempotent. It never deletes
  a document, never rewrites a stored version, and refuses to make an older
  version current.

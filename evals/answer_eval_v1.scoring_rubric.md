# Answer Evaluation v1 Scoring Rubric

This rubric was fixed before either scorer saw any model output. Each scorer
receives the same labelled record and the same prediction, and must not see the
other scorer's scores.

## Scale

Use `0`, `0.5`, or `1` for applicable metrics. Use `null` when a metric does not
apply. `null` means "not applicable", never zero.

## Metrics

### `answer_relevance`

- `1`: directly answers the question and includes every required expected fact.
  For an unanswerable record, a correct abstention scores 1.
- `0.5`: partially answers, omits a required fact, or adds a materially
  distracting claim.
- `0`: does not answer, answers the wrong question, or answers an unanswerable
  question as if evidence existed.

### `faithfulness`

- `1`: every factual claim is supported by the supplied evidence.
- `0.5`: the core answer is supported, but a minor detail is unsupported or
  overstated.
- `0`: includes an invented procedure, command, threshold, identifier, number,
  credential, or root cause; or contradicts the supplied evidence.

For an unanswerable record, a correct abstention with no unsupported claims
scores 1.

### `citation_correctness`

- `1`: every citation maps to a supplied evidence item, and the cited passage
  supports the claim it is attached to.
- `0.5`: all citations map to supplied evidence, but at least one is weak,
  irrelevant, or does not fully support its claim.
- `0`: any citation is fabricated or maps to a chunk that was not supplied.

Use `null` only when the prediction contains no citations and no citation was
expected. An answerable record with no citations scores 0.

### `citation_completeness`

- `1`: every required expected source/version is represented in the citations.
- `0.5`: some required source/version is missing, but at least one required
  citation is present.
- `0`: no required citation is present, or the answer cites only non-required
  sources.
- `null`: not applicable (abstention or unanswerable record).

### `abstention_correctness`

Binary and applicable to every record.

- `1`: abstains exactly when the record is unanswerable; answers when the record
  is answerable.
- `0`: answers an unanswerable record, abstains on an answerable record, or
  fails to produce a response (`status="error"`).

### `conflict_handling`

Only applicable when `conflict_expected` is true or the category is
`version_conflict`/`distractor` with mixed-version evidence. Otherwise `null`.

- `1`: identifies the conflict, names the versions involved, and does not present
  stale guidance as current.
- `0.5`: mentions both versions but does not clearly identify the conflict, or
  identifies the conflict without naming the version.
- `0`: silently chooses one version, attributes a version-specific step to the
  wrong version, or presents stale guidance as current.

## Claim-level citation coverage

For each answerable, non-abstaining prediction, list up to five factual claims.
For each claim record:

- `claim`: short paraphrase of the claim.
- `citation_evidence_id`: the evidence ID the answer cites for that claim, or
  `null` if it cites nothing.
- `supported`: `true` if the cited evidence supports the claim, `false` if it does
  not, or `null` if no citation exists.

Do not invent claims that are not in the answer. If the answer is an abstention
or a provider error, return an empty `claims` list.

## Required output

Return exactly one JSON object:

```json
{
  "records": [
    {
      "record_id": "answer-v1-example",
      "scores": {
        "answer_relevance": 1,
        "faithfulness": 1,
        "citation_correctness": 1,
        "citation_completeness": 1,
        "abstention_correctness": 1,
        "conflict_handling": null
      },
      "claims": [
        {
          "claim": "The operator checks the service health endpoint.",
          "citation_evidence_id": "E3",
          "supported": true
        }
      ],
      "rationale": "One or two sentences naming the evidence used."
    }
  ]
}
```

Do not wrap the object in Markdown fences. Do not add fields. Score every
record in the batch exactly once.

"""Tests for resumable answer-evaluation scoring runs."""

from __future__ import annotations

import json

from scripts.score_answer_eval import (
    load_existing_scores,
    reusable_scores,
    write_scores_ordered,
)


def test_resume_keeps_only_successful_scores(tmp_path) -> None:
    path = tmp_path / "scores.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"record_id": "one", "scoring_error": None}),
                json.dumps({"record_id": "two", "scoring_error": "ProviderError"}),
                "",
            ]
        ),
        encoding="utf-8",
    )

    existing = load_existing_scores(path)
    reusable = reusable_scores(existing)

    assert set(existing) == {"one", "two"}
    assert set(reusable) == {"one"}


def test_write_scores_ordered_uses_prediction_order(tmp_path) -> None:
    path = tmp_path / "scores.jsonl"
    predictions = [{"record_id": "a"}, {"record_id": "b"}, {"record_id": "c"}]
    scores = {
        "c": {"record_id": "c", "scoring_error": None},
        "a": {"record_id": "a", "scoring_error": None},
    }

    write_scores_ordered(path, predictions, scores)

    assert [json.loads(line)["record_id"] for line in path.read_text().splitlines()] == [
        "a",
        "c",
    ]


def test_load_existing_scores_ignores_missing_file(tmp_path) -> None:
    assert load_existing_scores(tmp_path / "missing.jsonl") == {}

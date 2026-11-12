"""Tests for resumable answer-evaluation generation runs."""

from __future__ import annotations

import json

from scripts.run_answer_eval import (
    load_existing_predictions,
    reusable_predictions,
    write_predictions_ordered,
)


def test_resume_keeps_only_successful_predictions(tmp_path) -> None:
    path = tmp_path / "predictions.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"record_id": "one", "status": "ok"}),
                json.dumps({"record_id": "two", "status": "error"}),
                "",
            ]
        ),
        encoding="utf-8",
    )

    existing = load_existing_predictions(path)
    reusable = reusable_predictions(existing)

    assert set(existing) == {"one", "two"}
    assert set(reusable) == {"one"}


def test_write_predictions_ordered_uses_dataset_order(tmp_path) -> None:
    path = tmp_path / "predictions.jsonl"
    records = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    predictions = {
        "c": {"record_id": "c", "status": "ok"},
        "a": {"record_id": "a", "status": "ok"},
    }

    write_predictions_ordered(path, records, predictions)

    written = load_existing_predictions(path)
    assert list(written) == ["a", "c"]
    assert [json.loads(line)["record_id"] for line in path.read_text().splitlines()] == [
        "a",
        "c",
    ]


def test_load_existing_predictions_ignores_missing_file(tmp_path) -> None:
    assert load_existing_predictions(tmp_path / "missing.jsonl") == {}

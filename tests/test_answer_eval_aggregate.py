"""Tests for repeated-run aggregation of answer_eval_v1 artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.aggregate_answer_eval_runs import aggregate_runs, run_summary


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def make_run(
    root: Path,
    *,
    run_id: str,
    dataset_sha: str = "dataset-sha",
    corpus_fingerprint: str = "corpus-sha",
    prompt_version: str = "grounded_answer_v2",
    retrieval: dict | None = None,
) -> Path:
    run_dir = root / run_id
    run_dir.mkdir()
    write_json(
        run_dir / "config.json",
        {
            "run_id": run_id,
            "dataset": {"sha256": dataset_sha},
            "corpus": {"fingerprint": corpus_fingerprint},
            "code": {"pipeline_sha256": "pipeline-sha"},
            "retrieval": retrieval or {"strategy": "hybrid"},
            "generation": {
                "provider": "opencode_cli",
                "model": "opencode/test-model",
                "requested_model": "opencode/test-model",
                "temperature": 0.0,
                "temperature_enforced": False,
                "max_output_tokens": 700,
                "max_output_tokens_enforced": "soft prompt limit only",
                "prompt_version": prompt_version,
                "provider_capabilities": {"temperature": "not_enforced"},
            },
            "scoring": {
                "scorers": [
                    {"id": "scorer_a", "model": "opencode/scorer-a"},
                    {"id": "scorer_b", "model": "opencode/scorer-b"},
                ],
                "configured_scorers": [
                    {"id": "scorer_a", "model": "opencode/scorer-a"},
                    {"id": "scorer_b", "model": "opencode/scorer-b"},
                ],
                "model_deviations": [],
            },
        },
    )
    write_json(
        run_dir / "summary.json",
        {
            "records_ok": 23,
            "records_error": 3,
            "provider_failures": 1,
            "generation_contract_failures": 2,
            "provider_failure_rate": 0.0385,
            "provider_attempts_total": 27,
            "provider_retries": 2,
            "provider_usage_reported": 0,
            "latency_ms": {"p50": 1000.0, "p95": 2000.0, "mean": 1200.0, "max": 2500.0},
            "tokens_estimate": {"input": 1000, "output": 100},
        },
    )
    write_json(
        run_dir / "report.json",
        {
            "status": "completed",
            "aggregate": {
                "provider_failure_rate": 0.0385,
                "sample_counts": {"records": 26},
                "overall": {
                    "answer_relevance": 0.9,
                    "faithfulness": 0.9,
                    "citation_correctness": 0.8,
                    "citation_completeness": 0.9,
                    "abstention_correctness": 1.0,
                    "conflict_handling": 0.5,
                    "n": 26,
                },
            },
        },
    )
    write_json(
        run_dir / "failure_analysis.json",
        {"counts": {"provider_failure": 1, "incomplete_generation": 2}},
    )
    write_json(
        run_dir / "scorer_metrics.json",
        {"inter_scorer_agreement": {"exact_rate": 0.9}},
    )
    write_json(
        run_dir / "disagreements.json",
        [{"record_id": "r1", "metric": "answer_relevance"}],
    )
    return run_dir


def test_run_summary_reads_required_artifacts(tmp_path: Path) -> None:
    run_dir = make_run(tmp_path, run_id="run1")

    summary = run_summary(run_dir)

    assert summary["run_id"] == "run1"
    assert summary["provider_failure_rate"] == 0.0385
    assert summary["provider_failures"] == 1
    assert summary["generation_contract_failures"] == 2
    assert summary["metrics"]["answer_relevance"] == 0.9
    assert summary["failure_counts"]["incomplete_generation"] == 2
    assert summary["scorer_disagreements"] == 1


def test_aggregate_reports_ranges_and_failure_totals(tmp_path: Path) -> None:
    runs = [
        run_summary(make_run(tmp_path, run_id="run1")),
        run_summary(make_run(tmp_path, run_id="run2")),
    ]

    aggregate = aggregate_runs(runs)

    assert aggregate["run_count"] == 2
    assert aggregate["comparability"]["frozen_configuration_held"] is True
    assert aggregate["metric_summary"]["answer_relevance"]["mean"] == 0.9
    assert aggregate["metric_summary"]["answer_relevance"]["stdev"] == 0.0
    assert aggregate["failure_counts"]["incomplete_generation"]["total"] == 4
    assert aggregate["provider_failure"]["total_errors"] == 6
    assert aggregate["provider_failure"]["total_provider_failures"] == 2
    assert aggregate["provider_failure"]["total_generation_contract_failures"] == 4
    assert aggregate["provider_failure"]["overall_rate"] == 0.0385
    assert aggregate["scorer_disagreements"]["total"] == 2


def test_aggregate_flags_configuration_deviations(tmp_path: Path) -> None:
    runs = [
        run_summary(make_run(tmp_path, run_id="run1")),
        run_summary(
            make_run(
                tmp_path,
                run_id="run2",
                dataset_sha="different-dataset",
                prompt_version="grounded_answer_v3",
            )
        ),
    ]

    aggregate = aggregate_runs(runs)

    assert aggregate["comparability"]["frozen_configuration_held"] is False
    assert "dataset_sha256" in aggregate["comparability"]["deviations"]
    assert "prompt_version" in aggregate["comparability"]["deviations"]

import json
import math
from pathlib import Path
from uuid import uuid4

import pytest

from evals.retrieval_eval import (
    EvaluationRunError,
    EvalValidationError,
    all_expected_at_k,
    any_hit_at_k,
    compare_reports,
    compute_corpus_fingerprint,
    evaluate_query_result,
    load_dataset_profile,
    load_golden_dataset,
    load_source_map,
    map_ranked_sources,
    ndcg_at_k,
    preflight_source_map,
    recall_at_k,
    reciprocal_rank_at_k,
    run_evaluation,
    validate_dataset_profile,
    write_report,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def valid_record(**overrides):
    record = {
        "id": "query-001",
        "question": "How do we handle payment timeouts?",
        "expected_source_ids": ["runbook-payments"],
        "answerable": True,
        "category": "runbook",
        "notes": "Synthetic fixture.",
    }
    record.update(overrides)
    return record


def write_dataset(tmp_path, records, name="golden_v0.jsonl"):
    path = tmp_path / name
    path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    return path


def write_source_map(tmp_path, sources, name="golden_v0.source_map.json"):
    path = tmp_path / name
    path.write_text(
        json.dumps(
            {
                "mapping_version": 1,
                "dataset_id": "golden_v0",
                "sources": sources,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_load_dataset_accepts_valid_jsonl_and_preserves_order(tmp_path) -> None:
    path = write_dataset(
        tmp_path,
        [
            valid_record(id="first"),
            valid_record(id="second", expected_source_ids=["source-b"]),
        ],
    )

    dataset = load_golden_dataset(path)

    assert dataset.dataset_id == "golden_v0"
    assert [record.id for record in dataset.records] == ["first", "second"]
    assert len(dataset.sha256) == 64


def test_load_dataset_rejects_missing_required_field(tmp_path) -> None:
    record = valid_record()
    del record["notes"]
    path = write_dataset(tmp_path, [record])

    with pytest.raises(EvalValidationError, match="missing required field.*notes"):
        load_golden_dataset(path)


def test_load_dataset_rejects_duplicate_ids(tmp_path) -> None:
    path = write_dataset(
        tmp_path,
        [valid_record(id="duplicate"), valid_record(id="duplicate")],
    )

    with pytest.raises(EvalValidationError, match="duplicate record id 'duplicate'"):
        load_golden_dataset(path)


def test_load_dataset_rejects_answerable_without_expected_sources(tmp_path) -> None:
    path = write_dataset(
        tmp_path,
        [valid_record(expected_source_ids=[], answerable=True)],
    )

    with pytest.raises(
        EvalValidationError,
        match="answerable records must have at least one expected source ID",
    ):
        load_golden_dataset(path)


def test_load_dataset_rejects_unanswerable_with_expected_sources(tmp_path) -> None:
    path = write_dataset(
        tmp_path,
        [
            valid_record(
                expected_source_ids=["source-a"],
                answerable=False,
            )
        ],
    )

    with pytest.raises(
        EvalValidationError,
        match="unanswerable records must have an empty",
    ):
        load_golden_dataset(path)


def test_load_dataset_rejects_duplicate_expected_source_ids(tmp_path) -> None:
    path = write_dataset(
        tmp_path,
        [
            valid_record(
                expected_source_ids=["source-a", "source-a"],
            )
        ],
    )

    with pytest.raises(EvalValidationError, match="contains duplicate IDs"):
        load_golden_dataset(path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("id", "", "must be a nonempty string"),
        ("question", 42, "must be a nonempty string"),
        ("answerable", "yes", "must be a JSON boolean"),
        ("expected_source_ids", "source-a", "must be a list"),
        ("category", None, "must be a nonempty string"),
        ("notes", 123, "must be a string"),
    ],
)
def test_load_dataset_rejects_invalid_field_types(
    tmp_path,
    field: str,
    value,
    message: str,
) -> None:
    path = write_dataset(tmp_path, [valid_record(**{field: value})])

    with pytest.raises(EvalValidationError, match=message):
        load_golden_dataset(path)


def test_load_dataset_rejects_non_string_expected_source_id(tmp_path) -> None:
    path = write_dataset(
        tmp_path,
        [valid_record(expected_source_ids=["source-a", 7])],
    )

    with pytest.raises(
        EvalValidationError,
        match="every 'expected_source_ids' entry must be a nonempty string",
    ):
        load_golden_dataset(path)


def test_load_dataset_rejects_malformed_json_with_line_number(tmp_path) -> None:
    path = tmp_path / "broken.jsonl"
    path.write_text(
        json.dumps(valid_record()) + "\n" + "{not json}\n",
        encoding="utf-8",
    )

    with pytest.raises(EvalValidationError, match=r"broken\.jsonl:2: invalid JSON"):
        load_golden_dataset(path)


def test_load_dataset_rejects_empty_file(tmp_path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")

    with pytest.raises(EvalValidationError, match="contains no evaluation records"):
        load_golden_dataset(path)


def test_load_dataset_accepts_repository_golden_v0() -> None:
    dataset = load_golden_dataset(
        REPO_ROOT / "evals" / "datasets" / "golden_v0.jsonl"
    )

    assert dataset.dataset_id == "golden_v0"
    assert len(dataset.records) == 6
    assert len(dataset.answerable_records) == 4
    assert len(dataset.unanswerable_records) == 2
    assert len({record.id for record in dataset.records}) == 6


def test_source_map_accepts_explicit_mapping_and_optional_version(tmp_path) -> None:
    dataset_path = write_dataset(
        tmp_path,
        [valid_record(expected_source_ids=["source-a", "source-b"])],
    )
    dataset = load_golden_dataset(dataset_path)
    source_map_path = write_source_map(
        tmp_path,
        {
            "source-a": {"source_key": "runbooks/a.md", "version_number": 2},
            "source-b": {"source_key": "runbooks/b.md"},
        },
    )

    source_map = load_source_map(source_map_path, dataset)

    assert source_map.dataset_id == "golden_v0"
    assert source_map.sources["source-a"].source_key == "runbooks/a.md"
    assert source_map.sources["source-a"].version_number == 2
    assert source_map.sources["source-b"].version_number is None


def test_source_map_requires_mapping_for_every_expected_source_id(tmp_path) -> None:
    dataset_path = write_dataset(
        tmp_path,
        [valid_record(expected_source_ids=["source-a", "source-b"])],
    )
    dataset = load_golden_dataset(dataset_path)
    source_map_path = write_source_map(
        tmp_path,
        {"source-a": {"source_key": "runbooks/a.md"}},
    )

    with pytest.raises(
        EvalValidationError,
        match="missing explicit mappings for: source-b",
    ) as exc_info:
        load_source_map(source_map_path, dataset)

    assert "No retrieval was executed and no scores were produced" in str(exc_info.value)


def test_source_map_rejects_unknown_source_id(tmp_path) -> None:
    dataset_path = write_dataset(tmp_path, [valid_record()])
    dataset = load_golden_dataset(dataset_path)
    source_map_path = write_source_map(
        tmp_path,
        {
            "runbook-payments": {"source_key": "runbooks/payments.md"},
            "unexpected-source": {"source_key": "runbooks/other.md"},
        },
    )

    with pytest.raises(
        EvalValidationError,
        match="unknown source IDs not present in the dataset: unexpected-source",
    ):
        load_source_map(source_map_path, dataset)


def test_source_map_rejects_duplicate_source_key_aliases(tmp_path) -> None:
    dataset_path = write_dataset(
        tmp_path,
        [valid_record(expected_source_ids=["source-a", "source-b"])],
    )
    dataset = load_golden_dataset(dataset_path)
    source_map_path = write_source_map(
        tmp_path,
        {
            "source-a": {"source_key": "runbooks/shared.md"},
            "source-b": {"source_key": "runbooks/shared.md"},
        },
    )

    with pytest.raises(EvalValidationError, match="is already mapped to 'source-a'"):
        load_source_map(source_map_path, dataset)


@pytest.mark.parametrize("version_number", [0, -1, "1", True])
def test_source_map_rejects_invalid_version_number(
    tmp_path,
    version_number,
) -> None:
    dataset_path = write_dataset(tmp_path, [valid_record()])
    dataset = load_golden_dataset(dataset_path)
    source_map_path = write_source_map(
        tmp_path,
        {
            "runbook-payments": {
                "source_key": "runbooks/payments.md",
                "version_number": version_number,
            }
        },
    )

    with pytest.raises(
        EvalValidationError,
        match="'version_number' must be an integer >= 1",
    ):
        load_source_map(source_map_path, dataset)


def test_source_map_rejects_dataset_mismatch(tmp_path) -> None:
    dataset_path = write_dataset(tmp_path, [valid_record()])
    dataset = load_golden_dataset(dataset_path)
    source_map_path = tmp_path / "map.json"
    source_map_path.write_text(
        json.dumps(
            {
                "mapping_version": 1,
                "dataset_id": "golden_v1",
                "sources": {
                    "runbook-payments": {
                        "source_key": "runbooks/payments.md",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        EvalValidationError,
        match="dataset_id 'golden_v1' does not match dataset 'golden_v0'",
    ):
        load_source_map(source_map_path, dataset)


def test_recall_at_5_uses_fractional_recall_for_multi_document_queries() -> None:
    assert recall_at_k(
        ["source-a", "source-x"],
        ["source-a", "source-b"],
    ) == pytest.approx(0.5)


def test_recall_at_5_uses_top_five_ranked_sources() -> None:
    ranked = ["source-1", "source-2", "source-3", "source-4", "source-5", "source-6"]

    assert recall_at_k(ranked, ["source-6"]) == 0.0
    assert recall_at_k(ranked, ["source-5"]) == 1.0


def test_mrr_uses_first_relevant_rank() -> None:
    assert reciprocal_rank_at_k(
        ["source-x", "source-a", "source-b"],
        ["source-a", "source-b"],
    ) == pytest.approx(0.5)


def test_hit_metrics_report_any_and_all_expected_sources() -> None:
    ranked = ["source-a", "source-x"]

    assert any_hit_at_k(ranked, ["source-a", "source-b"]) == 1
    assert all_expected_at_k(ranked, ["source-a", "source-b"]) == 0
    assert all_expected_at_k(ranked, ["source-a"]) == 1


def test_metrics_reject_empty_expected_source_sets() -> None:
    with pytest.raises(ValueError, match="expected_source_ids must not be empty"):
        recall_at_k(["source-a"], [])

    with pytest.raises(ValueError, match="expected_source_ids must not be empty"):
        reciprocal_rank_at_k(["source-a"], [])

    with pytest.raises(ValueError, match="expected_source_ids must not be empty"):
        any_hit_at_k(["source-a"], [])

    with pytest.raises(ValueError, match="expected_source_ids must not be empty"):
        all_expected_at_k(["source-a"], [])


def test_ndcg_at_5_with_synthetic_graded_relevance() -> None:
    grades = {"source-a": 2, "source-b": 1}

    ideal = ndcg_at_k(["source-a", "source-b"], grades)
    reversed_order = ndcg_at_k(["source-b", "source-a"], grades)

    assert ideal == pytest.approx(1.0)
    expected_reversed = (1.0 + 3.0 / math.log2(3)) / (
        3.0 + 1.0 / math.log2(3)
    )
    assert reversed_order == pytest.approx(expected_reversed)


def test_ndcg_returns_none_without_positive_grades() -> None:
    assert ndcg_at_k(["source-a"], {"source-a": 0}) is None
    assert ndcg_at_k(["source-a"], {}) is None


def make_result(
    source_key: str,
    *,
    version_number: int = 1,
    similarity: float = 0.8,
    hybrid_score: float = 0.03,
) -> dict:
    return {
        "chunk_id": uuid4(),
        "document_id": uuid4(),
        "version_id": uuid4(),
        "source_key": source_key,
        "version_number": version_number,
        "similarity": similarity,
        "hybrid_score": hybrid_score,
    }


def build_inputs(tmp_path, records, sources):
    dataset_path = write_dataset(tmp_path, records)
    dataset = load_golden_dataset(dataset_path)
    source_map_path = write_source_map(tmp_path, sources)
    source_map = load_source_map(source_map_path, dataset)
    return dataset, source_map


def test_result_mapping_uses_source_key_not_title_or_document_id(tmp_path) -> None:
    dataset, source_map = build_inputs(
        tmp_path,
        [valid_record(expected_source_ids=["source-a"])],
        {"source-a": {"source_key": "runbooks/a.md"}},
    )

    unmapped = make_result("runbooks/other.md")
    unmapped["title"] = "source-a"
    ranked_unmapped = map_ranked_sources([unmapped], source_map)
    assert ranked_unmapped[0].mapping_status == "unmapped"
    assert ranked_unmapped[0].mapped_source_id is None

    matched = make_result("runbooks/a.md")
    matched["title"] = "Completely different display title"
    ranked_matched = map_ranked_sources([matched], source_map)
    assert ranked_matched[0].mapping_status == "matched"
    assert ranked_matched[0].mapped_source_id == "source-a"
    assert dataset.answerable_records[0].expected_source_ids == ("source-a",)


def test_result_mapping_deduplicates_multiple_chunks_from_same_source(tmp_path) -> None:
    _, source_map = build_inputs(
        tmp_path,
        [
            valid_record(
                expected_source_ids=["source-a", "source-b"],
            )
        ],
        {
            "source-a": {"source_key": "runbooks/a.md"},
            "source-b": {"source_key": "runbooks/b.md"},
        },
    )

    ranked = map_ranked_sources(
        [
            make_result("runbooks/a.md"),
            make_result("runbooks/a.md"),
            make_result("runbooks/b.md"),
        ],
        source_map,
    )

    assert [(source.rank, source.source_key) for source in ranked] == [
        (1, "runbooks/a.md"),
        (2, "runbooks/b.md"),
    ]


def test_result_mapping_marks_unmapped_and_version_mismatch_results(tmp_path) -> None:
    _, source_map = build_inputs(
        tmp_path,
        [valid_record(expected_source_ids=["source-a"])],
        {
            "source-a": {
                "source_key": "runbooks/a.md",
                "version_number": 2,
            }
        },
    )

    ranked = map_ranked_sources(
        [
            make_result("runbooks/a.md", version_number=1),
            make_result("runbooks/unknown.md"),
            make_result("runbooks/a.md", version_number=2),
        ],
        source_map,
    )

    assert [source.mapping_status for source in ranked] == [
        "version_mismatch",
        "unmapped",
        "matched",
    ]
    assert ranked[0].mapped_source_id == "source-a"
    assert ranked[2].mapped_source_id == "source-a"


def test_evaluate_query_result_records_per_query_metrics(tmp_path) -> None:
    dataset, source_map = build_inputs(
        tmp_path,
        [
            valid_record(
                expected_source_ids=["source-a", "source-b"],
            )
        ],
        {
            "source-a": {"source_key": "runbooks/a.md"},
            "source-b": {"source_key": "runbooks/b.md"},
        },
    )

    evaluation = evaluate_query_result(
        dataset.answerable_records[0],
        [make_result("runbooks/a.md")],
        source_map,
        score_field="similarity",
    )

    assert evaluation["ranked_source_ids"] == ["source-a"]
    assert evaluation["matched_source_ids"] == ["source-a"]
    assert evaluation["missing_source_ids"] == ["source-b"]
    assert evaluation["metrics"]["recall_at_5"] == pytest.approx(0.5)
    assert evaluation["metrics"]["mrr"] == pytest.approx(1.0)
    assert evaluation["metrics"]["any_hit_at_5"] == 1
    assert evaluation["metrics"]["all_expected_at_5"] == 0
    assert evaluation["metrics"]["ndcg_at_5"] is None
    assert (
        evaluation["metrics"]["ndcg_status"]
        == "not_applicable_binary_relevance_only"
    )
    assert "content" not in evaluation["results"][0]
    assert "title" not in evaluation["results"][0]


def test_runner_calls_both_retrievers_with_same_query_and_limit(tmp_path) -> None:
    dataset, source_map = build_inputs(
        tmp_path,
        [
            valid_record(id="q1", question="first", expected_source_ids=["source-a"]),
            valid_record(id="q2", question="second", expected_source_ids=["source-b"]),
        ],
        {
            "source-a": {"source_key": "runbooks/a.md"},
            "source-b": {"source_key": "runbooks/b.md"},
        },
    )
    tenant_id = uuid4()
    calls: list[dict] = []

    def semantic(**kwargs):
        calls.append({"retriever": "semantic", **kwargs})
        return {
            "first": [make_result("runbooks/a.md")],
            "second": [make_result("runbooks/unknown.md")],
        }[kwargs["query"]]

    def hybrid(**kwargs):
        calls.append({"retriever": "hybrid", **kwargs})
        return {
            "first": [make_result("runbooks/a.md")],
            "second": [make_result("runbooks/b.md")],
        }[kwargs["query"]]

    run_evaluation(
        dataset=dataset,
        source_map=source_map,
        tenant_id=tenant_id,
        retrievers={"semantic": semantic, "hybrid": hybrid},
        retrieval_config={"k": 5},
    )

    assert len(calls) == 4
    for call in calls:
        assert call["tenant_id"] == tenant_id
        assert call["limit"] == 5

    first_calls = [call for call in calls if call["query"] == "first"]
    second_calls = [call for call in calls if call["query"] == "second"]
    assert [call["retriever"] for call in first_calls] == ["semantic", "hybrid"]
    assert [call["retriever"] for call in second_calls] == ["semantic", "hybrid"]


def test_runner_builds_per_query_comparison_and_aggregates(tmp_path) -> None:
    dataset, source_map = build_inputs(
        tmp_path,
        [
            valid_record(id="q1", question="first", expected_source_ids=["source-a"]),
            valid_record(id="q2", question="second", expected_source_ids=["source-b"]),
        ],
        {
            "source-a": {"source_key": "runbooks/a.md"},
            "source-b": {"source_key": "runbooks/b.md"},
        },
    )

    def semantic(**kwargs):
        return {
            "first": [make_result("runbooks/a.md")],
            "second": [make_result("runbooks/unknown.md")],
        }[kwargs["query"]]

    def hybrid(**kwargs):
        return {
            "first": [make_result("runbooks/a.md")],
            "second": [make_result("runbooks/b.md")],
        }[kwargs["query"]]

    report = run_evaluation(
        dataset=dataset,
        source_map=source_map,
        tenant_id=uuid4(),
        retrievers={"semantic": semantic, "hybrid": hybrid},
        retrieval_config={"k": 5},
    )

    assert report["aggregate"]["scored_query_count"] == 2
    assert report["aggregate"]["semantic"]["recall_at_5"] == pytest.approx(0.5)
    assert report["aggregate"]["hybrid"]["recall_at_5"] == pytest.approx(1.0)
    assert report["aggregate"]["delta"]["recall_at_5_delta"] == pytest.approx(0.5)
    assert report["queries"][0]["comparison"]["recall_at_5_delta"] == pytest.approx(0.0)
    assert report["queries"][1]["comparison"]["recall_at_5_delta"] == pytest.approx(1.0)
    assert report["aggregate"]["semantic"]["ndcg_at_5"] is None
    assert (
        report["aggregate"]["semantic"]["ndcg_status"]
        == "not_applicable_binary_relevance_only"
    )


def test_runner_excludes_unanswerable_records_from_retrieval_metrics(tmp_path) -> None:
    dataset, source_map = build_inputs(
        tmp_path,
        [
            valid_record(
                id="answerable",
                question="answerable",
                expected_source_ids=["source-a"],
            ),
            valid_record(
                id="unanswerable",
                question="unanswerable",
                expected_source_ids=[],
                answerable=False,
            ),
        ],
        {"source-a": {"source_key": "runbooks/a.md"}},
    )
    calls: list[str] = []

    def semantic(**kwargs):
        calls.append(kwargs["query"])
        return [make_result("runbooks/a.md")]

    def hybrid(**kwargs):
        calls.append(kwargs["query"])
        return [make_result("runbooks/a.md")]

    report = run_evaluation(
        dataset=dataset,
        source_map=source_map,
        tenant_id=uuid4(),
        retrievers={"semantic": semantic, "hybrid": hybrid},
        retrieval_config={"k": 5},
    )

    assert calls == ["answerable", "answerable"]
    assert report["aggregate"]["scored_query_count"] == 1
    assert len(report["unanswerable"]) == 1
    assert report["unanswerable"][0]["scored"] is False
    assert "abstention" in report["unanswerable"][0]["reason"]


def test_empty_answerable_set_produces_null_aggregates(tmp_path) -> None:
    dataset, source_map = build_inputs(
        tmp_path,
        [
            valid_record(
                id="unanswerable",
                question="unanswerable",
                expected_source_ids=[],
                answerable=False,
            )
        ],
        {},
    )

    def fail_if_called(**kwargs):
        raise AssertionError("retrievers must not be called without answerable records")

    report = run_evaluation(
        dataset=dataset,
        source_map=source_map,
        tenant_id=uuid4(),
        retrievers={"semantic": fail_if_called, "hybrid": fail_if_called},
        retrieval_config={"k": 5},
    )

    assert report["aggregate"]["scored_query_count"] == 0
    assert report["aggregate"]["semantic"]["query_count"] == 0
    assert report["aggregate"]["semantic"]["recall_at_5"] is None
    assert report["aggregate"]["hybrid"]["mrr"] is None


def test_report_contains_dataset_hash_and_retrieval_configuration(tmp_path) -> None:
    dataset, source_map = build_inputs(
        tmp_path,
        [valid_record(expected_source_ids=["source-a"])],
        {"source-a": {"source_key": "runbooks/a.md"}},
    )

    def retriever(**kwargs):
        return [make_result("runbooks/a.md")]

    report = run_evaluation(
        dataset=dataset,
        source_map=source_map,
        tenant_id=uuid4(),
        retrievers={"semantic": retriever, "hybrid": retriever},
        retrieval_config={
            "k": 5,
            "semantic": {"embedding_model": "test-model", "dimensions": 384},
            "hybrid": {"rrf_k": 60, "candidate_multiplier": 3},
        },
    )

    assert report["dataset"]["id"] == "golden_v0"
    assert report["dataset"]["sha256"] == dataset.sha256
    assert report["source_map"]["sha256"] == source_map.sha256
    assert report["retrieval_configuration"]["k"] == 5
    assert report["retrieval_configuration"]["hybrid"]["rrf_k"] == 60
    assert "database_url" not in json.dumps(report).lower()


def test_report_ordering_is_deterministic(tmp_path) -> None:
    dataset, source_map = build_inputs(
        tmp_path,
        [
            valid_record(id="q1", question="first", expected_source_ids=["source-a"]),
            valid_record(id="q2", question="second", expected_source_ids=["source-b"]),
        ],
        {
            "source-a": {"source_key": "runbooks/a.md"},
            "source-b": {"source_key": "runbooks/b.md"},
        },
    )

    semantic_result = make_result("runbooks/a.md")
    hybrid_result = make_result("runbooks/b.md")

    def semantic(**kwargs):
        return [semantic_result]

    def hybrid(**kwargs):
        return [hybrid_result]

    first = run_evaluation(
        dataset=dataset,
        source_map=source_map,
        tenant_id="tenant-fixed",
        retrievers={"semantic": semantic, "hybrid": hybrid},
        retrieval_config={"k": 5},
    )
    second = run_evaluation(
        dataset=dataset,
        source_map=source_map,
        tenant_id="tenant-fixed",
        retrievers={"semantic": semantic, "hybrid": hybrid},
        retrieval_config={"k": 5},
    )

    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    write_report(first, first_path)
    write_report(second, second_path)

    assert first_path.read_text(encoding="utf-8") == second_path.read_text(
        encoding="utf-8"
    )
    assert [query["id"] for query in first["queries"]] == ["q1", "q2"]


def test_runner_fails_fast_without_leaking_exception_details(tmp_path) -> None:
    dataset, source_map = build_inputs(
        tmp_path,
        [valid_record(id="q1", expected_source_ids=["source-a"])],
        {"source-a": {"source_key": "runbooks/a.md"}},
    )

    def semantic(**kwargs):
        raise RuntimeError("postgresql://user:secret@localhost/private")

    def hybrid(**kwargs):
        return [make_result("runbooks/a.md")]

    with pytest.raises(EvaluationRunError) as exc_info:
        run_evaluation(
            dataset=dataset,
            source_map=source_map,
            tenant_id=uuid4(),
            retrievers={"semantic": semantic, "hybrid": hybrid},
            retrieval_config={"k": 5},
        )

    message = str(exc_info.value)
    assert "query id 'q1'" in message
    assert "RuntimeError" in message
    assert "No evaluation report was written" in message
    assert "secret" not in message
    assert "postgresql://" not in message


def test_cli_validate_only_succeeds_with_valid_fixture(tmp_path, capsys) -> None:
    from scripts.run_retrieval_eval import main

    dataset_path = write_dataset(tmp_path, [valid_record()])
    source_map_path = write_source_map(
        tmp_path,
        {"runbook-payments": {"source_key": "runbooks/payments.md"}},
    )

    exit_code = main(
        [
            "--validate-only",
            "--dataset",
            str(dataset_path),
            "--source-map",
            str(source_map_path),
        ]
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Validated dataset 'golden_v0'" in output
    assert "No retrieval was executed" in output


def test_cli_reports_missing_mappings_actionably(tmp_path, capsys) -> None:
    from scripts.run_retrieval_eval import main

    dataset_path = write_dataset(
        tmp_path,
        [
            valid_record(id="mapped", expected_source_ids=["runbook-payments"]),
            valid_record(
                id="unmapped",
                question="What does PAY-503 mean?",
                expected_source_ids=["runbook-payments-errors-v1"],
            ),
        ],
    )
    source_map_path = write_source_map(
        tmp_path,
        {"runbook-payments": {"source_key": "runbooks/payments.md"}},
    )

    exit_code = main(
        [
            "--validate-only",
            "--dataset",
            str(dataset_path),
            "--source-map",
            str(source_map_path),
        ]
    )

    assert exit_code == 2
    error = capsys.readouterr().err
    assert "Evaluation validation failed" in error
    assert "missing explicit mappings for:" in error
    assert "runbook-payments-errors-v1" in error
    assert "No retrieval was executed and no scores were produced" in error


def test_cli_requires_tenant_for_live_run(tmp_path, capsys) -> None:
    from scripts.run_retrieval_eval import main

    dataset_path = write_dataset(tmp_path, [valid_record()])
    source_map_path = write_source_map(
        tmp_path,
        {"runbook-payments": {"source_key": "runbooks/payments.md"}},
    )

    exit_code = main(
        [
            "--dataset",
            str(dataset_path),
            "--source-map",
            str(source_map_path),
        ]
    )

    assert exit_code == 2
    assert "requires --tenant-id" in capsys.readouterr().err


def test_unmapped_result_keeps_rank_slot_and_counts_as_irrelevant(tmp_path) -> None:
    dataset, source_map = build_inputs(
        tmp_path,
        [valid_record(expected_source_ids=["source-a"])],
        {"source-a": {"source_key": "runbooks/a.md"}},
    )

    evaluation = evaluate_query_result(
        dataset.answerable_records[0],
        [make_result("runbooks/unknown.md"), make_result("runbooks/a.md")],
        source_map,
        score_field="similarity",
    )

    assert evaluation["metrics"]["recall_at_5"] == pytest.approx(1.0)
    assert evaluation["metrics"]["mrr"] == pytest.approx(0.5)
    assert evaluation["metrics"]["first_expected_rank"] == 2
    assert evaluation["ranked_units"][0]["mapping_status"] == "unmapped"
    assert evaluation["ranked_units"][0]["relevant"] is False


def test_version_mismatch_keeps_rank_slot_and_counts_as_irrelevant(tmp_path) -> None:
    dataset, source_map = build_inputs(
        tmp_path,
        [valid_record(expected_source_ids=["source-a"])],
        {"source-a": {"source_key": "runbooks/a.md", "version_number": 2}},
    )

    evaluation = evaluate_query_result(
        dataset.answerable_records[0],
        [
            make_result("runbooks/a.md", version_number=1),
            make_result("runbooks/a.md", version_number=2),
        ],
        source_map,
        score_field="similarity",
    )

    assert evaluation["metrics"]["mrr"] == pytest.approx(0.5)
    assert evaluation["metrics"]["first_expected_rank"] == 2
    assert evaluation["ranked_units"][0]["mapping_status"] == "version_mismatch"
    assert evaluation["ranked_units"][0]["relevant"] is False


def test_source_metrics_collapse_duplicate_chunks_before_the_cutoff(tmp_path) -> None:
    dataset, source_map = build_inputs(
        tmp_path,
        [
            valid_record(id="q1", expected_source_ids=["source-b"]),
            valid_record(
                id="q2",
                question="What does the database runbook say?",
                expected_source_ids=["source-a"],
            ),
        ],
        {
            "source-a": {"source_key": "runbooks/a.md"},
            "source-b": {"source_key": "runbooks/b.md"},
        },
    )

    results = [make_result("runbooks/a.md") for _ in range(5)]
    results.append(make_result("runbooks/b.md"))

    evaluation = evaluate_query_result(
        dataset.answerable_records[0],
        results,
        source_map,
        score_field="similarity",
        fetch_limit=10,
    )

    assert evaluation["metrics"]["recall_at_5"] == 0.0
    assert evaluation["metrics"]["any_hit_at_5"] == 0
    assert evaluation["source_metrics"]["recall_at_5"] == pytest.approx(1.0)
    assert evaluation["source_metrics"]["mrr"] == pytest.approx(0.5)
    assert [unit["rank"] for unit in evaluation["source_units"]] == [1, 2]


def test_report_contains_chunk_and_source_metrics_with_definitions(tmp_path) -> None:
    dataset, source_map = build_inputs(
        tmp_path,
        [valid_record(expected_source_ids=["source-a"])],
        {"source-a": {"source_key": "runbooks/a.md"}},
    )

    def retriever(**kwargs):
        return [make_result("runbooks/a.md")]

    report = run_evaluation(
        dataset=dataset,
        source_map=source_map,
        tenant_id=uuid4(),
        retrievers={"semantic": retriever, "hybrid": retriever},
        retrieval_config={"k": 5},
    )

    assert set(report["aggregate"]["by_rank_unit"]) == {"chunk", "source"}
    assert report["aggregate"]["by_rank_unit"]["chunk"]["semantic"][
        "rank_unit"
    ] == "chunk"
    assert report["aggregate"]["by_rank_unit"]["source"]["semantic"][
        "rank_unit"
    ] == "source"
    assert report["queries"][0]["retrievers"]["semantic"]["source_metrics"][
        "recall_at_5"
    ] == pytest.approx(1.0)
    assert "recall_at_5" in report["metric_definitions"]
    assert "reported only when" in report["metric_definitions"]["ndcg_at_5"].lower()
    assert report["relevance_grading"]["provided"] is False
    definitions = " ".join(report["metric_definitions"].values())
    assert "not an abstention" in definitions.lower()
    comparison = report["queries"][0]["comparison"]
    assert "recall_at_5_delta" in comparison
    assert set(comparison["by_rank_unit"]) == {"chunk", "source"}
    assert "content" not in report["queries"][0]["retrievers"]["semantic"]["results"][0]


def test_corpus_fingerprint_is_order_independent_and_content_free() -> None:
    rows = [
        {
            "source_key": "runbooks/a.md",
            "version_number": 1,
            "chunk_index": 0,
            "content_hash": "a" * 64,
        },
        {
            "source_key": "runbooks/b.md",
            "version_number": 2,
            "chunk_index": 0,
            "content_hash": "b" * 64,
        },
    ]

    fingerprint = compute_corpus_fingerprint(rows)
    reversed_fingerprint = compute_corpus_fingerprint(list(reversed(rows)))

    assert fingerprint["sha256"] == reversed_fingerprint["sha256"]
    assert fingerprint["document_count"] == 2
    assert fingerprint["current_version_count"] == 2
    assert fingerprint["chunk_count"] == 2
    assert "content" not in fingerprint


def test_corpus_fingerprint_rejects_invalid_hash() -> None:
    with pytest.raises(ValueError, match="invalid 'content_hash'"):
        compute_corpus_fingerprint(
            [
                {
                    "source_key": "runbooks/a.md",
                    "version_number": 1,
                    "chunk_index": 0,
                    "content_hash": "short",
                }
            ]
        )


def test_preflight_reports_pinned_version_mismatch_actionably(tmp_path) -> None:
    dataset, source_map = build_inputs(
        tmp_path,
        [valid_record(expected_source_ids=["source-a"])],
        {"source-a": {"source_key": "runbooks/a.md", "version_number": 2}},
    )

    issues = preflight_source_map(
        source_map,
        [
            {
                "source_key": "runbooks/a.md",
                "version_number": 1,
                "is_current": True,
                "status": "active",
                "chunk_count": 3,
            }
        ],
    )

    assert issues
    assert "pins version 2" in issues[0]
    assert "current version of 'runbooks/a.md' is 1" in issues[0]
    assert dataset.dataset_id == "golden_v0"


def test_preflight_reports_missing_document_and_empty_chunks(tmp_path) -> None:
    _, source_map = build_inputs(
        tmp_path,
        [valid_record(expected_source_ids=["source-a", "source-b"])],
        {
            "source-a": {"source_key": "runbooks/a.md"},
            "source-b": {"source_key": "runbooks/b.md"},
        },
    )

    issues = preflight_source_map(
        source_map,
        [
            {
                "source_key": "runbooks/a.md",
                "version_number": 1,
                "is_current": True,
                "status": "active",
                "chunk_count": 0,
            }
        ],
    )

    assert any("runbooks/a.md" in issue and "no chunks" in issue for issue in issues)
    assert any("runbooks/b.md" in issue and "does not exist" in issue for issue in issues)


def test_saturation_warning_fires_when_every_metric_is_perfect(tmp_path) -> None:
    dataset, source_map = build_inputs(
        tmp_path,
        [valid_record(expected_source_ids=["source-a"])],
        {"source-a": {"source_key": "runbooks/a.md"}},
    )

    def retriever(**kwargs):
        return [make_result("runbooks/a.md")]

    report = run_evaluation(
        dataset=dataset,
        source_map=source_map,
        tenant_id=uuid4(),
        retrievers={"semantic": retriever, "hybrid": retriever},
        retrieval_config={"k": 5},
    )

    warning = report["aggregate"]["saturation_warning"]
    assert warning is not None
    assert "does not discriminate" in warning
    assert "production readiness" in warning


def test_compare_reports_marks_missing_fingerprint_informational(tmp_path) -> None:
    dataset, source_map = build_inputs(
        tmp_path,
        [valid_record(expected_source_ids=["source-a"])],
        {"source-a": {"source_key": "runbooks/a.md"}},
    )

    def retriever(**kwargs):
        return [make_result("runbooks/a.md")]

    report = run_evaluation(
        dataset=dataset,
        source_map=source_map,
        tenant_id=uuid4(),
        retrievers={"semantic": retriever, "hybrid": retriever},
        retrieval_config={"k": 5},
        corpus_metadata={"sha256": "c" * 64, "status": "computed"},
    )

    comparison = compare_reports(report, {"dataset": {"id": "golden_v0"}})

    assert comparison["comparable"] is False
    assert comparison["informational_only"] is True
    assert comparison["metric_deltas"] is None
    assert any("fingerprint missing" in reason for reason in comparison["reasons"])


def test_keyword_diagnostics_report_contribution_to_hybrid(tmp_path) -> None:
    dataset, source_map = build_inputs(
        tmp_path,
        [valid_record(expected_source_ids=["source-a"])],
        {"source-a": {"source_key": "runbooks/a.md"}},
    )
    shared = make_result("runbooks/a.md")

    def retriever(**kwargs):
        return [shared]

    report = run_evaluation(
        dataset=dataset,
        source_map=source_map,
        tenant_id=uuid4(),
        retrievers={
            "semantic": retriever,
            "hybrid": retriever,
            "keyword": retriever,
        },
        retrieval_config={"k": 5},
    )

    keyword = report["queries"][0]["diagnostics"]["keyword"]
    assert keyword["provided"] is True
    assert keyword["chunks_contributing_to_hybrid_top_k"] == 1
    assert keyword["chunks_contributing_to_semantic_top_k"] == 1
    assert report["aggregate"]["diagnostics"][
        "queries_with_keyword_contribution"
    ] == 1


def test_source_map_requires_pinned_versions_for_version_selection(tmp_path) -> None:
    dataset_path = write_dataset(
        tmp_path,
        [
            valid_record(
                category="current_version_selection",
                expected_source_ids=["source-a"],
            )
        ],
    )
    dataset = load_golden_dataset(dataset_path)
    source_map_path = write_source_map(
        tmp_path,
        {"source-a": {"source_key": "runbooks/a.md"}},
    )

    with pytest.raises(EvalValidationError, match="without pinned versions"):
        load_source_map(source_map_path, dataset)


def test_profile_validation_rejects_insufficient_categories(tmp_path) -> None:
    dataset = load_golden_dataset(
        write_dataset(tmp_path, [valid_record(category="exact_identifier")])
    )
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(
        json.dumps({"min_category_counts": {"exact_identifier": 2}}),
        encoding="utf-8",
    )

    with pytest.raises(EvalValidationError, match="at least 2 required"):
        validate_dataset_profile(dataset, load_dataset_profile(profile_path))


def test_profile_loader_rejects_unknown_fields(tmp_path) -> None:
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(
        json.dumps({"min_category_count": {"exact_identifier": 1}}),
        encoding="utf-8",
    )

    with pytest.raises(EvalValidationError, match="unknown field"):
        load_dataset_profile(profile_path)


def test_golden_v1_checked_in_dataset_profile_and_map_validate() -> None:
    dataset = load_golden_dataset(REPO_ROOT / "evals" / "datasets" / "golden_v1.jsonl")
    profile = load_dataset_profile(
        REPO_ROOT / "evals" / "datasets" / "golden_v1.profile.json"
    )
    validate_dataset_profile(dataset, profile)
    source_map = load_source_map(
        REPO_ROOT / "evals" / "datasets" / "golden_v1.source_map.json",
        dataset,
    )

    assert dataset.dataset_id == "golden_v1"
    assert len(dataset.answerable_records) == 16
    assert len(dataset.unanswerable_records) == 8
    assert all(
        "injection" not in record.question.lower() for record in dataset.records
    )
    pinned = {
        expected_source_id: target.version_number
        for expected_source_id, target in source_map.sources.items()
        if target.version_number is not None
    }
    assert pinned == {
        "payment-api-runbook-v2": 2,
        "payments-migration-guide-v2": 2,
    }


def test_golden_v0_dataset_and_report_are_preserved() -> None:
    dataset = load_golden_dataset(REPO_ROOT / "evals" / "datasets" / "golden_v0.jsonl")
    assert dataset.sha256 == (
        "85146dcfa0d97480907cccc05b29dfd5958d30ea33f43cc9d110096faab956a2"
    )
    assert len(dataset.records) == 6

    report_path = (
        REPO_ROOT / "evals" / "reports" / "golden_v0_semantic_vs_hybrid.json"
    )
    assert report_path.is_file()
    assert json.loads(report_path.read_text(encoding="utf-8"))["report_version"] == 1


def test_report_paths_are_repository_relative(tmp_path) -> None:
    dataset, source_map = build_inputs(
        tmp_path,
        [valid_record(expected_source_ids=["source-a"])],
        {"source-a": {"source_key": "runbooks/a.md"}},
    )

    def retriever(**kwargs):
        return [make_result("runbooks/a.md")]

    report = run_evaluation(
        dataset=dataset,
        source_map=source_map,
        tenant_id=uuid4(),
        retrievers={"semantic": retriever, "hybrid": retriever},
        retrieval_config={"k": 5},
        corpus_metadata={"sha256": "d" * 64, "status": "computed"},
        repo_root=tmp_path,
    )

    assert report["dataset"]["path"] == "golden_v0.jsonl"
    assert report["source_map"]["path"] == "golden_v0.source_map.json"
    assert not Path(report["dataset"]["path"]).is_absolute()


def test_runner_scores_all_optional_lexical_retrievers(tmp_path) -> None:
    dataset, source_map = build_inputs(
        tmp_path,
        [
            valid_record(
                id="q1",
                category="exact_identifier",
                expected_source_ids=["source-a"],
            )
        ],
        {"source-a": {"source_key": "runbooks/a.md"}},
    )

    def retriever(**kwargs):
        return [make_result("runbooks/a.md")]

    def empty_keyword(**kwargs):
        return []

    def lexical(**kwargs):
        result = make_result("runbooks/a.md")
        result["lexical_score"] = 0.5
        return [result]

    report = run_evaluation(
        dataset=dataset,
        source_map=source_map,
        tenant_id=uuid4(),
        retrievers={
            "semantic": retriever,
            "keyword": empty_keyword,
            "lexical": lexical,
            "hybrid": retriever,
            "hybrid_source": retriever,
        },
        retrieval_config={"k": 5},
    )

    assert report["aggregate"]["retrievers"] == [
        "semantic",
        "keyword",
        "lexical",
        "hybrid",
        "hybrid_source",
    ]
    assert report["aggregate"]["by_retriever"]["keyword"]["source"][
        "recall_at_5"
    ] == 0.0
    assert report["aggregate"]["by_retriever"]["lexical"]["source"][
        "recall_at_5"
    ] == pytest.approx(1.0)
    assert report["aggregate"]["category_metrics"]["exact_identifier"][
        "retrievers"
    ]["lexical"]["source"]["recall_at_5"] == pytest.approx(1.0)
    assert set(report["queries"][0]["retrievers"]) == {
        "semantic",
        "keyword",
        "lexical",
        "hybrid",
        "hybrid_source",
    }
    assert report["queries"][0]["retrievers"]["lexical"]["results"][0][
        "score_field"
    ] == "lexical_score"
    assert report["aggregate"]["saturation_warning"] is None


def test_cli_parser_keeps_lexical_comparison_opt_in() -> None:
    from scripts.run_retrieval_eval import build_parser

    default_args = build_parser().parse_args([])
    comparison_args = build_parser().parse_args(["--include-lexical-comparison"])

    assert default_args.include_lexical_comparison is False
    assert comparison_args.include_lexical_comparison is True


def test_cli_parser_keeps_fusion_experiments_opt_in() -> None:
    from scripts.run_retrieval_eval import build_parser

    default_args = build_parser().parse_args([])
    fusion_args = build_parser().parse_args(["--include-fusion-experiments"])

    assert default_args.include_fusion_experiments is False
    assert fusion_args.include_fusion_experiments is True


def test_run_evaluation_scores_experimental_fusion_retrievers(tmp_path) -> None:
    dataset, source_map = build_inputs(
        tmp_path,
        [
            valid_record(
                category="exact_identifier",
                expected_source_ids=["source-a"],
            )
        ],
        {"source-a": {"source_key": "runbooks/a.md"}},
    )

    def retriever(**kwargs):
        return [make_result("runbooks/a.md")]

    lexical_result = make_result("runbooks/a.md")
    lexical_result["lexical_score"] = 0.5

    def lexical(**kwargs):
        return [dict(lexical_result)]

    def weighted(**kwargs):
        result = dict(lexical_result)
        result["ranking_origin"] = "fused"
        return [result]

    def lexical_primary(**kwargs):
        result = dict(lexical_result)
        result["ranking_origin"] = "lexical"
        return [result]

    report = run_evaluation(
        dataset=dataset,
        source_map=source_map,
        tenant_id=uuid4(),
        retrievers={
            "semantic": retriever,
            "lexical": lexical,
            "hybrid": retriever,
            "hybrid_weighted_semantic": weighted,
            "hybrid_weighted_lexical": weighted,
            "hybrid_lexical_primary": lexical_primary,
        },
        retrieval_config={"k": 5},
    )

    assert report["aggregate"]["retrievers"] == [
        "semantic",
        "lexical",
        "hybrid",
        "hybrid_weighted_semantic",
        "hybrid_weighted_lexical",
        "hybrid_lexical_primary",
    ]
    contributions = report["queries"][0]["diagnostics"]["fusion_contributions"]
    assert set(contributions) == {
        "hybrid",
        "hybrid_weighted_semantic",
        "hybrid_weighted_lexical",
        "hybrid_lexical_primary",
    }
    assert contributions["hybrid_lexical_primary"]["origin_counts"] == {
        "lexical": 1
    }
    assert report["aggregate"]["diagnostics"]["fusion_contributions"][
        "hybrid_lexical_primary"
    ]["queries_with_lexical_only_contribution"] == 1



def test_relevance_grades_must_cover_every_expected_source(tmp_path) -> None:
    path = write_dataset(
        tmp_path,
        [
            valid_record(
                expected_source_ids=["source-a", "source-b"],
                relevance_grades={"source-a": 3},
            )
        ],
    )

    with pytest.raises(
        EvalValidationError,
        match="must grade every expected source; missing: source-b",
    ):
        load_golden_dataset(path)


def test_relevance_grades_reject_unknown_sources_and_bad_values(tmp_path) -> None:
    unknown = write_dataset(
        tmp_path,
        [
            valid_record(
                expected_source_ids=["source-a"],
                relevance_grades={"source-a": 3, "source-b": 2},
            )
        ],
        name="unknown.jsonl",
    )
    with pytest.raises(EvalValidationError, match="unknown source ID 'source-b'"):
        load_golden_dataset(unknown)

    bad_grade = write_dataset(
        tmp_path,
        [
            valid_record(
                expected_source_ids=["source-a"],
                relevance_grades={"source-a": True},
            )
        ],
        name="bad_grade.jsonl",
    )
    with pytest.raises(EvalValidationError, match="integer between 1 and 3"):
        load_golden_dataset(bad_grade)


def test_graded_relevance_produces_source_ndcg_only(tmp_path) -> None:
    dataset, source_map = build_inputs(
        tmp_path,
        [
            valid_record(
                expected_source_ids=["source-a", "source-b"],
                relevance_grades={"source-a": 3, "source-b": 2},
            )
        ],
        {
            "source-a": {"source_key": "runbooks/a.md"},
            "source-b": {"source_key": "runbooks/b.md"},
        },
    )
    record = dataset.answerable_records[0]
    results = [make_result("runbooks/a.md"), make_result("runbooks/b.md")]

    ideal = evaluate_query_result(
        record,
        results,
        source_map,
        score_field="similarity",
    )
    reversed_order = evaluate_query_result(
        record,
        list(reversed(results)),
        source_map,
        score_field="similarity",
    )

    assert ideal["source_metrics"]["ndcg_at_5"] == pytest.approx(1.0)
    assert ideal["source_metrics"]["ndcg_status"] == "computed_graded_relevance"
    assert ideal["metrics"]["ndcg_at_5"] is None
    assert reversed_order["source_metrics"]["ndcg_at_5"] < 1.0


def test_lexical_diagnostics_report_candidate_counts(tmp_path) -> None:
    dataset, source_map = build_inputs(
        tmp_path,
        [valid_record(expected_source_ids=["source-a"])],
        {"source-a": {"source_key": "runbooks/a.md"}},
    )

    def semantic(**_kwargs):
        return [make_result("runbooks/a.md")]

    def hybrid(**_kwargs):
        return [make_result("runbooks/a.md")]

    def keyword(**_kwargs):
        return []

    def lexical(**_kwargs):
        return [make_result("runbooks/a.md"), make_result("runbooks/b.md")]

    report = run_evaluation(
        dataset=dataset,
        source_map=source_map,
        tenant_id=uuid4(),
        retrievers={
            "semantic": semantic,
            "hybrid": hybrid,
            "keyword": keyword,
            "lexical": lexical,
        },
        retrieval_config={"k": 5},
    )

    diagnostics = report["queries"][0]["diagnostics"]
    assert diagnostics["keyword"]["result_count"] == 0
    assert diagnostics["lexical"]["result_count"] == 2
    assert report["aggregate"]["diagnostics"][
        "queries_with_lexical_candidates"
    ] == 1
    assert report["aggregate"]["diagnostics"][
        "queries_with_keyword_candidates"
    ] == 0


def test_golden_v2_dataset_is_valid_and_independent() -> None:
    dataset = load_golden_dataset(REPO_ROOT / "evals/datasets/golden_v2.jsonl")
    profile = load_dataset_profile(REPO_ROOT / "evals/datasets/golden_v2.profile.json")
    validate_dataset_profile(dataset, profile)
    load_source_map(
        REPO_ROOT / "evals/datasets/golden_v2.source_map.json",
        dataset,
    )

    assert len(dataset.records) == 35
    assert len(dataset.answerable_records) == 29
    assert len(dataset.unanswerable_records) == 6
    assert all(
        record.relevance_grades for record in dataset.answerable_records
    )
    assert any(
        len(record.expected_source_ids) == 3
        for record in dataset.answerable_records
    )

    golden_v1 = load_golden_dataset(REPO_ROOT / "evals/datasets/golden_v1.jsonl")
    v1_questions = {
        " ".join(record.question.lower().split())
        for record in golden_v1.records
    }
    v2_questions = {
        " ".join(record.question.lower().split())
        for record in dataset.records
    }
    assert not v1_questions & v2_questions


def test_golden_v3_dataset_is_valid_and_independent() -> None:
    dataset = load_golden_dataset(REPO_ROOT / "evals/datasets/golden_v3.jsonl")
    profile = load_dataset_profile(REPO_ROOT / "evals/datasets/golden_v3.profile.json")
    validate_dataset_profile(dataset, profile)
    source_map = load_source_map(
        REPO_ROOT / "evals/datasets/golden_v3.source_map.json",
        dataset,
    )

    assert len(dataset.records) == 62
    assert len(dataset.answerable_records) == 50
    assert len(dataset.unanswerable_records) == 12
    assert len(source_map.sources) == 10
    assert all(
        record.relevance_grades for record in dataset.answerable_records
    )
    assert all(
        all(grade in (2, 3) for grade in record.relevance_grades.values())
        for record in dataset.answerable_records
    )
    assert any(
        len(record.expected_source_ids) == 3
        for record in dataset.answerable_records
    )

    categories = {record.category for record in dataset.answerable_records}
    assert {
        "exact_identifier",
        "status_vs_identifier",
        "api_version_url",
        "paraphrase_no_rare_term",
        "multi_source",
        "shared_identifier",
        "bare_numeric_distractor",
        "long_noisy",
        "multi_chunk_source",
        "distractor_overlap",
    } <= categories

    prior_questions = {
        " ".join(record.question.lower().split())
        for name in ("golden_v1.jsonl", "golden_v2.jsonl")
        for record in load_golden_dataset(
            REPO_ROOT / "evals/datasets" / name
        ).records
    }
    v3_questions = {
        " ".join(record.question.lower().split())
        for record in dataset.records
    }
    assert not prior_questions & v3_questions


def test_cli_parser_keeps_abstention_analysis_opt_in() -> None:
    from scripts.run_retrieval_eval import build_parser

    default_args = build_parser().parse_args([])
    abstention_args = build_parser().parse_args(["--include-abstention-analysis"])

    assert default_args.include_abstention_analysis is False
    assert abstention_args.include_abstention_analysis is True


def test_cli_hybrid_keyword_strategy_defaults_to_and() -> None:
    from scripts.run_retrieval_eval import build_parser

    default_args = build_parser().parse_args([])
    comparison_args = build_parser().parse_args(
        ["--hybrid-keyword-strategy", "or_idf"]
    )

    assert default_args.hybrid_keyword_strategy == "and"
    assert comparison_args.hybrid_keyword_strategy == "or_idf"

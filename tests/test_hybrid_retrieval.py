from uuid import UUID, uuid4

import pytest

from app.services import retrieval
from app.services.retrieval import RRF_K, fuse_ranked_results, hybrid_search


def make_result(
    chunk_id: UUID,
    *,
    title: str = "Payment API Runbook",
    similarity: float | None = None,
    keyword_score: float | None = None,
) -> dict:
    result = {
        "chunk_id": chunk_id,
        "document_id": uuid4(),
        "version_id": uuid4(),
        "title": title,
        "source_key": "runbooks/payment-api.md",
        "version_number": 1,
        "authority_level": 80,
        "chunk_index": 0,
        "content": "Check upstream timeouts and database pool saturation.",
    }
    if similarity is not None:
        result["cosine_distance"] = 1.0 - similarity
        result["similarity"] = similarity
    if keyword_score is not None:
        result["keyword_score"] = keyword_score
    return result


def test_fuse_ranked_results_returns_empty_list_for_empty_inputs() -> None:
    assert fuse_ranked_results([], [], limit=5) == []


@pytest.mark.parametrize("limit", [0, -1, 21])
def test_fuse_ranked_results_rejects_invalid_limit(limit: int) -> None:
    with pytest.raises(ValueError, match="limit must be between 1 and 20"):
        fuse_ranked_results([], [], limit=limit)


@pytest.mark.parametrize("rrf_k", [0, -1])
def test_fuse_ranked_results_rejects_invalid_rrf_k(rrf_k: int) -> None:
    with pytest.raises(ValueError, match="rrf_k must be at least 1"):
        fuse_ranked_results([], [], limit=5, rrf_k=rrf_k)


def test_fuse_ranked_results_deduplicates_chunk_present_in_both_lists() -> None:
    chunk_id = uuid4()

    fused = fuse_ranked_results(
        [make_result(chunk_id, similarity=0.8)],
        [make_result(chunk_id, keyword_score=0.4)],
        limit=5,
    )

    assert len(fused) == 1
    assert fused[0]["chunk_id"] == chunk_id
    assert fused[0]["semantic_rank"] == 1
    assert fused[0]["keyword_rank"] == 1
    assert fused[0]["semantic_similarity"] == pytest.approx(0.8)
    assert fused[0]["keyword_score"] == pytest.approx(0.4)
    assert fused[0]["hybrid_score"] == pytest.approx(2.0 / (RRF_K + 1))


def test_fuse_ranked_results_rewards_agreement_across_lists() -> None:
    semantic_only = uuid4()
    shared = uuid4()
    keyword_only = uuid4()

    fused = fuse_ranked_results(
        [make_result(semantic_only), make_result(shared)],
        [make_result(keyword_only), make_result(shared)],
        limit=5,
    )

    assert fused[0]["chunk_id"] == shared
    assert fused[0]["hybrid_score"] == pytest.approx(2.0 / (RRF_K + 2))
    assert fused[0]["semantic_rank"] == 2
    assert fused[0]["keyword_rank"] == 2


def test_fuse_ranked_results_includes_results_missing_from_one_list() -> None:
    semantic_only = uuid4()
    keyword_only = uuid4()

    fused = fuse_ranked_results(
        [make_result(semantic_only, similarity=0.9)],
        [make_result(keyword_only, keyword_score=0.7)],
        limit=5,
    )

    assert {item["chunk_id"] for item in fused} == {semantic_only, keyword_only}

    semantic_item = next(item for item in fused if item["chunk_id"] == semantic_only)
    assert semantic_item["semantic_rank"] == 1
    assert semantic_item["keyword_rank"] is None
    assert semantic_item["semantic_similarity"] == pytest.approx(0.9)
    assert semantic_item["keyword_score"] is None

    keyword_item = next(item for item in fused if item["chunk_id"] == keyword_only)
    assert keyword_item["semantic_rank"] is None
    assert keyword_item["keyword_rank"] == 1
    assert keyword_item["semantic_similarity"] is None
    assert keyword_item["keyword_score"] == pytest.approx(0.7)


def test_fuse_ranked_results_truncates_after_sorting() -> None:
    first, second, third = uuid4(), uuid4(), uuid4()

    fused = fuse_ranked_results(
        [make_result(first), make_result(second), make_result(third)],
        [],
        limit=2,
    )

    assert [item["chunk_id"] for item in fused] == [first, second]


def test_fuse_ranked_results_is_deterministic_when_scores_and_best_rank_tie() -> None:
    later_id = UUID("00000000-0000-0000-0000-0000000000ff")
    earlier_id = UUID("00000000-0000-0000-0000-000000000001")

    fused = fuse_ranked_results(
        [make_result(later_id)],
        [make_result(earlier_id)],
        limit=5,
    )

    assert [item["chunk_id"] for item in fused] == [earlier_id, later_id]


def test_fuse_ranked_results_ignores_duplicate_chunk_within_one_list() -> None:
    repeated = uuid4()
    next_chunk = uuid4()

    fused = fuse_ranked_results(
        [make_result(repeated), make_result(repeated), make_result(next_chunk)],
        [],
        limit=5,
    )

    assert [item["chunk_id"] for item in fused] == [repeated, next_chunk]
    assert fused[0]["hybrid_score"] == pytest.approx(1.0 / (RRF_K + 1))
    assert fused[1]["hybrid_score"] == pytest.approx(1.0 / (RRF_K + 2))


def test_fuse_ranked_results_preserves_semantic_metadata_for_duplicate() -> None:
    chunk_id = uuid4()
    semantic_result = make_result(chunk_id, title="Semantic title", similarity=0.8)
    keyword_result = make_result(chunk_id, title="Keyword title", keyword_score=0.4)

    fused = fuse_ranked_results([semantic_result], [keyword_result], limit=5)

    assert fused[0]["title"] == "Semantic title"
    assert fused[0]["document_id"] == semantic_result["document_id"]
    assert fused[0]["version_id"] == semantic_result["version_id"]
    assert fused[0]["content"] == semantic_result["content"]


def patch_retrievers(
    monkeypatch,
    *,
    semantic: list[dict] | None = None,
    keyword: list[dict] | None = None,
    captured: dict | None = None,
) -> None:
    def fake_semantic_search(**kwargs):
        if captured is not None:
            captured["semantic"] = kwargs
        return [] if semantic is None else semantic

    def fake_keyword_search(**kwargs):
        if captured is not None:
            captured["keyword"] = kwargs
        return [] if keyword is None else keyword

    monkeypatch.setattr(retrieval, "semantic_search", fake_semantic_search)
    monkeypatch.setattr(retrieval, "keyword_search", fake_keyword_search)


def test_hybrid_search_returns_empty_list_when_retrievers_return_nothing(monkeypatch) -> None:
    patch_retrievers(monkeypatch)

    assert hybrid_search(tenant_id=uuid4(), query="payment", limit=5) == []


def test_hybrid_search_rejects_blank_query() -> None:
    with pytest.raises(ValueError, match="query must not be empty"):
        hybrid_search(tenant_id=uuid4(), query="   ")


@pytest.mark.parametrize("limit", [0, -1, 21])
def test_hybrid_search_rejects_invalid_limit(limit: int) -> None:
    with pytest.raises(ValueError, match="limit must be between 1 and 20"):
        hybrid_search(tenant_id=uuid4(), query="payment", limit=limit)


@pytest.mark.parametrize("rrf_k", [0, -1])
def test_hybrid_search_rejects_invalid_rrf_k(rrf_k: int) -> None:
    with pytest.raises(ValueError, match="rrf_k must be at least 1"):
        hybrid_search(tenant_id=uuid4(), query="payment", rrf_k=rrf_k)


@pytest.mark.parametrize(
    ("limit", "expected_candidate_limit"),
    [(1, 3), (5, 15), (7, 20), (20, 20)],
)
def test_hybrid_search_passes_tenant_query_and_candidate_limit(
    monkeypatch,
    limit: int,
    expected_candidate_limit: int,
) -> None:
    tenant_id = uuid4()
    captured: dict = {}
    patch_retrievers(monkeypatch, captured=captured)

    hybrid_search(tenant_id=tenant_id, query="payment", limit=limit)

    for retriever in ("semantic", "keyword"):
        assert captured[retriever]["tenant_id"] == tenant_id
        assert captured[retriever]["query"] == "payment"
        assert captured[retriever]["limit"] == expected_candidate_limit


def test_hybrid_search_fuses_duplicate_chunk_once(monkeypatch) -> None:
    chunk_id = uuid4()
    patch_retrievers(
        monkeypatch,
        semantic=[make_result(chunk_id, similarity=0.8)],
        keyword=[make_result(chunk_id, keyword_score=0.4)],
    )

    results = hybrid_search(tenant_id=uuid4(), query="payment", limit=5)

    assert len(results) == 1
    assert results[0]["chunk_id"] == chunk_id
    assert results[0]["hybrid_score"] == pytest.approx(2.0 / (RRF_K + 1))
    assert results[0]["semantic_rank"] == 1
    assert results[0]["keyword_rank"] == 1


def test_hybrid_search_rewards_agreement_across_lists(monkeypatch) -> None:
    semantic_only = uuid4()
    shared = uuid4()
    keyword_only = uuid4()
    patch_retrievers(
        monkeypatch,
        semantic=[make_result(semantic_only), make_result(shared)],
        keyword=[make_result(keyword_only), make_result(shared)],
    )

    results = hybrid_search(tenant_id=uuid4(), query="payment", limit=5)

    assert results[0]["chunk_id"] == shared
    assert results[0]["hybrid_score"] == pytest.approx(2.0 / (RRF_K + 2))


def test_hybrid_search_includes_results_missing_from_one_list(monkeypatch) -> None:
    semantic_only = uuid4()
    keyword_only = uuid4()
    patch_retrievers(
        monkeypatch,
        semantic=[make_result(semantic_only, similarity=0.9)],
        keyword=[make_result(keyword_only, keyword_score=0.7)],
    )

    results = hybrid_search(tenant_id=uuid4(), query="payment", limit=5)

    assert {item["chunk_id"] for item in results} == {semantic_only, keyword_only}


def test_hybrid_search_truncates_to_requested_limit_after_sorting(monkeypatch) -> None:
    chunks = [uuid4() for _ in range(4)]
    patch_retrievers(
        monkeypatch,
        semantic=[make_result(chunk_id) for chunk_id in chunks],
    )

    results = hybrid_search(tenant_id=uuid4(), query="payment", limit=2)

    assert len(results) == 2
    assert [item["chunk_id"] for item in results] == chunks[:2]


def test_hybrid_search_is_deterministic_when_scores_and_best_rank_tie(monkeypatch) -> None:
    later_id = UUID("00000000-0000-0000-0000-0000000000ff")
    earlier_id = UUID("00000000-0000-0000-0000-000000000001")
    patch_retrievers(
        monkeypatch,
        semantic=[make_result(later_id)],
        keyword=[make_result(earlier_id)],
    )

    results = hybrid_search(tenant_id=uuid4(), query="payment", limit=5)

    assert [item["chunk_id"] for item in results] == [earlier_id, later_id]


def test_hybrid_search_preserves_chunk_metadata(monkeypatch) -> None:
    chunk_id = uuid4()
    semantic_result = make_result(chunk_id, title="Semantic title", similarity=0.8)
    patch_retrievers(monkeypatch, semantic=[semantic_result])

    results = hybrid_search(tenant_id=uuid4(), query="payment", limit=5)

    assert results[0]["document_id"] == semantic_result["document_id"]
    assert results[0]["version_id"] == semantic_result["version_id"]
    assert results[0]["title"] == "Semantic title"
    assert results[0]["source_key"] == semantic_result["source_key"]
    assert results[0]["version_number"] == semantic_result["version_number"]
    assert results[0]["authority_level"] == semantic_result["authority_level"]
    assert results[0]["chunk_index"] == semantic_result["chunk_index"]
    assert results[0]["content"] == semantic_result["content"]

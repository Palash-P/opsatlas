from uuid import UUID, uuid4

import pytest

from app.services import retrieval
from app.services.retrieval import (
    FUSION_STRATEGIES,
    FUSION_WEIGHT_PROFILES,
    IDENTIFIER_WEIGHT,
    KEYWORD_STRATEGIES,
    RRF_K,
    detect_version_request,
    fuse_lexical_primary_results,
    fuse_ranked_results,
    fuse_source_ranked_results,
    fuse_weighted_ranked_results,
    hybrid_search,
    lexical_search,
    prepare_lexical_query,
    resolve_version_request,
)


def make_result(
    source_key: str,
    *,
    version_number: int = 1,
    chunk_index: int = 0,
    similarity: float | None = None,
    keyword_score: float | None = None,
) -> dict:
    result = {
        "chunk_id": uuid4(),
        "document_id": uuid4(),
        "version_id": uuid4(),
        "title": source_key,
        "source_key": source_key,
        "version_number": version_number,
        "authority_level": 80,
        "chunk_index": chunk_index,
        "content": f"content for {source_key} chunk {chunk_index}",
    }
    if similarity is not None:
        result["similarity"] = similarity
    if keyword_score is not None:
        result["keyword_score"] = keyword_score
    return result


class FakeResult:
    def __init__(self, rows: list[dict] | None = None) -> None:
        self._rows = rows or []

    def mappings(self):
        return self

    def all(self):
        return self._rows


class FakeConnection:
    def __init__(self, captured: dict, rows: list[dict] | None = None) -> None:
        self._captured = captured
        self._rows = rows

    def execute(self, statement, params):
        self._captured["sql"] = str(statement)
        self._captured["params"] = params
        return FakeResult(self._rows)


class FakeEngine:
    def __init__(self, captured: dict, rows: list[dict] | None = None) -> None:
        self._captured = captured
        self._rows = rows

    def connect(self):
        captured = self._captured
        rows = self._rows

        class Context:
            def __enter__(self):
                return FakeConnection(captured, rows)

            def __exit__(self, *args):
                return False

        return Context()


@pytest.mark.parametrize(
    ("query", "kind", "value"),
    [
        ("What checks matter for PAY-503?", "identifier", "PAY-503"),
        ("What does PAY-409 mean?", "identifier", "PAY-409"),
        ("What does PAY-400 mean?", "identifier", "PAY-400"),
        ("Payments return HTTP 503.", "http_status", "503"),
        ("Is the v1 endpoint deprecated?", "api_version", "v1"),
        ("Should new clients use v2?", "api_version", "v2"),
    ],
)
def test_prepare_lexical_query_recognizes_technical_identifiers(
    query: str,
    kind: str,
    value: str,
) -> None:
    prepared = prepare_lexical_query(query)

    assert [(term.kind, term.value) for term in prepared.identifiers] == [
        (kind, value)
    ]
    assert value.lower() not in prepared.terms_query.lower()


def test_prepare_lexical_query_keeps_http_status_separate_from_pay_identifier() -> None:
    pay = prepare_lexical_query("PAY-503")
    http = prepare_lexical_query("HTTP 503")

    assert [(term.kind, term.value) for term in pay.identifiers] == [
        ("identifier", "PAY-503")
    ]
    assert [(term.kind, term.value) for term in http.identifiers] == [
        ("http_status", "503")
    ]
    assert pay.identifiers[0].sql_pattern != http.identifiers[0].sql_pattern
    assert "[^A-Za-z0-9-]" in http.identifiers[0].sql_pattern


def test_prepare_lexical_query_does_not_treat_generic_text_as_identifier() -> None:
    for query in (
        "What does the error-code runbook say?",
        "retrying non-idempotent payment operations",
        "aggregate connection budget",
        "error code",
    ):
        prepared = prepare_lexical_query(query)
        assert prepared.identifiers == ()
        assert prepared.terms_query == query


def test_prepare_lexical_query_leaves_punctuation_without_terms() -> None:
    prepared = prepare_lexical_query("???")

    assert prepared.terms_query == "???"
    assert prepared.identifiers == ()


def test_lexical_search_rejects_blank_query() -> None:
    with pytest.raises(ValueError, match="query must not be empty"):
        lexical_search(tenant_id=uuid4(), query="   ")


@pytest.mark.parametrize("limit", [0, -1, 21])
def test_lexical_search_rejects_invalid_limit(limit: int) -> None:
    with pytest.raises(ValueError, match="limit must be between 1 and 20"):
        lexical_search(tenant_id=uuid4(), query="PAY-503", limit=limit)


def test_lexical_search_returns_empty_for_punctuation_only_without_querying(
    monkeypatch,
) -> None:
    def fail_if_called():
        raise AssertionError("database must not be queried for punctuation-only input")

    monkeypatch.setattr(retrieval, "get_engine", fail_if_called)

    assert lexical_search(tenant_id=uuid4(), query="???", limit=5) == []


def test_lexical_search_uses_bound_parameters_and_tenant_filters(monkeypatch) -> None:
    tenant_id = uuid4()
    captured: dict = {}
    monkeypatch.setattr(
        retrieval,
        "get_engine",
        lambda: FakeEngine(captured),
    )

    rows = lexical_search(
        tenant_id=tenant_id,
        query="What checks matter for PAY-503?",
        limit=5,
    )

    assert rows == []
    sql = " ".join(captured["sql"].lower().split())
    assert "c.tenant_id = :tenant_id" in sql
    assert "d.tenant_id = :tenant_id" in sql
    assert "d.status = 'active'" in sql
    assert "dv.is_current = true" in sql
    assert "cast(:identifier_values as text[])" in sql
    assert "cast(:identifier_patterns as text[])" in sql
    assert ":lexical_query" in sql
    assert "pay-503" not in sql
    assert "count(distinct document_id)" in sql
    assert "term_coverage_score" in sql
    assert "ts_rank_cd(" in sql
    assert "order by" in sql
    assert "lexical_score desc" in sql
    assert "matched_term_count desc" in sql
    assert "rank_cd desc" in sql
    assert "chunk_id" in sql.split("order by", 1)[1]

    params = captured["params"]
    assert params["tenant_id"] == tenant_id
    assert params["result_limit"] == 5
    assert params["identifier_values"] == ["PAY-503"]
    assert params["identifier_patterns"][0].startswith("(^|")


def test_lexical_search_sends_cleaned_query_without_identifier(monkeypatch) -> None:
    captured: dict = {}
    monkeypatch.setattr(
        retrieval,
        "get_engine",
        lambda: FakeEngine(captured),
    )

    lexical_search(
        tenant_id=uuid4(),
        query="Payments return HTTP 503 after deploy",
        limit=5,
    )

    assert "503" not in captured["params"]["lexical_query"]
    assert captured["params"]["identifier_values"] == ["503"]


@pytest.mark.parametrize("identifier_weight", [0, -1.0, float("inf"), float("nan")])
def test_lexical_search_rejects_invalid_identifier_weight(
    identifier_weight: float,
) -> None:
    with pytest.raises(ValueError, match="identifier_weight"):
        lexical_search(
            tenant_id=uuid4(),
            query="PAY-503",
            identifier_weight=identifier_weight,
        )


def test_lexical_search_passes_identifier_weight_to_the_query(monkeypatch) -> None:
    captured: dict = {}
    monkeypatch.setattr(
        retrieval,
        "get_engine",
        lambda: FakeEngine(captured),
    )

    lexical_search(
        tenant_id=uuid4(),
        query="What checks matter for PAY-503?",
        identifier_weight=3.5,
    )

    assert captured["params"]["identifier_weight"] == pytest.approx(3.5)
    assert "cast(:identifier_weight as double precision)" in " ".join(
        captured["sql"].lower().split()
    )


def test_identifier_weight_default_is_explicit() -> None:
    assert IDENTIFIER_WEIGHT == 2.0


def test_would_pay503_identifier_mass_exceed_generic_rare_terms() -> None:
    """The correction must keep an exact identifier competitive with rare terms.

    On the synthetic corpus, the shared identifier IDF is about 2.0116 and the
    best coincidental generic term IDF is about 2.7047. The configured weight
    must be large enough to close that gap without changing matched-term counts.
    """

    assert IDENTIFIER_WEIGHT * 2.0116 > 2.7047


def test_hybrid_strategy_constants_are_explicit() -> None:
    assert KEYWORD_STRATEGIES == ("and", "or_idf")
    assert FUSION_STRATEGIES == ("chunk", "source", "weighted", "lexical_primary")


def test_fusion_weight_profiles_are_explicit() -> None:
    assert FUSION_WEIGHT_PROFILES == {
        "equal": {"semantic": 1.0, "lexical": 1.0},
        "semantic_heavy": {"semantic": 2.0, "lexical": 1.0},
        "lexical_heavy": {"semantic": 1.0, "lexical": 2.0},
    }


def test_hybrid_search_rejects_unknown_keyword_strategy() -> None:
    with pytest.raises(ValueError, match="keyword_strategy must be one of"):
        hybrid_search(
            tenant_id=uuid4(),
            query="payment",
            keyword_strategy="unknown",
        )


def test_hybrid_search_rejects_unknown_fusion_strategy() -> None:
    with pytest.raises(ValueError, match="fusion_strategy must be one of"):
        hybrid_search(
            tenant_id=uuid4(),
            query="payment",
            fusion_strategy="unknown",
        )


def test_hybrid_search_defaults_to_and_keyword_and_chunk_fusion(monkeypatch) -> None:
    captured: dict = {}
    shared = make_result("runbooks/a.md", similarity=0.9)

    def fake_semantic_search(**kwargs):
        captured["semantic"] = kwargs
        return [shared]

    def fake_keyword_search(**kwargs):
        captured["keyword"] = kwargs
        return []

    def fake_lexical_search(**kwargs):
        raise AssertionError("OR lexical strategy must not run by default")

    monkeypatch.setattr(retrieval, "semantic_search", fake_semantic_search)
    monkeypatch.setattr(retrieval, "keyword_search", fake_keyword_search)
    monkeypatch.setattr(retrieval, "lexical_search", fake_lexical_search)

    results = hybrid_search(tenant_id=uuid4(), query="payment", limit=5)

    assert [result["chunk_id"] for result in results] == [shared["chunk_id"]]
    assert captured["keyword"]["query"] == "payment"
    assert captured["semantic"]["limit"] == 15


def test_hybrid_search_can_select_or_keyword_and_source_fusion(monkeypatch) -> None:
    captured: dict = {}
    semantic_result = make_result("runbooks/a.md", similarity=0.9)
    lexical_result = make_result("runbooks/b.md")
    lexical_result["lexical_score"] = 0.8

    def fake_semantic_search(**kwargs):
        captured["semantic"] = kwargs
        return [semantic_result]

    def fake_keyword_search(**kwargs):
        raise AssertionError("AND keyword strategy must not run when OR is selected")

    def fake_lexical_search(**kwargs):
        captured["lexical"] = kwargs
        return [lexical_result]

    monkeypatch.setattr(retrieval, "semantic_search", fake_semantic_search)
    monkeypatch.setattr(retrieval, "keyword_search", fake_keyword_search)
    monkeypatch.setattr(retrieval, "lexical_search", fake_lexical_search)

    results = hybrid_search(
        tenant_id=uuid4(),
        query="payment",
        limit=5,
        keyword_strategy="or_idf",
        fusion_strategy="source",
    )

    assert captured["lexical"]["query"] == "payment"
    assert {result["source_key"] for result in results} == {
        "runbooks/a.md",
        "runbooks/b.md",
    }
    assert all("semantic_source_rank" in result for result in results)
    assert semantic_result["source_key"] == "runbooks/a.md"
    lexical_item = next(
        result for result in results if result["source_key"] == "runbooks/b.md"
    )
    assert lexical_item["keyword_score"] == pytest.approx(0.8)


def test_source_aware_fusion_prevents_duplicate_chunks_from_crowding() -> None:
    a_chunks = [
        make_result("runbooks/a.md", chunk_index=index) for index in range(2)
    ]
    b_chunk = make_result("runbooks/b.md", chunk_index=0)

    semantic = [*a_chunks, b_chunk]

    chunk_level = fuse_ranked_results(semantic, [], limit=2)
    source_level = fuse_source_ranked_results(semantic, [], limit=2)

    assert [item["source_key"] for item in chunk_level] == [
        "runbooks/a.md",
        "runbooks/a.md",
    ]
    assert [item["source_key"] for item in source_level] == [
        "runbooks/a.md",
        "runbooks/b.md",
    ]


def test_source_aware_fusion_keeps_best_chunk_rank_and_counts() -> None:
    first = make_result("runbooks/a.md", chunk_index=0)
    second = make_result("runbooks/a.md", chunk_index=1)
    other = make_result("runbooks/b.md", chunk_index=0)

    fused = fuse_source_ranked_results(
        [first, second, other],
        [],
        limit=5,
    )

    assert [item["source_key"] for item in fused] == [
        "runbooks/a.md",
        "runbooks/b.md",
    ]
    assert fused[0]["semantic_source_rank"] == 1
    assert fused[0]["semantic_rank"] == 1
    assert fused[0]["semantic_source_chunk_count"] == 2
    assert fused[0]["chunk_id"] == first["chunk_id"]
    assert fused[0]["hybrid_score"] == pytest.approx(1.0 / (RRF_K + 1))


def test_source_aware_fusion_is_deterministic_when_source_scores_tie() -> None:
    later = make_result("runbooks/z.md")
    earlier = make_result("runbooks/a.md")

    first = fuse_source_ranked_results(
        [later],
        [earlier],
        limit=5,
    )
    second = fuse_source_ranked_results(
        [later],
        [earlier],
        limit=5,
    )

    assert [item["source_key"] for item in first] == ["runbooks/a.md", "runbooks/z.md"]
    assert [item["source_key"] for item in second] == [
        "runbooks/a.md",
        "runbooks/z.md",
    ]


@pytest.mark.parametrize("limit", [0, -1, 21])
def test_source_aware_fusion_rejects_invalid_limit(limit: int) -> None:
    with pytest.raises(ValueError, match="limit must be between 1 and 20"):
        fuse_source_ranked_results([], [], limit=limit)


@pytest.mark.parametrize("rrf_k", [0, -1])
def test_source_aware_fusion_rejects_invalid_rrf_k(rrf_k: int) -> None:
    with pytest.raises(ValueError, match="rrf_k must be at least 1"):
        fuse_source_ranked_results([], [], limit=5, rrf_k=rrf_k)


def test_source_aware_fusion_deduplicates_source_within_one_list() -> None:
    source = "runbooks/a.md"
    results = [make_result(source, chunk_index=index) for index in range(3)]

    fused = fuse_source_ranked_results(results, [], limit=5)

    assert len(fused) == 1
    assert fused[0]["hybrid_score"] == pytest.approx(1.0 / (RRF_K + 1))


def test_lexical_search_result_contract_is_stable_when_rows_returned(monkeypatch) -> None:
    chunk_id = uuid4()
    rows = [
        {
            "chunk_id": chunk_id,
            "document_id": uuid4(),
            "version_id": uuid4(),
            "title": "Payment errors",
            "source_key": "runbooks/payment-errors.md",
            "version_number": 1,
            "authority_level": 80,
            "chunk_index": 0,
            "content": "PAY-503 means upstream timeout.",
            "term_coverage_score": 4.5,
            "identifier_score": 2.0,
            "lexical_score": 6.5,
            "matched_term_count": 3,
            "rank_cd": 0.1,
        }
    ]
    captured: dict = {}
    monkeypatch.setattr(
        retrieval,
        "get_engine",
        lambda: FakeEngine(captured, rows),
    )

    results = lexical_search(
        tenant_id=UUID("00000000-0000-0000-0000-000000000001"),
        query="PAY-503",
        limit=5,
    )

    assert results == rows
    assert "content" in results[0]


def test_weighted_fusion_semantic_heavy_favours_the_semantic_ranking() -> None:
    semantic = make_result("runbooks/semantic.md", similarity=0.9)
    lexical = make_result("runbooks/lexical.md")
    lexical["lexical_score"] = 10.0

    fused = fuse_weighted_ranked_results(
        [semantic],
        [lexical],
        limit=5,
        semantic_weight=2.0,
        lexical_weight=1.0,
    )

    assert [item["source_key"] for item in fused] == [
        "runbooks/semantic.md",
        "runbooks/lexical.md",
    ]
    assert fused[0]["semantic_weight"] == pytest.approx(2.0)
    assert fused[0]["lexical_weight"] == pytest.approx(1.0)
    assert fused[0]["hybrid_score"] == pytest.approx(2.0 / (RRF_K + 1))


def test_weighted_fusion_lexical_heavy_favours_the_lexical_ranking() -> None:
    semantic = make_result("runbooks/semantic.md", similarity=0.9)
    lexical = make_result("runbooks/lexical.md")
    lexical["lexical_score"] = 10.0

    fused = fuse_weighted_ranked_results(
        [semantic],
        [lexical],
        limit=5,
        semantic_weight=1.0,
        lexical_weight=2.0,
    )

    assert [item["source_key"] for item in fused] == [
        "runbooks/lexical.md",
        "runbooks/semantic.md",
    ]
    assert fused[0]["keyword_score"] == pytest.approx(10.0)
    assert fused[0]["hybrid_score"] == pytest.approx(2.0 / (RRF_K + 1))


def test_weighted_fusion_accumulates_both_contributions_for_shared_chunks() -> None:
    shared = make_result("runbooks/shared.md", similarity=0.9)
    shared["lexical_score"] = 10.0

    fused = fuse_weighted_ranked_results(
        [shared],
        [shared],
        limit=5,
        semantic_weight=2.0,
        lexical_weight=3.0,
    )

    assert len(fused) == 1
    assert fused[0]["hybrid_score"] == pytest.approx(
        (2.0 + 3.0) / (RRF_K + 1)
    )
    assert fused[0]["semantic_rank"] == 1
    assert fused[0]["keyword_rank"] == 1


def test_weighted_fusion_is_deterministic_when_scores_tie() -> None:
    later = make_result("runbooks/z.md", similarity=0.9)
    earlier = make_result("runbooks/a.md")
    earlier["lexical_score"] = 1.0

    first = fuse_weighted_ranked_results(
        [later],
        [earlier],
        limit=5,
        semantic_weight=1.0,
        lexical_weight=1.0,
    )
    second = fuse_weighted_ranked_results(
        [later],
        [earlier],
        limit=5,
        semantic_weight=1.0,
        lexical_weight=1.0,
    )

    assert [item["chunk_id"] for item in first] == [
        item["chunk_id"] for item in second
    ]
    assert {item["source_key"] for item in first} == {
        "runbooks/a.md",
        "runbooks/z.md",
    }


@pytest.mark.parametrize(
    ("semantic_weight", "lexical_weight"),
    [(0.0, 1.0), (1.0, 0.0), (-1.0, 1.0), (1.0, -1.0)],
)
def test_weighted_fusion_rejects_non_positive_weights(
    semantic_weight: float,
    lexical_weight: float,
) -> None:
    with pytest.raises(ValueError, match="must be greater than 0"):
        fuse_weighted_ranked_results(
            [],
            [],
            limit=5,
            semantic_weight=semantic_weight,
            lexical_weight=lexical_weight,
        )


def test_lexical_primary_returns_lexical_order_then_semantic_fallback() -> None:
    first = make_result("runbooks/lexical-1.md")
    first["lexical_score"] = 10.0
    second = make_result("runbooks/lexical-2.md")
    second["lexical_score"] = 5.0
    fallback = make_result("runbooks/semantic-only.md", similarity=0.9)
    shared = make_result("runbooks/lexical-1.md")
    shared["similarity"] = 0.8

    fused = fuse_lexical_primary_results(
        [fallback, shared],
        [first, second],
        limit=3,
    )

    assert [item["source_key"] for item in fused] == [
        "runbooks/lexical-1.md",
        "runbooks/lexical-2.md",
        "runbooks/semantic-only.md",
    ]
    assert [item["ranking_origin"] for item in fused] == [
        "lexical",
        "lexical",
        "semantic_fallback",
    ]
    assert fused[2]["semantic_rank"] == 1
    assert fused[2]["lexical_rank"] is None


def test_lexical_primary_stops_at_limit_without_semantic_fallback() -> None:
    lexical = [
        make_result(f"runbooks/lexical-{index}.md") for index in range(3)
    ]
    for index, item in enumerate(lexical, start=1):
        item["lexical_score"] = float(10 - index)
    semantic = make_result("runbooks/semantic.md", similarity=0.9)

    fused = fuse_lexical_primary_results(
        [semantic],
        lexical,
        limit=2,
    )

    assert [item["source_key"] for item in fused] == [
        "runbooks/lexical-0.md",
        "runbooks/lexical-1.md",
    ]
    assert all(item["ranking_origin"] == "lexical" for item in fused)


def test_lexical_primary_keeps_unique_chunks_only() -> None:
    first = make_result("runbooks/a.md", chunk_index=0)
    duplicate = make_result("runbooks/a.md", chunk_index=0)
    duplicate["chunk_id"] = first["chunk_id"]
    other = make_result("runbooks/a.md", chunk_index=1)

    fused = fuse_lexical_primary_results([], [first, duplicate, other], limit=5)

    assert [item["chunk_id"] for item in fused] == [
        first["chunk_id"],
        other["chunk_id"],
    ]


def test_hybrid_search_weighted_strategy_passes_explicit_weights(monkeypatch) -> None:
    semantic = make_result("runbooks/a.md", similarity=0.9)
    lexical = make_result("runbooks/b.md")
    lexical["lexical_score"] = 1.0

    monkeypatch.setattr(
        retrieval,
        "semantic_search",
        lambda **kwargs: [semantic],
    )
    monkeypatch.setattr(
        retrieval,
        "lexical_search",
        lambda **kwargs: [lexical],
    )

    results = hybrid_search(
        tenant_id=uuid4(),
        query="payment",
        limit=5,
        keyword_strategy="or_idf",
        fusion_strategy="weighted",
        semantic_weight=2.0,
        lexical_weight=3.0,
    )

    assert results[0]["semantic_weight"] == pytest.approx(2.0)
    assert results[0]["lexical_weight"] == pytest.approx(3.0)


def test_hybrid_search_lexical_primary_uses_semantic_fallback(monkeypatch) -> None:
    lexical = make_result("runbooks/lexical.md")
    lexical["lexical_score"] = 5.0
    semantic = make_result("runbooks/semantic.md", similarity=0.9)

    monkeypatch.setattr(
        retrieval,
        "semantic_search",
        lambda **kwargs: [semantic],
    )
    monkeypatch.setattr(
        retrieval,
        "lexical_search",
        lambda **kwargs: [lexical],
    )

    results = hybrid_search(
        tenant_id=uuid4(),
        query="payment",
        limit=2,
        keyword_strategy="or_idf",
        fusion_strategy="lexical_primary",
    )

    assert [item["source_key"] for item in results] == [
        "runbooks/lexical.md",
        "runbooks/semantic.md",
    ]
    assert [item["ranking_origin"] for item in results] == [
        "lexical",
        "semantic_fallback",
    ]


@pytest.mark.parametrize("weight", [0.0, -1.0, float("inf"), float("nan")])
def test_hybrid_search_rejects_invalid_fusion_weights(weight: float) -> None:
    with pytest.raises(ValueError, match="weight"):
        hybrid_search(
            tenant_id=uuid4(),
            query="payment",
            semantic_weight=weight,
        )


@pytest.mark.parametrize(
    ("query", "scope", "numbers", "include_historical"),
    [
        (
            (
                "According to version 2 of the payment troubleshooting runbook, "
                "when should rollback run?"
            ),
            "explicit",
            (2,),
            True,
        ),
        (
            "What did revision 1 say about retries?",
            "explicit",
            (1,),
            True,
        ),
        (
            "What does the current payment runbook say about retries?",
            "current",
            (),
            False,
        ),
        (
            "What did the deprecated payment runbook say about retries?",
            "historical",
            (),
            True,
        ),
        (
            "How did the rollback change between versions?",
            "historical",
            (),
            True,
        ),
        (
            "What did the old runbook say, and what is the current guidance now?",
            "ambiguous",
            (),
            True,
        ),
    ],
)
def test_detect_version_request_classifies_question_intent(
    query: str,
    scope: str,
    numbers: tuple[int, ...],
    include_historical: bool,
) -> None:
    request = detect_version_request(query)

    assert request.scope == scope
    assert request.version_numbers == numbers
    assert request.include_historical is include_historical
    assert request.filter_numbers == ()


def test_detect_version_request_ignores_api_path_versions() -> None:
    request = detect_version_request(
        "Which POST path should a new payment client target under API version 3?"
    )

    assert request.version_numbers == ()
    assert request.scope == "current"


def test_resolve_version_request_explicit_filter_is_opt_in() -> None:
    auto = resolve_version_request("version 2 of the runbook")
    filtered = resolve_version_request("version 2 of the runbook", version_numbers=(2,), filter_versions=True)

    assert auto.filter_numbers == ()
    assert auto.include_historical is True
    assert filtered.filter_numbers == (2,)
    assert filtered.include_historical is True


def test_resolve_version_request_rejects_filter_without_numbers() -> None:
    with pytest.raises(ValueError, match="requires at least one version number"):
        resolve_version_request("what changed?", mode="explicit_filter")

    with pytest.raises(ValueError, match="requires at least one version number"):
        resolve_version_request("what changed?", filter_versions=True)


def test_lexical_version_ordering_does_not_reference_out_of_scope_cte_alias(
    monkeypatch,
) -> None:
    captured: dict = {}
    monkeypatch.setattr(
        retrieval,
        "get_engine",
        lambda: FakeEngine(captured),
    )

    lexical_search(
        tenant_id=uuid4(),
        query="version 2 of the payment runbook",
        include_historical=True,
        rank_version_numbers=(2,),
        filter_version_numbers=(2,),
    )

    sql = " ".join(captured["sql"].lower().split())
    order_by = sql.split("order by", 1)[1]
    assert "scored.version_number" in order_by
    assert "dv.version_number" not in order_by
    assert "scored.is_current" not in order_by
    assert "cast(:rank_version_numbers as integer[])" in order_by
    assert "dv.is_current = true" not in sql
    assert "dv.version_number = any(cast(:version_numbers as integer[]))" in sql
    assert captured["params"]["rank_version_numbers"] == [2]
    assert captured["params"]["version_numbers"] == [2]


def test_lexical_historical_ordering_uses_cte_columns(monkeypatch) -> None:
    captured: dict = {}
    monkeypatch.setattr(
        retrieval,
        "get_engine",
        lambda: FakeEngine(captured),
    )

    lexical_search(
        tenant_id=uuid4(),
        query="what changed between versions",
        include_historical=True,
    )

    order_by = " ".join(captured["sql"].lower().split()).split("order by", 1)[1]
    assert "scored.is_current" in order_by
    assert "dv.is_current" not in order_by


def test_lexical_search_default_remains_current_only(monkeypatch) -> None:
    captured: dict = {}
    monkeypatch.setattr(
        retrieval,
        "get_engine",
        lambda: FakeEngine(captured),
    )

    lexical_search(tenant_id=uuid4(), query="payment runbook")

    sql = " ".join(captured["sql"].lower().split())
    assert "dv.is_current = true" in sql
    assert "rank_version_numbers" not in captured["params"]
    assert "version_numbers" not in captured["params"]
    assert "scored.version_number" not in sql.split("order by", 1)[1]


def test_version_order_prefers_named_version_without_current_bias() -> None:
    order = retrieval._version_order_sql(
        rank_numbers=(1,),
        include_historical=True,
        version_number_expr="scored.version_number",
        is_current_expr="scored.is_current",
    )

    assert "scored.version_number" in order
    assert "scored.is_current" not in order

    current_only = retrieval._version_order_sql(
        rank_numbers=(),
        include_historical=True,
    )
    assert "dv.is_current" in current_only


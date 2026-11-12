"""Read-only live checks for the OR+IDF lexical strategy.

These tests need the seeded synthetic corpus and embedding service. They are
skipped unless ``OPSATLAS_LIVE_TESTS=1`` and ``OPSATLAS_TEST_TENANT_ID`` are set,
so the default unit suite stays independent of PostgreSQL.
"""

from __future__ import annotations

import os
from uuid import UUID, uuid4

import pytest

from app.services.retrieval import (
    detect_version_request,
    hybrid_search,
    keyword_search,
    lexical_search,
    prepare_lexical_query,
    resolve_version_request,
)

LIVE = os.environ.get("OPSATLAS_LIVE_TESTS") == "1"
TENANT_VALUE = os.environ.get("OPSATLAS_TEST_TENANT_ID")

pytestmark = [
    pytest.mark.skipif(not LIVE, reason="OPSATLAS_LIVE_TESTS is not 1"),
    pytest.mark.skipif(
        not TENANT_VALUE,
        reason="OPSATLAS_TEST_TENANT_ID is not set",
    ),
]


def tenant_id() -> UUID:
    return UUID(TENANT_VALUE)


def source_keys(query: str, *, limit: int = 20) -> list[str]:
    return [
        result["source_key"]
        for result in lexical_search(
            tenant_id=tenant_id(),
            query=query,
            limit=limit,
        )
    ]


def unique_units(results: list[dict]) -> list[tuple[str, int]]:
    units: list[tuple[str, int]] = []
    for result in results:
        unit = (result["source_key"], result["version_number"])
        if unit not in units:
            units.append(unit)
    return units


@pytest.mark.parametrize("identifier", ["PAY-503", "PAY-409", "PAY-400"])
def test_live_identifier_queries_rank_the_error_runbook_first(identifier: str) -> None:
    results = lexical_search(
        tenant_id=tenant_id(),
        query=f"What does the error-code runbook say about {identifier}?",
        limit=5,
    )

    assert results
    assert results[0]["source_key"] == "runbooks/payment-errors.md"


def test_live_http_status_is_not_treated_as_a_pay_identifier() -> None:
    pay_results = lexical_search(
        tenant_id=tenant_id(),
        query="What does PAY-503 mean?",
        limit=5,
    )
    http_results = lexical_search(
        tenant_id=tenant_id(),
        query="Payments return HTTP 503 after a deployment.",
        limit=5,
    )

    assert pay_results[0]["source_key"] == "runbooks/payment-errors.md"
    assert http_results[0]["source_key"] == "runbooks/payment-api.md"
    assert http_results[0]["version_number"] == 2
    assert all(
        result["source_key"] != "runbooks/payment-api.md"
        for result in pay_results
    )

    pay = prepare_lexical_query("PAY-503")
    http = prepare_lexical_query("HTTP 503")
    assert pay.identifiers[0].kind == "identifier"
    assert http.identifiers[0].kind == "http_status"


def test_live_error_code_and_error_code_space_are_not_special_cased() -> None:
    hyphenated = lexical_search(
        tenant_id=tenant_id(),
        query="What does the error-code runbook list?",
        limit=5,
    )
    spaced = lexical_search(
        tenant_id=tenant_id(),
        query="What does the error code runbook list?",
        limit=5,
    )

    assert hyphenated[0]["source_key"] == "runbooks/payment-errors.md"
    assert spaced[0]["source_key"] == "runbooks/payment-errors.md"


def test_live_rare_terms_rank_their_expected_runbooks_first() -> None:
    idempotency = lexical_search(
        tenant_id=tenant_id(),
        query="What does the runbook say about idempotency for payment retries?",
        limit=5,
    )
    non_idempotent = lexical_search(
        tenant_id=tenant_id(),
        query=(
            "What does the error-code runbook say about retrying "
            "non-idempotent payment operations?"
        ),
        limit=5,
    )
    aggregate_budget = lexical_search(
        tenant_id=tenant_id(),
        query=(
            "Why can increasing each instance's pool size make connection "
            "exhaustion worse?"
        ),
        limit=5,
    )

    assert idempotency[0]["source_key"] == "runbooks/payment-errors.md"
    assert non_idempotent[0]["source_key"] == "runbooks/payment-errors.md"
    assert aggregate_budget[0]["source_key"] == "runbooks/postgres-pool.md"


def test_live_api_version_selects_the_current_migration_guide() -> None:
    results = lexical_search(
        tenant_id=tenant_id(),
        query=(
            "Is the v1 payments endpoint deprecated, and what should new "
            "clients use instead?"
        ),
        limit=5,
    )

    assert results[0]["source_key"] == "api-specs/payments.md"
    assert results[0]["version_number"] == 2


def test_live_multi_document_query_returns_both_expected_sources() -> None:
    results = lexical_search(
        tenant_id=tenant_id(),
        query=(
            "What mitigation does the synthetic postmortem describe, and what "
            "does the current operations runbook recommend?"
        ),
        limit=20,
    )

    units = unique_units(results)
    keys = [source_key for source_key, _ in units]
    assert "runbooks/postgres-pool.md" in keys[:5]
    assert "postmortems/db-pool-2025-02.md" in keys[:5]


@pytest.mark.parametrize(
    ("query", "expected_source"),
    [
        (
            "Kafka consumer lag keeps growing after a deployment; what should we check?",
            "runbooks/kafka-consumer-lag.md",
        ),
        (
            "Redis failover caused readiness checks to fail; what is the recovery procedure?",
            "runbooks/redis-failover.md",
        ),
        (
            "What severity levels does the incident escalation process define?",
            "runbooks/incident-escalation.md",
        ),
    ],
)
def test_live_distractor_overlap_queries_still_rank_their_runbook_first(
    query: str,
    expected_source: str,
) -> None:
    results = lexical_search(tenant_id=tenant_id(), query=query, limit=5)

    assert results[0]["source_key"] == expected_source


def test_live_tenant_and_current_version_filters_are_preserved() -> None:
    assert lexical_search(
        tenant_id=uuid4(),
        query="PAY-503 error code",
        limit=5,
    ) == []

    results = lexical_search(
        tenant_id=tenant_id(),
        query="payment troubleshooting runbook database connectivity",
        limit=20,
    )
    units = set(unique_units(results))

    assert ("runbooks/payment-api.md", 2) in units
    assert ("runbooks/payment-api.md", 1) not in units
    assert ("api-specs/payments.md", 1) not in units


def test_live_and_keyword_baseline_is_still_available() -> None:
    results = keyword_search(
        tenant_id=tenant_id(),
        query="What checks matter for PAY-503?",
        limit=5,
    )

    assert results == []


def test_live_results_are_deterministic_and_punctuation_is_empty() -> None:
    first = lexical_search(tenant_id=tenant_id(), query="PAY-503", limit=5)
    second = lexical_search(tenant_id=tenant_id(), query="PAY-503", limit=5)

    assert [result["chunk_id"] for result in first] == [
        result["chunk_id"] for result in second
    ]
    assert lexical_search(tenant_id=tenant_id(), query="???", limit=5) == []


def test_live_source_aware_fusion_collapses_duplicate_chunks() -> None:
    results = hybrid_search(
        tenant_id=tenant_id(),
        query="Requests are queuing for a database connection under load.",
        limit=5,
        keyword_strategy="or_idf",
        fusion_strategy="source",
    )

    units = unique_units(results)
    assert len(units) == len(results)
    assert all(result["semantic_source_chunk_count"] is not None for result in results)


def test_live_http503_vs_pay503_query_recovers_both_expected_sources() -> None:
    query = (
        "A monitor fires for HTTP 503 while a different alert fires for PAY-503. "
        "Are they the same signal?"
    )
    results = lexical_search(tenant_id=tenant_id(), query=query, limit=5)

    units = unique_units(results)
    keys = [source_key for source_key, _ in units]
    assert "runbooks/payment-api.md" in keys
    assert "runbooks/payment-errors.md" in keys


def test_live_shared_identifier_keeps_all_matching_sources_ranked() -> None:
    results = lexical_search(tenant_id=tenant_id(), query="PAY-503", limit=20)

    keys = {result["source_key"] for result in results}
    assert "runbooks/payment-errors.md" in keys
    assert "api-specs/payments.md" in keys
    assert "api-specs/payments-v3.md" in keys


def test_live_identifier_weight_does_not_hide_semantic_or_generic_matches() -> None:
    results = lexical_search(
        tenant_id=tenant_id(),
        query=(
            "A monitor fires for HTTP 503 while a different alert fires for PAY-503. "
            "Are they the same signal?"
        ),
        limit=20,
    )

    # Identifiers are ranking signals, not filters: unrelated generic-term
    # matches must still be returned below the identifier-bearing sources.
    keys = {result["source_key"] for result in results}
    assert "runbooks/search-deploy.md" in keys
    assert "runbooks/postgres-pool.md" in keys


def test_live_unrelated_numeric_values_are_not_identifiers() -> None:
    prepared = prepare_lexical_query(
        "The 2025-02 incident affected 1000 requests on port 8080."
    )

    assert prepared.identifiers == ()
    assert "2025-02" in prepared.terms_query


def test_live_bare_http_status_is_not_an_error_identifier() -> None:
    prepared = prepare_lexical_query("Why does the service return 503?")
    assert [(term.kind, term.value) for term in prepared.identifiers] == [
        ("http_status", "503")
    ]
    assert "[^A-Za-z0-9-]" in prepared.identifiers[0].sql_pattern

    results = lexical_search(
        tenant_id=tenant_id(),
        query="Why does the service return 503?",
        limit=5,
    )
    assert results


@pytest.mark.parametrize(
    ("query", "expected_version"),
    [
        (
            (
                "According to version 1 of the payment troubleshooting runbook, "
                "when should the approved rollback procedure be followed?"
            ),
            1,
        ),
        (
            (
                "According to version 2 of the payment troubleshooting runbook, "
                "when should the approved rollback procedure be followed?"
            ),
            2,
        ),
    ],
)
def test_live_explicit_version_retrieves_the_named_runbook_version(
    query: str,
    expected_version: int,
) -> None:
    request = detect_version_request(query)
    results = lexical_search(
        tenant_id=tenant_id(),
        query=query,
        limit=5,
        include_historical=request.include_historical,
        rank_version_numbers=request.version_numbers or None,
    )

    units = [(result["source_key"], result["version_number"]) for result in results]
    assert ("runbooks/payment-api.md", expected_version) in units[:3]


def test_live_explicit_v2_rollback_is_retrieved_by_hybrid_search() -> None:
    query = (
        "According to version 2 of the payment troubleshooting runbook, "
        "when should the approved rollback procedure be followed?"
    )
    request = detect_version_request(query)
    results = hybrid_search(
        tenant_id=tenant_id(),
        query=query,
        limit=5,
        keyword_strategy="or_idf",
        version_request=request,
    )

    assert any(
        result["source_key"] == "runbooks/payment-api.md"
        and result["version_number"] == 2
        for result in results
    )


def test_live_current_question_excludes_superseded_versions() -> None:
    results = lexical_search(
        tenant_id=tenant_id(),
        query="What does the current payment troubleshooting runbook say about rollback?",
        limit=20,
    )

    units = unique_units(results)
    assert ("runbooks/payment-api.md", 2) in units
    assert ("runbooks/payment-api.md", 1) not in units


def test_live_deprecated_question_includes_historical_versions() -> None:
    results = lexical_search(
        tenant_id=tenant_id(),
        query="What did the deprecated payment troubleshooting runbook say about rollback?",
        limit=20,
        include_historical=True,
    )

    versions = {
        result["version_number"]
        for result in results
        if result["source_key"] == "runbooks/payment-api.md"
    }
    assert 1 in versions


def test_live_ambiguous_version_question_is_detected_and_bounded() -> None:
    query = (
        "What did the old payment runbook say about rollback, and what is the "
        "current guidance now?"
    )
    request = detect_version_request(query)

    assert request.scope == "ambiguous"
    assert request.include_historical is True

    results = lexical_search(
        tenant_id=tenant_id(),
        query=query,
        limit=20,
        include_historical=True,
    )
    assert results


def test_live_explicit_version_filter_only_returns_that_version() -> None:
    request = resolve_version_request(
        "payment troubleshooting runbook rollback",
        version_numbers=(2,),
        filter_versions=True,
    )
    results = lexical_search(
        tenant_id=tenant_id(),
        query="payment troubleshooting runbook rollback",
        limit=20,
        include_historical=request.include_historical,
        filter_version_numbers=request.filter_numbers,
        rank_version_numbers=request.version_numbers,
    )

    assert results
    assert {result["version_number"] for result in results} == {2}


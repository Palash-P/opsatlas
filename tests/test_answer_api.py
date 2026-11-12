"""API tests for POST /api/v1/answer using mocked retrieval and generation."""

from __future__ import annotations

import json
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient

from app.api import answers
from app.core.config import Settings
from app.main import app
from app.services.embedding_client import EmbeddingServiceError
from app.services.llm_providers import (
    GenerationConfigurationError,
    GenerationProviderError,
    GenerationRateLimitError,
    GenerationTimeoutError,
    ProviderResponse,
)

client = TestClient(app)
TEST_TENANT_ID = UUID("11111111-1111-4111-8111-111111111111")


def make_result(**overrides) -> dict:
    result = {
        "chunk_id": uuid4(),
        "document_id": uuid4(),
        "version_id": uuid4(),
        "title": "Payment API Runbook",
        "source_key": "runbooks/payment-api.md",
        "version_number": 2,
        "authority_level": 80,
        "chunk_index": 0,
        "content": "Check upstream timeouts before rolling back.",
        "hybrid_score": 0.42,
        "semantic_rank": 1,
        "keyword_rank": 1,
        "similarity": 0.8,
        "keyword_score": 0.3,
    }
    result.update(overrides)
    return result


def grounded_payload(**overrides) -> str:
    payload = {
        "answer": "Check upstream timeouts.",
        "used_evidence_ids": ["E1"],
        "abstained": False,
        "abstention_reason": None,
        "insufficient_evidence": False,
        "conflicts": [],
        "uncertainty": None,
    }
    payload.update(overrides)
    return json.dumps(payload)


class FakeProvider:
    name = "fake"
    model = "fake-model"

    def __init__(self, text: str) -> None:
        self.text = text

    def generate(self, request):
        return ProviderResponse(text=self.text, provider=self.name, model=self.model)


def patch_generation(monkeypatch, *, text=None, error=None):
    monkeypatch.setattr(
        answers,
        "get_settings",
        lambda: Settings(llm_retry_backoff_seconds=0.0, llm_max_retries=0),
    )

    def fake_build_provider(_settings):
        if error is not None:
            raise error
        return FakeProvider(text or grounded_payload())

    monkeypatch.setattr(answers, "build_llm_provider", fake_build_provider)


def post_answer(**overrides):
    body = {
        "tenant_id": str(TEST_TENANT_ID),
        "question": "What should I check before rolling back?",
    }
    body.update(overrides)
    return client.post("/api/v1/answer", json=body)


def test_answer_endpoint_returns_answer_citations_and_diagnostics(monkeypatch) -> None:
    result = make_result()
    monkeypatch.setattr(
        answers,
        "retrieve_for_answer",
        lambda **kwargs: ([result], {"strategy": "hybrid", "limit": 5}),
    )
    patch_generation(monkeypatch)

    response = post_answer()

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Check upstream timeouts."
    assert len(body["citations"]) == 1
    citation = body["citations"][0]
    assert citation["evidence_id"] == "E1"
    assert citation["document_id"] == str(result["document_id"])
    assert citation["version_id"] == str(result["version_id"])
    assert citation["source_key"] == "runbooks/payment-api.md"
    assert citation["title"] == "Payment API Runbook"
    assert citation["version_number"] == 2
    assert citation["chunk_id"] == str(result["chunk_id"])
    assert citation["excerpt"]

    assert body["evidence"][0]["chunk_id"] == str(result["chunk_id"])
    assert body["evidence"][0]["version_number"] == 2
    assert body["retrieval"]["result_count"] == 1
    assert body["retrieval"]["results"][0]["score"] == pytest.approx(0.42)
    assert body["retrieval"]["results"][0]["score_field"] == "hybrid_score"
    assert body["generation"]["provider"] == "fake"
    assert body["generation"]["prompt_version"]
    assert body["generation"]["attempts"] == 1
    assert body["abstention"] == {
        "abstained": False,
        "reason": None,
        "insufficient_evidence": False,
    }
    assert body["conflicts"] == []
    assert body["diagnostics"]["evidence_count"] == 1
    assert body["diagnostics"]["mixed_versions"] is False
    assert body["diagnostics"]["uncited_evidence_ids"] == []


def test_answer_endpoint_preserves_existing_retrieval_defaults(monkeypatch) -> None:
    captured = {}

    def fake_hybrid_search(**kwargs):
        captured.update(kwargs)
        return [make_result()]

    monkeypatch.setattr(answers, "hybrid_search", fake_hybrid_search)
    patch_generation(monkeypatch)

    response = post_answer()

    assert response.status_code == 200
    assert captured["limit"] == 5
    assert captured["keyword_strategy"] == "and"
    assert captured["fusion_strategy"] == "chunk"
    assert captured["semantic_weight"] == pytest.approx(1.0)
    assert captured["lexical_weight"] == pytest.approx(1.0)
    assert captured["rrf_k"] == 60


def test_answer_endpoint_accepts_retrieval_overrides(monkeypatch) -> None:
    captured = {}

    def fake_hybrid_search(**kwargs):
        captured.update(kwargs)
        return [make_result()]

    monkeypatch.setattr(answers, "hybrid_search", fake_hybrid_search)
    patch_generation(monkeypatch)

    response = post_answer(
        retrieval={
            "strategy": "hybrid",
            "limit": 3,
            "keyword_strategy": "or_idf",
            "fusion_strategy": "lexical_primary",
            "context_token_budget": 500,
            "max_evidence": 2,
        }
    )

    assert response.status_code == 200
    assert captured["limit"] == 3
    assert captured["keyword_strategy"] == "or_idf"
    assert captured["fusion_strategy"] == "lexical_primary"
    assert response.json()["diagnostics"]["context_token_budget"] == 500


def test_answer_endpoint_rejects_blank_question() -> None:
    response = post_answer(question="   ")

    assert response.status_code == 422
    assert "blank" in response.json()["detail"].lower()


def test_answer_endpoint_returns_abstention_when_no_evidence(monkeypatch) -> None:
    monkeypatch.setattr(
        answers,
        "retrieve_for_answer",
        lambda **kwargs: ([], {"strategy": "hybrid", "limit": 5}),
    )
    patch_generation(monkeypatch)

    response = post_answer()

    assert response.status_code == 200
    body = response.json()
    assert body["abstention"]["abstained"] is True
    assert body["abstention"]["reason"] == "no_evidence"
    assert body["citations"] == []
    assert body["evidence"] == []


def test_answer_endpoint_maps_embedding_failure_to_503(monkeypatch) -> None:
    def fail(**kwargs):
        raise EmbeddingServiceError("down")

    monkeypatch.setattr(answers, "retrieve_for_answer", fail)

    response = post_answer()

    assert response.status_code == 503
    assert "Embedding service" in response.json()["detail"]


def test_answer_endpoint_maps_generation_timeout_to_504(monkeypatch) -> None:
    monkeypatch.setattr(
        answers,
        "retrieve_for_answer",
        lambda **kwargs: ([make_result()], {"strategy": "hybrid", "limit": 5}),
    )
    patch_generation(monkeypatch, error=GenerationTimeoutError("timed out"))

    response = post_answer()

    assert response.status_code == 504


def test_answer_endpoint_maps_rate_limit_to_429(monkeypatch) -> None:
    monkeypatch.setattr(
        answers,
        "retrieve_for_answer",
        lambda **kwargs: ([make_result()], {"strategy": "hybrid", "limit": 5}),
    )
    patch_generation(monkeypatch, error=GenerationRateLimitError("slow down"))

    response = post_answer()

    assert response.status_code == 429


def test_answer_endpoint_maps_provider_error_to_502(monkeypatch) -> None:
    monkeypatch.setattr(
        answers,
        "retrieve_for_answer",
        lambda **kwargs: ([make_result()], {"strategy": "hybrid", "limit": 5}),
    )
    patch_generation(monkeypatch, error=GenerationProviderError("failed"))

    response = post_answer()

    assert response.status_code == 502


def test_answer_endpoint_maps_configuration_error_to_503(monkeypatch) -> None:
    monkeypatch.setattr(
        answers,
        "retrieve_for_answer",
        lambda **kwargs: ([make_result()], {"strategy": "hybrid", "limit": 5}),
    )
    patch_generation(monkeypatch, error=GenerationConfigurationError("missing key"))

    response = post_answer()

    assert response.status_code == 503


def test_answer_endpoint_rejects_fabricated_citation(monkeypatch) -> None:
    monkeypatch.setattr(
        answers,
        "retrieve_for_answer",
        lambda **kwargs: ([make_result()], {"strategy": "hybrid", "limit": 5}),
    )
    patch_generation(
        monkeypatch,
        text=grounded_payload(used_evidence_ids=["E99"]),
    )

    response = post_answer()

    assert response.status_code == 502
    assert "not supplied" in response.json()["detail"]


def test_answer_endpoint_rejects_invalid_model_output(monkeypatch) -> None:
    monkeypatch.setattr(
        answers,
        "retrieve_for_answer",
        lambda **kwargs: ([make_result()], {"strategy": "hybrid", "limit": 5}),
    )
    patch_generation(monkeypatch, text="not json")

    response = post_answer()

    assert response.status_code == 502


def test_answer_endpoint_reports_conflicts_and_mixed_versions(monkeypatch) -> None:
    old = make_result(version_number=1, chunk_index=0, content="Restart workers.")
    new = make_result(version_number=2, chunk_index=1, content="Roll workers.")
    monkeypatch.setattr(
        answers,
        "retrieve_for_answer",
        lambda **kwargs: ([old, new], {"strategy": "hybrid", "limit": 5}),
    )
    patch_generation(
        monkeypatch,
        text=grounded_payload(
            used_evidence_ids=["E1", "E2"],
            conflicts=[
                {
                    "evidence_ids": ["E1", "E2"],
                    "description": "Version 1 restarts; version 2 rolls.",
                }
            ],
        ),
    )

    response = post_answer()

    assert response.status_code == 200
    body = response.json()
    assert body["diagnostics"]["mixed_versions"] is True
    assert body["conflicts"] == [
        {
            "evidence_ids": ["E1", "E2"],
            "description": "Version 1 restarts; version 2 rolls.",
        }
    ]
    assert len(body["citations"]) == 2


def test_answer_endpoint_returns_version_analysis(monkeypatch) -> None:
    old = make_result(version_number=1, chunk_index=0, content="Restart workers.")
    new = make_result(version_number=2, chunk_index=1, content="Roll workers.")
    monkeypatch.setattr(
        answers,
        "retrieve_for_answer",
        lambda **kwargs: ([old, new], {"strategy": "hybrid", "limit": 5}),
    )
    patch_generation(
        monkeypatch,
        text=grounded_payload(
            used_evidence_ids=["E1", "E2"],
            version_analysis={
                "versions_identified": [
                    {
                        "source_key": "runbooks/payment-api.md",
                        "version_number": 1,
                        "status": "historical",
                        "evidence_ids": ["E1"],
                    },
                    {
                        "source_key": "runbooks/payment-api.md",
                        "version_number": 2,
                        "status": "current",
                        "evidence_ids": ["E2"],
                    },
                ],
                "differences": ["Version 2 rolls workers instead of restarting them."],
                "conflicting_instructions": [
                    "Version 1 restarts workers; version 2 rolls them."
                ],
                "recommendation_applies_to": "version 2",
                "supporting_evidence_ids": ["E1", "E2"],
            },
        ),
    )

    response = post_answer()

    assert response.status_code == 200
    analysis = response.json()["version_analysis"]
    assert analysis["recommendation_applies_to"] == "version 2"
    assert [item["version_number"] for item in analysis["versions_identified"]] == [1, 2]
    assert analysis["conflicting_instructions"]
    assert response.json()["generation"]["provider"] == "fake"


def test_answer_endpoint_omits_version_analysis_when_not_applicable(monkeypatch) -> None:
    monkeypatch.setattr(
        answers,
        "retrieve_for_answer",
        lambda **kwargs: ([make_result()], {"strategy": "hybrid", "limit": 5}),
    )
    patch_generation(monkeypatch)

    response = post_answer()

    assert response.status_code == 200
    assert response.json()["version_analysis"] is None

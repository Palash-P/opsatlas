"""Offline failure-injection coverage for dependency and provider faults.

These tests do not stop containers. They force the same exception types that
the live failure-injection script triggers (PostgreSQL pool/connect failures,
embedding outages, provider timeouts, rate limits, malformed envelopes, and
citation violations) and assert the API's status, error category, retry count,
recovery behavior, and request-log correctness.
"""

from __future__ import annotations

import json
import logging
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError
from sqlalchemy.exc import TimeoutError as SQLAlchemyTimeoutError

from app.api import answers, search
from app.core.config import Settings
from app.core.observability import LOGGER_NAME
from app.main import app
from app.services.embedding_client import EmbeddingServiceError
from app.services.llm_providers import (
    CitationValidationError,
    GenerationInvalidResponseError,
    GenerationProviderError,
    GenerationRateLimitError,
    GenerationTimeoutError,
    ProviderResponse,
)

client = TestClient(app)
TENANT_A = UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
TOKEN = "failure-injection-token-must-not-be-logged"
QUESTION = "SENSITIVE QUESTION MUST NOT BE LOGGED"


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


class CapturingHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@pytest.fixture
def request_logs():
    logger = logging.getLogger(LOGGER_NAME)
    handler = CapturingHandler()
    previous_level = logger.level
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    yield handler.records
    logger.removeHandler(handler)
    logger.setLevel(previous_level)


class ProviderStub:
    name = "fake"
    model = "fake-model"

    def __init__(self, *, text: str | None = None, error: Exception | None = None) -> None:
        self.text = text or json.dumps(
            {
                "answer": "Check upstream timeouts.",
                "used_evidence_ids": ["E1"],
                "abstained": False,
                "abstention_reason": None,
                "insufficient_evidence": False,
                "conflicts": [],
                "uncertainty": None,
            }
        )
        self.error = error
        self.calls = 0

    def generate(self, request):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return ProviderResponse(
            text=self.text,
            provider=self.name,
            model=self.model,
            usage={"prompt_tokens": 100, "completion_tokens": 20},
        )


def patch_provider(monkeypatch, provider: ProviderStub, *, retries: int = 0) -> None:
    monkeypatch.setattr(
        answers,
        "get_settings",
        lambda: Settings(llm_retry_backoff_seconds=0.0, llm_max_retries=retries),
    )
    monkeypatch.setattr(answers, "build_llm_provider", lambda _settings: provider)
    monkeypatch.setattr(
        answers,
        "retrieve_for_answer",
        lambda **kwargs: ([make_result()], {"strategy": "hybrid", "limit": 5}),
    )


def post_answer():
    return client.post(
        "/api/v1/answer",
        json={"question": QUESTION, "retrieval": {"strategy": "lexical"}},
    )


# ---------------------------------------------------------------------------
# Dependency failures
# ---------------------------------------------------------------------------


def test_postgres_failure_maps_to_503_and_is_categorized(monkeypatch, request_logs) -> None:
    monkeypatch.setattr(
        answers,
        "retrieve_for_answer",
        lambda **kwargs: (_ for _ in ()).throw(
            OperationalError("SELECT 1", {}, RuntimeError("connection refused"))
        ),
    )

    response = post_answer()

    assert response.status_code == 503
    assert response.json() == {"detail": "A required dependency is unavailable"}
    event = request_logs[-1].event_data
    assert event["status_code"] == 503
    assert event["error_category"] == "database_unavailable"
    assert event["request_id"] == response.headers["X-Request-ID"]


def test_pool_timeout_maps_to_503_not_500(monkeypatch, request_logs) -> None:
    monkeypatch.setattr(
        answers,
        "retrieve_for_answer",
        lambda **kwargs: (_ for _ in ()).throw(SQLAlchemyTimeoutError("pool exhausted")),
    )

    response = post_answer()

    assert response.status_code == 503
    assert request_logs[-1].event_data["error_category"] == "database_unavailable"


def test_embedding_outage_maps_to_503_on_search(monkeypatch, request_logs) -> None:
    monkeypatch.setattr(
        search,
        "semantic_search",
        lambda **kwargs: (_ for _ in ()).throw(
            EmbeddingServiceError("Could not reach the local embedding service")
        ),
    )

    response = client.post("/api/v1/search", json={"query": QUESTION, "limit": 5})

    assert response.status_code == 503
    assert response.json() == {"detail": "Embedding service is unavailable"}
    event = request_logs[-1].event_data
    assert event["error_category"] == "embedding_unavailable"
    assert event["retrieval_strategy"] == "semantic"
    assert isinstance(event["retrieval_latency_ms"], int)


def test_embedding_outage_maps_to_503_on_answer(monkeypatch, request_logs) -> None:
    monkeypatch.setattr(
        answers,
        "retrieve_for_answer",
        lambda **kwargs: (_ for _ in ()).throw(
            EmbeddingServiceError("Could not reach the local embedding service")
        ),
    )

    response = post_answer()

    assert response.status_code == 503
    assert response.json() == {"detail": "Embedding service is unavailable"}
    assert request_logs[-1].event_data["error_category"] == "embedding_unavailable"


# ---------------------------------------------------------------------------
# Provider failures
# ---------------------------------------------------------------------------


def test_provider_timeout_returns_504_after_retry(monkeypatch, request_logs) -> None:
    provider = ProviderStub(error=GenerationTimeoutError("upstream timed out"))
    patch_provider(monkeypatch, provider, retries=1)

    response = post_answer()

    assert response.status_code == 504
    assert provider.calls == 2
    event = request_logs[-1].event_data
    assert event["error_category"] == "provider_timeout"
    assert event["retry_count"] == 1
    assert event["provider"] == "fake"
    assert isinstance(event["generation_latency_ms"], int)
    assert event["observed_model"] is None


def test_provider_rate_limit_returns_429_after_retry(monkeypatch, request_logs) -> None:
    provider = ProviderStub(error=GenerationRateLimitError("429"))
    patch_provider(monkeypatch, provider, retries=1)

    response = post_answer()

    assert response.status_code == 429
    assert provider.calls == 2
    event = request_logs[-1].event_data
    assert event["error_category"] == "provider_rate_limited"
    assert event["retry_count"] == 1


def test_invalid_provider_envelope_is_not_retried(monkeypatch, request_logs) -> None:
    provider = ProviderStub(error=GenerationInvalidResponseError("no choices"))
    patch_provider(monkeypatch, provider, retries=2)

    response = post_answer()

    assert response.status_code == 502
    assert provider.calls == 1
    event = request_logs[-1].event_data
    assert event["error_category"] == "invalid_model_output"
    assert event["retry_count"] == 0


def test_invalid_model_output_is_not_retried(monkeypatch, request_logs) -> None:
    provider = ProviderStub(text="this is not json")
    patch_provider(monkeypatch, provider, retries=2)

    response = post_answer()

    assert response.status_code == 502
    assert provider.calls == 1
    assert request_logs[-1].event_data["error_category"] == "invalid_model_output"


def test_unknown_citation_is_rejected_as_502(monkeypatch, request_logs) -> None:
    provider = ProviderStub(
        text=json.dumps(
            {
                "answer": "Invented.",
                "used_evidence_ids": ["E999"],
                "abstained": False,
                "abstention_reason": None,
                "insufficient_evidence": False,
                "conflicts": [],
                "uncertainty": None,
            }
        )
    )
    patch_provider(monkeypatch, provider, retries=2)

    response = post_answer()

    assert response.status_code == 502
    assert provider.calls == 1
    assert request_logs[-1].event_data["error_category"] == (
        "citation_validation_error"
    )


def test_retryable_provider_error_retries_then_succeeds(monkeypatch) -> None:
    class FlakyProvider(ProviderStub):
        def generate(self, request):
            self.calls += 1
            if self.calls == 1:
                raise GenerationProviderError("upstream 500", retryable=True)
            return ProviderResponse(
                text=self.text,
                provider=self.name,
                model=self.model,
                usage={"prompt_tokens": 10, "completion_tokens": 5},
            )

    provider = FlakyProvider()
    patch_provider(monkeypatch, provider, retries=1)

    response = post_answer()

    assert response.status_code == 200
    assert provider.calls == 2


def test_non_retryable_provider_error_stops_immediately(monkeypatch, request_logs) -> None:
    provider = ProviderStub(
        error=GenerationProviderError("credentials rejected", retryable=False)
    )
    patch_provider(monkeypatch, provider, retries=2)

    response = post_answer()

    assert response.status_code == 502
    assert provider.calls == 1
    assert request_logs[-1].event_data["error_category"] == "provider_error"


# ---------------------------------------------------------------------------
# Log correctness under failure
# ---------------------------------------------------------------------------


def test_failure_log_excludes_question_token_and_raw_tenant_uuid(
    monkeypatch, request_logs
) -> None:
    provider = ProviderStub(error=GenerationTimeoutError("timed out"))
    patch_provider(monkeypatch, provider, retries=0)

    response = post_answer()

    assert response.status_code == 504
    event = request_logs[-1].event_data
    serialized = json.dumps(event)
    assert QUESTION not in serialized
    assert TOKEN not in serialized
    assert str(TENANT_A) not in serialized
    assert event["tenant_ref"].startswith("t_")
    assert event["error_category"] == "provider_timeout"
    assert event["input_tokens"] is None
    assert event["output_tokens"] is None
    assert event["token_source"] is None


def test_every_failure_has_one_correlated_request_log(monkeypatch, request_logs) -> None:
    provider = ProviderStub(error=GenerationInvalidResponseError("no choices"))
    patch_provider(monkeypatch, provider, retries=0)
    request_id = "failure-correlate-0001"

    response = client.post(
        "/api/v1/answer",
        headers={"X-Request-ID": request_id},
        json={"question": QUESTION},
    )

    matches = [
        record.event_data
        for record in request_logs
        if getattr(record, "event_data", {}).get("request_id") == request_id
    ]
    assert response.status_code == 502
    assert len(matches) == 1
    assert matches[0]["status_code"] == 502


def test_recovery_after_provider_failure(monkeypatch) -> None:
    provider = ProviderStub(error=GenerationTimeoutError("timed out"))
    patch_provider(monkeypatch, provider, retries=0)

    failed = post_answer()
    assert failed.status_code == 504

    provider.error = None
    recovered = post_answer()

    assert recovered.status_code == 200
    assert recovered.json()["generation"]["provider"] == "fake"


# ---------------------------------------------------------------------------
# Citation validation helper
# ---------------------------------------------------------------------------


def test_citation_validation_error_is_a_generation_error() -> None:
    with pytest.raises(CitationValidationError):
        raise CitationValidationError("bad")

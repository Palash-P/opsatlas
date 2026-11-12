"""Phase D adversarial security and observability tests.

Coverage:

- missing/invalid authentication and consistent unauthorized responses;
- authenticated tenant resolution and body-tenant forgery;
- tenant isolation in retrieval (including a defense-in-depth row check);
- document ACL scope in retrieval SQL;
- cross-tenant ingestion and non-disclosure of tenant existence;
- forged version-analysis metadata;
- provider error non-leakage into responses and request logs;
- structured request telemetry without prompts, content, tokens, or raw tenants;
- bounded rate limiting and concurrency.
"""

from __future__ import annotations

import json
import logging
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient

from app.api import answers
from app.core import auth as auth_module
from app.core.auth import Principal, acl_filter_sql
from app.core.config import Settings
from app.core.limits import (
    get_concurrency_semaphore,
    reset_limiters,
)
from app.core.observability import LOGGER_NAME
from app.main import app
from app.services import retrieval
from app.services.answer_generation import parse_model_answer
from app.services.evidence import assemble_evidence
from app.services.llm_providers import (
    CitationValidationError,
    GenerationProviderError,
    ProviderResponse,
)

client = TestClient(app)
TENANT_A = UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
TENANT_B = UUID("bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb")
TOKEN_A = "token-tenant-a-should-never-be-logged"


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


class RecordingProvider:
    name = "fake"
    model = "fake-model"

    def __init__(
        self,
        text: str | None = None,
        *,
        usage: dict[str, int] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.text = text or grounded_payload()
        self.usage = usage
        self.error = error

    def generate(self, request):
        if self.error is not None:
            raise self.error
        return ProviderResponse(
            text=self.text,
            provider=self.name,
            model=self.model,
            usage=self.usage,
        )


def patch_generation(monkeypatch, provider: RecordingProvider) -> None:
    monkeypatch.setattr(
        answers,
        "get_settings",
        lambda: Settings(llm_retry_backoff_seconds=0.0, llm_max_retries=0),
    )
    monkeypatch.setattr(answers, "build_llm_provider", lambda _settings: provider)


def configure_auth(monkeypatch, *, token: str = TOKEN_A, tenant_id: UUID = TENANT_A) -> None:
    monkeypatch.setattr(
        auth_module,
        "get_settings",
        lambda: Settings(
            auth_tokens={
                token: {
                    "tenant_id": str(tenant_id),
                    "subject": "alice",
                    "roles": ["member"],
                    "groups": ["on-call"],
                }
            }
        ),
    )


def auth_header(token: str = TOKEN_A) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


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


# ---------------------------------------------------------------------------
# Authentication and tenant resolution
# ---------------------------------------------------------------------------


def test_missing_bearer_token_returns_a_consistent_401(real_auth, monkeypatch) -> None:
    configure_auth(monkeypatch)

    response = client.post("/api/v1/search", json={"query": "payments failing"})

    assert response.status_code == 401
    assert response.json() == {"detail": "Unauthorized"}
    assert response.headers["WWW-Authenticate"] == "Bearer"
    assert response.headers["X-Request-ID"]


def test_invalid_bearer_token_returns_401(real_auth, monkeypatch) -> None:
    configure_auth(monkeypatch)

    response = client.post(
        "/api/v1/search",
        headers=auth_header("not-a-configured-token"),
        json={"query": "payments failing"},
    )

    assert response.status_code == 401
    assert response.json() == {"detail": "Unauthorized"}


def test_body_tenant_mismatch_is_forbidden_before_retrieval(
    real_auth, monkeypatch
) -> None:
    configure_auth(monkeypatch)
    called = {"retrieval": False}

    def fail_if_called(**kwargs):
        called["retrieval"] = True
        raise AssertionError("retrieval must not run for a tenant mismatch")

    monkeypatch.setattr(answers, "retrieve_for_answer", fail_if_called)

    response = client.post(
        "/api/v1/answer",
        headers=auth_header(),
        json={
            "tenant_id": str(TENANT_B),
            "question": "What should I check before rolling back?",
        },
    )

    assert response.status_code == 403
    assert response.json() == {"detail": "Forbidden"}
    assert called["retrieval"] is False


def test_forged_body_tenant_does_not_disclose_tenant_existence(
    real_auth, monkeypatch
) -> None:
    configure_auth(monkeypatch)
    responses = []
    for tenant_id in (TENANT_B, uuid4()):
        responses.append(
            client.post(
                "/api/v1/answer",
                headers=auth_header(),
                json={"tenant_id": str(tenant_id), "question": "anything"},
            )
        )

    assert [response.status_code for response in responses] == [403, 403]
    assert responses[0].json() == responses[1].json() == {"detail": "Forbidden"}


def test_authenticated_tenant_is_used_when_body_tenant_is_omitted(
    real_auth, monkeypatch
) -> None:
    configure_auth(monkeypatch)
    captured: dict = {}

    def fake_retrieve(**kwargs):
        captured.update(kwargs)
        return [], {"strategy": "hybrid", "limit": 5}

    monkeypatch.setattr(answers, "retrieve_for_answer", fake_retrieve)
    patch_generation(monkeypatch, RecordingProvider())

    response = client.post(
        "/api/v1/answer",
        headers=auth_header(),
        json={"question": "What should I check before rolling back?"},
    )

    assert response.status_code == 200
    assert captured["tenant_id"] == TENANT_A
    assert captured["principal"].tenant_id == TENANT_A


def test_cross_tenant_ingest_is_forbidden(real_auth, monkeypatch) -> None:
    configure_auth(monkeypatch)

    response = client.post(
        "/api/v1/documents/ingest",
        headers=auth_header(),
        json={
            "tenant_id": str(TENANT_B),
            "source_key": "runbooks/foreign.md",
            "title": "Foreign runbook",
            "content": "This must never be written into tenant B.",
        },
    )

    assert response.status_code == 403
    assert response.json() == {"detail": "Forbidden"}


# ---------------------------------------------------------------------------
# Retrieval and citation isolation
# ---------------------------------------------------------------------------


def test_retrieval_defense_in_depth_rejects_a_foreign_tenant_row(monkeypatch) -> None:
    captured: dict = {}
    monkeypatch.setattr(
        retrieval,
        "get_engine",
        lambda: FakeEngine(
            captured,
            [{"tenant_id": TENANT_B, "source_key": "foreign.md"}],
        ),
    )

    with pytest.raises(retrieval.TenantIsolationError):
        retrieval.lexical_search(
            tenant_id=TENANT_A,
            query="PAY-503",
            limit=5,
            principal=Principal(tenant_id=TENANT_A, subject="alice"),
        )


def test_retrieval_sql_applies_tenant_and_document_acl_scope(monkeypatch) -> None:
    captured: dict = {}
    monkeypatch.setattr(retrieval, "get_engine", lambda: FakeEngine(captured))
    principal = Principal(
        tenant_id=TENANT_A,
        subject="alice",
        user_id=UUID("33333333-3333-4333-8333-333333333333"),
        roles=frozenset({"member"}),
        groups=frozenset({"on-call"}),
    )

    retrieval.keyword_search(
        tenant_id=TENANT_A,
        query="payment",
        limit=5,
        principal=principal,
    )

    sql = " ".join(captured["sql"].lower().split())
    assert "c.tenant_id = :tenant_id" in sql
    assert "d.tenant_id = :tenant_id" in sql
    assert "document_acl" in sql
    assert "acl.permission in ('read', 'manage')" in sql
    assert "acl.principal_type = 'user'" in sql
    assert "acl.principal_type = 'group'" in sql
    assert "acl.principal_type = 'role'" in sql

    params = captured["params"]
    assert params["tenant_id"] == TENANT_A
    assert params["acl_user_ids"] == [
        "alice",
        "33333333-3333-4333-8333-333333333333",
    ]
    assert params["acl_groups"] == ["on-call"]
    assert params["acl_roles"] == ["member"]


def test_acl_filter_is_absent_for_internal_calls_without_a_principal(monkeypatch) -> None:
    captured: dict = {}
    monkeypatch.setattr(retrieval, "get_engine", lambda: FakeEngine(captured))

    retrieval.keyword_search(tenant_id=TENANT_A, query="payment", limit=5)

    assert "document_acl" not in captured["sql"]


def test_acl_filter_predicate_requires_an_explicit_grant_when_acls_exist() -> None:
    sql = " ".join(acl_filter_sql("d").lower().split())

    assert "not exists" in sql
    assert "acl.permission in ('read', 'manage')" in sql
    assert "acl.principal_type = 'user'" in sql


def test_version_analysis_cannot_relabel_supplied_evidence_metadata() -> None:
    result = make_result(source_key="runbooks/payment-api.md", version_number=2)
    bundle = assemble_evidence([result], context_token_budget=4000)
    forged = grounded_payload(
        version_analysis={
            "versions_identified": [
                {
                    "source_key": "secrets/other-tenant.md",
                    "version_number": 99,
                    "status": "current",
                    "evidence_ids": ["E1"],
                }
            ],
            "differences": [],
            "conflicting_instructions": [],
            "recommendation_applies_to": "version 99",
            "supporting_evidence_ids": ["E1"],
        }
    )

    with pytest.raises(CitationValidationError, match="does not match supplied evidence"):
        parse_model_answer(forged, bundle)


def test_citations_are_built_from_server_evidence_not_model_metadata(
    monkeypatch, set_identity
) -> None:
    set_identity(TENANT_A)
    result = make_result(source_key="runbooks/tenant-a.md", version_number=2)
    monkeypatch.setattr(
        answers,
        "retrieve_for_answer",
        lambda **kwargs: ([result], {"strategy": "hybrid", "limit": 5}),
    )
    # The model may only name an evidence ID; it cannot supply metadata.
    patch_generation(monkeypatch, RecordingProvider())

    response = client.post(
        "/api/v1/answer",
        json={
            "tenant_id": str(TENANT_A),
            "question": "What should I check before rolling back?",
        },
    )

    assert response.status_code == 200
    citation = response.json()["citations"][0]
    assert citation["document_id"] == str(result["document_id"])
    assert citation["source_key"] == "runbooks/tenant-a.md"
    assert citation["version_number"] == 2


# ---------------------------------------------------------------------------
# Observability
# ---------------------------------------------------------------------------


def test_request_log_contains_operational_fields_and_no_sensitive_content(
    monkeypatch, request_logs, set_identity
) -> None:
    set_identity(TENANT_A)
    secret_content = "TOPSECRET DOCUMENT BODY MUST NOT BE LOGGED"
    secret_question = "TOPSECRET QUESTION MUST NOT BE LOGGED"
    result = make_result(content=secret_content)
    monkeypatch.setattr(
        answers,
        "retrieve_for_answer",
        lambda **kwargs: ([result], {"strategy": "hybrid", "limit": 5}),
    )
    patch_generation(
        monkeypatch,
        RecordingProvider(usage={"prompt_tokens": 120, "completion_tokens": 30}),
    )

    response = client.post(
        "/api/v1/answer",
        json={"tenant_id": str(TENANT_A), "question": secret_question},
    )

    assert response.status_code == 200
    event = request_logs[-1].event_data
    assert event["request_id"] == response.headers["X-Request-ID"]
    assert event["path"] == "/api/v1/answer"
    assert event["status_code"] == 200
    assert event["retrieval_strategy"] == "hybrid"
    assert isinstance(event["retrieval_latency_ms"], int)
    assert isinstance(event["evidence_latency_ms"], int)
    assert isinstance(event["generation_latency_ms"], int)
    assert isinstance(event["total_latency_ms"], int)
    assert event["provider"] == "fake"
    assert event["observed_model"] == "fake-model"
    assert event["retry_count"] == 0
    assert event["input_tokens"] == 120
    assert event["output_tokens"] == 30
    assert event["token_source"] == "actual"
    assert event["error_category"] is None
    assert event["tenant_ref"].startswith("t_")
    assert str(TENANT_A) not in json.dumps(event)
    serialized = json.dumps(event)
    assert secret_content not in serialized
    assert secret_question not in serialized


def test_token_usage_is_marked_estimated_when_provider_does_not_report(
    monkeypatch, request_logs, set_identity
) -> None:
    set_identity(TENANT_A)
    monkeypatch.setattr(
        answers,
        "retrieve_for_answer",
        lambda **kwargs: ([make_result()], {"strategy": "hybrid", "limit": 5}),
    )
    patch_generation(monkeypatch, RecordingProvider())

    response = client.post(
        "/api/v1/answer",
        json={"tenant_id": str(TENANT_A), "question": "What happens next?"},
    )

    assert response.status_code == 200
    event = request_logs[-1].event_data
    assert event["token_source"] == "estimated"
    assert event["input_tokens"] > 0
    assert event["output_tokens"] > 0


def test_provider_error_is_not_leaked_into_the_response_or_request_log(
    monkeypatch, request_logs, set_identity
) -> None:
    set_identity(TENANT_A)
    secret = "api_key=sk-live-do-not-log tenant=tenant-b-internal"
    monkeypatch.setattr(
        answers,
        "retrieve_for_answer",
        lambda **kwargs: ([make_result()], {"strategy": "hybrid", "limit": 5}),
    )
    patch_generation(
        monkeypatch,
        RecordingProvider(error=GenerationProviderError(secret, retryable=True)),
    )

    response = client.post(
        "/api/v1/answer",
        json={"tenant_id": str(TENANT_A), "question": "What happens next?"},
    )

    assert response.status_code == 502
    assert response.json() == {"detail": "Answer generation provider failed"}
    assert secret not in response.text
    event = request_logs[-1].event_data
    assert event["error_category"] == "provider_error"
    assert event["status_code"] == 502
    assert secret not in json.dumps(event)
    assert "sk-live" not in json.dumps(event)


def test_bearer_token_is_never_logged(real_auth, monkeypatch, request_logs) -> None:
    configure_auth(monkeypatch)

    from app.api import search

    monkeypatch.setattr(search, "semantic_search", lambda **kwargs: [])

    response = client.post(
        "/api/v1/search",
        headers=auth_header(),
        json={"tenant_id": str(TENANT_A), "query": "payment errors"},
    )

    assert response.status_code == 200
    event = request_logs[-1].event_data
    assert event["tenant_ref"].startswith("t_")
    serialized = json.dumps(event)
    assert TOKEN_A not in serialized
    assert "alice" not in serialized
    assert str(TENANT_A) not in serialized


# ---------------------------------------------------------------------------
# Bounded request limits
# ---------------------------------------------------------------------------


def test_rate_limit_returns_429_with_retry_after(monkeypatch, set_identity) -> None:
    from app.api import search
    from app.core import limits as limits_module

    set_identity(TENANT_A)
    reset_limiters()
    monkeypatch.setattr(
        limits_module,
        "get_settings",
        lambda: Settings(
            rate_limit_per_minute=1,
            rate_limit_window_seconds=60.0,
            max_concurrent_requests=0,
        ),
    )
    monkeypatch.setattr(search, "semantic_search", lambda **kwargs: [])

    first = client.post(
        "/api/v1/search",
        json={"tenant_id": str(TENANT_A), "query": "payment errors"},
    )
    second = client.post(
        "/api/v1/search",
        json={"tenant_id": str(TENANT_A), "query": "payment errors"},
    )

    assert first.status_code == 200
    assert second.status_code == 429
    assert second.json() == {"detail": "Too many requests"}
    assert int(second.headers["Retry-After"]) >= 1


def test_concurrency_limit_returns_429_when_no_slot_is_available(monkeypatch) -> None:
    from app.api import search
    from app.core import limits as limits_module

    reset_limiters()
    settings = Settings(
        rate_limit_per_minute=0,
        max_concurrent_requests=1,
        concurrency_acquire_timeout_seconds=0.0,
    )
    monkeypatch.setattr(limits_module, "get_settings", lambda: settings)
    monkeypatch.setattr(search, "semantic_search", lambda **kwargs: [])

    semaphore = get_concurrency_semaphore(settings)
    assert semaphore is not None
    semaphore.acquire()
    try:
        response = client.post(
            "/api/v1/search",
            json={"tenant_id": str(TENANT_A), "query": "payment errors"},
        )
    finally:
        semaphore.release()

    assert response.status_code == 429
    assert response.json() == {"detail": "Too many concurrent requests"}

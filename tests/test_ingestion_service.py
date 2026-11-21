"""Characterization tests for synchronous ingestion (Phase F, Step 1).

These tests pin the current behavior of ``app.services.ingestion.ingest_document``
and the existing ingest endpoint before the staged, job-driven refactor.

They are deliberately DB-backed. The current contract depends on PostgreSQL
uniqueness constraints, ``SELECT ... FOR UPDATE`` row locks, transaction
rollback, and real serialization/JSON behavior that mocks cannot reproduce.

Phase F, Step 3 contract note: a new version stays non-current while its chunks
are embedded and becomes current only after the activation transaction verifies
that every chunk has a valid embedding. These tests use deterministic local
embeddings; they never call a paid provider.

Every test uses a throwaway tenant and deletes it afterwards. The seeded local
corpus is never read or modified.
"""

from __future__ import annotations

import hashlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from app.db.connections import check_postgres, get_engine
from app.main import app
from app.services.chunking import chunk_text, normalize_document_text
from app.services.ingestion import (
    EmptyDocumentError,
    TenantNotFoundError,
    ingest_document,
)
from tests.conftest import principal_for

client = TestClient(app)
INGEST_URL = "/api/v1/documents/ingest"

RESPONSE_KEYS = {
    "document_id",
    "version_id",
    "version_number",
    "content_hash",
    "chunk_count",
    "deduplicated",
    "is_current",
    "embedding_status",
}


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def db_engine():
    """Skip DB-backed tests when local PostgreSQL is unavailable."""

    last_error: Exception | None = None
    for attempt in range(3):
        try:
            check_postgres()
        except Exception as exc:  # noqa: BLE001 - absence must skip, not error
            last_error = exc
            if attempt < 2:
                time.sleep(0.25)
            continue
        return get_engine()

    assert last_error is not None
    pytest.skip(
        f"PostgreSQL is not available for ingestion integration tests: {type(last_error).__name__}"
    )


@pytest.fixture(autouse=True)
def _deterministic_embedding(deterministic_embedding):
    return deterministic_embedding


@pytest.fixture
def tenant_id(db_engine) -> UUID:
    """Create a throwaway tenant; deleting it cascades to all owned rows."""

    tenant = uuid4()
    with db_engine.begin() as connection:
        connection.execute(
            text("INSERT INTO tenants (id, name) VALUES (:id, :name)"),
            {"id": tenant, "name": f"phase-f-ingestion-{tenant}"},
        )
    try:
        yield tenant
    finally:
        with db_engine.begin() as connection:
            connection.execute(
                text("DELETE FROM tenants WHERE id = :id"),
                {"id": tenant},
            )


def _long_content(prefix: str, lines: int = 120) -> str:
    """Return content long enough to produce multiple chunks (> 1200 chars)."""

    return "\n".join(
        f"{prefix} step {index}: verify dependency {index} before escalation."
        for index in range(lines)
    )


def _ingest(
    tenant_id: UUID,
    *,
    source_key: str = "runbooks/phase-f-characterization.md",
    content: str = "Investigate PAY-503 before escalation.",
    title: str = "Phase F characterization runbook",
    source_type: str = "runbook",
    **overrides,
) -> dict:
    kwargs = {
        "principal": principal_for(tenant_id),
        "tenant_id": tenant_id,
        "source_key": source_key,
        "title": title,
        "source_type": source_type,
        "content": content,
    }
    kwargs.update(overrides)
    return ingest_document(**kwargs)


def _document(tenant_id: UUID, source_key: str) -> dict | None:
    with get_engine().connect() as connection:
        row = (
            connection.execute(
                text(
                    """
                SELECT id, tenant_id, source_key, title, source_type,
                       current_version_id, status
                FROM documents
                WHERE tenant_id = :tenant_id AND source_key = :source_key
                """
                ),
                {"tenant_id": tenant_id, "source_key": source_key},
            )
            .mappings()
            .first()
        )
    return dict(row) if row is not None else None


def _versions(document_id: UUID) -> list[dict]:
    with get_engine().connect() as connection:
        rows = (
            connection.execute(
                text(
                    """
                SELECT id, version_number, content_hash, source_uri,
                       authority_level, is_current, published_at, ingested_at
                FROM document_versions
                WHERE document_id = :document_id
                ORDER BY version_number
                """
                ),
                {"document_id": document_id},
            )
            .mappings()
            .all()
        )
    return [dict(row) for row in rows]


def _chunks(version_id: UUID) -> list[dict]:
    with get_engine().connect() as connection:
        rows = (
            connection.execute(
                text(
                    """
                SELECT chunk_index, content, content_hash, embedding,
                       embedding_model, embedded_at
                FROM chunks
                WHERE document_version_id = :version_id
                ORDER BY chunk_index
                """
                ),
                {"version_id": version_id},
            )
            .mappings()
            .all()
        )
    return [dict(row) for row in rows]


def _document_row_count(tenant_id: UUID) -> int:
    with get_engine().connect() as connection:
        return connection.execute(
            text("SELECT count(*) FROM documents WHERE tenant_id = :tenant_id"),
            {"tenant_id": tenant_id},
        ).scalar_one()


def _chunk_row_count(document_id: UUID) -> int:
    with get_engine().connect() as connection:
        return connection.execute(
            text("SELECT count(*) FROM chunks WHERE document_id = :document_id"),
            {"document_id": document_id},
        ).scalar_one()


# ---------------------------------------------------------------------------
# 1. New document ingestion
# ---------------------------------------------------------------------------


def test_new_document_creates_document_version_and_chunks(tenant_id: UUID) -> None:
    content = _long_content("New runbook")
    normalized = normalize_document_text(content)
    expected_chunks = chunk_text(normalized)

    result = _ingest(
        tenant_id,
        source_key="runbooks/new.md",
        title="New Runbook",
        content=content,
        source_uri="https://example.test/runbooks/new",
        authority_level=70,
    )

    assert set(result) == RESPONSE_KEYS
    assert isinstance(result["document_id"], UUID)
    assert isinstance(result["version_id"], UUID)
    assert result["version_number"] == 1
    assert result["content_hash"] == hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    assert result["chunk_count"] == len(expected_chunks)
    assert len(expected_chunks) > 1
    assert result["deduplicated"] is False
    assert result["is_current"] is True
    assert result["embedding_status"] == "complete"

    document = _document(tenant_id, "runbooks/new.md")
    assert document is not None
    assert document["id"] == result["document_id"]
    assert document["current_version_id"] == result["version_id"]
    assert document["status"] == "active"
    assert document["title"] == "New Runbook"
    assert document["source_type"] == "runbook"

    versions = _versions(document["id"])
    assert len(versions) == 1
    version = versions[0]
    assert version["id"] == result["version_id"]
    assert version["version_number"] == 1
    assert version["content_hash"] == result["content_hash"]
    assert version["source_uri"] == "https://example.test/runbooks/new"
    assert version["authority_level"] == 70
    assert version["is_current"] is True
    assert version["published_at"] is None

    chunks = _chunks(version["id"])
    assert len(chunks) == result["chunk_count"]
    assert [chunk["chunk_index"] for chunk in chunks] == list(range(len(expected_chunks)))
    assert [chunk["content"] for chunk in chunks] == expected_chunks
    assert [chunk["content_hash"] for chunk in chunks] == [
        hashlib.sha256(chunk.encode("utf-8")).hexdigest() for chunk in expected_chunks
    ]

    # Phase F, Step 3: activation requires complete, model-tagged embeddings.
    assert all(chunk["embedding"] is not None for chunk in chunks)
    assert {chunk["embedding_model"] for chunk in chunks} == {
        "sentence-transformers/all-MiniLM-L6-v2"
    }
    assert all(chunk["embedded_at"] is not None for chunk in chunks)


def test_new_document_defaults_authority_and_source_uri(tenant_id: UUID) -> None:
    result = _ingest(tenant_id, source_key="runbooks/defaults.md")

    document = _document(tenant_id, "runbooks/defaults.md")
    assert document is not None
    version = _versions(document["id"])[0]
    assert version["authority_level"] == 50
    assert version["source_uri"] is None
    assert result["version_number"] == 1


# ---------------------------------------------------------------------------
# 2. Duplicate content
# ---------------------------------------------------------------------------


def test_duplicate_content_reuses_version_and_refreshes_document_metadata(
    tenant_id: UUID,
) -> None:
    content = _long_content("Duplicate runbook")
    first = _ingest(
        tenant_id,
        source_key="runbooks/duplicate.md",
        title="Original title",
        content=content,
    )
    chunks_before = _chunks(first["version_id"])

    second = _ingest(
        tenant_id,
        source_key="runbooks/duplicate.md",
        title="Renamed title",
        source_type="api_spec",
        content=content,
    )

    assert second["deduplicated"] is True
    assert second["document_id"] == first["document_id"]
    assert second["version_id"] == first["version_id"]
    assert second["version_number"] == first["version_number"] == 1
    assert second["content_hash"] == first["content_hash"]
    assert second["chunk_count"] == first["chunk_count"]
    assert second["is_current"] is True

    assert len(_versions(first["document_id"])) == 1
    assert _chunks(first["version_id"]) == chunks_before

    document = _document(tenant_id, "runbooks/duplicate.md")
    assert document is not None
    assert document["title"] == "Renamed title"
    assert document["source_type"] == "api_spec"
    assert document["current_version_id"] == first["version_id"]


# ---------------------------------------------------------------------------
# 3. Changed content
# ---------------------------------------------------------------------------


def test_changed_content_creates_next_version_and_demotes_previous(
    tenant_id: UUID,
) -> None:
    first = _ingest(
        tenant_id,
        source_key="runbooks/changed.md",
        content=_long_content("Original version"),
    )
    first_chunks = _chunks(first["version_id"])

    second = _ingest(
        tenant_id,
        source_key="runbooks/changed.md",
        content=_long_content("Replacement version"),
    )

    assert second["deduplicated"] is False
    assert second["document_id"] == first["document_id"]
    assert second["version_id"] != first["version_id"]
    assert second["version_number"] == 2

    versions = {version["version_number"]: version for version in _versions(first["document_id"])}
    assert set(versions) == {1, 2}
    assert versions[1]["id"] == first["version_id"]
    assert versions[1]["is_current"] is False
    assert versions[2]["id"] == second["version_id"]
    assert versions[2]["is_current"] is True

    document = _document(tenant_id, "runbooks/changed.md")
    assert document is not None
    assert document["current_version_id"] == second["version_id"]

    # Historical chunks remain untouched.
    assert _chunks(first["version_id"]) == first_chunks
    assert len(_chunks(second["version_id"])) == second["chunk_count"]
    assert _chunk_row_count(first["document_id"]) == (first["chunk_count"] + second["chunk_count"])


def test_version_numbers_continue_to_increment(tenant_id: UUID) -> None:
    results = [
        _ingest(
            tenant_id,
            source_key="runbooks/increment.md",
            content=_long_content(f"Revision {revision}"),
        )
        for revision in range(3)
    ]

    assert [result["version_number"] for result in results] == [1, 2, 3]
    document = _document(tenant_id, "runbooks/increment.md")
    assert document is not None
    versions = _versions(document["id"])
    assert [version["version_number"] for version in versions] == [1, 2, 3]
    assert [version["is_current"] for version in versions] == [False, False, True]


# ---------------------------------------------------------------------------
# 4. Historical reactivation
# ---------------------------------------------------------------------------


def test_resubmitting_historical_content_reactivates_it_without_new_version(
    tenant_id: UUID,
) -> None:
    original = _long_content("Original snapshot")
    replacement = _long_content("Replacement snapshot")

    first = _ingest(tenant_id, source_key="runbooks/reactivate.md", content=original)
    second = _ingest(tenant_id, source_key="runbooks/reactivate.md", content=replacement)
    third = _ingest(tenant_id, source_key="runbooks/reactivate.md", content=original)

    assert third["deduplicated"] is True
    assert third["document_id"] == first["document_id"]
    assert third["version_id"] == first["version_id"]
    assert third["version_number"] == 1
    assert third["content_hash"] == first["content_hash"]
    assert third["is_current"] is True

    document = _document(tenant_id, "runbooks/reactivate.md")
    assert document is not None
    assert document["current_version_id"] == first["version_id"]

    versions = {version["version_number"]: version for version in _versions(document["id"])}
    assert len(versions) == 2
    assert versions[1]["is_current"] is True
    assert versions[2]["is_current"] is False
    assert versions[2]["id"] == second["version_id"]

    # Reactivation does not delete the superseded version's chunks.
    assert len(_chunks(second["version_id"])) == second["chunk_count"]
    assert len(_chunks(first["version_id"])) == first["chunk_count"]


# ---------------------------------------------------------------------------
# 5. Tenant validation
# ---------------------------------------------------------------------------


def test_unknown_tenant_is_rejected_without_writes(db_engine) -> None:
    missing_tenant = uuid4()

    with pytest.raises(TenantNotFoundError):
        _ingest(missing_tenant, source_key="runbooks/unknown-tenant.md")

    with db_engine.connect() as connection:
        assert (
            connection.execute(
                text("SELECT count(*) FROM documents WHERE tenant_id = :tenant_id"),
                {"tenant_id": missing_tenant},
            ).scalar_one()
            == 0
        )
        assert (
            connection.execute(
                text("SELECT count(*) FROM tenants WHERE id = :tenant_id"),
                {"tenant_id": missing_tenant},
            ).scalar_one()
            == 0
        )


# ---------------------------------------------------------------------------
# 6. Empty or whitespace-only content
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("content", ["", "   ", "\n\t\r\n  "])
def test_empty_or_whitespace_content_raises_and_writes_nothing(
    tenant_id: UUID,
    content: str,
) -> None:
    source_key = "runbooks/empty.md"

    with pytest.raises(EmptyDocumentError, match="non-whitespace text"):
        _ingest(tenant_id, source_key=source_key, content=content)

    assert _document(tenant_id, source_key) is None
    assert _document_row_count(tenant_id) == 0


# ---------------------------------------------------------------------------
# 7. Transaction rollback
# ---------------------------------------------------------------------------


def _fail_on_chunk_hash(monkeypatch, *, fail_at_call: int) -> None:
    """Make ingestion fail while hashing a chunk, after earlier chunks inserted."""

    from app.services import ingestion as ingestion_module

    real_sha256 = ingestion_module.sha256_text
    calls = 0

    def flaky_sha256(value: str) -> str:
        nonlocal calls
        calls += 1
        if calls >= fail_at_call:
            raise RuntimeError("simulated chunk persistence failure")
        return real_sha256(value)

    monkeypatch.setattr(ingestion_module, "sha256_text", flaky_sha256)


def test_chunk_failure_rolls_back_and_keeps_last_known_good(
    tenant_id: UUID,
    monkeypatch,
) -> None:
    source_key = "runbooks/rollback.md"
    first = _ingest(
        tenant_id,
        source_key=source_key,
        title="Last known good",
        content=_long_content("Known good version"),
    )
    chunks_before = _chunks(first["version_id"])

    # Call 1 hashes the content. Call 2 hashes chunk 0 and inserts it. Call 3
    # raises before chunk 1 is inserted, leaving partial work in the transaction.
    _fail_on_chunk_hash(monkeypatch, fail_at_call=3)

    with pytest.raises(RuntimeError, match="simulated chunk persistence failure"):
        _ingest(
            tenant_id,
            source_key=source_key,
            title="Must roll back",
            content=_long_content("Broken replacement version"),
        )

    document = _document(tenant_id, source_key)
    assert document is not None
    assert document["current_version_id"] == first["version_id"]
    assert document["title"] == "Last known good"

    versions = _versions(document["id"])
    assert len(versions) == 1
    assert versions[0]["id"] == first["version_id"]
    assert versions[0]["is_current"] is True
    assert _chunks(first["version_id"]) == chunks_before
    assert _chunk_row_count(document["id"]) == first["chunk_count"]


def test_first_ingestion_failure_leaves_no_partial_records(
    tenant_id: UUID,
    monkeypatch,
) -> None:
    source_key = "runbooks/first-attempt-failure.md"
    _fail_on_chunk_hash(monkeypatch, fail_at_call=3)

    with pytest.raises(RuntimeError, match="simulated chunk persistence failure"):
        _ingest(
            tenant_id,
            source_key=source_key,
            content=_long_content("Never persisted"),
        )

    assert _document(tenant_id, source_key) is None
    assert _document_row_count(tenant_id) == 0


# ---------------------------------------------------------------------------
# 8. Concurrent ingestion against real PostgreSQL
# ---------------------------------------------------------------------------


def _run_concurrently(*submissions):
    barrier = threading.Barrier(len(submissions))

    def run(submission):
        barrier.wait(timeout=10)
        return submission()

    with ThreadPoolExecutor(max_workers=len(submissions)) as pool:
        futures = [pool.submit(run, submission) for submission in submissions]
        return [future.result(timeout=30) for future in futures]


def test_concurrent_identical_submissions_create_one_version(tenant_id: UUID) -> None:
    source_key = "runbooks/concurrent-identical.md"
    content = _long_content("Concurrent identical", lines=160)

    results = _run_concurrently(
        lambda: _ingest(tenant_id, source_key=source_key, content=content),
        lambda: _ingest(tenant_id, source_key=source_key, content=content),
    )

    assert len({result["version_id"] for result in results}) == 1
    assert sorted(result["deduplicated"] for result in results) == [False, True]
    assert {result["version_number"] for result in results} == {1}

    document = _document(tenant_id, source_key)
    assert document is not None
    versions = _versions(document["id"])
    assert len(versions) == 1
    assert versions[0]["is_current"] is True
    assert len(_chunks(versions[0]["id"])) == results[0]["chunk_count"]


def test_concurrent_different_submissions_create_two_versions_one_current(
    tenant_id: UUID,
) -> None:
    source_key = "runbooks/concurrent-different.md"
    first_content = _long_content("Concurrent variant A", lines=160)
    second_content = _long_content("Concurrent variant B", lines=160)

    results = _run_concurrently(
        lambda: _ingest(tenant_id, source_key=source_key, content=first_content),
        lambda: _ingest(tenant_id, source_key=source_key, content=second_content),
    )

    assert len({result["version_id"] for result in results}) == 2
    assert {result["deduplicated"] for result in results} == {False}
    assert {result["version_number"] for result in results} == {1, 2}

    document = _document(tenant_id, source_key)
    assert document is not None
    versions = _versions(document["id"])
    assert len(versions) == 2
    assert {version["version_number"] for version in versions} == {1, 2}
    assert sum(1 for version in versions if version["is_current"]) == 1

    current = next(version for version in versions if version["is_current"])
    assert document["current_version_id"] == current["id"]

    by_version_id = {result["version_id"]: result for result in results}
    for version in versions:
        result = by_version_id[version["id"]]
        assert version["content_hash"] == result["content_hash"]
        assert len(_chunks(version["id"])) == result["chunk_count"]


# ---------------------------------------------------------------------------
# Endpoint compatibility against real PostgreSQL
# ---------------------------------------------------------------------------


def test_ingest_endpoint_contract_against_real_database(
    tenant_id: UUID,
    set_identity,
) -> None:
    set_identity(tenant_id=tenant_id)
    payload = {
        "source_key": "runbooks/endpoint-contract.md",
        "title": "Endpoint contract",
        "source_type": "runbook",
        "content": _long_content("Endpoint contract"),
        "source_uri": "https://example.test/endpoint-contract",
        "authority_level": 65,
    }

    first = client.post(INGEST_URL, json=payload)
    assert first.status_code == 201
    first_body = first.json()
    assert set(first_body) == RESPONSE_KEYS
    assert UUID(first_body["document_id"])
    assert UUID(first_body["version_id"])
    assert first_body["version_number"] == 1
    assert first_body["deduplicated"] is False
    assert first_body["is_current"] is True
    assert first_body["embedding_status"] == "complete"

    second = client.post(INGEST_URL, json=payload)
    assert second.status_code == 200
    second_body = second.json()
    assert second_body["document_id"] == first_body["document_id"]
    assert second_body["version_id"] == first_body["version_id"]
    assert second_body["deduplicated"] is True
    assert second_body["is_current"] is True

    document = _document(tenant_id, payload["source_key"])
    assert document is not None
    assert str(document["id"]) == first_body["document_id"]
    assert len(_versions(document["id"])) == 1

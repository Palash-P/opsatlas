"""Phase F, Step 3: embedding completeness and safe version activation.

These are real PostgreSQL tests. Embeddings are deterministic local values; no
paid provider or local model server is required. Every test uses a throwaway
tenant whose deletion cascades to documents, versions, chunks, and jobs.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from uuid import UUID, uuid4

import pytest
from sqlalchemy import inspect, text
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from app.db.connections import check_postgres, get_engine
from app.services import ingestion as ingestion_module
from app.services.embedding_client import (
    EmbeddingUnavailableError,
    EmbeddingValidationError,
    expected_dimensions,
    expected_model,
)
from app.services.ingestion import (
    ActivationIntegrityError,
    EmbeddingCompletenessError,
    _activate_version,
    ingest_document,
)
from app.services.retrieval import hybrid_search, keyword_search, semantic_search
from tests.conftest import deterministic_embeddings, principal_for

# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def db_engine():
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            check_postgres()
        except Exception as exc:  # noqa: BLE001 - absence must skip, not error
            last_error = exc
            if attempt < 2:
                time.sleep(0.25)
            continue

        engine = get_engine()
        job_columns = {column["name"] for column in inspect(engine).get_columns("ingestion_jobs")}
        document_columns = {column["name"] for column in inspect(engine).get_columns("documents")}
        if (
            "activation_generation" not in job_columns
            or "last_activation_generation" not in document_columns
        ):
            pytest.fail(
                "Activation tests require sql/004_activation_generation.sql",
                pytrace=False,
            )
        return engine

    assert last_error is not None
    pytest.skip(f"PostgreSQL is not available for activation tests: {type(last_error).__name__}")


@pytest.fixture(autouse=True)
def _deterministic_embedding(deterministic_embedding):
    return deterministic_embedding


@pytest.fixture
def tenant_id(db_engine) -> UUID:
    tenant = uuid4()
    with db_engine.begin() as connection:
        connection.execute(
            text("INSERT INTO tenants (id, name) VALUES (:id, :name)"),
            {"id": tenant, "name": f"phase-f-activation-{tenant}"},
        )
    try:
        yield tenant
    finally:
        with db_engine.begin() as connection:
            connection.execute(
                text("DELETE FROM tenants WHERE id = :id"),
                {"id": tenant},
            )


def _insert_tenant(engine) -> UUID:
    tenant = uuid4()
    with engine.begin() as connection:
        connection.execute(
            text("INSERT INTO tenants (id, name) VALUES (:id, :name)"),
            {"id": tenant, "name": f"phase-f-activation-other-{tenant}"},
        )
    return tenant


def _long_content(prefix: str, lines: int = 120) -> str:
    return "\n".join(
        f"{prefix} step {index}: verify dependency {index} before escalation."
        for index in range(lines)
    )


def _ingest(
    tenant_id: UUID,
    *,
    source_key: str = "runbooks/phase-f-activation.md",
    content: str = "Investigate PAY-503 before escalation.",
    title: str = "Phase F activation runbook",
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
                       current_version_id, status, last_activation_generation
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
                SELECT id, version_number, content_hash, is_current
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
                SELECT chunk_index, content, embedding, embedding_model,
                       embedded_at, vector_dims(embedding) AS dimensions
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


def _jobs(tenant_id: UUID) -> list[dict]:
    with get_engine().connect() as connection:
        rows = (
            connection.execute(
                text(
                    """
                SELECT id, document_id, version_id, status, attempts,
                       activation_generation, error_code, error_summary,
                       created_at, started_at, finished_at
                FROM ingestion_jobs
                WHERE tenant_id = :tenant_id
                ORDER BY created_at, id
                """
                ),
                {"tenant_id": tenant_id},
            )
            .mappings()
            .all()
        )
    return [dict(row) for row in rows]


def _job_for_version(tenant_id: UUID, version_id: UUID) -> dict:
    return next(job for job in _jobs(tenant_id) if job["version_id"] == version_id)


def _clear_embeddings(version_id: UUID) -> None:
    with get_engine().begin() as connection:
        connection.execute(
            text(
                """
                UPDATE chunks
                SET embedding = NULL,
                    embedding_model = NULL,
                    embedded_at = NULL
                WHERE document_version_id = :version_id
                """
            ),
            {"version_id": version_id},
        )


class RecordingEmbedder:
    """Deterministic embedder that records every batch the pipeline sends."""

    def __init__(self) -> None:
        self.batch_sizes: list[int] = []
        self.texts: list[str] = []

    @property
    def call_count(self) -> int:
        return len(self.batch_sizes)

    def __call__(self, texts: list[str]) -> list[list[float]]:
        self.batch_sizes.append(len(texts))
        self.texts.extend(texts)
        return deterministic_embeddings(texts)


# ---------------------------------------------------------------------------
# 1-3. Non-current until complete, every chunk embedded, batching > 32
# ---------------------------------------------------------------------------


def test_new_version_is_non_current_until_embeddings_complete(
    tenant_id: UUID,
    monkeypatch,
) -> None:
    observations: list[tuple[UUID | None, list[bool], str]] = []

    def inspecting_embedder(texts: list[str]) -> list[list[float]]:
        document = _document(tenant_id, "runbooks/activation-staging.md")
        assert document is not None
        versions = _versions(document["id"])
        observations.append(
            (
                document["current_version_id"],
                [version["is_current"] for version in versions],
                document["status"],
            )
        )
        return deterministic_embeddings(texts)

    monkeypatch.setattr(ingestion_module, "embed_texts", inspecting_embedder)

    result = _ingest(
        tenant_id,
        source_key="runbooks/activation-staging.md",
        content=_long_content("Staged activation", lines=200),
    )

    assert observations
    assert all(current is None for current, _flags, _status in observations)
    assert all(not any(flags) for _current, flags, _status in observations)
    assert all(status == "active" for _current, _flags, status in observations)

    document = _document(tenant_id, "runbooks/activation-staging.md")
    assert document is not None
    assert document["current_version_id"] == result["version_id"]
    assert result["is_current"] is True
    assert result["embedding_status"] == "complete"


def test_every_activated_chunk_has_a_complete_embedding(tenant_id: UUID) -> None:
    result = _ingest(
        tenant_id,
        source_key="runbooks/activation-complete.md",
        content=_long_content("Complete embeddings"),
    )

    chunks = _chunks(result["version_id"])
    assert len(chunks) == result["chunk_count"]
    assert all(chunk["embedding"] is not None for chunk in chunks)
    assert {chunk["embedding_model"] for chunk in chunks} == {expected_model()}
    assert all(chunk["embedded_at"] is not None for chunk in chunks)
    assert {chunk["dimensions"] for chunk in chunks} == {expected_dimensions()}


def test_more_than_32_chunks_are_embedded_in_multiple_batches(
    tenant_id: UUID,
    monkeypatch,
) -> None:
    embedder = RecordingEmbedder()
    monkeypatch.setattr(ingestion_module, "embed_texts", embedder)

    result = _ingest(
        tenant_id,
        source_key="runbooks/activation-batches.md",
        content=_long_content("Batch processing", lines=1200),
    )

    assert result["chunk_count"] > 32
    assert embedder.call_count > 1
    assert all(size <= 32 for size in embedder.batch_sizes)
    assert sum(embedder.batch_sizes) == result["chunk_count"]
    assert len(embedder.texts) == result["chunk_count"]


# ---------------------------------------------------------------------------
# 4-6. Invalid provider output is rejected
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("invalid_kind", ["count", "dimensions", "non_finite"])
def test_invalid_embedding_output_is_rejected(
    tenant_id: UUID,
    monkeypatch,
    invalid_kind: str,
) -> None:
    source_key = "runbooks/activation-invalid-output.md"
    first = _ingest(tenant_id, source_key=source_key, content=_long_content("Known good"))
    chunks_before = _chunks(first["version_id"])

    def invalid_embedder(texts: list[str]) -> list[list[float]]:
        vectors = deterministic_embeddings(texts)
        if invalid_kind == "count":
            return vectors[:-1]
        if invalid_kind == "dimensions":
            return [vector[:-1] for vector in vectors]
        broken = list(vectors[0])
        broken[0] = float("nan")
        return [broken, *vectors[1:]]

    monkeypatch.setattr(ingestion_module, "embed_texts", invalid_embedder)

    with pytest.raises(EmbeddingValidationError):
        _ingest(
            tenant_id,
            source_key=source_key,
            content=_long_content("Invalid embedding output"),
        )

    document = _document(tenant_id, source_key)
    assert document is not None
    assert document["current_version_id"] == first["version_id"]
    versions = _versions(document["id"])
    assert len(versions) == 2
    assert versions[0]["is_current"] is True
    assert versions[1]["is_current"] is False
    assert _chunks(first["version_id"]) == chunks_before

    failed_job = _job_for_version(tenant_id, versions[1]["id"])
    assert failed_job["status"] == "failed"
    assert failed_job["error_code"] == "embedding_invalid_response"
    assert all(chunk["embedding"] is None for chunk in _chunks(versions[1]["id"]))


# ---------------------------------------------------------------------------
# 7, 9. Partial failure and retry reuse
# ---------------------------------------------------------------------------


class FailingAfterOneBatch:
    def __init__(self) -> None:
        self.batches: list[list[str]] = []

    def __call__(self, texts: list[str]) -> list[list[float]]:
        self.batches.append(list(texts))
        if len(self.batches) == 2:
            raise EmbeddingUnavailableError("dependency timeout")
        return deterministic_embeddings(texts)


def test_partial_embedding_failure_leaves_previous_version_current(
    tenant_id: UUID,
    monkeypatch,
) -> None:
    source_key = "runbooks/activation-partial-failure.md"
    first = _ingest(tenant_id, source_key=source_key, content=_long_content("Known good"))
    failing = FailingAfterOneBatch()
    monkeypatch.setattr(ingestion_module, "embed_texts", failing)

    with pytest.raises(EmbeddingUnavailableError):
        _ingest(
            tenant_id,
            source_key=source_key,
            content=_long_content("Partially embedded", lines=1200),
        )

    assert len(failing.batches) == 2
    assert all(len(batch) <= 32 for batch in failing.batches)

    document = _document(tenant_id, source_key)
    assert document is not None
    assert document["current_version_id"] == first["version_id"]
    versions = _versions(document["id"])
    assert len(versions) == 2
    assert versions[0]["is_current"] is True
    assert versions[1]["is_current"] is False

    failed_chunks = _chunks(versions[1]["id"])
    embedded = [chunk for chunk in failed_chunks if chunk["embedding"] is not None]
    missing = [chunk for chunk in failed_chunks if chunk["embedding"] is None]
    assert len(embedded) == 32
    assert missing
    assert all(chunk["embedded_at"] is not None for chunk in embedded)

    failed_job = _job_for_version(tenant_id, versions[1]["id"])
    assert failed_job["status"] == "failed"
    assert failed_job["error_code"] == "embedding_service_unavailable"
    assert failed_job["error_summary"] == "Embedding service was unavailable"


def test_retry_reuses_inactive_version_and_valid_embeddings(
    tenant_id: UUID,
    monkeypatch,
) -> None:
    source_key = "runbooks/activation-retry.md"
    first = _ingest(tenant_id, source_key=source_key, content=_long_content("Known good"))
    failing = FailingAfterOneBatch()
    monkeypatch.setattr(ingestion_module, "embed_texts", failing)

    replacement = _long_content("Retryable replacement", lines=1200)
    with pytest.raises(EmbeddingUnavailableError):
        _ingest(tenant_id, source_key=source_key, content=replacement)

    document = _document(tenant_id, source_key)
    assert document is not None
    failed_version = _versions(document["id"])[1]
    chunks_after_failure = _chunks(failed_version["id"])
    first_batch_timestamps = {
        chunk["chunk_index"]: chunk["embedded_at"]
        for chunk in chunks_after_failure
        if chunk["embedding"] is not None
    }
    first_batch_indices = set(first_batch_timestamps)

    retry = RecordingEmbedder()
    monkeypatch.setattr(ingestion_module, "embed_texts", retry)
    result = _ingest(tenant_id, source_key=source_key, content=replacement)

    assert result["version_id"] == failed_version["id"]
    assert result["is_current"] is True
    assert result["deduplicated"] is True
    document = _document(tenant_id, source_key)
    assert document is not None
    assert document["current_version_id"] == failed_version["id"]
    assert len(_versions(document["id"])) == 2
    assert _job_for_version(tenant_id, failed_version["id"])["attempts"] == 2

    retried_texts = set(retry.texts)
    assert len(retry.texts) == result["chunk_count"] - len(first_batch_indices)
    assert all(size <= 32 for size in retry.batch_sizes)

    chunks_after_retry = _chunks(failed_version["id"])
    for chunk in chunks_after_retry:
        if chunk["chunk_index"] in first_batch_indices:
            assert chunk["embedded_at"] == first_batch_timestamps[chunk["chunk_index"]]
        assert chunk["embedding"] is not None
    # The retry must not have re-sent the first batch.
    assert not (
        {
            chunk["content"]
            for chunk in chunks_after_retry
            if chunk["chunk_index"] in first_batch_indices
        }
        & retried_texts
    )
    assert first["version_id"] != failed_version["id"]


# ---------------------------------------------------------------------------
# 8. Activation transaction rollback
# ---------------------------------------------------------------------------


def test_database_failure_during_activation_rolls_back_pointer_and_flags(
    tenant_id: UUID,
    monkeypatch,
) -> None:
    source_key = "runbooks/activation-rollback.md"
    first = _ingest(tenant_id, source_key=source_key, content=_long_content("Known good"))

    def exploding_finish(*args, **kwargs):
        raise SQLAlchemyError("simulated activation commit failure")

    monkeypatch.setattr(ingestion_module, "_mark_job_succeeded", exploding_finish)

    with pytest.raises(SQLAlchemyError):
        _ingest(
            tenant_id,
            source_key=source_key,
            content=_long_content("Activation must roll back"),
        )

    document = _document(tenant_id, source_key)
    assert document is not None
    assert document["current_version_id"] == first["version_id"]
    versions = _versions(document["id"])
    assert len(versions) == 2
    assert versions[0]["is_current"] is True
    assert versions[1]["is_current"] is False

    failed_job = _job_for_version(tenant_id, versions[1]["id"])
    assert failed_job["status"] == "failed"
    assert failed_job["error_code"] == "database_error"


# ---------------------------------------------------------------------------
# 10-11. Historical reactivation
# ---------------------------------------------------------------------------


def test_historical_reactivation_reuses_valid_embeddings(
    tenant_id: UUID,
    monkeypatch,
) -> None:
    source_key = "runbooks/activation-historical.md"
    original = _long_content("Historical snapshot")
    first = _ingest(tenant_id, source_key=source_key, content=original)
    second = _ingest(tenant_id, source_key=source_key, content=_long_content("Replacement"))
    chunks_before = _chunks(first["version_id"])

    embedder = RecordingEmbedder()
    monkeypatch.setattr(ingestion_module, "embed_texts", embedder)
    third = _ingest(tenant_id, source_key=source_key, content=original)

    assert third["deduplicated"] is True
    assert third["version_id"] == first["version_id"]
    assert third["is_current"] is True
    assert embedder.call_count == 0
    assert _chunks(first["version_id"]) == chunks_before

    document = _document(tenant_id, source_key)
    assert document is not None
    assert document["current_version_id"] == first["version_id"]
    versions = {version["version_number"]: version for version in _versions(document["id"])}
    assert len(versions) == 2
    assert versions[1]["is_current"] is True
    assert versions[2]["id"] == second["version_id"]
    assert versions[2]["is_current"] is False


def test_historical_version_without_embeddings_is_embedded_before_activation(
    tenant_id: UUID,
    monkeypatch,
) -> None:
    source_key = "runbooks/activation-historical-gap.md"
    original = _long_content("Historical unembedded snapshot")
    first = _ingest(tenant_id, source_key=source_key, content=original)
    _ingest(tenant_id, source_key=source_key, content=_long_content("Replacement"))
    _clear_embeddings(first["version_id"])

    embedder = RecordingEmbedder()
    monkeypatch.setattr(ingestion_module, "embed_texts", embedder)
    third = _ingest(tenant_id, source_key=source_key, content=original)

    assert third["version_id"] == first["version_id"]
    assert third["is_current"] is True
    assert sum(embedder.batch_sizes) == first["chunk_count"]
    assert embedder.call_count > 0
    assert all(chunk["embedding"] is not None for chunk in _chunks(first["version_id"]))

    document = _document(tenant_id, source_key)
    assert document is not None
    assert document["current_version_id"] == first["version_id"]
    assert len(_versions(document["id"])) == 2


# ---------------------------------------------------------------------------
# 12. A late older execution cannot replace a newer active version
# ---------------------------------------------------------------------------


def test_older_late_finishing_version_cannot_replace_newer_current(
    tenant_id: UUID,
    monkeypatch,
) -> None:
    source_key = "runbooks/activation-late-finish.md"
    older_content = _long_content("LATE-MARKER-OLDER", lines=200)
    newer_content = _long_content("LATE-MARKER-NEWER", lines=200)

    older_started = threading.Event()
    release_older = threading.Event()

    def gated_embedder(texts: list[str]) -> list[list[float]]:
        if any("LATE-MARKER-OLDER" in text for text in texts):
            older_started.set()
            if not release_older.wait(timeout=20):
                raise AssertionError("timed out waiting to release the older ingestion")
        return deterministic_embeddings(texts)

    monkeypatch.setattr(ingestion_module, "embed_texts", gated_embedder)

    with ThreadPoolExecutor(max_workers=1) as pool:
        older_future = pool.submit(
            _ingest,
            tenant_id,
            source_key=source_key,
            content=older_content,
        )
        assert older_started.wait(timeout=20)

        newer = _ingest(tenant_id, source_key=source_key, content=newer_content)
        assert newer["is_current"] is True

        release_older.set()
        older = older_future.result(timeout=30)

    assert older["is_current"] is False
    assert older["embedding_status"] == "complete"

    document = _document(tenant_id, source_key)
    assert document is not None
    assert document["current_version_id"] == newer["version_id"]
    versions = _versions(document["id"])
    assert len(versions) == 2
    assert sum(1 for version in versions if version["is_current"]) == 1
    current = next(version for version in versions if version["is_current"])
    assert current["id"] == newer["version_id"]

    assert _job_for_version(tenant_id, older["version_id"])["status"] == "succeeded"
    assert _job_for_version(tenant_id, newer["version_id"])["status"] == "succeeded"
    assert all(chunk["embedding"] is not None for chunk in _chunks(older["version_id"]))


# ---------------------------------------------------------------------------
# 13. At most one current version per document
# ---------------------------------------------------------------------------


def test_at_most_one_current_version_is_enforced_by_the_database(
    tenant_id: UUID,
) -> None:
    source_key = "runbooks/activation-unique-current.md"
    first = _ingest(tenant_id, source_key=source_key, content=_long_content("First"))
    second = _ingest(tenant_id, source_key=source_key, content=_long_content("Second"))

    document = _document(tenant_id, source_key)
    assert document is not None
    versions = _versions(document["id"])
    assert sum(1 for version in versions if version["is_current"]) == 1
    assert document["current_version_id"] == second["version_id"]

    with get_engine().connect() as connection:
        index_names = {
            row[0]
            for row in connection.execute(
                text(
                    "SELECT indexname FROM pg_indexes "
                    "WHERE schemaname = 'public' AND tablename = 'document_versions'"
                )
            )
        }
    assert "document_versions_one_current_idx" in index_names

    with pytest.raises(IntegrityError), get_engine().begin() as connection:
        connection.execute(
            text("UPDATE document_versions SET is_current = TRUE WHERE id = :version_id"),
            {"version_id": first["version_id"]},
        )

    versions = _versions(document["id"])
    assert sum(1 for version in versions if version["is_current"]) == 1
    assert (
        next(version for version in versions if version["is_current"])["id"]
        == (second["version_id"])
    )


# ---------------------------------------------------------------------------
# 14. A succeeded job is not downgraded by redundant execution
# ---------------------------------------------------------------------------


def test_redundant_success_is_a_pure_read(tenant_id: UUID, monkeypatch) -> None:
    source_key = "runbooks/activation-pure-read.md"
    content = _long_content("Idempotent success")
    first = _ingest(tenant_id, source_key=source_key, content=content)
    job_before = _job_for_version(tenant_id, first["version_id"])
    chunks_before = _chunks(first["version_id"])

    def exploding_embedder(texts: list[str]) -> list[list[float]]:
        raise AssertionError("a pure duplicate must not call the embedding service")

    monkeypatch.setattr(ingestion_module, "embed_texts", exploding_embedder)
    second = _ingest(tenant_id, source_key=source_key, content=content, title="Renamed")

    assert second["deduplicated"] is True
    assert second["version_id"] == first["version_id"]
    assert second["is_current"] is True
    job_after = _job_for_version(tenant_id, first["version_id"])
    assert job_after["status"] == "succeeded"
    assert job_after["attempts"] == job_before["attempts"] == 1
    assert job_after["finished_at"] == job_before["finished_at"]
    assert _chunks(first["version_id"]) == chunks_before


def test_failed_redundant_reactivation_does_not_downgrade_succeeded_job(
    tenant_id: UUID,
    monkeypatch,
) -> None:
    source_key = "runbooks/activation-preserve-success.md"
    original = _long_content("Original success")
    first = _ingest(tenant_id, source_key=source_key, content=original)
    second = _ingest(tenant_id, source_key=source_key, content=_long_content("Newer"))
    _clear_embeddings(first["version_id"])

    def exploding_embedder(texts: list[str]) -> list[list[float]]:
        raise EmbeddingUnavailableError("dependency unavailable")

    monkeypatch.setattr(ingestion_module, "embed_texts", exploding_embedder)

    with pytest.raises(EmbeddingUnavailableError):
        _ingest(tenant_id, source_key=source_key, content=original)

    job = _job_for_version(tenant_id, first["version_id"])
    assert job["status"] == "succeeded"
    assert job["error_code"] is None
    assert job["error_summary"] is None

    document = _document(tenant_id, source_key)
    assert document is not None
    assert document["current_version_id"] == second["version_id"]
    assert all(chunk["embedding"] is None for chunk in _chunks(first["version_id"]))


# ---------------------------------------------------------------------------
# 15. Cross-tenant activation is rejected
# ---------------------------------------------------------------------------


def test_cross_tenant_activation_is_rejected(tenant_id: UUID, db_engine) -> None:
    source_key = "runbooks/activation-tenant-boundary.md"
    result = _ingest(tenant_id, source_key=source_key, content=_long_content("Tenant A"))
    other_tenant = _insert_tenant(db_engine)
    document = _document(tenant_id, source_key)
    assert document is not None

    try:
        with pytest.raises(ActivationIntegrityError):
            _activate_version(
                job_id=_job_for_version(tenant_id, result["version_id"])["id"],
                tenant_id=other_tenant,
                document_id=result["document_id"],
                version_id=result["version_id"],
                version_number=result["version_number"],
                content_hash=result["content_hash"],
                expected_chunk_count=result["chunk_count"],
            )

        unchanged = _document(tenant_id, source_key)
        assert unchanged is not None
        assert unchanged["current_version_id"] == result["version_id"]
        assert (
            next(version for version in _versions(document["id"]) if version["is_current"])["id"]
            == result["version_id"]
        )
    finally:
        with db_engine.begin() as connection:
            connection.execute(
                text("DELETE FROM tenants WHERE id = :id"),
                {"id": other_tenant},
            )


# ---------------------------------------------------------------------------
# 17-18. Retrieval sees exactly the activated version
# ---------------------------------------------------------------------------


def test_keyword_semantic_and_hybrid_retrieval_use_the_activated_version(
    tenant_id: UUID,
) -> None:
    source_key = "runbooks/activation-retrieval.md"
    _ingest(
        tenant_id,
        source_key=source_key,
        content=_long_content("zqxoldmarker", lines=60),
    )
    second = _ingest(
        tenant_id,
        source_key=source_key,
        content=_long_content("zqxcurrentmarker", lines=60),
    )

    keyword = keyword_search(tenant_id=tenant_id, query="zqxcurrentmarker", limit=20)
    semantic = semantic_search(tenant_id=tenant_id, query="zqxcurrentmarker", limit=20)
    hybrid = hybrid_search(tenant_id=tenant_id, query="zqxcurrentmarker", limit=20)

    assert keyword
    assert semantic
    assert hybrid
    assert {result["version_id"] for result in keyword} == {second["version_id"]}
    assert {result["version_id"] for result in semantic} == {second["version_id"]}
    assert {result["version_id"] for result in hybrid} == {second["version_id"]}


def test_failed_incomplete_version_is_not_exposed_by_default_retrieval(
    tenant_id: UUID,
    monkeypatch,
) -> None:
    source_key = "runbooks/activation-hidden-failure.md"
    first = _ingest(
        tenant_id,
        source_key=source_key,
        content=_long_content("zqxvisiblecurrent", lines=60),
    )

    def invalid_embedder(texts: list[str]) -> list[list[float]]:
        return deterministic_embeddings(texts)[:-1]

    monkeypatch.setattr(ingestion_module, "embed_texts", invalid_embedder)
    with pytest.raises(EmbeddingValidationError):
        _ingest(
            tenant_id,
            source_key=source_key,
            content=_long_content("zqxhiddenfailedversion", lines=60),
        )

    document = _document(tenant_id, source_key)
    assert document is not None
    assert document["current_version_id"] == first["version_id"]

    keyword = keyword_search(tenant_id=tenant_id, query="zqxhiddenfailedversion", limit=20)
    semantic = semantic_search(
        tenant_id=tenant_id,
        query="zqxhiddenfailedversion",
        limit=20,
    )
    hybrid = hybrid_search(tenant_id=tenant_id, query="zqxhiddenfailedversion", limit=20)

    assert keyword == []
    assert semantic
    assert hybrid
    for results in (semantic, hybrid):
        assert {result["version_id"] for result in results} == {first["version_id"]}


# ---------------------------------------------------------------------------
# Defensive: a direct activation of an incomplete version is rejected
# ---------------------------------------------------------------------------


def test_direct_activation_rejects_incomplete_embeddings(
    tenant_id: UUID,
    monkeypatch,
) -> None:
    source_key = "runbooks/activation-direct-check.md"

    def invalid_embedder(texts: list[str]) -> list[list[float]]:
        return deterministic_embeddings(texts)[:-1]

    monkeypatch.setattr(ingestion_module, "embed_texts", invalid_embedder)
    with pytest.raises(EmbeddingValidationError):
        _ingest(tenant_id, source_key=source_key, content=_long_content("Incomplete"))

    document = _document(tenant_id, source_key)
    assert document is not None
    version = _versions(document["id"])[0]
    job = _job_for_version(tenant_id, version["id"])
    chunk_count = len(_chunks(version["id"]))

    with pytest.raises(EmbeddingCompletenessError):
        _activate_version(
            job_id=job["id"],
            tenant_id=tenant_id,
            document_id=document["id"],
            version_id=version["id"],
            version_number=version["version_number"],
            content_hash=version["content_hash"],
            expected_chunk_count=chunk_count,
        )

    assert _document(tenant_id, source_key)["current_version_id"] is None

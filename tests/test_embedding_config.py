"""Phase F, Step 4: one embedding dimension contract.

The tests cover the configuration boundary, provider response validation, the
retrieval query path, the live PostgreSQL column, and ingestion's refusal to act
on a drifted schema. They use deterministic embeddings where a document must be
ingested; no provider or model server is required.
"""

from __future__ import annotations

import time
from uuid import uuid4

import pytest
from pydantic import ValidationError
from sqlalchemy import text

from app.core.config import DATABASE_EMBEDDING_DIMENSIONS, Settings
from app.db.connections import check_postgres, get_engine
from app.db.embedding_schema import (
    EmbeddingSchemaError,
    database_embedding_dimensions,
    verify_embedding_schema,
)
from app.services import embedding_client, retrieval
from app.services import ingestion as ingestion_module
from app.services.embedding_client import (
    DEFAULT_EMBEDDING_DIMENSIONS,
    EmbeddingValidationError,
    embed_texts,
    expected_dimensions,
    expected_model,
    validate_embeddings,
    vector_sql_type,
)
from app.services.ingestion import ingest_document
from tests.conftest import principal_for


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
        return get_engine()

    assert last_error is not None
    pytest.skip(f"PostgreSQL is not available for schema tests: {type(last_error).__name__}")


def test_default_configuration_uses_the_384_dimensional_contract() -> None:
    settings = Settings(_env_file=None)

    assert DATABASE_EMBEDDING_DIMENSIONS == 384
    assert DEFAULT_EMBEDDING_DIMENSIONS == 384
    assert settings.embedding_dimensions == 384
    assert settings.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
    assert expected_dimensions() == 384
    assert expected_model() == "sentence-transformers/all-MiniLM-L6-v2"
    assert vector_sql_type() == "vector(384)"


def test_configuration_rejects_a_dimension_the_schema_cannot_store() -> None:
    with pytest.raises(ValidationError) as excinfo:
        Settings(_env_file=None, embedding_dimensions=768)

    message = str(excinfo.value)
    assert "embedding_dimensions" in message
    assert "schema migration" in message


def test_provider_response_with_wrong_vector_length_is_rejected(monkeypatch) -> None:
    monkeypatch.setattr(
        embedding_client,
        "_post",
        lambda payload: {
            "model": expected_model(),
            "dimensions": expected_dimensions(),
            "embeddings": [[0.0] * (expected_dimensions() - 1)],
        },
    )

    with pytest.raises(EmbeddingValidationError, match="dimensions"):
        embed_texts(["a runbook body"])


def test_provider_response_with_wrong_declared_dimension_is_rejected(monkeypatch) -> None:
    monkeypatch.setattr(
        embedding_client,
        "_post",
        lambda payload: {
            "model": expected_model(),
            "dimensions": expected_dimensions() + 1,
            "embeddings": [[0.0] * expected_dimensions()],
        },
    )

    with pytest.raises(EmbeddingValidationError, match="dimensions"):
        embed_texts(["a runbook body"])


def test_retrieval_rejects_a_query_vector_with_the_wrong_dimension(monkeypatch) -> None:
    connection_used = False

    class FakeEngine:
        def connect(self):
            nonlocal connection_used
            connection_used = True
            raise AssertionError("PostgreSQL must not see a mismatched query vector")

    monkeypatch.setattr(
        retrieval,
        "embed_texts",
        lambda texts: [[0.0] * (expected_dimensions() + 1)],
    )
    monkeypatch.setattr(retrieval, "get_engine", lambda: FakeEngine())

    with pytest.raises(EmbeddingValidationError, match="dimensions"):
        retrieval.semantic_search(
            tenant_id=uuid4(),
            query="payments failing after deployment",
            limit=5,
        )

    assert connection_used is False


def test_validate_embeddings_rejects_wrong_dimension_directly() -> None:
    with pytest.raises(EmbeddingValidationError, match="dimensions"):
        validate_embeddings(
            [[0.0] * (expected_dimensions() + 1)],
            expected_count=1,
            expected_dimensions=expected_dimensions(),
        )


def test_database_vector_column_matches_the_configured_dimension(db_engine) -> None:
    with db_engine.connect() as connection:
        declared = connection.execute(
            text(
                """
                SELECT format_type(a.atttypid, a.atttypmod)
                FROM pg_attribute AS a
                WHERE a.attrelid = 'chunks'::regclass
                  AND a.attname = 'embedding'
                  AND NOT a.attisdropped
                """
            )
        ).scalar_one()

        assert declared == f"vector({expected_dimensions()})"
        assert database_embedding_dimensions(connection) == expected_dimensions()
        assert verify_embedding_schema(connection) == expected_dimensions()

        incompatible_rows = connection.execute(
            text(
                """
                SELECT count(*)
                FROM chunks
                WHERE embedding IS NOT NULL
                  AND vector_dims(embedding) IS DISTINCT FROM :dimensions
                """
            ),
            {"dimensions": expected_dimensions()},
        ).scalar_one()
        assert incompatible_rows == 0


class _FakeScalarResult:
    def __init__(self, value: int | None) -> None:
        self.value = value

    def scalar_one_or_none(self) -> int | None:
        return self.value


class _FakeConnection:
    def __init__(self, typmod: int | None) -> None:
        self.typmod = typmod

    def execute(self, statement) -> _FakeScalarResult:
        return _FakeScalarResult(self.typmod)


@pytest.mark.parametrize("typmod", [768, None, -1])
def test_schema_drift_from_the_configuration_is_detected(typmod: int | None) -> None:
    with pytest.raises(EmbeddingSchemaError, match="do not match"):
        verify_embedding_schema(_FakeConnection(typmod), dimensions=384)


def test_ingestion_refuses_schema_drift_and_keeps_the_current_version(
    db_engine,
    deterministic_embedding,
    monkeypatch,
) -> None:
    tenant = uuid4()
    source_key = "runbooks/embedding-schema-contract.md"

    with db_engine.begin() as connection:
        connection.execute(
            text("INSERT INTO tenants (id, name) VALUES (:id, :name)"),
            {"id": tenant, "name": f"phase-f-embedding-schema-{tenant}"},
        )

    try:
        first = ingest_document(
            principal=principal_for(tenant),
            tenant_id=tenant,
            source_key=source_key,
            title="Embedding schema contract",
            source_type="runbook",
            content="Check PAY-503 before escalating to the payments on-call.",
        )

        with db_engine.connect() as connection:
            current_before = connection.execute(
                text(
                    """
                    SELECT current_version_id
                    FROM documents
                    WHERE tenant_id = :tenant_id AND source_key = :source_key
                    """
                ),
                {"tenant_id": tenant, "source_key": source_key},
            ).scalar_one()

        def reject_schema(connection) -> None:
            raise EmbeddingSchemaError("simulated schema drift")

        monkeypatch.setattr(ingestion_module, "verify_embedding_schema", reject_schema)

        with pytest.raises(EmbeddingSchemaError, match="schema drift"):
            ingest_document(
                principal=principal_for(tenant),
                tenant_id=tenant,
                source_key=source_key,
                title="Embedding schema contract",
                source_type="runbook",
                content="A different runbook body that would create version two.",
            )

        with db_engine.connect() as connection:
            current_after = connection.execute(
                text(
                    """
                    SELECT current_version_id
                    FROM documents
                    WHERE tenant_id = :tenant_id AND source_key = :source_key
                    """
                ),
                {"tenant_id": tenant, "source_key": source_key},
            ).scalar_one()
            latest_job = (
                connection.execute(
                    text(
                        """
                    SELECT status, error_code
                    FROM ingestion_jobs
                    WHERE tenant_id = :tenant_id
                    ORDER BY created_at DESC, id DESC
                    LIMIT 1
                    """
                    ),
                    {"tenant_id": tenant},
                )
                .mappings()
                .one()
            )

        assert current_before == first["version_id"]
        assert current_after == first["version_id"]
        assert latest_job["status"] == "failed"
        assert latest_job["error_code"] == "embedding_schema_mismatch"
    finally:
        with db_engine.begin() as connection:
            connection.execute(
                text("DELETE FROM tenants WHERE id = :id"),
                {"id": tenant},
            )

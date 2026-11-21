"""HTTP client for the local OpsAtlas embedding service."""

from __future__ import annotations

import json
import math
import os
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from app.core.config import (
    DATABASE_EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL_DEFAULT,
    get_settings,
)

# Backwards-compatible environment override used by evaluation scripts.
EMBEDDING_URL = os.getenv("OPSATLAS_EMBEDDING_URL")

# Defaults exposed for scripts and reports. The effective model and dimensions
# are read from application settings at call time so deployments can configure
# them without changing the provider implementation. Both defaults come from the
# single contract in app.core.config.
DEFAULT_EMBEDDING_MODEL = EMBEDDING_MODEL_DEFAULT
DEFAULT_EMBEDDING_DIMENSIONS = DATABASE_EMBEDDING_DIMENSIONS
EXPECTED_MODEL = DEFAULT_EMBEDDING_MODEL
EXPECTED_DIMENSIONS = DEFAULT_EMBEDDING_DIMENSIONS

# The embedding service accepts at most this many texts per request. Ingestion
# must batch longer documents instead of silently sending an oversized payload.
MAX_BATCH_SIZE = 32


class EmbeddingServiceError(RuntimeError):
    """Raised when the embedding service is unavailable or returns invalid data.

    ``retryable`` distinguishes a transient dependency failure (timeout,
    connection error, 429/5xx) from a permanent request or response problem.
    Callers map the two classes to different HTTP status codes and job failure
    codes without exposing provider details.
    """

    retryable = False


class EmbeddingUnavailableError(EmbeddingServiceError):
    """The embedding service could not be reached or asked the caller to retry."""

    retryable = True


class EmbeddingValidationError(EmbeddingServiceError):
    """The embedding service returned a malformed or unusable response."""


def _base_url() -> str:
    return (EMBEDDING_URL or get_settings().embedding_url).rstrip("/")


def expected_model() -> str:
    """Return the configured embedding model name."""

    return get_settings().embedding_model


def expected_dimensions() -> int:
    """Return the configured embedding vector dimension."""

    return get_settings().embedding_dimensions


def vector_sql_type() -> str:
    """Return the pgvector SQL type for the configured dimension.

    ``Settings`` rejects any dimension that differs from the fixed database
    contract, so this value is always compatible with ``chunks.embedding``.
    The interpolated value is an integer, never user-supplied text.
    """

    return f"vector({expected_dimensions()})"


def vector_literal(vector: list[float]) -> str:
    """Format a vector as the text literal accepted by a pgvector cast."""

    return "[" + ",".join(repr(float(value)) for value in vector) + "]"


def _post(payload: bytes) -> dict:
    settings = get_settings()
    request = Request(
        f"{_base_url()}/embed",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    attempts = 1 + max(0, settings.embedding_max_retries)
    backoff = max(0.0, settings.embedding_retry_backoff_seconds)

    for attempt in range(attempts):
        try:
            with urlopen(
                request,
                timeout=settings.embedding_timeout_seconds,
            ) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            retryable = exc.code in (408, 429) or exc.code >= 500
            if retryable and attempt + 1 < attempts:
                if backoff:
                    time.sleep(backoff * (2**attempt))
                continue
            if retryable:
                raise EmbeddingUnavailableError(
                    f"Embedding service returned HTTP {exc.code}"
                ) from exc
            raise EmbeddingServiceError(f"Embedding service returned HTTP {exc.code}") from exc
        except (URLError, TimeoutError, OSError) as exc:
            if attempt + 1 < attempts:
                if backoff:
                    time.sleep(backoff * (2**attempt))
                continue
            raise EmbeddingUnavailableError("Could not reach the local embedding service") from exc
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise EmbeddingValidationError("Embedding service returned invalid JSON") from exc

    # The loop either returns or raises; this keeps type checkers satisfied.
    raise EmbeddingUnavailableError("Embedding service is unavailable")


def validate_embeddings(
    vectors: Any,
    *,
    expected_count: int,
    expected_dimensions: int | None = None,
) -> list[list[float]]:
    """Validate a provider embedding response and return plain float vectors.

    This is shared by the HTTP client and ingestion. Ingestion calls it again
    after every batch so a test fake, a proxy, or a future client change cannot
    persist a response that violates the activation contract.
    """

    if expected_dimensions is None:
        expected_dimensions = get_settings().embedding_dimensions

    if expected_count < 1:
        raise ValueError("expected_count must be at least 1")

    if not isinstance(vectors, list) or len(vectors) != expected_count:
        raise EmbeddingValidationError("Embedding count does not match input count")

    validated: list[list[float]] = []

    for vector in vectors:
        if not isinstance(vector, list) or len(vector) != expected_dimensions:
            raise EmbeddingValidationError("Invalid embedding vector dimensions")

        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in vector
        ):
            raise EmbeddingValidationError("Embedding contains invalid numeric values")

        validated.append([float(value) for value in vector])

    return validated


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Return validated embeddings for at most ``MAX_BATCH_SIZE`` texts."""

    if not texts or len(texts) > MAX_BATCH_SIZE:
        raise ValueError(f"texts must contain between 1 and {MAX_BATCH_SIZE} items")

    if any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("every text must be a nonempty string")

    result = _post(json.dumps({"texts": texts}).encode("utf-8"))

    if not isinstance(result, dict):
        raise EmbeddingValidationError("Embedding service returned an invalid response")

    model = expected_model()
    dimensions = expected_dimensions()

    if result.get("model") != model:
        raise EmbeddingValidationError("Unexpected embedding model")

    if result.get("dimensions") != dimensions:
        raise EmbeddingValidationError("Unexpected embedding dimensions")

    return validate_embeddings(
        result.get("embeddings"),
        expected_count=len(texts),
        expected_dimensions=dimensions,
    )

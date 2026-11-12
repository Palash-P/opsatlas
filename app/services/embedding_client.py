"""HTTP client for the local OpsAtlas embedding service."""

from __future__ import annotations

import json
import math
import os
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from app.core.config import get_settings

# Backwards-compatible environment override used by evaluation scripts.
EMBEDDING_URL = os.getenv("OPSATLAS_EMBEDDING_URL")
EXPECTED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EXPECTED_DIMENSIONS = 384


class EmbeddingServiceError(RuntimeError):
    """Raised when the embedding service is unavailable or returns invalid data."""


def _base_url() -> str:
    return (EMBEDDING_URL or get_settings().embedding_url).rstrip("/")


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
            raise EmbeddingServiceError(
                f"Embedding service returned HTTP {exc.code}"
            ) from exc
        except (URLError, TimeoutError, OSError) as exc:
            if attempt + 1 < attempts:
                if backoff:
                    time.sleep(backoff * (2**attempt))
                continue
            raise EmbeddingServiceError(
                "Could not reach the local embedding service"
            ) from exc
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise EmbeddingServiceError(
                "Embedding service returned invalid JSON"
            ) from exc

    # The loop either returns or raises; this keeps type checkers satisfied.
    raise EmbeddingServiceError("Embedding service is unavailable")


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Return validated, normalized 384-dimensional embeddings."""

    if not texts or len(texts) > 32:
        raise ValueError("texts must contain between 1 and 32 items")

    if any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("every text must be a nonempty string")

    result = _post(json.dumps({"texts": texts}).encode("utf-8"))

    if result.get("model") != EXPECTED_MODEL:
        raise EmbeddingServiceError("Unexpected embedding model")

    if result.get("dimensions") != EXPECTED_DIMENSIONS:
        raise EmbeddingServiceError("Unexpected embedding dimensions")

    vectors = result.get("embeddings")
    if not isinstance(vectors, list) or len(vectors) != len(texts):
        raise EmbeddingServiceError("Embedding count does not match input count")

    validated: list[list[float]] = []

    for vector in vectors:
        if not isinstance(vector, list) or len(vector) != EXPECTED_DIMENSIONS:
            raise EmbeddingServiceError("Invalid embedding vector dimensions")

        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in vector
        ):
            raise EmbeddingServiceError("Embedding contains invalid numeric values")

        validated.append([float(value) for value in vector])

    return validated

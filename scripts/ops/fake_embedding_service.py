"""Deterministic local embedding service for tests and CI.

The production embedding service is a separate process the operator supplies.
This script implements the same tiny HTTP contract without a model download or
paid provider so live tests, CI, and Compose smoke checks can exercise the full
ingestion path deterministically:

    GET  /health -> {"status": "ok", "model": ..., "dimensions": ...}
    POST /embed  {"texts": ["...", ...]}
                 -> {"model": ..., "dimensions": 384, "embeddings": [[...], ...]}

Vectors are normalized and derived only from the text's SHA-256 digest, so the
same text always produces the same vector across processes. Request content is
never logged.

Run it directly:

    python scripts/ops/fake_embedding_service.py --port 8091 \
        --model sentence-transformers/all-MiniLM-L6-v2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

MAX_BATCH_SIZE = 32
DEFAULT_DIMENSIONS = 384


def deterministic_vector(text: str, dimensions: int) -> list[float]:
    seed = hashlib.sha256(text.encode("utf-8")).digest()
    values: list[float] = []
    counter = 0
    while len(values) < dimensions:
        digest = hashlib.sha256(seed + counter.to_bytes(4, "big")).digest()
        for offset in range(0, 32, 4):
            raw = int.from_bytes(digest[offset : offset + 4], "big")
            values.append((raw / 2**32) * 2.0 - 1.0)
        counter += 1
    values = values[:dimensions]
    norm = math.sqrt(sum(value * value for value in values)) or 1.0
    return [value / norm for value in values]


def _handler(model: str, dimensions: int):
    class Handler(BaseHTTPRequestHandler):
        server_version = "OpsAtlasFakeEmbedding/1.0"

        def _send(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path.rstrip("/") == "/health":
                self._send(
                    200,
                    {"status": "ok", "model": model, "dimensions": dimensions},
                )
            else:
                self._send(404, {"detail": "not found"})

        def do_POST(self) -> None:
            if self.path.rstrip("/") != "/embed":
                self._send(404, {"detail": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 1_000_000:
                    raise ValueError("invalid content length")
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                texts = payload.get("texts")
                if (
                    not isinstance(texts, list)
                    or not texts
                    or len(texts) > MAX_BATCH_SIZE
                    or any(not isinstance(item, str) or not item.strip() for item in texts)
                ):
                    raise ValueError("texts must be a nonempty list of strings")
            except (ValueError, TypeError, json.JSONDecodeError, UnicodeDecodeError):
                self._send(400, {"detail": "invalid request"})
                return

            self._send(
                200,
                {
                    "model": model,
                    "dimensions": dimensions,
                    "embeddings": [
                        deterministic_vector(text, dimensions) for text in texts
                    ],
                },
            )

        def log_message(self, format: str, *args: Any) -> None:
            # Never log request paths with query strings or bodies. A single
            # startup line is emitted by ``main`` instead.
            return

    return Handler


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8091)
    parser.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    parser.add_argument("--dimensions", type=int, default=DEFAULT_DIMENSIONS)
    args = parser.parse_args(argv)

    if args.dimensions < 1:
        parser.error("--dimensions must be positive")

    server = ThreadingHTTPServer(
        (args.host, args.port), _handler(args.model, args.dimensions)
    )
    print(
        f"fake embedding service listening on http://{args.host}:{server.server_port} "
        f"(model={args.model}, dimensions={args.dimensions})",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

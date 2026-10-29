"""Local embedding service for OpsAtlas; bind to loopback only."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging

from sentence_transformers import SentenceTransformer

HOST = "127.0.0.1"
PORT = 8091
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MAX_TEXTS = 32
MAX_BODY_BYTES = 1_000_000

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("opsatlas-embeddings")

logger.info("Loading embedding model: %s", MODEL_NAME)
model = SentenceTransformer(MODEL_NAME, device="cpu")
logger.info("Embedding model loaded")


class Handler(BaseHTTPRequestHandler):
    def send_json(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path != "/health":
            self.send_json(404, {"error": "not_found"})
            return

        self.send_json(200, {
            "status": "ok",
            "model": MODEL_NAME,
            "dimensions": 384,
            "device": "cpu",
        })

    def do_POST(self):
        if self.path != "/embed":
            self.send_json(404, {"error": "not_found"})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > MAX_BODY_BYTES:
                self.send_json(413, {"error": "invalid_body_size"})
                return

            payload = json.loads(self.rfile.read(length))
            texts = payload.get("texts") if isinstance(payload, dict) else None

            if (
                not isinstance(texts, list)
                or not 1 <= len(texts) <= MAX_TEXTS
                or any(not isinstance(t, str) or not t.strip() for t in texts)
            ):
                self.send_json(400, {
                    "error": "texts_must_be_1_to_32_nonempty_strings"
                })
                return

            if any(len(t) > 20_000 for t in texts):
                self.send_json(400, {"error": "text_too_long"})
                return

            vectors = model.encode(
                texts,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )

            self.send_json(200, {
                "model": MODEL_NAME,
                "dimensions": int(vectors.shape[1]),
                "embeddings": vectors.tolist(),
            })

        except (ValueError, json.JSONDecodeError):
            self.send_json(400, {"error": "invalid_json_or_payload"})
        except Exception:
            logger.exception("Embedding request failed")
            self.send_json(500, {"error": "embedding_failed"})

    def log_message(self, fmt, *args):
        logger.info("%s - %s", self.address_string(), fmt % args)


if __name__ == "__main__":
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    logger.info("Embedding service listening on http://%s:%s", HOST, PORT)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Stopping embedding service")
    finally:
        server.server_close()
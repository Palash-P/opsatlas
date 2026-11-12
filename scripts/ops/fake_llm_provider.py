"""Deterministic OpenAI-compatible provider used only by local failure tests.

The server chooses its behavior from a ``[[mode=...]]`` marker in the user
message. This lets one API process exercise timeout, rate-limit, invalid
response, citation, and server-error paths without restarting the provider.

Supported modes: ``ok``, ``slow``, ``timeout``, ``rate_limit``,
``invalid_envelope``, ``invalid_output``, ``bad_citation``, ``server_error``.

This is a test double. It is never imported by the application.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

logger = logging.getLogger("opsatlas-fake-provider")

MODE_RE = re.compile(r"\[\[mode=([a-z_]+)\]\]")

GROUNDED_OK = {
    "answer": "Check upstream timeouts before rolling back.",
    "used_evidence_ids": ["E1"],
    "abstained": False,
    "abstention_reason": None,
    "insufficient_evidence": False,
    "conflicts": [],
    "uncertainty": None,
}

GROUNDED_BAD_CITATION = dict(GROUNDED_OK)
GROUNDED_BAD_CITATION["used_evidence_ids"] = ["E999"]


def _mode_from_body(body: dict) -> str:
    messages = body.get("messages")
    if not isinstance(messages, list):
        return "ok"
    for message in reversed(messages):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        match = MODE_RE.search(content)
        if match:
            return match.group(1)
    return "ok"


class Handler(BaseHTTPRequestHandler):
    server_version = "OpsAtlasFakeProvider/1.0"

    def _send(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _completion(self, content: str, *, model: str = "fake-model") -> dict:
        return {
            "id": "chatcmpl-fake",
            "object": "chat.completion",
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 111, "completion_tokens": 22, "total_tokens": 133},
        }

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send(200, {"status": "ok"})
            return
        self._send(404, {"error": "not_found"})

    def do_POST(self) -> None:
        if not self.path.endswith("/chat/completions"):
            self._send(404, {"error": "not_found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, json.JSONDecodeError):
            self._send(400, {"error": "invalid_json"})
            return

        mode = _mode_from_body(body if isinstance(body, dict) else {})
        logger.info("mode=%s", mode)

        if mode == "slow":
            time.sleep(1.5)
            self._send(200, self._completion(json.dumps(GROUNDED_OK)))
            return
        if mode == "timeout":
            time.sleep(3.0)
            self._send(200, self._completion(json.dumps(GROUNDED_OK)))
            return
        if mode == "rate_limit":
            self._send(429, {"error": {"message": "rate limit exceeded"}})
            return
        if mode == "invalid_envelope":
            self._send(200, {"choices": []})
            return
        if mode == "invalid_output":
            self._send(200, self._completion("this is not json"))
            return
        if mode == "bad_citation":
            self._send(200, self._completion(json.dumps(GROUNDED_BAD_CITATION)))
            return
        if mode == "server_error":
            self._send(500, {"error": {"message": "upstream failure"}})
            return
        self._send(200, self._completion(json.dumps(GROUNDED_OK)))

    def log_message(self, fmt: str, *args: object) -> None:
        logger.info("%s - %s", self.address_string(), fmt % args)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8099)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    logger.info("Fake provider listening on http://%s:%s", args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

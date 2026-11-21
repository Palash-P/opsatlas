"""Structured request telemetry without prompts, secrets, or document text.

The request log is one JSON object per completed request. It records stage
timings, a privacy-safe tenant reference, retrieval strategy, provider/model,
retry count, token accounting, and an error category. It never records the
Authorization header, API keys, request/response bodies, prompts, retrieved
content, or exception messages.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import re
import time
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any
from uuid import UUID, uuid4

from fastapi import Request

from app.core.config import Settings

LOGGER_NAME = "opsatlas.request"
WORKER_LOGGER_NAME = "opsatlas.worker"
RECONCILIATION_LOGGER_NAME = "opsatlas.reconciliation"
SCHEMA_LOGGER_NAME = "opsatlas.schema"
_OPERATIONAL_LOGGER_NAMES = (
    LOGGER_NAME,
    WORKER_LOGGER_NAME,
    RECONCILIATION_LOGGER_NAME,
    SCHEMA_LOGGER_NAME,
)
_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9._-]{8,64}$")

logger = logging.getLogger(LOGGER_NAME)
worker_logger = logging.getLogger(WORKER_LOGGER_NAME)
reconciliation_logger = logging.getLogger(RECONCILIATION_LOGGER_NAME)
schema_logger = logging.getLogger(SCHEMA_LOGGER_NAME)


@dataclass
class RequestTelemetry:
    """Mutable per-request record populated by middleware and endpoints."""

    request_id: str
    method: str
    path: str
    status_code: int | None = None
    tenant_ref: str | None = None
    retrieval_strategy: str | None = None
    retrieval_latency_ms: int | None = None
    evidence_latency_ms: int | None = None
    generation_latency_ms: int | None = None
    total_latency_ms: int | None = None
    provider: str | None = None
    observed_model: str | None = None
    requested_model: str | None = None
    retry_count: int | None = None
    error_category: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    token_source: str | None = None


_current_telemetry: ContextVar[RequestTelemetry | None] = ContextVar(
    "opsatlas_request_telemetry",
    default=None,
)


def get_telemetry() -> RequestTelemetry | None:
    return _current_telemetry.get()


def current_request_id() -> str:
    telemetry = get_telemetry()
    return telemetry.request_id if telemetry is not None else uuid4().hex


def set_error_category(category: str | None) -> None:
    telemetry = get_telemetry()
    if telemetry is not None and category:
        telemetry.error_category = category


def tenant_reference(tenant_id: UUID, settings: Settings | None = None) -> str:
    """Return a stable, salted tenant reference instead of the raw UUID.

    Worker logs have no request context, so this helper is shared by HTTP
    telemetry and background task events.
    """

    if settings is None:
        from app.core.config import get_settings

        settings = get_settings()
    digest = hmac.new(
        settings.auth_log_tenant_salt.encode("utf-8"),
        str(tenant_id).encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return f"t_{digest[:16]}"


def record_tenant_reference(tenant_id: UUID, settings: Settings | None = None) -> None:
    """Record a stable, salted tenant reference instead of the raw UUID."""

    telemetry = get_telemetry()
    if telemetry is None:
        return
    telemetry.tenant_ref = tenant_reference(tenant_id, settings)


def _request_id(request: Request) -> str:
    supplied = request.headers.get("x-request-id")
    if supplied and _REQUEST_ID_RE.fullmatch(supplied):
        return supplied
    return uuid4().hex


def categorize_status(status_code: int) -> str | None:
    if status_code < 400:
        return None
    return {
        400: "invalid_request",
        401: "unauthorized",
        403: "forbidden",
        404: "not_found",
        409: "conflict",
        422: "validation_error",
        429: "rate_limited",
        500: "internal_error",
        502: "provider_error",
        503: "dependency_unavailable",
        504: "timeout",
    }.get(status_code, "request_failed")


def build_event_payload(telemetry: RequestTelemetry) -> dict[str, Any]:
    return {
        "event": "request",
        "request_id": telemetry.request_id,
        "method": telemetry.method,
        "path": telemetry.path,
        "status_code": telemetry.status_code,
        "tenant_ref": telemetry.tenant_ref,
        "retrieval_strategy": telemetry.retrieval_strategy,
        "retrieval_latency_ms": telemetry.retrieval_latency_ms,
        "evidence_latency_ms": telemetry.evidence_latency_ms,
        "generation_latency_ms": telemetry.generation_latency_ms,
        "total_latency_ms": telemetry.total_latency_ms,
        "provider": telemetry.provider,
        "observed_model": telemetry.observed_model,
        "requested_model": telemetry.requested_model,
        "retry_count": telemetry.retry_count,
        "error_category": telemetry.error_category,
        "input_tokens": telemetry.input_tokens,
        "output_tokens": telemetry.output_tokens,
        "token_source": telemetry.token_source,
    }


class JsonFormatter(logging.Formatter):
    """Render ``event_data`` as one JSON object per log line."""

    def format(self, record: logging.LogRecord) -> str:
        data = getattr(record, "event_data", None)
        if not isinstance(data, dict):
            return super().format(record)
        payload = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            **data,
        }
        return json.dumps(payload, sort_keys=True, default=str)


def _configure_json_logger(target: logging.Logger, level: str) -> None:
    target.setLevel(level)
    if not any(getattr(handler, "_opsatlas_json", False) for handler in target.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        handler._opsatlas_json = True  # type: ignore[attr-defined]
        target.addHandler(handler)
    target.propagate = False


def configure_logging(settings: Settings) -> None:
    """Install one idempotent JSON handler for request and worker loggers."""

    level = settings.log_level.upper()
    for name in _OPERATIONAL_LOGGER_NAMES:
        _configure_json_logger(logging.getLogger(name), level)


def emit_operational_event(
    target: logging.Logger,
    event: str,
    *,
    level: int = logging.INFO,
    message: str = "operational event",
    **fields: Any,
) -> None:
    """Emit one sanitized JSON event without prompts, content, or secrets.

    Callers pass only bounded identifiers and counters. This helper exists so
    API-side operational events (for example an async ingestion submission) use
    the same shape as worker events instead of ad-hoc log strings.
    """

    payload: dict[str, Any] = {"event": event}
    telemetry = get_telemetry()
    if telemetry is not None:
        payload["request_id"] = telemetry.request_id
    payload.update(fields)
    target.log(level, message, extra={"event_data": payload})


def emit_request_log(telemetry: RequestTelemetry) -> None:
    status = telemetry.status_code or 500
    if status >= 500:
        level = logging.ERROR
    elif status >= 400:
        level = logging.WARNING
    else:
        level = logging.INFO
    logger.log(
        level,
        "request",
        extra={"event_data": build_event_payload(telemetry)},
    )


async def observe_request(request: Request, call_next):
    """HTTP middleware: assign a request ID, time the request, and log once."""

    telemetry = RequestTelemetry(
        request_id=_request_id(request),
        method=request.method,
        path=request.url.path,
    )
    token = _current_telemetry.set(telemetry)
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        telemetry.total_latency_ms = int((time.perf_counter() - started) * 1000)
        telemetry.status_code = 500
        telemetry.error_category = telemetry.error_category or "internal_error"
        emit_request_log(telemetry)
        raise
    else:
        telemetry.total_latency_ms = int((time.perf_counter() - started) * 1000)
        telemetry.status_code = response.status_code
        if telemetry.error_category is None:
            telemetry.error_category = categorize_status(response.status_code)
        response.headers["X-Request-ID"] = telemetry.request_id
        emit_request_log(telemetry)
        return response
    finally:
        _current_telemetry.reset(token)

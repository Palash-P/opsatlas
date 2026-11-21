import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from sqlalchemy.exc import OperationalError
from sqlalchemy.exc import TimeoutError as SQLAlchemyTimeoutError

from app.api.answers import router as answers_router
from app.api.documents import router as documents_router
from app.api.search import router as search_router
from app.core.auth import AuthenticationError, AuthorizationError
from app.core.config import get_settings
from app.core.limits import (
    ConcurrencyLimitExceededError,
    RateLimitBackendUnavailableError,
    RateLimitExceededError,
)
from app.core.observability import (
    SCHEMA_LOGGER_NAME,
    configure_logging,
    emit_operational_event,
    observe_request,
    set_error_category,
)
from app.db.connections import check_postgres, check_redis
from app.db.schema_validation import check_schema_readiness
from app.version import APP_VERSION

settings = get_settings()
configure_logging(settings)
logger = logging.getLogger(SCHEMA_LOGGER_NAME)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Log schema state at startup without hiding a debuggable process.

    The process still starts so ``/health`` can prove liveness and ``/ready`` can
    return an actionable 503. Readiness, not process startup, is the gate that
    removes an incompatible instance from rotation.
    """

    try:
        check_schema_readiness()
    except Exception as exc:  # noqa: BLE001 - readiness reports the failure
        logger.error(
            "database schema is not ready; /ready will fail",
            extra={
                "event_data": {
                    "event": "schema_validation",
                    "outcome": "failed",
                    "error_category": "schema_incompatible",
                    "detail": str(exc),
                }
            },
        )

    configured_tokens = sum(
        1
        for token in settings.auth_tokens
        if isinstance(token, str) and token.strip()
    )
    emit_operational_event(
        logger,
        "auth_tokens_loaded",
        message="bearer token map loaded",
        configured_tokens=configured_tokens,
    )
    emit_operational_event(
        logger,
        "rate_limit_backend",
        message="request rate limiter configured",
        backend=settings.rate_limit_backend,
    )
    yield


app = FastAPI(title=settings.app_name, version=APP_VERSION, lifespan=lifespan)
app.middleware("http")(observe_request)
app.include_router(documents_router)
app.include_router(search_router)
app.include_router(answers_router)


@app.exception_handler(AuthenticationError)
async def _authentication_error_handler(
    request: Request, exc: AuthenticationError
) -> JSONResponse:
    set_error_category("unauthorized")
    return JSONResponse(
        status_code=401,
        content={"detail": "Unauthorized"},
        headers={"WWW-Authenticate": "Bearer"},
    )


@app.exception_handler(AuthorizationError)
async def _authorization_error_handler(
    request: Request, exc: AuthorizationError
) -> JSONResponse:
    # The reason is deliberately not returned: a tenant mismatch must not
    # disclose whether a target tenant or document exists.
    set_error_category("forbidden")
    return JSONResponse(status_code=403, content={"detail": "Forbidden"})


@app.exception_handler(RateLimitExceededError)
async def _rate_limit_error_handler(
    request: Request, exc: RateLimitExceededError
) -> JSONResponse:
    set_error_category("rate_limited")
    return JSONResponse(
        status_code=429,
        content={"detail": "Too many requests"},
        headers={"Retry-After": str(exc.retry_after)},
    )


@app.exception_handler(ConcurrencyLimitExceededError)
async def _concurrency_error_handler(
    request: Request, exc: ConcurrencyLimitExceededError
) -> JSONResponse:
    set_error_category("concurrency_limited")
    return JSONResponse(
        status_code=429,
        content={"detail": "Too many concurrent requests"},
        headers={"Retry-After": str(exc.retry_after)},
    )


@app.exception_handler(RateLimitBackendUnavailableError)
async def _rate_limit_backend_error_handler(
    request: Request, exc: RateLimitBackendUnavailableError
) -> JSONResponse:
    # Fail closed when the shared limiter is required but unreachable. The
    # response is constant so it cannot disclose the Redis endpoint or a key.
    set_error_category("rate_limit_backend_unavailable")
    return JSONResponse(
        status_code=503,
        content={"detail": "A required dependency is unavailable"},
        headers={"Retry-After": str(exc.retry_after)},
    )


@app.exception_handler(OperationalError)
@app.exception_handler(SQLAlchemyTimeoutError)
async def _database_unavailable_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    # Exception text can contain DSNs or SQL. Return a constant detail only.
    set_error_category("database_unavailable")
    return JSONResponse(
        status_code=503,
        content={"detail": "A required dependency is unavailable"},
    )


@app.get("/health", tags=["operations"])
def health() -> dict[str, str]:
    """Liveness check: confirms the API process can serve requests.

    The version is the single authoritative value from ``app.version`` and is
    also the FastAPI/OpenAPI version, so a running container can be matched to
    the package metadata it was built from.
    """

    return {"status": "ok", "service": settings.app_name, "version": APP_VERSION}


@app.get("/ready", tags=["operations"])
def ready() -> JSONResponse:
    """Readiness check: PostgreSQL and its schema are required; Redis is optional.

    Redis is not on the answer/retrieval path. Losing it degrades background
    capabilities but must not remove a serving API instance from rotation.
    PostgreSQL is required for every data-bearing request, and a reachable
    database with missing migrations, a non-384 embedding column, or missing
    Phase F objects is not ready.
    """

    checks: dict[str, str] = {}
    required_failures: list[str] = []
    optional_failures: list[str] = []

    postgres_ok = False
    try:
        check_postgres()
        checks["postgres"] = "ok"
        postgres_ok = True
    except Exception:  # noqa: BLE001 - readiness must report any dependency failure
        checks["postgres"] = "error"
        required_failures.append("postgres")
        emit_operational_event(
            logger,
            "readiness",
            level=logging.ERROR,
            message="readiness dependency failed",
            check="postgres",
            outcome="unavailable",
            error_category="database_unavailable",
        )

    if postgres_ok:
        try:
            check_schema_readiness()
            checks["schema"] = "ok"
        except Exception as exc:  # noqa: BLE001 - readiness must report any schema failure
            checks["schema"] = "error"
            required_failures.append("schema")
            logger.error(
                "database schema is not ready",
                extra={
                    "event_data": {
                        "event": "schema_validation",
                        "outcome": "failed",
                        "error_category": "schema_incompatible",
                        "detail": str(exc),
                    }
                },
            )
    else:
        # Do not perform a second introspection against an unreachable database.
        checks["schema"] = "unavailable"

    try:
        check_redis()
        checks["redis"] = "ok"
    except Exception:  # noqa: BLE001 - readiness must report any dependency failure
        checks["redis"] = "error"
        optional_failures.append("redis")
        emit_operational_event(
            logger,
            "readiness",
            level=logging.WARNING,
            message="optional readiness dependency failed",
            check="redis",
            outcome="degraded",
            error_category="cache_unavailable",
        )

    if required_failures:
        status = "not_ready"
    elif optional_failures:
        status = "degraded"
    else:
        status = "ready"

    return JSONResponse(
        status_code=503 if required_failures else 200,
        content={"status": status, "checks": checks},
    )

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from sqlalchemy.exc import OperationalError
from sqlalchemy.exc import TimeoutError as SQLAlchemyTimeoutError

from app.api.answers import router as answers_router
from app.api.documents import router as documents_router
from app.api.search import router as search_router
from app.core.auth import AuthenticationError, AuthorizationError
from app.core.config import get_settings
from app.core.limits import ConcurrencyLimitExceededError, RateLimitExceededError
from app.core.observability import (
    configure_logging,
    observe_request,
    set_error_category,
)
from app.db.connections import check_postgres, check_redis

settings = get_settings()
configure_logging(settings)

app = FastAPI(title=settings.app_name, version="0.5.0")
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
    """Liveness check: confirms the API process can serve requests."""
    return {"status": "ok", "service": settings.app_name}


@app.get("/ready", tags=["operations"])
def ready() -> JSONResponse:
    """Readiness check: PostgreSQL is required, Redis is optional.

    Redis is not on the answer/retrieval path. Losing it degrades background
    capabilities (once they exist) but must not remove a serving API instance
    from rotation. PostgreSQL is required for every data-bearing request.
    """

    checks: dict[str, str] = {}
    required_failures: list[str] = []
    optional_failures: list[str] = []

    try:
        check_postgres()
        checks["postgres"] = "ok"
    except Exception:  # noqa: BLE001 - readiness must report any dependency failure
        checks["postgres"] = "error"
        required_failures.append("postgres")

    try:
        check_redis()
        checks["redis"] = "ok"
    except Exception:  # noqa: BLE001 - readiness must report any dependency failure
        checks["redis"] = "error"
        optional_failures.append("redis")

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

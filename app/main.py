from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.api.documents import router as documents_router
from app.api.search import router as search_router
from app.core.config import get_settings
from app.db.connections import check_postgres, check_redis

settings = get_settings()
app = FastAPI(title=settings.app_name, version="0.3.0")
app.include_router(documents_router)
app.include_router(search_router)


@app.get("/health", tags=["operations"])
def health() -> dict[str, str]:
    """Liveness check: confirms the API process can serve requests."""
    return {"status": "ok", "service": settings.app_name}


@app.get("/ready", tags=["operations"])
def ready() -> JSONResponse:
    """Readiness check: verifies the dependencies needed by this scaffold."""
    checks: dict[str, str] = {}
    failures: list[str] = []

    for name, check in (("postgres", check_postgres), ("redis", check_redis)):
        try:
            check()
            checks[name] = "ok"
        except Exception:
            checks[name] = "error"
            failures.append(name)

    status = "ready" if not failures else "not_ready"
    payload = {"status": status, "checks": checks}
    return JSONResponse(status_code=200 if not failures else 503, content=payload)
from functools import lru_cache

from redis import Redis
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from app.core.config import get_settings


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """Return the process-wide SQLAlchemy engine and connection pool.

    The pool bounds concurrent database work, recycles connections, checks stale
    connections before use, and gives a request a finite wait when the pool is
    exhausted. ``statement_timeout`` bounds a single query so a slow retrieval
    cannot hold a worker indefinitely.
    """

    settings = get_settings()
    connect_args: dict[str, object] = {
        "connect_timeout": settings.db_connect_timeout_seconds,
    }
    if settings.db_statement_timeout_ms > 0:
        connect_args["options"] = (
            f"-c statement_timeout={settings.db_statement_timeout_ms}"
        )

    return create_engine(
        settings.database_url,
        pool_pre_ping=True,
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
        pool_timeout=settings.db_pool_timeout_seconds,
        pool_recycle=settings.db_pool_recycle_seconds,
        connect_args=connect_args,
    )


def check_postgres() -> None:
    with get_engine().connect() as connection:
        connection.execute(text("SELECT 1"))


def check_redis() -> None:
    settings = get_settings()
    client = Redis.from_url(settings.redis_url, socket_connect_timeout=1, socket_timeout=1)
    try:
        if client.ping() is not True:
            raise RuntimeError("Redis ping did not return success")
    finally:
        client.close()

from functools import lru_cache

from redis import Redis
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from app.core.config import get_settings


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """Return the process-wide SQLAlchemy engine and connection pool."""
    settings = get_settings()
    return create_engine(
        settings.database_url,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=5,
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

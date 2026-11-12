from functools import lru_cache
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "OpsAtlas"
    app_env: str = "local"
    database_url: str = "postgresql+psycopg://opsatlas:opsatlas_dev_password@localhost:5432/opsatlas"
    redis_url: str = "redis://localhost:6379/0"
    log_level: str = "INFO"

    # Authentication. Tokens are configured, not stored in the database. Each
    # entry maps an opaque bearer token to a tenant and principal. The token
    # itself is never logged or returned. Example (JSON):
    #   AUTH_TOKENS={"token-value":{"tenant_id":"<uuid>","subject":"alice",
    #                              "roles":["member"],"groups":["on-call"]}}
    # An empty map means every protected request is unauthenticated (401).
    auth_tokens: dict[str, Any] = Field(default_factory=dict)
    # Salt used before hashing a tenant UUID for request logs. This keeps the
    # tenant correlation-stable without writing the raw tenant identifier.
    auth_log_tenant_salt: str = "opsatlas-local-observability-salt"

    # Request protection. These limits are per process, not distributed.
    rate_limit_per_minute: int = 240
    rate_limit_window_seconds: float = 60.0
    max_concurrent_requests: int = 16
    concurrency_acquire_timeout_seconds: float = 0.25

    # Answer generation. The default provider is deterministic and local so the
    # complete RAG path works without a paid API key. Provider-specific HTTP
    # calls live in app.services.llm_providers, never in business logic.
    llm_provider: str = "extractive"
    llm_model: str = "extractive-local-v1"
    llm_temperature: float = 0.0
    llm_max_output_tokens: int = 700
    llm_timeout_seconds: float = 20.0
    llm_max_retries: int = 1
    llm_retry_backoff_seconds: float = 0.25
    llm_api_base: str | None = None
    llm_api_key: str | None = None
    llm_api_path: str = "/chat/completions"

    # Embedding service client. The old OPSATLAS_EMBEDDING_URL environment
    # variable is still honored by the client for evaluation scripts.
    embedding_url: str = "http://127.0.0.1:8091"
    embedding_timeout_seconds: float = 10.0
    embedding_max_retries: int = 1
    embedding_retry_backoff_seconds: float = 0.2

    # Database pool and statement limits. pool_pre_ping detects stale
    # connections; the pool timeout and statement timeout bound request waits.
    db_pool_size: int = 5
    db_max_overflow: int = 5
    db_pool_timeout_seconds: float = 5.0
    db_pool_recycle_seconds: int = 1800
    db_connect_timeout_seconds: int = 5
    db_statement_timeout_ms: int = 15_000

    # Evidence assembly defaults. Token estimates use a documented character
    # approximation, not a model tokenizer.
    answer_context_token_budget: int = 4000
    answer_max_evidence: int = 8
    answer_citation_excerpt_chars: int = 600

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)


@lru_cache
def get_settings() -> Settings:
    return Settings()

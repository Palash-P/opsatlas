from functools import lru_cache
from typing import Any

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.auth_tokens import validate_token_map

# The database embedding contract. ``chunks.embedding`` is declared as
# ``vector(384)`` by sql/002_embedding_384.sql and retrieval casts query vectors
# to the same type. This is deliberately a compile-time contract, not a runtime
# schema discovery mechanism: changing it requires a migration, a backfill or
# re-embedding plan, and matching retrieval/index support. Keeping one constant
# here means the application cannot silently start with a dimension the database
# cannot store.
DATABASE_EMBEDDING_DIMENSIONS = 384

# Default model used by the local embedding service and by evaluation scripts
# that need a stable label when settings are not loaded.
EMBEDDING_MODEL_DEFAULT = "sentence-transformers/all-MiniLM-L6-v2"

# Environment names that require deployment-grade secrets. Anything else is
# treated as a local development environment, where the placeholder defaults in
# ``.env.example`` are intentional.
PRODUCTION_ENVIRONMENTS = frozenset({"production", "prod"})

# Values that only exist so a fresh checkout works locally. They must never be
# accepted when APP_ENV declares a production deployment.
LOCAL_DEVELOPMENT_DATABASE_PASSWORD = "opsatlas_dev_password"
LOCAL_DEVELOPMENT_TENANT_SALT = "opsatlas-local-observability-salt"


class Settings(BaseSettings):
    app_name: str = "OpsAtlas"
    app_env: str = "local"
    # Host-run fallback only. It matches the Compose-published host port
    # (POSTGRES_PORT, default 5433); containers receive an explicit
    # postgres:5432 URL from docker-compose.yml. Always override with
    # DATABASE_URL in real deployments.
    database_url: str = (
        "postgresql+psycopg://opsatlas:opsatlas_dev_password@localhost:5433/opsatlas"
    )
    redis_url: str = "redis://localhost:6379/0"
    log_level: str = "INFO"

    # Background ingestion queue. Celery uses Redis as a broker only: durable
    # job state lives in PostgreSQL (ingestion_jobs), so no result backend is
    # configured. Keep the broker on a separate Redis database from the
    # readiness/cache connection to avoid key collisions and accidental FLUSHDB.
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_task_default_queue: str = "ingestion"
    # Reconciliation has its own queue and worker. A dedicated queue keeps
    # periodic recovery from waiting behind a busy ingestion backlog. The two
    # names must differ so a misconfiguration cannot silently collapse the
    # isolation.
    celery_reconciliation_queue: str = Field(default="reconciliation", min_length=1)
    celery_task_serializer: str = "json"
    celery_accept_content: list[str] = Field(default_factory=lambda: ["json"])
    celery_task_acks_late: bool = True
    celery_task_reject_on_worker_lost: bool = True
    celery_worker_prefetch_multiplier: int = 1
    celery_worker_concurrency: int = 2
    celery_task_soft_time_limit_seconds: int = 1800
    celery_task_time_limit_seconds: int = 1860
    celery_visibility_timeout_seconds: int = 2400
    celery_broker_connection_retry_on_startup: bool = True

    # Bounded background retries. Only errors that cannot corrupt or duplicate
    # durable state are retried; the ingestion service itself remains
    # idempotent and records every execution attempt in PostgreSQL.
    ingestion_task_max_retries: int = 3
    ingestion_task_retry_backoff_seconds: float = 5.0
    ingestion_task_retry_backoff_max_seconds: float = 120.0
    ingestion_task_unexpected_max_retries: int = 1

    # Reconciliation and execution ownership. The worker mints a fresh fencing
    # token when it claims a job and renews the short lease between embedding
    # batches. A job whose lease expired can be redispatched; its old token can
    # no longer activate or fail the job. Dispatch leases keep overlapping
    # reconcilers from publishing the same job simultaneously.
    ingestion_execution_lease_seconds: int = Field(default=300, ge=1)
    ingestion_reconcile_enabled: bool = True
    ingestion_reconcile_interval_seconds: int = Field(default=60, ge=5)
    ingestion_reconcile_batch_size: int = Field(default=50, ge=1, le=1000)
    ingestion_reconcile_dispatch_lease_seconds: int = Field(default=300, ge=1)
    ingestion_reconcile_dispatch_backoff_seconds: float = Field(default=15.0, ge=0)
    ingestion_reconcile_dispatch_backoff_max_seconds: float = Field(default=300.0, ge=0)
    # A transiently failed worker job is normally retried by Celery. Wait longer
    # than the largest Celery backoff before reconciliation assumes the retry
    # message was lost, so recovery does not race a healthy retry.
    ingestion_reconcile_transient_grace_seconds: float = Field(default=300.0, ge=0)

    # Authentication. Tokens are configured, not stored in the database. Each
    # entry maps an opaque bearer token to a tenant and principal. The token
    # itself is never logged or returned. Example (JSON):
    #   AUTH_TOKENS={"token-value":{"tenant_id":"<uuid>","subject":"alice",
    #                              "roles":["member","document_writer"],
    #                              "groups":["on-call"]}}
    # An empty map means every protected request is unauthenticated (401).
    # The explicit "document_writer" role is required to create documents or
    # update documents that have no ACL rows; tenant membership alone is not a
    # write permission. Documents with ACL rows require a matching "manage"
    # grant instead.
    auth_tokens: dict[str, Any] = Field(default_factory=dict)
    # Salt used before hashing a tenant UUID for request logs. This keeps the
    # tenant correlation-stable without writing the raw tenant identifier.
    auth_log_tenant_salt: str = "opsatlas-local-observability-salt"

    # Request protection. The concurrency slot is always per process. The rate
    # limit is per process when ``rate_limit_backend=memory`` (the default for
    # local development and tests) and shared across API replicas when
    # ``rate_limit_backend=redis``. Production deployments with more than one
    # replica must use Redis; the limiter keys are derived from a salted hash of
    # the tenant UUID, never from the token or the raw tenant identifier.
    rate_limit_backend: str = "memory"
    rate_limit_per_minute: int = 240
    rate_limit_window_seconds: float = 60.0
    rate_limit_redis_key_prefix: str = "opsatlas:ratelimit"
    rate_limit_redis_timeout_seconds: float = Field(default=0.25, gt=0)
    # A Redis outage must not silently disable limiting. The default degrades to
    # the in-memory limiter for the current process and emits a warning; setting
    # this to false fails closed with 503 instead.
    rate_limit_redis_fallback_to_local: bool = True
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
    # variable is still honored by the client for evaluation scripts. The model
    # and dimensions are configuration, not business logic: ingestion persists
    # the model name with every vector and refuses to activate a version whose
    # vectors do not match both values. The model may change after re-embedding,
    # but the dimension must match the fixed PostgreSQL vector contract.
    embedding_url: str = "http://127.0.0.1:8091"
    embedding_model: str = EMBEDDING_MODEL_DEFAULT
    embedding_dimensions: int = Field(default=DATABASE_EMBEDDING_DIMENSIONS, ge=1)
    embedding_timeout_seconds: float = 10.0
    embedding_max_retries: int = 1
    embedding_retry_backoff_seconds: float = 0.2

    @field_validator("embedding_dimensions")
    @classmethod
    def _require_database_dimension(cls, value: int) -> int:
        """Reject a configured dimension the database schema cannot store.

        ``chunks.embedding`` is ``vector(384)``. Accepting a different value here
        would delay the failure until the first insert or retrieval, where the
        provider response is already paid for and a partial target version may
        exist. Failing while settings are constructed keeps the contract
        explicit and puts the remedy in the error text.
        """

        if value != DATABASE_EMBEDDING_DIMENSIONS:
            raise ValueError(
                "embedding_dimensions must match the PostgreSQL vector column "
                f"dimension ({DATABASE_EMBEDDING_DIMENSIONS}); changing it requires "
                "a schema migration and matching retrieval support"
            )
        return value

    @model_validator(mode="after")
    def _require_valid_task_timing(self) -> "Settings":
        """Keep the hard worker time limit above the soft limit.

        A hard limit at or below the soft limit would let Celery kill the
        process before the task can persist a sanitized failure and schedule a
        bounded retry. The visibility timeout must also exceed the hard limit so
        Redis does not redeliver a still-running task.
        """

        if self.celery_task_time_limit_seconds <= self.celery_task_soft_time_limit_seconds:
            raise ValueError(
                "celery_task_time_limit_seconds must be greater than "
                "celery_task_soft_time_limit_seconds"
            )
        if self.celery_visibility_timeout_seconds <= self.celery_task_time_limit_seconds:
            raise ValueError(
                "celery_visibility_timeout_seconds must be greater than "
                "celery_task_time_limit_seconds"
            )
        if self.ingestion_execution_lease_seconds >= self.celery_visibility_timeout_seconds:
            raise ValueError(
                "ingestion_execution_lease_seconds must be less than "
                "celery_visibility_timeout_seconds so reconciliation can recover "
                "an abandoned worker before or alongside broker redelivery"
            )
        if (
            self.ingestion_reconcile_dispatch_backoff_max_seconds
            < self.ingestion_reconcile_dispatch_backoff_seconds
        ):
            raise ValueError(
                "ingestion_reconcile_dispatch_backoff_max_seconds must be greater "
                "than or equal to ingestion_reconcile_dispatch_backoff_seconds"
            )
        if self.celery_reconciliation_queue == self.celery_task_default_queue:
            raise ValueError(
                "celery_reconciliation_queue must differ from "
                "celery_task_default_queue so reconciliation cannot be starved by "
                "the ingestion backlog"
            )
        return self

    @model_validator(mode="after")
    def _require_production_secrets(self) -> "Settings":
        """Fail closed when a production deployment still uses local defaults.

        ``.env.example`` is deliberately usable out of the box, so it contains a
        known development database password, an empty token map, and a fixed
        tenant-log salt. Those values are safe only for local development. A
        process that claims ``APP_ENV=production`` must receive real secrets from
        the deployment environment; starting with placeholders would expose an
        unauthenticated API and log correlation that anyone can reverse.
        """

        if self.app_env.lower() not in PRODUCTION_ENVIRONMENTS:
            return self

        problems: list[str] = []
        if LOCAL_DEVELOPMENT_DATABASE_PASSWORD in self.database_url:
            problems.append(
                "DATABASE_URL still uses the local development password; "
                "inject a deployment secret"
            )
        if not self.auth_tokens:
            problems.append(
                "AUTH_TOKENS is empty; every protected endpoint would reject or "
                "run without an authenticated principal"
            )
        if self.auth_log_tenant_salt == LOCAL_DEVELOPMENT_TENANT_SALT:
            problems.append(
                "AUTH_LOG_TENANT_SALT still uses the local default; set a "
                "deployment-specific random salt"
            )
        if problems:
            raise ValueError(
                "unsafe production configuration (APP_ENV="
                f"{self.app_env}): " + "; ".join(problems)
            )
        return self

    @field_validator("rate_limit_backend", mode="before")
    @classmethod
    def _normalize_rate_limit_backend(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip().lower()
        return value

    @model_validator(mode="after")
    def _require_shared_rate_limit_in_production(self) -> "Settings":
        """Refuse a multi-replica deployment that limits each process separately.

        A process-local limiter still bounds one process correctly, but with N
        replicas it allows up to N times the configured rate per tenant. Redis is
        already a required background dependency, so requiring the shared limiter
        in production is a configuration assertion, not a new datastore.
        """

        if self.rate_limit_backend not in {"memory", "redis"}:
            raise ValueError(
                "rate_limit_backend must be 'memory' or 'redis' "
                f"(received {self.rate_limit_backend!r})"
            )
        if (
            self.app_env.lower() in PRODUCTION_ENVIRONMENTS
            and self.rate_limit_backend != "redis"
        ):
            raise ValueError(
                "unsafe production configuration: rate_limit_backend must be "
                "'redis' so multiple API replicas share one per-tenant limit"
            )
        return self

    @model_validator(mode="after")
    def _require_valid_auth_tokens(self) -> "Settings":
        """Validate the bearer-token map before the process serves traffic.

        Structural problems (unknown fields, invalid UUIDs or timestamps) fail
        in every environment so a typo cannot silently produce a token that may
        never authenticate. Production additionally requires long non-placeholder
        tokens and at least one currently usable entry. Error text never contains
        a token value or configured identity.
        """

        problems = validate_token_map(
            self.auth_tokens,
            production=self.app_env.lower() in PRODUCTION_ENVIRONMENTS,
        )
        if problems:
            raise ValueError("invalid AUTH_TOKENS configuration: " + "; ".join(problems))
        return self

    # Database pool and statement limits. pool_pre_ping detects stale
    # connections; the pool timeout and statement timeout bound request waits.
    db_pool_size: int = 5
    db_max_overflow: int = 5
    db_pool_timeout_seconds: float = 5.0
    db_pool_recycle_seconds: int = 1800
    db_connect_timeout_seconds: int = 5
    db_statement_timeout_ms: int = 15_000

    # Readiness schema validation is cached only on success. A failed check is
    # retried on the next readiness probe so repairing the schema is visible
    # immediately; a successful introspection is reused briefly to keep a busy
    # /ready endpoint from scanning the catalog on every scrape.
    schema_validation_cache_seconds: float = Field(default=30.0, ge=0)

    # Evidence assembly defaults. Token estimates use a documented character
    # approximation, not a model tokenizer.
    answer_context_token_budget: int = 4000
    answer_max_evidence: int = 8
    answer_citation_excerpt_chars: int = 600

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)


@lru_cache
def get_settings() -> Settings:
    return Settings()

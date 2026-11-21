"""Deployment checks for the Compose API service and startup ordering.

These tests read ``docker-compose.yml`` directly so they run without a Docker
daemon. The final test invokes ``docker compose config`` when the Compose CLI is
installed, which also validates interpolation and YAML structure.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from urllib.parse import urlsplit

import pytest

yaml = pytest.importorskip("yaml")

ROOT = Path(__file__).resolve().parents[1]
COMPOSE_PATH = ROOT / "docker-compose.yml"
ENV_EXAMPLE_PATH = ROOT / ".env.example"


def _env_example() -> dict[str, str]:
    """Parse simple ``KEY=value`` lines from ``.env.example``."""

    values: dict[str, str] = {}
    for raw in ENV_EXAMPLE_PATH.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip()
    return values


def _url_port(url: str) -> tuple[str, int | None]:
    parsed = urlsplit(url.replace("postgresql+psycopg://", "postgresql://", 1))
    return parsed.hostname or "", parsed.port


@pytest.fixture(scope="module")
def compose() -> dict:
    data = yaml.safe_load(COMPOSE_PATH.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


@pytest.fixture(scope="module")
def services(compose: dict) -> dict:
    return compose["services"]


def test_api_builds_from_the_shared_application_image(services: dict) -> None:
    api = services["api"]

    assert api["build"] == {"context": ".", "dockerfile": "Dockerfile"}
    # The API must reuse the same image definition as the background services.
    for name in ("migrate", "worker", "reconciliation-worker", "beat"):
        assert services[name]["build"] == api["build"]


def test_api_command_runs_uvicorn_on_the_container_interface(services: dict) -> None:
    command = services["api"]["command"]

    assert command == [
        "uvicorn",
        "app.main:app",
        "--host=0.0.0.0",
        "--port=8000",
    ]
    assert "app.db.migrations" not in " ".join(command)


def test_api_host_port_is_configurable_and_defaults_to_8000(services: dict) -> None:
    assert services["api"]["ports"] == [
        "${API_BIND_ADDRESS:-127.0.0.1}:${API_PORT:-8000}:8000"
    ]


def test_api_starts_only_after_healthy_dependencies_and_migration(services: dict) -> None:
    depends_on = services["api"]["depends_on"]

    assert depends_on["postgres"] == {"condition": "service_healthy"}
    assert depends_on["redis"] == {"condition": "service_healthy"}
    assert depends_on["migrate"] == {"condition": "service_completed_successfully"}
    assert services["api"]["restart"] == "unless-stopped"


def test_api_health_check_uses_readiness_not_liveness(services: dict) -> None:
    healthcheck = services["api"]["healthcheck"]
    command = " ".join(healthcheck["test"])

    assert "http://127.0.0.1:8000/ready" in command
    assert "/health" not in command
    assert healthcheck["interval"] == "15s"
    assert healthcheck["timeout"] == "5s"
    assert healthcheck["retries"] == 5
    assert healthcheck["start_period"] == "30s"


def test_api_environment_matches_worker_and_migration_configuration(services: dict) -> None:
    api = services["api"]["environment"]
    worker = services["worker"]["environment"]
    reconciliation_worker = services["reconciliation-worker"]["environment"]
    migrate = services["migrate"]["environment"]
    beat = services["beat"]["environment"]

    assert (
        api["DATABASE_URL"]
        == worker["DATABASE_URL"]
        == reconciliation_worker["DATABASE_URL"]
        == beat["DATABASE_URL"]
    )
    assert api["DATABASE_URL"] == migrate["DATABASE_URL"]
    assert "@postgres:5432/" in api["DATABASE_URL"]

    assert api["REDIS_URL"] == worker["REDIS_URL"] == reconciliation_worker["REDIS_URL"]
    assert (
        api["CELERY_BROKER_URL"]
        == worker["CELERY_BROKER_URL"]
        == reconciliation_worker["CELERY_BROKER_URL"]
        == beat["CELERY_BROKER_URL"]
    )
    assert (
        api["CELERY_TASK_DEFAULT_QUEUE"]
        == worker["CELERY_TASK_DEFAULT_QUEUE"]
        == reconciliation_worker["CELERY_TASK_DEFAULT_QUEUE"]
        == beat["CELERY_TASK_DEFAULT_QUEUE"]
    )
    assert (
        api["CELERY_RECONCILIATION_QUEUE"]
        == worker["CELERY_RECONCILIATION_QUEUE"]
        == reconciliation_worker["CELERY_RECONCILIATION_QUEUE"]
        == beat["CELERY_RECONCILIATION_QUEUE"]
    )

    assert api["APP_ENV"] == worker["APP_ENV"] == migrate["APP_ENV"]
    assert api["LOG_LEVEL"] == worker["LOG_LEVEL"] == migrate["LOG_LEVEL"]

    # API-only settings still come from the same environment contract.
    assert "AUTH_TOKENS" in api
    assert api["LLM_PROVIDER"] == "${LLM_PROVIDER:-extractive}"
    assert api["SCHEMA_VALIDATION_CACHE_SECONDS"] == "${SCHEMA_VALIDATION_CACHE_SECONDS:-30}"


def test_migration_worker_and_beat_topology_is_preserved(services: dict) -> None:
    migrate = services["migrate"]
    assert migrate["command"] == ["python", "-m", "app.db.migrations"]
    assert migrate["restart"] == "no"
    assert migrate["depends_on"]["postgres"] == {"condition": "service_healthy"}

    worker = services["worker"]
    assert worker["command"][:4] == [
        "celery",
        "-A",
        "app.celery_app:celery_app",
        "worker",
    ]
    assert worker["depends_on"]["migrate"] == {
        "condition": "service_completed_successfully"
    }

    beat = services["beat"]
    assert beat["command"][:4] == [
        "celery",
        "-A",
        "app.celery_app:celery_app",
        "beat",
    ]
    assert beat["depends_on"]["migrate"] == {
        "condition": "service_completed_successfully"
    }


def test_workers_consume_disjoint_single_queues(services: dict) -> None:
    worker_command = services["worker"]["command"]
    reconciliation_command = services["reconciliation-worker"]["command"]

    assert "--queues=${CELERY_TASK_DEFAULT_QUEUE:-ingestion}" in worker_command
    assert "--queues=${CELERY_RECONCILIATION_QUEUE:-reconciliation}" in reconciliation_command
    assert "--hostname=ingestion@%h" in worker_command
    assert "--hostname=reconciliation@%h" in reconciliation_command

    assert not any("RECONCILIATION_QUEUE" in part for part in worker_command)
    assert not any("DEFAULT_QUEUE" in part for part in reconciliation_command)


def test_reconciliation_worker_is_bounded_and_migration_gated(services: dict) -> None:
    reconciliation_worker = services["reconciliation-worker"]
    command = reconciliation_worker["command"]

    assert command[:4] == [
        "celery",
        "-A",
        "app.celery_app:celery_app",
        "worker",
    ]
    assert "--concurrency=${CELERY_RECONCILIATION_WORKER_CONCURRENCY:-1}" in command
    assert "--max-tasks-per-child=1000" in command
    assert reconciliation_worker["mem_limit"] == "512m"
    assert reconciliation_worker["depends_on"]["postgres"] == {
        "condition": "service_healthy"
    }
    assert reconciliation_worker["depends_on"]["redis"] == {
        "condition": "service_healthy"
    }
    assert reconciliation_worker["depends_on"]["migrate"] == {
        "condition": "service_completed_successfully"
    }
    assert reconciliation_worker["restart"] == "unless-stopped"


def test_worker_health_checks_target_their_own_node_and_validate_schema(
    services: dict,
) -> None:
    expected_destinations = {
        "worker": "ingestion@$(hostname)",
        "reconciliation-worker": "reconciliation@$(hostname)",
    }

    for name, destination in expected_destinations.items():
        healthcheck = services[name]["healthcheck"]
        command = " ".join(healthcheck["test"])
        assert f"--destination {destination}" in command
        assert "python -m app.db.schema_validation --quiet" in command
        assert healthcheck["start_period"] == "30s"


def test_postgres_container_port_stays_5432_and_host_port_is_configurable(
    services: dict,
) -> None:
    assert services["postgres"]["ports"] == [
        "${POSTGRES_BIND_ADDRESS:-127.0.0.1}:${POSTGRES_PORT:-5433}:5432"
    ]


def test_datastores_and_api_publish_on_loopback_by_default(services: dict) -> None:
    assert services["postgres"]["ports"][0].startswith(
        "${POSTGRES_BIND_ADDRESS:-127.0.0.1}:"
    )
    assert services["redis"]["ports"] == [
        "${REDIS_BIND_ADDRESS:-127.0.0.1}:${REDIS_PORT:-6379}:6379"
    ]
    assert services["api"]["ports"][0].startswith("${API_BIND_ADDRESS:-127.0.0.1}:")


def test_all_application_containers_use_the_internal_postgres_address(
    services: dict,
) -> None:
    for name in ("migrate", "api", "worker", "reconciliation-worker", "beat"):
        url = services[name]["environment"]["DATABASE_URL"]
        assert "@postgres:5432/" in url, name
        assert "localhost" not in url, name
        assert "5433" not in url, name


def test_migration_service_uses_the_internal_postgres_address(services: dict) -> None:
    url = services["migrate"]["environment"]["DATABASE_URL"]

    # Container-to-container access never uses the host-published port.
    assert "@postgres:5432/" in url
    assert "postgres:5433" not in url
    assert "localhost:5433" not in url


def test_env_example_documents_the_same_default_host_port(services: dict) -> None:
    env = _env_example()

    assert env["POSTGRES_PORT"] == "5433"
    assert env["POSTGRES_BIND_ADDRESS"] == "127.0.0.1"
    assert services["postgres"]["ports"] == [
        "${POSTGRES_BIND_ADDRESS:-" + env["POSTGRES_BIND_ADDRESS"] + "}:"
        "${POSTGRES_PORT:-" + env["POSTGRES_PORT"] + "}:5432"
    ]

    hostname, port = _url_port(env["DATABASE_URL"])
    assert hostname == "localhost"
    assert port == int(env["POSTGRES_PORT"])


def test_settings_default_matches_the_documented_host_port() -> None:
    from app.core.config import Settings

    default = Settings.model_fields["database_url"].default

    hostname, port = _url_port(default)
    assert hostname == "localhost"
    assert port == int(_env_example()["POSTGRES_PORT"])


def test_readme_documents_both_postgres_access_modes() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    # Host-run processes use the published port; containers use the service
    # address. The override variable and conflict diagnosis must be documented.
    assert "POSTGRES_PORT" in readme
    assert "localhost:5433" in readme
    assert "postgres:5432" in readme
    assert "5432" in readme
    assert "conflict" in readme.lower()


def test_production_relevant_images_are_pinned(services: dict) -> None:
    assert services["postgres"]["image"] == "pgvector/pgvector:0.8.0-pg16"
    assert services["redis"]["image"] == "redis:7.4.2-alpine"

    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "FROM python:3.13.9-slim-bookworm" in dockerfile
    assert ":latest" not in dockerfile

    for name, service in services.items():
        image = service.get("image")
        if image:
            assert image != "latest", name
            assert not image.endswith(":latest"), name


def test_database_migration_doc_matches_the_pinned_pgvector_image(services: dict) -> None:
    doc = (ROOT / "docs" / "DATABASE_MIGRATIONS.md").read_text(encoding="utf-8")

    pinned = services["postgres"]["image"]
    assert pinned == "pgvector/pgvector:0.8.0-pg16"

    # The documentation must not advertise a floating tag that Compose and CI
    # have already pinned.
    assert f"`{pinned}`" in doc
    assert "pgvector/pgvector:pg16" not in doc


def test_database_migration_doc_does_not_pass_a_sqlalchemy_url_to_pg_dump() -> None:
    doc = (ROOT / "docs" / "DATABASE_MIGRATIONS.md").read_text(encoding="utf-8")

    # pg_dump rejects the SQLAlchemy driver suffix (postgresql+psycopg://).
    assert '"$DATABASE_URL"' not in doc
    assert "postgresql+psycopg://" in doc  # the incompatible form is explained
    # The documented path converts the URL or uses libpq parameters.
    assert "scripts/ops/db_backup.py" in doc
    assert "PGPASSFILE" in doc


def test_application_image_runs_as_a_non_root_user() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "useradd" in dockerfile
    assert "USER opsatlas" in dockerfile
    assert "USER root" not in dockerfile
    # The application tree must be writable by the runtime user for Celery beat
    # and Python's import/cache behavior even though bytecode is disabled.
    assert "chown -R opsatlas:opsatlas /app" in dockerfile


def test_application_containers_drop_privileges_and_are_bounded(services: dict) -> None:
    expected_limits = {
        "migrate": "512m",
        "api": "512m",
        "worker": "1g",
        "reconciliation-worker": "512m",
        "beat": "256m",
    }

    for name, limit in expected_limits.items():
        service = services[name]
        assert service["security_opt"] == ["no-new-privileges:true"], name
        assert service["cap_drop"] == ["ALL"], name
        assert service["mem_limit"] == limit, name

    # Datastores keep their own runtime defaults but are never privileged.
    for name in ("postgres", "redis"):
        assert services[name].get("privileged") is not True


def test_compose_configuration_parses_with_the_compose_cli() -> None:
    if shutil.which("docker") is None:
        pytest.skip("docker CLI is not installed")

    version = subprocess.run(
        ["docker", "compose", "version"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if version.returncode != 0:
        pytest.skip("docker compose plugin is not available")

    result = subprocess.run(
        ["docker", "compose", "config", "--quiet"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

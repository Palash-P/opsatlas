"""Host versus container embedding addresses (B-1 regression tests).

The embedding service runs on the host, outside Compose. Host-run processes can
reach it on loopback; containers cannot, because a container's loopback is the
container. ``.env.example`` therefore documents two variables:

* ``EMBEDDING_URL`` -- host-run processes (read by ``Settings``);
* ``CONTAINER_EMBEDDING_URL`` -- mapped by Compose into the container's
  ``EMBEDDING_URL``.

These tests read the repository files directly, so they run without a Docker
daemon. The final test also renders the Compose configuration with the example
env file when the Compose CLI is installed.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

ROOT = Path(__file__).resolve().parents[1]
COMPOSE_PATH = ROOT / "docker-compose.yml"
ENV_EXAMPLE_PATH = ROOT / ".env.example"
README_PATH = ROOT / "README.md"
EMBEDDING_DOC_PATH = ROOT / "docs" / "EMBEDDING_CONFIGURATION.md"

HOST_EMBEDDING_URL = "http://127.0.0.1:8091"
CONTAINER_EMBEDDING_URL = "http://host.docker.internal:8091"


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


def _resolve(value: str, env: dict[str, str]) -> str:
    """Resolve Compose-style ``${VAR:-default}`` in a single environment value."""

    pattern = re.compile(r"\$\{([A-Z0-9_]+)(?::-([^}]*))?\}")

    def replace(match: re.Match[str]) -> str:
        name, default = match.group(1), match.group(2)
        if name in env and env[name] != "":
            return env[name]
        return default or ""

    return pattern.sub(replace, value)


@pytest.fixture(scope="module")
def services() -> dict:
    data = yaml.safe_load(COMPOSE_PATH.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data["services"]


def test_env_example_separates_host_and_container_addresses() -> None:
    env = _env_example()

    assert env["EMBEDDING_URL"] == HOST_EMBEDDING_URL
    assert env["CONTAINER_EMBEDDING_URL"] == CONTAINER_EMBEDDING_URL
    # The host value must never be the container-gateway alias.
    assert "host.docker.internal" not in env["EMBEDDING_URL"]


def test_host_run_settings_use_the_host_url(monkeypatch) -> None:
    from app.core.config import Settings

    monkeypatch.delenv("EMBEDDING_URL", raising=False)
    monkeypatch.delenv("OPSATLAS_EMBEDDING_URL", raising=False)
    monkeypatch.delenv("CONTAINER_EMBEDDING_URL", raising=False)

    settings = Settings(_env_file=ENV_EXAMPLE_PATH)

    assert settings.embedding_url == HOST_EMBEDDING_URL
    assert "host.docker.internal" not in settings.embedding_url
    # The container-only variable is Compose interpolation input, not an
    # application setting; it must not leak into Settings.
    assert "container_embedding_url" not in Settings.model_fields


def test_api_container_receives_the_container_url(services: dict) -> None:
    env = _env_example()
    api_environment = services["api"]["environment"]

    raw = api_environment["EMBEDDING_URL"]
    assert raw == "${CONTAINER_EMBEDDING_URL:-" + CONTAINER_EMBEDDING_URL + "}"
    assert _resolve(raw, env) == CONTAINER_EMBEDDING_URL


def test_worker_receives_the_container_url(services: dict) -> None:
    env = _env_example()
    worker_environment = services["worker"]["environment"]

    raw = worker_environment["EMBEDDING_URL"]
    assert raw == "${CONTAINER_EMBEDDING_URL:-" + CONTAINER_EMBEDDING_URL + "}"
    assert _resolve(raw, env) == CONTAINER_EMBEDDING_URL


def test_every_service_that_embeds_uses_the_container_address(services: dict) -> None:
    env = _env_example()

    # API and ingestion worker call the provider. Reconciliation only dispatches
    # jobs, and migrate/beat never embed, so they must not carry the variable.
    embedding_services = {
        name
        for name, service in services.items()
        if "EMBEDDING_URL" in service.get("environment", {})
    }
    assert embedding_services == {"api", "worker"}

    for name in embedding_services:
        resolved = _resolve(services[name]["environment"]["EMBEDDING_URL"], env)
        assert resolved == CONTAINER_EMBEDDING_URL, name
        assert resolved != HOST_EMBEDDING_URL, name


def test_documented_env_example_workflow_resolves_both_modes(services: dict) -> None:
    """Copying .env.example to .env must work for host and container callers."""

    env = _env_example()

    host_url = env["EMBEDDING_URL"]
    assert host_url == HOST_EMBEDDING_URL

    for name in ("api", "worker"):
        container_url = _resolve(services[name]["environment"]["EMBEDDING_URL"], env)
        assert container_url == CONTAINER_EMBEDDING_URL, name

    # Both modes must be documented where an operator will look.
    readme = README_PATH.read_text(encoding="utf-8")
    embedding_doc = EMBEDDING_DOC_PATH.read_text(encoding="utf-8")
    for document in (readme, embedding_doc):
        assert "EMBEDDING_URL" in document
        assert "CONTAINER_EMBEDDING_URL" in document
        assert "host.docker.internal" in document


def test_compose_renders_the_container_url_with_the_example_env_file() -> None:
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
        ["docker", "compose", "--env-file", ".env.example", "config"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    rendered = yaml.safe_load(result.stdout)
    for name in ("api", "worker"):
        assert (
            rendered["services"][name]["environment"]["EMBEDDING_URL"]
            == CONTAINER_EMBEDDING_URL
        ), name

# Dependency and Version Reproducibility

## Authoritative application version

`app/version.py` defines `APP_VERSION`. The package version is dynamic:

```toml
[project]
dynamic = ["version"]

[tool.setuptools.dynamic]
version = { attr = "app.version.__version__" }
```

FastAPI/OpenAPI metadata, `/health`, evaluation report defaults, and installed
distribution metadata (`importlib.metadata.version("opsatlas")`) all read this
one value. Do not add a second hardcoded version to documentation, Compose, or
Python modules. To release a new version, edit `app/version.py` only, then
rebuild the image and verify:

```bash
python -c "from importlib.metadata import version; from app.version import APP_VERSION; print(version('opsatlas'), APP_VERSION)"
```

`pyproject.toml` keeps direct dependency ranges; `requirements.lock` pins every
resolved runtime and development dependency, including transitives.

## Lockfile

`requirements.lock` is generated with `uv` from `pyproject.toml` for Python 3.13,
matching the Docker base image (`python:3.13.9-slim-bookworm`). It includes the
`dev` extra so CI can install one verified set for lint and tests. Docker installs
the runtime project with the lock as a constraint:

```dockerfile
RUN pip install --no-cache-dir --constraint requirements.lock .
```

### Transitive pins that look unused

`opentelemetry-api==1.45.1` is pinned because it is a required transitive
dependency, not because the application imports it directly:

```text
fastapi 0.161.0
  -> opentelemetry-api>=1.44.0   (unconditional Requires-Dist)
```

`redis` and `starlette` only pull OpenTelemetry through optional extras, so
FastAPI is the binding chain. The application does not install an SDK, exporter,
or tracing provider. Do not remove this pin unless FastAPI drops the hard
requirement or the FastAPI range is changed and the lock is regenerated and
re-verified.

### Regenerate without changing verified pins

Run from the repository root with the lockfile present:

```bash
uv pip compile pyproject.toml --extra dev --universal --python-version 3.13 \
  --no-header --no-annotate -o requirements.lock
```

`uv` uses the existing output as version preferences, so this refreshes only what
the declared ranges require. Verified against `requirements.lock` at the time of
writing: the command reproduces the same 49 pins. It also rewrites the file from
scratch, which removes the explanatory header comment block at the top; restore
that header while reviewing the diff. Never hand-edit individual pins.

### Deliberate dependency upgrade

```bash
# 1. Change a range in pyproject.toml if needed.
# 2. Resolve fresh.
uv pip compile pyproject.toml --extra dev --universal --python-version 3.13 \
  --upgrade --no-header --no-annotate -o requirements.lock
# 3. Verify in a clean environment (below), run Ruff and the full suite,
#    then build the Docker image.
```

Do not upgrade PostgreSQL, Redis, pgvector, or the Python major version as part
of a lock refresh. Those are deployment contract changes and need their own
migration, compatibility, and rollback review.

### Verify a clean install

```bash
python -m venv .venv-verify
# Windows: .venv-verify\Scripts\activate
# macOS/Linux: source .venv-verify/bin/activate
python -m pip install --upgrade pip
pip install --constraint requirements.lock ".[dev]"
pip check
python -c "from importlib.metadata import version; from app.version import APP_VERSION; assert version('opsatlas') == APP_VERSION"
pytest -ra
```

The Docker build performs the runtime half of this check using only
`requirements.lock`.

## CI

`.github/workflows/ci.yml` runs:

- `ruff check .`;
- the offline suite (live tests report as skipped, with `-ra`);
- a PostgreSQL/pgvector + Redis integration job that starts from an empty
  database, runs the migration runner twice, validates schema readiness, seeds
  the synthetic corpus through the deterministic embedding service, and runs the
  full live suite plus API liveness/readiness probes;
- `docker compose config` and a real image build with in-image version/import
  verification.

CI never uses a production `DATABASE_URL`. The integration database is the
service container's `opsatlas_ci`; tests that need isolation create uniquely
named `opsatlas_test_<uuid>` databases and drop only those.

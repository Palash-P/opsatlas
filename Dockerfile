# Shared application image for the FastAPI API, the one-shot migration job, and
# the Celery worker/beat processes. The host virtual environment remains
# available for local development.
#
# The base image is pinned to an explicit patch release. Operators upgrading it
# should change this tag deliberately, then run the full test suite and
# ``docker compose build`` before deploying. Digest pinning (``python@sha256:...``)
# is supported by Docker but makes security updates harder to review; if used,
# record the digest in the change ticket.
FROM python:3.13.9-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/home/opsatlas

WORKDIR /app

# Install the exact dependency set resolved in requirements.lock. The lock is
# generated from pyproject.toml and pins every transitive dependency; regenerate
# it with the procedure in docs/DEPENDENCIES.md rather than editing it by hand.
COPY pyproject.toml README.md requirements.lock ./
COPY app ./app
COPY sql ./sql

RUN pip install --no-cache-dir --constraint requirements.lock . \
    && groupadd --gid 10001 opsatlas \
    && useradd --uid 10001 --gid opsatlas --create-home --shell /usr/sbin/nologin opsatlas \
    && chown -R opsatlas:opsatlas /app /home/opsatlas

# Run as an unprivileged user. Celery beat writes its schedule into /app, which
# is owned by this user; application logs go to stdout.
USER opsatlas

EXPOSE 8000

CMD ["celery", "-A", "app.celery_app:celery_app", "worker", "--loglevel=INFO"]

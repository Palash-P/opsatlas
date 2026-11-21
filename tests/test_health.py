from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_reports_process_is_alive() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_ready_reports_dependency_failures_without_leaking_details(monkeypatch) -> None:
    from app import main

    def fail() -> None:
        raise RuntimeError("secret connection string should not be returned")

    monkeypatch.setattr(main, "check_postgres", fail)
    monkeypatch.setattr(main, "check_redis", lambda: None)
    response = client.get("/ready")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "not_ready"
    assert body["checks"] == {"postgres": "error", "schema": "unavailable", "redis": "ok"}
    assert "secret connection string" not in response.text


def test_ready_returns_200_when_dependencies_are_healthy(monkeypatch) -> None:
    from app import main

    monkeypatch.setattr(main, "check_postgres", lambda: None)
    monkeypatch.setattr(main, "check_schema_readiness", lambda: None)
    monkeypatch.setattr(main, "check_redis", lambda: None)

    response = client.get("/ready")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "checks": {
            "postgres": "ok",
            "schema": "ok",
            "redis": "ok",
        },
    }


def test_ready_degrades_but_stays_serving_when_redis_fails(monkeypatch) -> None:
    from app import main

    def fail() -> None:
        raise RuntimeError("redis unavailable")

    monkeypatch.setattr(main, "check_postgres", lambda: None)
    monkeypatch.setattr(main, "check_schema_readiness", lambda: None)
    monkeypatch.setattr(main, "check_redis", fail)

    response = client.get("/ready")

    assert response.status_code == 200
    assert response.json() == {
        "status": "degraded",
        "checks": {
            "postgres": "ok",
            "schema": "ok",
            "redis": "error",
        },
    }
    assert "redis unavailable" not in response.text


def test_ready_fails_when_database_schema_is_incompatible(monkeypatch) -> None:
    from app import main

    def fail_schema() -> None:
        raise RuntimeError("schema detail must not be returned to clients")

    monkeypatch.setattr(main, "check_postgres", lambda: None)
    monkeypatch.setattr(main, "check_schema_readiness", fail_schema)
    monkeypatch.setattr(main, "check_redis", lambda: None)

    response = client.get("/ready")

    assert response.status_code == 503
    assert response.json() == {
        "status": "not_ready",
        "checks": {
            "postgres": "ok",
            "schema": "error",
            "redis": "ok",
        },
    }
    assert "schema detail" not in response.text


def test_health_is_not_authenticated_and_has_a_request_id() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.headers["X-Request-ID"]


def test_health_is_liveness_only_even_when_dependencies_are_down(monkeypatch) -> None:
    from app import main

    def fail() -> None:
        raise RuntimeError("database is down")

    monkeypatch.setattr(main, "check_postgres", fail)
    monkeypatch.setattr(main, "check_redis", fail)
    monkeypatch.setattr(main, "check_schema_readiness", fail)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def _capture_events(logger):
    import logging

    events: list[dict] = []

    class Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            data = getattr(record, "event_data", None)
            if isinstance(data, dict):
                events.append(data)

    handler = Capture()
    logger.addHandler(handler)
    return events, handler


def test_ready_logs_a_sanitized_dependency_failure_event(monkeypatch) -> None:
    from app import main

    events, handler = _capture_events(main.logger)

    def fail_postgres() -> None:
        raise RuntimeError("postgresql://user:super-secret@db:5432/opsatlas")

    monkeypatch.setattr(main, "check_postgres", fail_postgres)
    monkeypatch.setattr(main, "check_redis", lambda: None)

    try:
        response = client.get("/ready")
    finally:
        main.logger.removeHandler(handler)

    assert response.status_code == 503
    readiness = [event for event in events if event.get("event") == "readiness"]
    assert len(readiness) == 1
    assert readiness[0]["check"] == "postgres"
    assert readiness[0]["outcome"] == "unavailable"
    assert readiness[0]["error_category"] == "database_unavailable"
    assert "request_id" in readiness[0]
    assert "super-secret" not in str(readiness)
    assert "super-secret" not in response.text


def test_schema_logger_is_configured_for_json_events() -> None:
    import logging

    from app.core.config import Settings
    from app.core.observability import SCHEMA_LOGGER_NAME, configure_logging

    configure_logging(Settings(_env_file=None, app_env="local"))
    logger = logging.getLogger(SCHEMA_LOGGER_NAME)

    assert any(getattr(handler, "_opsatlas_json", False) for handler in logger.handlers)
    assert logger.propagate is False

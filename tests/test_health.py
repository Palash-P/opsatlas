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
    assert body["checks"] == {"postgres": "error", "redis": "ok"}
    assert "secret connection string" not in response.text

def test_ready_returns_200_when_dependencies_are_healthy(monkeypatch) -> None:
    from app import main

    monkeypatch.setattr(main, "check_postgres", lambda: None)
    monkeypatch.setattr(main, "check_redis", lambda: None)

    response = client.get("/ready")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "checks": {
            "postgres": "ok",
            "redis": "ok",
        },
    }


def test_ready_degrades_but_stays_serving_when_redis_fails(monkeypatch) -> None:
    from app import main

    def fail() -> None:
        raise RuntimeError("redis unavailable")

    monkeypatch.setattr(main, "check_postgres", lambda: None)
    monkeypatch.setattr(main, "check_redis", fail)

    response = client.get("/ready")

    assert response.status_code == 200
    assert response.json() == {
        "status": "degraded",
        "checks": {
            "postgres": "ok",
            "redis": "error",
        },
    }
    assert "redis unavailable" not in response.text


def test_health_is_not_authenticated_and_has_a_request_id() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.headers["X-Request-ID"]
from fastapi.testclient import TestClient

from app.main import create_app
from app.services.dependencies import DependencyHealthService


async def ready() -> bool:
    return True


async def unavailable() -> bool:
    return False


def test_health_reports_liveness_and_trace_id() -> None:
    with TestClient(create_app()) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.headers["X-Trace-ID"]


def test_ready_reports_each_dependency() -> None:
    app = create_app()
    app.state.dependencies = DependencyHealthService(ready, unavailable, ready)

    with TestClient(app) as client:
        response = client.get("/ready")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "dependencies": {"postgres": "ready", "qdrant": "unavailable", "neo4j": "ready"},
    }


def test_ready_requires_postgres() -> None:
    app = create_app()
    app.state.dependencies = DependencyHealthService(unavailable, ready, ready)

    with TestClient(app) as client:
        response = client.get("/ready")

    assert response.status_code == 503
    assert response.json()["status"] == "not_ready"

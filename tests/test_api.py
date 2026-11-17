"""
Integration tests for FastAPI endpoints.
"""
import pytest
from fastapi.testclient import TestClient
from app.main import app, seed_data_if_empty
from app.db.session import init_db


@pytest.fixture(scope="module", autouse=True)
def setup_app_state():
    init_db()
    seed_data_if_empty()


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "indexed_jobs" in data


def test_jobs_list_and_search(client):
    # List jobs
    res = client.get("/api/v1/jobs/")
    assert res.status_code == 200
    jobs = res.json()
    assert len(jobs) > 0

    # Semantic search
    search_res = client.post("/api/v1/jobs/search", json={"query": "AI systems engineer", "top_k": 3})
    assert search_res.status_code == 200
    results = search_res.json()
    assert len(results) > 0
    assert "score" in results[0]


def test_agent_chat_api(client):
    res = client.post("/api/v1/agent/chat", json={
        "message": "Find me Python jobs paying at least $150k",
        "session_id": "test_api_sess"
    })
    assert res.status_code == 200
    data = res.json()
    assert "reply" in data
    assert len(data["tool_calls"]) > 0


def test_evals_run_api(client):
    res = client.post("/api/v1/evals/run?k=5")
    assert res.status_code == 200
    data = res.json()
    assert "run_id" in data
    assert "pass_rate_percent" in data
    assert len(data["metrics"]) > 0

import pytest
from fastapi.testclient import TestClient

from app.agent import ConversationStore, PropertyAgent
from app.llm import RuleBasedLLM
from app.main import app, get_agent, get_store
from app.tools import PropertyTools


@pytest.fixture
def client(settings):
    agent = PropertyAgent(RuleBasedLLM(), PropertyTools(settings=settings), settings)
    store = ConversationStore()
    app.dependency_overrides[get_agent] = lambda: agent
    app.dependency_overrides[get_store] = lambda: store
    yield TestClient(app), agent
    app.dependency_overrides.clear()


def test_full_flow_with_confirmation(client):
    c, agent = client
    r = c.post("/chat", json={"message": "I need a 2 bed under £2400 from 2026-12-20"}).json()
    cid = r["conversation_id"]
    assert r["status"] == "ok" and r["preferences"]["budget_max_pcm"] == 2400 and r["preferences"]["bedrooms"] == 2
    assert "P002" in r["reply"]

    r = c.post("/chat", json={"conversation_id": cid, "message": "I'd like to view P002"}).json()
    assert "P002-S1" in r["reply"]

    r = c.post("/chat", json={"conversation_id": cid, "message": "book P002-S1 for Jane Doe jane@example.com"}).json()
    assert r["status"] == "awaiting_confirmation" and r["pending_action"]["tool"] == "book_viewing"
    assert agent.tools.bookings == []

    r = c.post("/chat", json={"conversation_id": cid, "message": "Yes", "confirm": True}).json()
    assert r["status"] == "ok" and "B0001" in r["reply"] and len(agent.tools.bookings) == 1
    assert r["metrics"]["conversation"]["llm_calls"] >= 4

    meta = c.get(f"/conversations/{cid}").json()
    assert meta["status"] == "active" and meta["metrics"]["tool_calls"] >= 4


def test_validation_and_404(client):
    c, _ = client
    assert c.post("/chat", json={}).status_code == 422
    assert c.get("/conversations/nope").status_code == 404
    assert c.get("/health").json() == {"ok": True}

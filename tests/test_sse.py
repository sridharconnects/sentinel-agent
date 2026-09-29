"""API + SSE event ordering (FastAPI TestClient, mock mode)."""
import json

from fastapi.testclient import TestClient

from app.agents import MAX_INPUT_CHARS
from app.api import app

client = TestClient(app)


def _sse_events(resp_text: str) -> list[dict]:
    events = []
    for line in resp_text.splitlines():
        if line.startswith("data:"):
            events.append(json.loads(line[5:].strip()))
    return events


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_stream_event_ordering():
    with client.stream("GET", "/stream",
                       params={"question": "What is agentic AI evaluation?"}) as r:
        assert r.status_code == 200
        text = r.read().decode()
    events = _sse_events(text)
    assert events, "expected SSE events"
    # First event starts the pipeline; last event completes it.
    assert events[0]["type"] == "node_start" and events[0]["node"] == "supervisor"
    assert events[-1]["type"] == "done"
    # Every node_start precedes its node_end; tokens only come from
    # analyst/finalizer; exactly one done; no error.
    seen_start: dict[str, int] = {}
    for i, e in enumerate(events):
        if e["type"] == "node_start":
            seen_start[e["node"]] = i
        if e["type"] == "node_end":
            assert e["node"] in seen_start and seen_start[e["node"]] < i
        if e["type"] == "token":
            assert e["node"] in ("analyst", "finalizer")
    assert sum(1 for e in events if e["type"] == "done") == 1
    assert not [e for e in events if e["type"] == "error"]


def test_brief_returns_cited_answer():
    r = client.post("/brief", json={"question": "What is RAG?"})
    assert r.status_code == 200
    body = r.json()
    assert "[1]" in body["final_answer"]
    assert body["events"][-1]["type"] == "done"


def test_stream_rejects_overlong_question():
    r = client.get("/stream", params={"question": "x" * (MAX_INPUT_CHARS + 1)})
    assert r.status_code == 422

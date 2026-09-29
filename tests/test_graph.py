"""LangGraph wiring compiles and runs end to end (mock mode)."""
import pytest

pytest.importorskip("langgraph")

from app.graph import build_graph
from app.llm import MockLLM


def test_langgraph_pipeline_completes():
    graph = build_graph(llm=MockLLM())
    final = graph.invoke({"question": "What is agentic AI?",
                          "events": [], "revision_count": 0})
    assert final["final_answer"]
    assert "[1]" in final["final_answer"]
    assert final["events"], "expected recorded node events"
    assert final["events"][0]["node"] == "supervisor"

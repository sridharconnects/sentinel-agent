"""LangGraph wiring for the sentinel-agent nodes.

This is the framework-orchestrated alternative to ``stream_briefing()`` in
``app/agents.py`` (which the API uses because it yields a uniform event
stream with or without LangGraph installed). Both paths execute the same node
functions from ``app/agents.py``.

Requires the ``langgraph`` package (``pip install -r requirements.txt``).
"""
from __future__ import annotations

from typing import Optional

try:
    from langgraph.graph import END, StateGraph

    LANGGRAPH_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only when installed
    LANGGRAPH_AVAILABLE = False

from .agents import (
    MAX_CRITIC_ROUNDS,
    analyst_node,
    critic_node,
    finalizer_node,
    researcher_node,
    supervisor_node,
)
from .llm import BaseLLM, get_llm


def _drain(node_fn, llm: BaseLLM):
    """Adapt an event-yielding node to LangGraph's ``(state) -> updates`` shape.

    Returns the *full* state (not just deltas): node functions mutate the
    state dict in place, and this langgraph version replaces rather than
    merges node outputs, so deltas alone would drop keys like ``question``.
    """

    def _node(state: dict) -> dict:
        updates = dict(state)  # propagate everything forward
        updates["events"] = state.get("events", []) + list(node_fn(state, llm))
        for key in ("subqueries", "evidence", "draft", "critique",
                    "critic_passed", "revision_count", "final_answer"):
            if key in state:
                updates[key] = state[key]
        return updates

    return _node


def _critic_router(state: dict) -> str:
    if state.get("critic_passed"):
        return "finalizer"
    if state.get("revision_count", 0) >= MAX_CRITIC_ROUNDS:
        return "finalizer"
    return "analyst"


def _critic_with_count(llm: BaseLLM):
    """Critic node wrapper that bumps ``revision_count`` on FAIL (the router
    itself must stay pure: router mutations are not persisted as updates)."""

    _base = _drain(critic_node, llm)

    def _node(state: dict) -> dict:
        updates = _base(state)
        if not updates.get("critic_passed"):
            updates["revision_count"] = updates.get("revision_count", 0) + 1
        return updates

    return _node


def build_graph(llm: Optional[BaseLLM] = None):
    """Build and compile the StateGraph. Raises if langgraph is missing."""
    if not LANGGRAPH_AVAILABLE:
        raise RuntimeError(
            "langgraph is not installed. Run: pip install -r requirements.txt"
        )
    llm = llm or get_llm()
    g: StateGraph = StateGraph(dict)
    g.add_node("supervisor", _drain(supervisor_node, llm))
    g.add_node("researcher", _drain(researcher_node, llm))
    g.add_node("analyst", _drain(analyst_node, llm))
    g.add_node("critic", _critic_with_count(llm))
    g.add_node("finalizer", _drain(finalizer_node, llm))
    g.set_entry_point("supervisor")
    g.add_edge("supervisor", "researcher")
    g.add_edge("researcher", "analyst")
    g.add_edge("analyst", "critic")
    g.add_conditional_edges("critic", _critic_router,
                            {"analyst": "analyst", "finalizer": "finalizer"})
    g.add_edge("finalizer", END)
    return g.compile()

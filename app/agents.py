"""Agent nodes, orchestration, guardrails, and the event stream.

Pipeline: supervisor -> researcher -> analyst -> critic -> (analyst again on
FAIL, up to MAX_CRITIC_ROUNDS revisions) -> finalizer.

``stream_briefing()`` is the single orchestration path used by the API and the
tests: it yields event dicts of the form::

    {"type": "node_start" | "node_end" | "token" | "tool_call" | "guardrail" | "error" | "done",
     "node": "<node name>",
     "data": {...}}

``app/graph.py`` wires these same node functions into a LangGraph StateGraph
for teams that prefer framework orchestration.
"""
from __future__ import annotations

import os
import re
import time
from typing import Callable, Iterator, Optional

from .llm import BaseLLM, get_llm
from .tools import web_search


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


MAX_CRITIC_ROUNDS = _env_int("MAX_CRITIC_ROUNDS", 3)
RUN_TIMEOUT_SECONDS = _env_float("RUN_TIMEOUT_SECONDS", 120.0)
MAX_INPUT_CHARS = _env_int("MAX_INPUT_CHARS", 2000)
BLOCKLIST = [w.strip().lower() for w in os.getenv("BLOCKLIST", "password,ssn,api_key").split(",") if w.strip()]


class InputTooLongError(ValueError):
    pass


Event = dict
NodeFn = Callable[..., Iterator[Event]]  # (state: dict, llm: BaseLLM) -> events


def _parse_numbered_lines(text: str) -> list[str]:
    items = []
    for line in text.splitlines():
        m = re.match(r"\s*\d+[.)]\s*(.+)", line)
        if m and m.group(1).strip():
            items.append(m.group(1).strip())
    return items


# ---------------------------------------------------------------- nodes ---


def supervisor_node(state: dict, llm: BaseLLM) -> Iterator[Event]:
    """Decompose the question into focused search sub-queries."""
    yield {"type": "node_start", "node": "supervisor", "data": {}}
    prompt = (
        "[SUPERVISOR]\n"
        f"QUESTION: {state['question']}\n"
        "Decompose into 2 focused web-search sub-queries, one per line as '1. ...' and '2. ...'."
    )
    out = llm.generate(prompt)
    subqueries = _parse_numbered_lines(out)[:3] or [state["question"]]
    state["subqueries"] = subqueries
    yield {"type": "node_end", "node": "supervisor", "data": {"subqueries": subqueries}}


def researcher_node(state: dict, llm: BaseLLM) -> Iterator[Event]:
    """Run web searches for each sub-query and collect evidence."""
    yield {"type": "node_start", "node": "researcher", "data": {}}
    evidence: list[dict] = []
    subqueries = state.get("subqueries") or [state["question"]]
    for q in subqueries:
        yield {"type": "tool_call", "node": "researcher",
               "data": {"tool": "web_search", "args": {"query": q}}}
        for r in web_search(q, max_results=4):
            evidence.append({
                "id": len(evidence) + 1,
                "query": q,
                "title": r.title,
                "url": r.url,
                "snippet": r.snippet,
            })
    state["evidence"] = evidence
    yield {"type": "node_end", "node": "researcher",
           "data": {"evidence_count": len(evidence)}}


def analyst_node(state: dict, llm: BaseLLM) -> Iterator[Event]:
    """Synthesize evidence into a cited draft (tokens streamed)."""
    yield {"type": "node_start", "node": "analyst", "data": {}}
    context = "\n".join(
        f"[[{e['id']}]] {e['title']} — {e['snippet']}" for e in state.get("evidence", [])
    )
    prompt = (
        "[ANALYST]\n"
        f"QUESTION: {state['question']}\n"
        f"CONTEXT:\n{context}\n"
    )
    if state.get("critique"):
        prompt += f"FEEDBACK:\n{state['critique']}\nRevise the draft to address the feedback.\n"
    prompt += "Write exactly 3 bullet findings, each ending with its citation like [1]."
    draft = llm.generate(prompt)
    state["draft"] = draft
    for token in llm.stream(prompt):
        yield {"type": "token", "node": "analyst", "data": {"token": token}}
    yield {"type": "node_end", "node": "analyst", "data": {"chars": len(draft)}}


def critic_node(state: dict, llm: BaseLLM) -> Iterator[Event]:
    """Groundedness check: every bullet must cite a real evidence id."""
    yield {"type": "node_start", "node": "critic", "data": {}}
    valid_ids = ",".join(str(e["id"]) for e in state.get("evidence", []))
    prompt = (
        "[CRITIC]\n"
        f"DRAFT:\n{state.get('draft', '')}\n"
        f"VALID_IDS: {valid_ids}\n"
        "Check: every bullet must cite a valid source id like [n]. "
        "Reply 'VERDICT: PASS' or 'VERDICT: FAIL' followed by 'REASON: ...'."
    )
    out = llm.generate(prompt)
    passed = "VERDICT: PASS" in out.upper()
    state["critique"] = out
    state["critic_passed"] = passed
    yield {"type": "node_end", "node": "critic", "data": {"passed": passed}}


def apply_blocklist(text: str, blocklist: Optional[list[str]] = None) -> tuple[str, list[str]]:
    """Redact blocklisted terms (case-insensitive). Returns (text, hits).

    Illustrative safeguard: a production system would use a proper content
    filter / DLP service instead of substring matching.
    """
    hits: list[str] = []
    redacted = text
    for term in blocklist if blocklist is not None else BLOCKLIST:
        if term and term in redacted.lower():
            hits.append(term)
            redacted = re.sub(re.escape(term), "[REDACTED]", redacted, flags=re.IGNORECASE)
    return redacted, hits


def finalizer_node(state: dict, llm: BaseLLM,
                   blocklist: Optional[list[str]] = None) -> Iterator[Event]:
    """Produce the final briefing; apply the output blocklist."""
    yield {"type": "node_start", "node": "finalizer", "data": {}}
    prompt = (
        "[FINAL]\n"
        f"DRAFT:\n{state.get('draft', '')}\n"
        "Produce the final briefing text. Keep all citations exactly as written."
    )
    final = llm.generate(prompt)
    final, hits = apply_blocklist(final, blocklist)
    if hits:
        yield {"type": "guardrail", "node": "finalizer",
               "data": {"action": "blocklist_redaction", "terms": hits}}
    state["final_answer"] = final
    for token in llm.stream(prompt):
        yield {"type": "token", "node": "finalizer", "data": {"token": token}}
    yield {"type": "node_end", "node": "finalizer", "data": {"chars": len(final)}}


# ---------------------------------------------------------- orchestration ---


def default_config() -> dict:
    return {
        "max_critic_rounds": MAX_CRITIC_ROUNDS,
        "timeout_seconds": RUN_TIMEOUT_SECONDS,
        "max_input_chars": MAX_INPUT_CHARS,
        "blocklist": BLOCKLIST,
    }


def stream_briefing(question: str, llm: Optional[BaseLLM] = None,
                    config: Optional[dict] = None) -> Iterator[Event]:
    """Run the full pipeline, yielding events as nodes execute.

    Guardrails enforced here: input length cap, per-run wall-clock timeout
    (checked between nodes and between streamed events), critic revision cap,
    and the output blocklist (in ``finalizer_node``).
    """
    cfg = default_config()
    cfg.update(config or {})
    if len(question) > cfg["max_input_chars"]:
        yield {"type": "error", "node": "guardrail", "data": {
            "reason": "input_too_long",
            "max_chars": cfg["max_input_chars"],
            "got_chars": len(question)}}
        return

    llm = llm or get_llm()
    state: dict = {"question": question, "revision_count": 0}
    deadline = time.monotonic() + cfg["timeout_seconds"]
    aborted = False

    def timeout_event() -> Event:
        return {"type": "error", "node": "guardrail", "data": {
            "reason": "timeout",
            "timeout_seconds": cfg["timeout_seconds"]}}

    def run_node(node_fn: NodeFn) -> Iterator[Event]:
        nonlocal aborted
        if time.monotonic() > deadline:
            aborted = True
            yield timeout_event()
            return
        for ev in node_fn(state, llm):
            yield ev
            if time.monotonic() > deadline:
                aborted = True
                yield timeout_event()
                return

    def finalizer_with_config(s: dict, l: BaseLLM) -> Iterator[Event]:
        yield from finalizer_node(s, l, blocklist=cfg["blocklist"])

    yield from run_node(supervisor_node)
    if aborted:
        return
    yield from run_node(researcher_node)
    if aborted:
        return

    rounds = 0
    while True:
        yield from run_node(analyst_node)
        if aborted:
            return
        yield from run_node(critic_node)
        if aborted:
            return
        if state.get("critic_passed") or rounds >= cfg["max_critic_rounds"]:
            break
        rounds += 1
        state["revision_count"] = rounds

    yield from run_node(finalizer_with_config)
    if aborted:
        return
    yield {"type": "done", "node": "pipeline", "data": {
        "final_answer": state.get("final_answer", ""),
        "revision_count": state.get("revision_count", 0),
        "critic_passed": state.get("critic_passed", False)}}

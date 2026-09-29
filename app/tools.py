"""Tool implementations shared by the agent, the MCP server, and the tests.

Every tool is a plain function with a pydantic input schema. The MCP server
(``app/mcp_server.py``) wraps these same functions, so there is one
implementation to test and maintain.

Set ``SENTINEL_OFFLINE=1`` to force the deterministic offline search mock
(used by the test suite); otherwise the real DuckDuckGo search is used when
the ``ddgs`` package is installed.
"""
from __future__ import annotations

import ast
import datetime
import hashlib
import os
import re

from pydantic import BaseModel, Field, ValidationError  # noqa: F401 (re-exported for tests)


class SearchResult(BaseModel):
    title: str
    url: str
    snippet: str


class WebSearchInput(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    max_results: int = Field(default=5, ge=1, le=10)


class CalculatorInput(BaseModel):
    expression: str = Field(min_length=1, max_length=200)


class TodayInput(BaseModel):
    pass


TOOL_SCHEMAS: dict[str, dict] = {
    "web_search": {
        "description": "Search the web for up-to-date information.",
        "input_schema": WebSearchInput.model_json_schema(),
    },
    "calculator": {
        "description": "Evaluate a basic arithmetic expression safely (no builtins).",
        "input_schema": CalculatorInput.model_json_schema(),
    },
    "today": {
        "description": "Return today's date (ISO format) for time-sensitive questions.",
        "input_schema": TodayInput.model_json_schema(),
    },
}


def _mock_search(query: str, max_results: int) -> list[SearchResult]:
    """Deterministic offline search results. URLs use ``mock.local`` so it is
    obvious these are synthetic — never real citations."""
    digest = hashlib.md5(query.encode()).hexdigest()[:6]
    topics = ["overview", "latest developments", "expert analysis", "market outlook"]
    out = []
    for i in range(max_results):
        topic = topics[i % len(topics)]
        out.append(
            SearchResult(
                title=f"{query.strip()} — {topic}",
                url=f"https://mock.local/search?q={digest}&r={i}",
                snippet=(
                    f"Key points about {query.strip()} ({topic}): recent developments, "
                    f"supporting data, and analyst outlook. Reference {digest}-{i}."
                ),
            )
        )
    return out


def _offline_mode() -> bool:
    return os.getenv("SENTINEL_OFFLINE", "").strip().lower() in ("1", "true", "yes")


def web_search(query: str, max_results: int = 5) -> list[SearchResult]:
    """Web search via DuckDuckGo (``ddgs`` package).

    Uses the deterministic offline mock when ``SENTINEL_OFFLINE=1``, when the
    ``ddgs`` package is missing, or when the network request fails — so the
    pipeline and tests stay offline-safe.
    """
    WebSearchInput(query=query, max_results=max_results)  # validate
    if _offline_mode():
        return _mock_search(query, max_results)
    try:
        from ddgs import DDGS
    except ImportError:
        return _mock_search(query, max_results)
    try:
        results: list[SearchResult] = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results) or []:
                results.append(
                    SearchResult(
                        title=r.get("title", "") or "",
                        url=r.get("href", "") or "",
                        snippet=r.get("body", "") or "",
                    )
                )
        return results or _mock_search(query, max_results)
    except Exception:
        return _mock_search(query, max_results)


_ALLOWED_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Constant,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.UAdd,
    ast.USub,
)


def calculator(expression: str) -> str:
    """Safely evaluate arithmetic like ``2 + 3*4``. Rejects names, calls, and
    non-numeric constants."""
    CalculatorInput(expression=expression)  # validate
    tree = ast.parse(expression, mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ValueError(f"unsupported expression element: {type(node).__name__}")
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)):
            raise ValueError("only numeric constants are allowed")
    return str(eval(compile(tree, "<calculator>", "eval"), {"__builtins__": {}}, {}))


def today() -> str:
    """Today's date as ``YYYY-MM-DD``."""
    return datetime.date.today().isoformat()


def is_mock_result(result: SearchResult) -> bool:
    """True when a result came from the offline mock (``mock.local`` URL)."""
    return result.url.startswith("https://mock.local/")

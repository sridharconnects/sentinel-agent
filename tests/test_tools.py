"""Tool schemas, safety, and determinism."""
import re

import pytest
from pydantic import ValidationError

from app.tools import (
    TOOL_SCHEMAS,
    WebSearchInput,
    calculator,
    is_mock_result,
    today,
    web_search,
)


def test_tool_schemas_present():
    assert set(TOOL_SCHEMAS) == {"web_search", "calculator", "today"}
    for name, spec in TOOL_SCHEMAS.items():
        assert spec["description"]
        assert spec["input_schema"]["type"] == "object"


def test_web_search_input_validation():
    with pytest.raises(ValidationError):
        WebSearchInput(query="", max_results=5)
    with pytest.raises(ValidationError):
        WebSearchInput(query="ok", max_results=0)
    with pytest.raises(ValidationError):
        WebSearchInput(query="ok", max_results=11)
    assert WebSearchInput(query="agentic ai").max_results == 5


def test_calculator_basic():
    assert calculator("2 + 3*4") == "14"
    assert calculator("(10 - 4) / 3") == "2.0"
    assert calculator("2 ** 8") == "256"


def test_calculator_rejects_unsafe():
    for expr in ["__import__('os').system('x')", "open('/etc/passwd').read()",
                 "len('abc')", "'a' + 'b'", "1/0 if False else x"]:
        with pytest.raises((ValueError, NameError, SyntaxError)):
            calculator(expr)


def test_today_format():
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", today())


def test_web_search_returns_results():
    results = web_search("agentic AI evaluation", max_results=3)
    assert len(results) == 3
    assert all(r.title and r.url and r.snippet for r in results)


def test_web_search_mock_is_deterministic_and_labeled():
    a = web_search("quantum error correction", max_results=2)
    b = web_search("quantum error correction", max_results=2)
    assert [r.model_dump() for r in a] == [r.model_dump() for r in b]
    # Offline fallback results are clearly labeled, never fake "real" citations.
    assert all(is_mock_result(r) for r in a)

"""MCP server exposing the sentinel-agent tools.

Run it directly (stdio transport)::

    python -m app.mcp_server

Requires the ``mcp`` package (``pip install -r requirements.txt``). The tools
are the same functions as in ``app/tools.py`` — the server is a thin wrapper.
"""
from __future__ import annotations

from .tools import calculator, today, web_search

try:
    from mcp.server.fastmcp import FastMCP

    _MCP_AVAILABLE = True
except ImportError:
    _MCP_AVAILABLE = False


def build_server():
    if not _MCP_AVAILABLE:
        raise RuntimeError(
            "The 'mcp' package is not installed. Run: pip install -r requirements.txt"
        )
    mcp = FastMCP("sentinel-tools")

    @mcp.tool()
    def web_search_tool(query: str, max_results: int = 5) -> str:
        """Search the web for up-to-date information on a query."""
        results = web_search(query, max_results=max_results)
        return "\n".join(
            f"- {r.title} ({r.url}): {r.snippet}" for r in results
        )

    @mcp.tool()
    def calculator_tool(expression: str) -> str:
        """Evaluate a basic arithmetic expression, e.g. '2 + 3*4'."""
        return calculator(expression)

    @mcp.tool()
    def today_tool() -> str:
        """Return today's date (YYYY-MM-DD)."""
        return today()

    return mcp


def main() -> None:
    build_server().run()  # stdio transport


if __name__ == "__main__":
    main()

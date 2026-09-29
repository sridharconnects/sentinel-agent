"""MCP client demo: spawn the tool server over stdio and call its tools.

Run from the repo root::

    python -m app.mcp_client

Requires the ``mcp`` package. This shows the client side of the same
client+server pair; the agent itself calls the plain functions in
``app/tools.py`` directly (no subprocess overhead), while external MCP hosts
can consume ``app/mcp_server.py``.
"""
from __future__ import annotations

import asyncio
import os
import sys


async def call_tool(tool_name: str, arguments: dict):
    """Call a tool on the in-repo MCP server over stdio; return the result."""
    try:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
    except ImportError:
        raise RuntimeError(
            "The 'mcp' package is not installed. Run: pip install -r requirements.txt"
        )
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "app.mcp_server"],
        cwd=repo_root,
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            return await session.call_tool(tool_name, arguments)


async def demo() -> None:
    try:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
    except ImportError:
        raise RuntimeError(
            "The 'mcp' package is not installed. Run: pip install -r requirements.txt"
        )
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    params = StdioServerParameters(
        command=sys.executable, args=["-m", "app.mcp_server"], cwd=repo_root
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            print("Tools exposed by app.mcp_server:")
            for t in tools.tools:
                print(f"  - {t.name}: {t.description}")
            res = await call_tool("calculator_tool", {"expression": "2 + 3*4"})
            print("calculator_tool('2 + 3*4') ->", res.content[0].text)


if __name__ == "__main__":
    asyncio.run(demo())

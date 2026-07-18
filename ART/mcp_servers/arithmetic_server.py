from __future__ import annotations

from mcp.server.fastmcp import FastMCP


server = FastMCP(
    name="arithmetic-lab",
    instructions=(
        "A tiny arithmetic MCP server exposing deterministic integer tools for "
        "local RL experiments."
    ),
)


@server.tool()
def add(a: int, b: int) -> int:
    """Return a + b."""
    return a + b


@server.tool()
def sub(a: int, b: int) -> int:
    """Return a - b."""
    return a - b


@server.tool()
def mul(a: int, b: int) -> int:
    """Return a * b."""
    return a * b


@server.tool()
def max2(a: int, b: int) -> int:
    """Return the larger of two integers."""
    return a if a >= b else b


if __name__ == "__main__":
    server.run(transport="stdio")

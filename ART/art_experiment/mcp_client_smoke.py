from __future__ import annotations

import asyncio
from pathlib import Path

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


SERVER_PATH = Path(__file__).resolve().parents[1] / "mcp_servers" / "arithmetic_server.py"


async def main() -> None:
    server_params = StdioServerParameters(
        command="python",
        args=[str(SERVER_PATH)],
        cwd=str(SERVER_PATH.parent.parent),
    )

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            tools = await session.list_tools()
            tool_names = [tool.name for tool in tools.tools]
            print("tools:", tool_names)
            result = await session.call_tool("mul", {"a": 6, "b": 7})
            print("mul(6,7):", result)


if __name__ == "__main__":
    asyncio.run(main())

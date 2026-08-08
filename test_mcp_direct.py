import sys, os
from interface.mcp_server.social.services import protonmail
from system.config import load_config
load_config()

# Mock MCP object
class MockMCP:
    def tool(self, **kwargs):
        def decorator(fn):
            self.fn = fn
            return fn
        return decorator

mcp = MockMCP()
protonmail.load_tools(mcp)

import asyncio
async def main():
    print("Testing read_protonmail directly...")
    result = await mcp.fn(max_results=2)
    print("Result:", result)

asyncio.run(main())

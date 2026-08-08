#!/bin/bash
# Pre-login script to create ProtonMail session cache
# Run this once to ensure no first-time login delay

set -e

cd /home/oppa-ai/Aiko-chan

echo "Pre-authenticating with ProtonMail to cache session..."

uv run python3 << 'EOF'
import asyncio
import sys
sys.path.insert(0, "/home/oppa-ai/Aiko-chan")

from system.config import load_config
load_config()

from agentic.mcp_client import init_mcp_client, get_mcp_client
from system.log import get_logger

log = get_logger(__name__)

async def pre_auth():
    progress = lambda msg: print(f"[{__import__('time').strftime('%H:%M:%S')}] {msg}", file=__import__('sys').stderr, flush=True)
    
    progress("Pre-auth: Connecting to MCP server...")
    client = init_mcp_client()
    if client is None:
        progress("ERROR: Failed to connect to MCP server")
        return 1
    
    progress("Pre-auth: Calling read_protonmail to create session cache...")
    result = await client.call_tool("read_protonmail", {"max_results": 1})
    
    if result.get("ok"):
        progress("Pre-auth: Session cache created successfully!")
        return 0
    else:
        progress(f"Pre-auth: Failed - {result.get('error')}")
        return 1

sys.exit(asyncio.run(pre_auth()))
EOF

"""MCP Studio backend — status and tool listing for MCP servers."""
from __future__ import annotations

from pathlib import Path
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import subprocess
import json
import os
import importlib.util
import sys

app = FastAPI(title="Aiko MCP Studio")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"


def _get_mcp_servers() -> list[dict]:
    """Get list of MCP servers from the interface/mcp_server directory."""
    mcp_root = Path(__file__).resolve().parents[5] / "interface" / "mcp_server"
    servers = []
    
    if mcp_root.exists():
        for server_dir in mcp_root.iterdir():
            if server_dir.is_dir() and not server_dir.name.startswith("."):
                server_info = {
                    "name": server_dir.name,
                    "path": str(server_dir),
                    "status": "unknown",
                    "tools": [],
                    "port": None,
                    "pid": None,
                    "description": "",
                }
                
                # Check for server.py to determine port
                server_py = server_dir / "server.py"
                if server_py.exists():
                    try:
                        content = server_py.read_text()
                        # Look for port configuration
                        for line in content.split('\n'):
                            if 'port' in line.lower() and ('=' in line or ':' in line):
                                import re
                                match = re.search(r'port[^0-9]*([0-9]{3,5})', line)
                                if match:
                                    server_info["port"] = int(match.group(1))
                                    break
                    except Exception:
                        pass
                    
                    # Get description from server
                    if "Aiko Social" in content:
                        server_info["description"] = "Social media posting (X, Threads, Bluesky, Mastodon, Reddit, Discord, YouTube, Pixelset, Email)"
                
                # Also check for tools in tools/ directory
                tools_dir = server_dir / "tools"
                if tools_dir.exists():
                    static_tools = _get_static_tools(tools_dir)
                    server_info["tools"] = static_tools
                
                servers.append(server_info)
    
    return servers


def _get_static_tools(tools_dir: Path) -> list[dict]:
    """Extract tool information from the tools directory."""
    tools = []
    try:
        for tool_file in tools_dir.glob("*.py"):
            if tool_file.name in ("__init__.py", "base.py"):
                continue
            try:
                content = tool_file.read_text()
                # Look for tool name and description
                import re
                # Find @mcp.tool or @server.tool decorators
                for match in re.finditer(r'@(?:mcp|server)\.tool\s*\(\s*name\s*=\s*["\']([^"\']+)["\']', content):
                    tool_name = match.group(1)
                    # Try to get description
                    desc_match = re.search(r'description\s*=\s*["\']([^"\']+)["\']', content[match.end():match.end()+500])
                    description = desc_match.group(1) if desc_match else ""
                    tools.append({
                        "name": tool_name,
                        "description": description,
                        "parameters": {}
                    })
            except Exception:
                pass
    except Exception:
        pass
    return tools


def _check_server_status(server: dict) -> dict:
    """Check if an MCP server is running."""
    import socket
    
    port = server.get("port")
    if port:
        try:
            with socket.create_connection(("localhost", port), timeout=1):
                server["status"] = "running"
        except Exception:
            server["status"] = "stopped"
    else:
        server["status"] = "unknown"
    
    return server


@app.get("/api/servers")
async def get_servers():
    """Get all MCP servers with their status and tools."""
    servers = _get_mcp_servers()
    for server in servers:
        _check_server_status(server)
    return {"servers": servers, "count": len(servers)}


@app.get("/api/servers/{server_name}")
async def get_server(server_name: str):
    """Get details for a specific MCP server."""
    servers = _get_mcp_servers()
    server = next((s for s in servers if s["name"] == server_name), None)
    if not server:
        raise HTTPException(status_code=404, detail="Server not found")
    
    _check_server_status(server)
    return server


@app.get("/api/health")
async def health():
    return {"ok": True, "service": "mcp-studio"}


@app.get("/")
async def serve_studio():
    return FileResponse(FRONTEND_DIR / "index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
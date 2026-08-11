"""MCP Studio backend — status and tool listing for MCP servers."""
from __future__ import annotations

from pathlib import Path
import re
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

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
SHARED_DIR = Path(__file__).resolve().parents[2] / "shared"

app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="mcp-frontend")

app.mount("/shared", StaticFiles(directory=str(SHARED_DIR), html=True), name="studio-shared")


def _get_mcp_servers() -> list[dict]:
    """Get list of MCP servers from the interface/mcp_server directory."""
    mcp_root = Path(__file__).resolve().parents[5] / "interface" / "mcp_server"
    servers = []

    _SKIP = {"__pycache__", "__init__"}

    if mcp_root.exists():
        for server_dir in sorted(mcp_root.iterdir()):
            if not server_dir.is_dir():
                continue
            if server_dir.name.startswith(".") or server_dir.name in _SKIP:
                continue

            server_info = {
                "name": server_dir.name,
                "path": str(server_dir),
                "status": "unknown",
                "tools": [],
                "port": None,
                "pid": None,
                "description": "",
            }

            # Parse server.py for port + title
            server_py = server_dir / "server.py"
            if server_py.exists():
                try:
                    content = server_py.read_text()
                    # PORT env default: int(os.getenv("...", "8100"))
                    port_match = re.search(
                        r'\bPORT\b[^=\n]*=\s*int\([^)]*["\'](\d{2,5})["\']',
                        content,
                    )
                    if port_match:
                        server_info["port"] = int(port_match.group(1))
                    # FastMCP("Aiko Social MCP Server", ...)
                    title_match = re.search(r'FastMCP\s*\(\s*["\']([^"\']+)["\']', content)
                    if title_match:
                        server_info["description"] = title_match.group(1)
                except Exception:
                    pass

            # Scan tools/ directory
            tools_dir = server_dir / "tools"
            if tools_dir.exists():
                server_info["tools"] = _get_static_tools(tools_dir)

            servers.append(server_info)

    return servers


def _get_static_tools(tools_dir: Path) -> list[dict]:
    """Extract tool information by scanning tool files for @mcp.tool decorators."""
    tools = []
    try:
        for tool_file in sorted(tools_dir.glob("*.py")):
            if tool_file.name in ("__init__.py", "base.py"):
                continue
            try:
                content = tool_file.read_text()
                # Match @mcp.tool( / @server.tool( at any indentation
                for match in re.finditer(
                    r'@(?:mcp|server)\.tool\s*\(\s*name\s*=\s*["\']([^"\']+)["\']',
                    content,
                ):
                    tool_name = match.group(1)
                    after = content[match.end(): match.end() + 500]
                    desc_match = re.search(r'description\s*=\s*["\']([^"\']+)["\']', after)
                    tools.append({
                        "name": tool_name,
                        "description": desc_match.group(1) if desc_match else "",
                        "parameters": {},
                        "source_file": tool_file.name,
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
def get_servers():
    """Get all MCP servers with their status and tools.

    sync def so the 1s-per-server socket probes run in FastAPI's threadpool
    instead of freezing the event loop.
    """
    servers = _get_mcp_servers()
    for server in servers:
        _check_server_status(server)
    return {"servers": servers, "count": len(servers)}


@app.get("/api/servers/{server_name}")
def get_server(server_name: str):
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

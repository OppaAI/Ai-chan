"""Simple README for Aiko MCP Studio.

This directory contains the frontend and backend for the MCP server status and tool listing.

# Running the Studio

1. Install Python dependencies:
   ```bash
   pip install fastapi uvicorn requests
   ```

2. Start the backend:
   ```bash
   python -m interface.webui.studio.mcp.backend.api
   ```

3. Access the studio at http://localhost:8003

# Features

- View all MCP servers from interface/mcp_server/
- Check server status (running/stopped/unknown)
- List available tools for each running server
- View tool parameters and descriptions
- Auto-refresh every 30 seconds

# Files

- `backend/api.py` - FastAPI backend serving the API and static files
- `frontend/index.html` - Single-page interface with sidebar for server selection
- `entrypoint.sh` - Startup script

# Design

- Left sidebar: List of MCP servers with status badges and tool counts
- Right panel: Server details (path, port, status) and available tools
- Similar visual design to LTM Graph Studio, DAG Studio, and Approval Studio
"""
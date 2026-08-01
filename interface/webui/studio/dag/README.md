"""Simple README for Aiko Graph Studio.

This directory contains the frontend and backend for the visual graph editor.

# Running the Studio

1. Install Python dependencies:
   ```bash
   pip install fastapi uvicorn
   ```

2. Start the backend:
   ```bash
   python -m interface.webui.studio.dag.backend.api
   ```

3. Access the studio at http://localhost:8000

# Features

- View all available playbooks
- Display DAG graphs with fixed level-based layout
- Nodes with input/output ports (left: depends_on; right: triggered by)
- Edge styling by type: depends_on (gray solid), loop_to (pink dashed), fallback_to (orange)
- Inspect node details (tool, run_if, max_visits, loop, fallback)
- Inspect edge details (type, source, target, tool call)
- Zoom and pan canvas (mouse wheel + buttons)
- Legend showing edge type colors

# Files

- `backend/api.py` - FastAPI backend serving the API and static files
- `frontend/index.html` - Single-page React-like interface with D3.js graph visualization
- `templates/index.html` - Redirect to frontend (kept for fallback)
- `static/` - Static assets (CSS, etc.)

# Note

This is a minimal implementation. For production use, you'd want to:
- Add authentication
- Add graph editing capabilities
- Create a proper React frontend
- Add node search/filter
- Add playbook execution controls
"""

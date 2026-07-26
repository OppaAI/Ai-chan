"""Simple README for Aiko Graph Studio.

This directory contains the frontend and backend for the visual graph editor.

# Running the Studio

1. Install Python dependencies:
   ```bash
   pip install fastapi uvicorn
   ```

2. Start the backend:
   ```bash
   python -m agentic.studio.backend.api
   ```

3. Access the studio at http://localhost:8000

# Features

- View all available playbooks
- Display graphs with nodes and edges
- Inspect node details (tool, conditions, reducers)
- Drag nodes to reposition
- Export graphs as JSON

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
- Add more visualization features
"""

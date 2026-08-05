"""Simple README for Aiko Approval Studio.

This directory contains the frontend and backend for the job post approval interface.

# Running the Studio

1. Install Python dependencies:
   ```bash
   pip install fastapi uvicorn
   ```

2. Start the backend:
   ```bash
   python -m interface.webui.studio.approval.backend.api
   ```

3. Access the studio at http://localhost:8002

# Features

- View all daily job post drafts
- Filter by status: pending, approved, posted
- Review draft content with full text display
- Approve/reject drafts with one click
- View draft metadata (date, category, LLM enrichment, etc.)

# Files

- `backend/api.py` - FastAPI backend serving the API and static files
- `frontend/index.html` - Single-page interface with sidebar for draft selection
- `entrypoint.sh` - Startup script

# Design

- Left sidebar: List of job post drafts with status badges
- Top right panel: Draft metadata and action buttons
- Bottom right panel: Full draft content display (larger area for reading URLs and content)
- Similar visual design to Memory Graph Studio and DAG Studio
"""

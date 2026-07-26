"""Studio backend for Aiko graph visualizer.

Serves the playbooks JSON and handles API requests for the studio frontend.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
import json

app = FastAPI(title="Aiko Graph Studio")

# Allow connections from the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files (CSS, JS, etc.)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup templates
templates_dir = Path(__file__).parent / "frontend"
app.mount("/frontend", StaticFiles(directory=templates_dir), name="frontend")

templates = Jinja2Templates(directory="templates")


# Load playbooks from graph_engine
from agentic.graph_engine import load_playbooks

# Get default playbooks as JSON for the studio
PLAYBOOKS = load_playbooks()


@app.get("/api/playbooks")
async def get_playbooks():
    """Get all playbooks for the studio."""
    return PLAYBOOKS


@app.get("/api/playbooks/{playbook_id}")
async def get_playbook(playbook_id: str):
    """Get a specific playbook by ID."""
    for playbook in PLAYBOOKS:
        if playbook["id"] == playbook_id:
            return playbook
    raise HTTPException(status_code=404, detail="Playbook not found")


@app.get("/")
async def serve_studio(request):
    """Serve the studio interface."""
    return templates.TemplateResponse("index.html", {"request": request})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

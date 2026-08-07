#!/usr/bin/env python3
"""Entrypoint for Grasp Studio."""
import os
from pathlib import Path

def main():
    os.chdir(Path(__file__).parent)
    print("Starting Aiko Grasp Studio at http://127.0.0.1:8003")
    import uvicorn
    from interface.webui.studio.grasp.backend.api import app
    # Local-only by default (mutable demo state; no auth).
    uvicorn.run(app, host="127.0.0.1", port=8003)

if __name__ == "__main__":
    main()

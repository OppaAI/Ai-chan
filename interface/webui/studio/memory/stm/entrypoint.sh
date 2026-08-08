#!/usr/bin/env python3
"""Entrypoint for STM Studio."""
import os
import sys
from pathlib import Path

def main():
    project_root = Path(__file__).resolve().parents[5]
    os.chdir(project_root)
    sys.path.insert(0, str(project_root))
    print("Starting Aiko STM Studio at http://127.0.0.1:8003")
    import uvicorn
    from interface.webui.studio.memory.stm.backend.api import app
    # Local-only by default (mutable demo state; no auth).
    uvicorn.run(app, host="127.0.0.1", port=8003)

if __name__ == "__main__":
    main()

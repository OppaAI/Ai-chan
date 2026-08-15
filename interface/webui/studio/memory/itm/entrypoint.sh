#!/usr/bin/env python3
"""Docker entrypoint for the ITM (Episodic Memory) Studio.

Usage:
    docker run aiko-itm-studio
"""
import os
import subprocess
import sys
from pathlib import Path


def check_dependencies():
    try:
        import fastapi
        import uvicorn
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "fastapi", "uvicorn"], check=True)


def main():
    check_dependencies()
    project_root = Path(__file__).resolve().parents[5]
    os.chdir(project_root)
    sys.path.insert(0, str(project_root))

    import uvicorn
    from interface.webui.studio.memory.itm.backend.api import app

    uvicorn.run(app, host="0.0.0.0", port=8004)


if __name__ == "__main__":
    main()
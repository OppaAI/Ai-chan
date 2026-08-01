#!/usr/bin/env python3
"""Docker entrypoint for Studio.

This script runs when the container starts. It ensures required dependencies
are installed and starts the FastAPI server.

Usage:
    docker run aiko-studio
"""

import subprocess
import sys
import os
from pathlib import Path

# Check for required dependencies
def check_dependencies():
    print("Checking dependencies...")

    # Check for Python
    try:
        import fastapi
        print("✓ FastAPI is installed")
    except ImportError:
        print("FastAPI is not installed. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "fastapi"], check=True)

    try:
        import uvicorn
        print("✓ Uvicorn is installed")
    except ImportError:
        print("Uvicorn is not installed. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "uvicorn"], check=True)

    # Check for config setup
    config_path = Path.home() / ".config" / "aiko" / "secrets.enc.age"
    if not config_path.exists():
        print(f"⚠ Config file not found at {config_path}")
        print("Please set up config secrets first:")
        print("  1. Create secrets.enc.age with your encrypted credentials")
        print("  2. Update .env to point to correct config location")

    return True


def main():
    """Run the studio server."""
    check_dependencies()

    # Load environment variables
    os.chdir(Path(__file__).parent)

    # Run the API server
    print("Starting Aiko Memory Graph Studio server at http://localhost:8001")
    print("Visit the URL to access the studio interface")

    import uvicorn
    from interface.webui.studio.memory.backend.api import app

    uvicorn.run(app, host="0.0.0.0", port=8001)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""Standalone entry for Spec Studio."""
import os
import uvicorn

if __name__ == "__main__":
    # Default to loopback-only for security; override via SPEC_STUDIO_HOST if needed
    host = os.getenv("SPEC_STUDIO_HOST", "127.0.0.1")
    uvicorn.run(
        "interface.webui.studio.spec.backend.api:app",
        host=host,
        port=8010,
        reload=False,
    )

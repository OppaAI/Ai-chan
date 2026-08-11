#!/usr/bin/env python3
"""Standalone entry for Spec Studio."""
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "interface.webui.studio.spec.backend.api:app",
        host="0.0.0.0",
        port=8010,
        reload=False,
    )

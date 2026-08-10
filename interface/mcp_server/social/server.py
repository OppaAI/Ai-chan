from __future__ import annotations

import os
import sys
import warnings

# Suppress CryptographyDeprecationWarning for TripleDES (used by protonmail-api-client)
# Note: CryptographyDeprecationWarning subclasses UserWarning, not DeprecationWarning.
warnings.filterwarnings(
    "ignore",
    category=Warning,
    message="TripleDES has been moved to cryptography.hazmat.decrepit.ciphers.algorithms.TripleDES",
)

# Load config FIRST, before anything else
from system.config import load_config
load_config()

from system.log import get_logger
from mcp.server.fastmcp import FastMCP

log = get_logger(__name__)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Initialize MCP server for stdio transport (no HTTP binding)
mcp = FastMCP("Aiko Social MCP Server")


# ── Apply middleware (rate limiting, audit logging) ────────────────────────
from social.guards import wrap_tool

_original_tool = mcp.tool


def _wrapped_tool(*args, **kwargs):
    """
    Decorator factory that wraps tool functions with rate limiting.

    Docstring: Intercept MCP tool registration, apply middleware
    to enforce quotas and log invocations before passing to MCP.
    """
    def decorator(fn):
        name = kwargs.get("name") or fn.__name__
        return _original_tool(*args, **kwargs)(wrap_tool(name, fn))

    return decorator


mcp.tool = _wrapped_tool


# ── Tool registration ──────────────────────────────────────────────────────

def _load_tools() -> None:
    """Load tool modules and register with MCP."""
    from social.services import x, threads, youtube, medium
    from social.services import reddit, bluesky, mastodon, pixelset, multipost
    from social.services import discord, email

    for mod in (x, threads, youtube, medium, reddit, bluesky, mastodon, pixelset, discord, email, multipost):
        if hasattr(mod, "load_tools"):
            mod.load_tools(mcp)


# ── Entry point ────────────────────────────────────────────────────────────

def main() -> None:
    """Boot MCP server on stdio transport."""
    log.info("Starting Aiko Social MCP Server (stdio transport)")
    _load_tools()
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
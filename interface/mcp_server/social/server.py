import os
import sys
from pathlib import Path

# Load config FIRST, before anything else
from system.config import load_config
load_config()

from system.log import get_logger
log = get_logger(__name__)

from mcp.server.fastmcp import FastMCP

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# HOST/PORT kept for reference (unused with stdio transport)
HOST = os.getenv("SOCIAL_MCP_HOST", "127.0.0.1")
PORT = int(os.getenv("SOCIAL_MCP_PORT", "8100"))

mcp = FastMCP("Aiko Social MCP Server", host=HOST, port=PORT)


# ── apply middleware (rate limiting, idempotency, tool logging) ──────────────
from social.middleware import wrap_tool

_original_tool = mcp.tool

def _wrapped_tool(*args, **kwargs):
    def decorator(fn):
        name = kwargs.get("name") or fn.__name__
        # Skip internal tools
        if name not in ("_inject_env",):
            fn = wrap_tool(name, fn)
        return _original_tool(*args, **kwargs)(fn)
    return decorator

mcp.tool = _wrapped_tool


# ── tool registration ─────────────────────────────────────────────────────

def _load_tools() -> None:
    from social.services import x, threads, youtube, medium
    from social.services import reddit, bluesky, mastodon, pixelset, multipost
    from social.services import discord, email

    for mod in (x, threads, youtube, medium, reddit, bluesky, mastodon, pixelset, discord, email, multipost):
        if hasattr(mod, "load_tools"):
            mod.load_tools(mcp)


# Internal tool: inject env vars at runtime (called by Aiko on connect)
@mcp.tool(
    name="_inject_env",
    description="INTERNAL: inject environment variables from Aiko process (not user-facing)",
)
def _inject_env(vars: dict[str, str]) -> dict:
    count = 0
    for k, v in vars.items():
        if k not in os.environ:
            os.environ[k] = v
            count += 1
    return {"ok": True, "injected": count}


# ── entry point ───────────────────────────────────────────────────────────

def main() -> None:
    _load_tools()
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
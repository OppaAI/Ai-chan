from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HOST = os.getenv("SOCIAL_MCP_HOST", "127.0.0.1")
PORT = int(os.getenv("SOCIAL_MCP_PORT", "8100"))

mcp = FastMCP("Aiko Social MCP Server", host=HOST, port=PORT)


# ── credential loading ────────────────────────────────────────────────────

def _decrypt_env_age() -> None:
    env_age = Path(".env.age")
    identity = Path(os.getenv("AGE_IDENTITY_PATH", "key.txt")).expanduser()
    if not env_age.exists():
        return
    try:
        result = subprocess.run(
            ["age", "--decrypt", "-i", str(identity), str(env_age)],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                if key and val and key not in os.environ:
                    os.environ[key] = val
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass


# ── DB + middleware ───────────────────────────────────────────────────────

def _init_db() -> None:
    from social.db import init_db
    init_db()
    from social.db import get_db
    db = get_db()
    db.cleanup()


def _apply_middleware() -> None:
    from social.middleware import wrap_tool

    for name, tool in list(mcp._tool_manager._tools.items()):
        original_fn = tool.fn
        wrapped = wrap_tool(name, original_fn)
        tool.fn = wrapped


# ── tool registration ─────────────────────────────────────────────────────

def _load_tools() -> None:
    from social.tools import x, threads, youtube
    from social.tools import reddit, bluesky, mastodon, pixelset, multipost
    from social.tools import discord, email

    for mod in (x, threads, youtube, reddit, bluesky, mastodon, pixelset, discord, email, multipost):
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
    _decrypt_env_age()
    _init_db()
    _load_tools()
    _apply_middleware()
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()

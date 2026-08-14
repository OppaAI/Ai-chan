from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
import warnings
from typing import Any

# Suppress pydantic-settings IncompleteFieldDefinitionWarning for the MCP library's
# ServerSettings.lifespan field (a Callable forward reference).
warnings.filterwarnings(
    "ignore",
    message="Field 'lifespan' has an incomplete definition.*",
)
from system.log import get_logger

log = get_logger(__name__)

_MCP_CLIENT: "MCPClient | None" = None


class MCPToolBridge:
    def __init__(self, client: "MCPClient", tool_name: str, input_schema: dict[str, Any]) -> None:
        self._client = client
        self.tool_name = tool_name
        self.input_schema = input_schema

    def __call__(self, **kwargs: Any) -> dict[str, Any]:
        try:
            return self._client.call_tool_sync(self.tool_name, kwargs)
        except Exception as e:
            log.error("[mcp] Bridge call to %s failed: %s", self.tool_name, e)
            return {"ok": False, "error": str(e), "tool": self.tool_name}


class MCPClient:
    def __init__(self, server_url: str = "") -> None:
        # stdio transport only: the MCP server is spawned as a subprocess
        # inheriting Aiko's environment (see _ensure_session). server_url is
        # accepted for API compatibility but no longer used.
        self.server_url = server_url
        self._tools: dict[str, dict[str, Any]] = {}
        self._bridges: list[Any] = []
        self._session = None
        self._transport = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: threading.Thread | None = None
        self._loop_ready = threading.Event()

    # ── persistent event loop ────────────────────────────────────────────
    # The client session and stdio transport spawn background tasks that must
    # keep running for the life of the connection. Running them inside a
    # throwaway anyio.run() loop closes them the moment that loop exits, so
    # every subsequent tool call fails with ClosedResourceError. Instead we
    # host the session in one dedicated loop thread and dispatch each call
    # onto it.
    def _ensure_loop(self) -> None:
        if self._loop is not None:
            return

        def _run() -> None:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._loop_ready.set()
            try:
                self._loop.run_forever()
            finally:
                try:
                    self._loop.run_until_complete(self._loop.shutdown_asyncgens())
                except Exception:
                    pass
                self._loop.close()

        self._loop_thread = threading.Thread(
            target=_run, name="aiko-mcp-client", daemon=True
        )
        self._loop_thread.start()
        if not self._loop_ready.wait(timeout=10):
            raise TimeoutError("mcp client event loop failed to start")

    def _run_coro(self, coro_factory, timeout: float = 60) -> Any:
        self._ensure_loop()
        future = asyncio.run_coroutine_threadsafe(coro_factory(), self._loop)
        return future.result(timeout=timeout)

    def list_tools_sync(self) -> list[dict[str, Any]]:
        return self._run_coro(self.list_tools)

    def call_tool_sync(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return self._run_coro(lambda: self.call_tool(name, arguments))

    def close_sync(self) -> None:
        self._run_coro(self.close)

    # ── session / tools ──────────────────────────────────────────────────

    async def _ensure_session(self):
        if self._session is not None:
            return self._session

        from mcp import StdioServerParameters
        from mcp.client.session import ClientSession
        from mcp.client.stdio import stdio_client

        # Repo root: client.py lives at <repo>/agentic/mcp_client/client.py
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        # Pass Aiko's full env explicitly: the SDK's stdio transport only
        # inherits a safe subset otherwise, which would drop the social
        # API tokens the server needs (AISA_API_KEY, THREADS_*, ...).
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "interface.mcp_server.social.server"],
            cwd=repo_root,
            env=dict(os.environ),
        )
        self._transport = stdio_client(params)
        read, write = await self._transport.__aenter__()

        session = ClientSession(read, write)
        await session.__aenter__()
        await session.initialize()
        self._session = session
        return session

    async def list_tools(self) -> list[dict[str, Any]]:
        try:
            session = await self._ensure_session()
            result = await session.list_tools()
            tools = []
            for t in result.tools:
                entry = {
                    "name": t.name,
                    "description": t.description,
                    "inputSchema": t.inputSchema if hasattr(t, "inputSchema") else t.parameters if hasattr(t, "parameters") else {},
                }
                tools.append(entry)
            self._tools = {t["name"]: t for t in tools}
            return tools
        except Exception as e:
            log.warning("[mcp] Failed to list tools: %s", e)
            return []

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        session = await self._ensure_session()
        result = await session.call_tool(name, arguments)
        if result.content:
            text = result.content[0].text if hasattr(result.content[0], "text") else str(result.content[0])
            try:
                return json.loads(text)
            except (json.JSONDecodeError, TypeError):
                return {"ok": True, "result": text}
        return {"ok": True}

    def get_bridge_tool_defs(self) -> list[tuple[str, str, dict[str, Any], list[str], Any]]:
        defs = []
        for name, info in self._tools.items():
            schema = info.get("inputSchema", {})
            props = schema.get("properties", {})
            required = schema.get("required", [])
            bridge = MCPToolBridge(self, name, schema)
            self._bridges.append(bridge)
            defs.append((name, info.get("description", ""), props, required, bridge))
        return defs

    async def close(self):
        try:
            if self._session is not None:
                await self._session.__aexit__(None, None, None)
        except Exception:
            log.warning("[mcp] session close failed")
        try:
            if self._transport is not None:
                await self._transport.__aexit__(None, None, None)
        except Exception:
            log.warning("[mcp] transport close failed")
        self._session = None
        self._transport = None


def get_mcp_client() -> MCPClient | None:
    return _MCP_CLIENT


def init_mcp_client(server_url: str = "") -> MCPClient | None:
    global _MCP_CLIENT
    try:
        client = MCPClient(server_url=server_url)
        tools = client.list_tools_sync()
        if not tools:
            log.warning("[mcp] Connected but no tools discovered")
        else:
            endpoint = f"stdio:{sys.executable} -m interface.mcp_server.social.server"
            log.info("[mcp] Discovered %d tools from %s", len(tools), endpoint)
            for t in tools:
                log.info("[mcp]   %s — %s", t["name"], t["description"][:60])

        _MCP_CLIENT = client
        return client
    except Exception as e:
        log.warning("[mcp] Failed to connect to MCP server: %s", e)
        return None


def shutdown_mcp_client() -> None:
    global _MCP_CLIENT
    if _MCP_CLIENT is not None:
        try:
            _MCP_CLIENT.close_sync()
        except Exception:
            log.warning("[mcp] shutdown close failed")
        _MCP_CLIENT = None

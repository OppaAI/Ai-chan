"""
agentic/toolkit/codebase.py

Codebase RAG tools for Jetson Orin Nano 8GB.

Exposes the separate per-user codebase DB at
<USER_SPACE_ROOT>/<user_id>/knowledge/codebase.db so Aiko can
answer "where is X / how does Y work" via regular RAG (hybrid
vec + FTS5 + RRF). Graph layer is approximated via entity boost
— no separate graph DB needed on 8GB.

Tools are lazy: ingestion runs once (incremental SHA1), search
embeds via Harrier (640d) on demand.
"""
from __future__ import annotations

from typing import Any
from agentic.registry import TOOLS, tool
from system.log import get_logger

log = get_logger(__name__)

@tool(
    TOOLS["codebase_search"] if "codebase_search" in TOOLS else "codebase_search",
    description="Search Aiko's codebase RAG (separate DB at <USER_SPACE_ROOT>/<user_id>/knowledge/codebase.db) for code/docs relevant to the query. Optimized for Jetson Orin Nano 8GB (harrier 640d, 900-char chunks). Use for 'where is X', 'how does Y work', 'find file for Z'.",
    graph=True,
    react=True,
    domain="codebase",
)
def codebase_search(query: str, limit: int = 5) -> str:
    """Hybrid search over codebase.db. Returns JSON blocks with path+snippet."""
    from cognition.knowledge.codebase import search_codebase
    from cognition.memory.vecstore import HarrierEmbedder
    from agentic.toolkit.common import json_block
    query = (query or "").strip()
    if not query:
        return json_block("codebase_search", {"ok": False, "error": "query required"})
    try:
        emb = HarrierEmbedder()
        hits = search_codebase(query, limit=max(1, min(int(limit), 10)), embedder=emb)
        payload = {
            "ok": True,
            "query": query,
            "count": len(hits),
            "hits": [
                {
                    "path": h.get("path", ""),
                    "chunk_index": h.get("chunk_index", 0),
                    "score": round(float(h.get("score", 0)), 4),
                    "text": (h.get("text", "") or "")[:900],
                }
                for h in hits
            ],
        }
        return json_block("codebase_search", payload)
    except Exception as e:
        log.warning("codebase_search failed: %s", e)
        return json_block("codebase_search", {"ok": False, "error": str(e), "query": query})

@tool(
    TOOLS["codebase_context"] if "codebase_context" in TOOLS else "codebase_context",
    description="Retrieve formatted <codebase_context> block for a codebase question (RAG). For prompt injection.",
    graph=True,
    react=True,
    domain="codebase",
)
def codebase_context(query: str, limit: int = 5, max_chars: int = 4000) -> str:
    from cognition.knowledge.codebase import codebase_context_for
    from cognition.memory.vecstore import HarrierEmbedder
    try:
        emb = HarrierEmbedder()
        return codebase_context_for(query, limit=max(1, min(int(limit), 8)), max_chars=max(1, min(int(max_chars), 8000)), embedder=emb)
    except Exception as e:
        return f"<codebase_context>\nError: {e}\n</codebase_context>"

@tool(
    TOOLS["codebase_ingest"] if "codebase_ingest" in TOOLS else "codebase_ingest",
    description="Ingest (or refresh) the entire Aiko codebase into the per-user codebase.db. Incremental by SHA1; safe to call repeatedly. Optimized for Jetson (batched 32, WAL, cosine).",
    graph=True,
    react=True,
    domain="codebase",
)
def codebase_ingest(force: bool = False, repo_root: str = "") -> str:
    from cognition.knowledge.codebase import ingest_codebase
    from agentic.toolkit.common import json_block
    from pathlib import Path
    try:
        root = Path(repo_root).expanduser().resolve() if (repo_root or "").strip() else None
        res = ingest_codebase(force=bool(force), repo_root=root)
        return json_block("codebase_ingest", res)
    except Exception as e:
        log.warning("codebase_ingest failed: %s", e)
        return json_block("codebase_ingest", {"ok": False, "error": str(e)})

__all__ = ["codebase_search", "codebase_context", "codebase_ingest"]

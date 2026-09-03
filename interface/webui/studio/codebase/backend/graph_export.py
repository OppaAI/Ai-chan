"""Export an explorable, body-shaped map of the indexed codebase."""
from __future__ import annotations

import hashlib
import math
import re
import sqlite3
from collections import defaultdict
from pathlib import Path

from system.userspace import current_user_id, user_state_path


_BODY_MAP = [
    (r"^cognition/(memory|knowledge|subliminal|attention|think|reason|consolidate)", "head", "brain", "#35e7f2"),
    (r"^cognition/", "head", "brain", "#6cecf2"),
    (r"^sensory/(vision|video)|^interface/webui|^training/persona", "head_eyes", "eyes", "#62f5e7"),
    (r"^sensory/(listen|audio)|^sensory/speak.*listen|sherpa", "head_ears", "ears", "#d5f57a"),
    (r"^sensory/speak|^interface/adapter|^agentic/toolkit/social", "head_mouth", "voice", "#74d7ff"),
    (r"^system/(userspace|config|secure|bioclock|log)|^main\.py", "chest", "core", "#9cb7ff"),
    (r"^agentic/(toolkit|workflows|graph_engine|registry)", "arms", "tools", "#7ed8ff"),
    (r"^system/(orchestrate|schedule|wakeup|prepare|turngate)", "legs", "mobility", "#54e4bd"),
    (r"^agentic/mcp|^interface/mcp|^backend", "spine", "spine", "#70a9d4"),
    (r".*", "tail", "support", "#8ba4b0"),
]

_BODY_ANCHORS = {
    "head": (0.50, 0.16), "head_eyes": (0.50, 0.12), "head_ears": (0.50, 0.15),
    "head_mouth": (0.50, 0.20), "chest": (0.50, 0.38), "arms": (0.50, 0.42),
    "spine": (0.50, 0.51), "legs": (0.50, 0.74), "tail": (0.50, 0.89),
}


def _body_for_path(path: str) -> tuple[str, str, str]:
    for pattern, anchor, label, color in _BODY_MAP:
        if re.search(pattern, path):
            return anchor, label, color
    return "tail", "support", "#8ba4b0"


def _module_for_path(path: str) -> str:
    """Keep related files together without collapsing the whole brain into one dot."""
    parts = Path(path).parts
    if len(parts) == 1:
        return path
    if parts[0] in {"cognition", "system", "sensory", "agentic"} and len(parts) >= 2:
        return "/".join(parts[:2])
    if parts[:2] == ("interface", "webui") and len(parts) >= 4:
        return "/".join(parts[:4])
    return "/".join(parts[:2])


def _node_id(module: str) -> str:
    return "module-" + hashlib.sha1(module.encode()).hexdigest()[:12]


def _connect(user_id: str) -> tuple[sqlite3.Connection | None, Path]:
    path = user_state_path("knowledge/codebase.db", user_id=user_id)
    if not path.exists():
        return None, path
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn, path


def export_codebase_graph(user_id: str | None = None, limit: int = 400) -> dict:
    uid = user_id or current_user_id()
    conn, path = _connect(uid)
    if conn is None:
        return {"nodes": [], "edges": [], "meta": {"user_id": uid, "exists": False, "path": str(path)}}
    try:
        docs = conn.execute("SELECT id, path, title FROM codebase_docs WHERE user_id=? ORDER BY path LIMIT ?", (uid, limit)).fetchall()
        modules: dict[str, list[sqlite3.Row]] = defaultdict(list)
        for doc in docs:
            modules[_module_for_path(doc["path"])].append(doc)
        nodes, body_counts, groups = [], defaultdict(int), defaultdict(list)
        for module, members in modules.items():
            anchor, label, color = _body_for_path(module)
            body_counts[label] += 1
            index = len(groups[anchor])
            groups[anchor].append(module)
            # A deterministic fan makes dense regions readable while staying on the figure.
            angle = (index * 2.399963229728653) - 1.57
            radius = 0.025 + min(index, 10) * 0.011
            ax, ay = _BODY_ANCHORS[anchor]
            nodes.append({
                "id": _node_id(module), "module": module, "path": module, "title": f"{len(members)} indexed file{'s' if len(members) != 1 else ''}",
                "file_count": len(members), "body_part": label, "anchor": anchor, "color": color,
                "x": max(.05, min(.95, ax + radius * math.cos(angle))),
                "y": max(.05, min(.95, ay + radius * math.sin(angle))),
            })
        edges = []
        for anchor, module_names in groups.items():
            for module in module_names[1:5]:
                edges.append({"source": _node_id(module_names[0]), "target": _node_id(module), "kind": "shared_body_system"})
        return {"nodes": nodes, "edges": edges, "meta": {"user_id": uid, "path": str(path), "exists": True, "count": len(nodes), "body_counts": dict(body_counts), "limit": limit}, "anchors": {key: {"x": x, "y": y} for key, (x, y) in _BODY_ANCHORS.items()}}
    finally:
        conn.close()


def module_details(module: str, user_id: str | None = None) -> dict:
    """Return a concise, deterministic natural-language explanation for one module."""
    uid = user_id or current_user_id()
    conn, _ = _connect(uid)
    if conn is None:
        return {"module": module, "error": "Codebase index is not available yet."}
    try:
        rows = conn.execute("SELECT id, path FROM codebase_docs WHERE user_id=? AND (path=? OR path LIKE ?) ORDER BY path", (uid, module, f"{module}/%")).fetchall()
        paths = [row["path"] for row in rows]
        if not rows:
            return {"module": module, "error": "This module is no longer in the codebase index."}
        placeholders = ",".join("?" * len(rows))
        chunks = conn.execute(f"SELECT text FROM codebase_chunks WHERE user_id=? AND doc_id IN ({placeholders}) ORDER BY chunk_index LIMIT 24", [uid, *[row["id"] for row in rows]]).fetchall()
        text = "\n".join(row["text"] for row in chunks)
        functions = []
        for name in re.findall(r"(?:^|\n)\s*(?:async\s+def|def|function|class)\s+([A-Za-z_]\w*)", text):
            if name not in functions:
                functions.append(name)
        anchor, body_part, _ = _body_for_path(module)
        function_phrase = ", ".join(functions[:8]) if functions else "No named functions were found in the indexed excerpts"
        summary = f"{module} is a {body_part} module in Aiko's codebase. It contains {len(paths)} indexed file{'s' if len(paths) != 1 else ''} and appears to provide {function_phrase}."
        return {"module": module, "body_part": body_part, "anchor": anchor, "summary": summary, "functions": functions[:12], "files": paths, "excerpt": re.sub(r"\s+", " ", text)[:360]}
    finally:
        conn.close()

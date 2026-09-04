"""Export an explorable neural-net map of the indexed codebase.

The front-end runs a D3 force simulation — the backend only classifies every
top-level package into a coherent system bucket and exposes dependency edges.
"""
from __future__ import annotations

import hashlib
import re
import sqlite3
import subprocess
import ast
import json
from collections import defaultdict
from pathlib import Path

from system.userspace import current_user_id, user_state_path


_BODY_MAP = [
    (r"^cognition/(memory|knowledge|subliminal|attention|think|reason|consolidate)", "brain", "#c651a8"),
    (r"^cognition/", "brain", "#d8bcff"),
    (r"^sensory/", "senses", "#51d4c8"),
    (r"^system/", "core", "#a888e8"),
    (r"^agentic/(workflows|graph_engine)", "core", "#a888e8"),
    (r"^agentic/(toolkit|registry|mcp|bridge)", "tools", "#7298e8"),
    (r"^interface/", "interface", "#8ab4ff"),
    (r"^training/", "training", "#e88c6a"),
    (r"^tests/", "tests", "#9aa0b4"),
    (r"^main\.py$", "core", "#a888e8"),
    (r".*", "support", "#887b9a"),
]

_SYSTEM_LABELS = {
    "brain":     "Brain",
    "senses":    "Senses",
    "core":      "Core",
    "tools":     "Arms",
    "interface": "Interface",
    "training":  "Training",
    "tests":     "Tests",
    "support":   "Support",
}

_REPO_ROOT = Path(__file__).resolve().parents[5]
_FUNCTION_RE = re.compile(r"(?:^|\n)\s*(?:async\s+def|def|function|class)\s+([A-Za-z_]\w*)")
_COMPLEXITY_RE = re.compile(r"\b(if|elif|for|while|except|case|and|or)\b")


def _source_for_path(path: str) -> str:
    """Read source from the checked-out repository, never from a request path."""
    candidate = (_REPO_ROOT / path).resolve()
    try:
        candidate.relative_to(_REPO_ROOT)
        return candidate.read_text(encoding="utf-8", errors="ignore")
    except (OSError, ValueError):
        return ""


def _symbols(source: str) -> list[str]:
    return list(dict.fromkeys(_FUNCTION_RE.findall(source)))


def _module_metrics(paths: list[str]) -> dict:
    sources = [_source_for_path(path) for path in paths]
    source = "\n".join(sources)
    lines = sum(text.count("\n") + bool(text) for text in sources)
    functions = list(dict.fromkeys(symbol for text in sources for symbol in _symbols(text)))
    return {
        "loc": lines,
        "function_count": len(functions),
        "functions": functions,
        # This is intentionally an estimate, not a substitute for a dedicated lizard/radon report.
        "complexity": 1 + len(_COMPLEXITY_RE.findall(source)),
    }


def _imports(source: str) -> set[str]:
    imports: set[str] = set()
    for target in re.findall(r"^\s*(?:from|import)\s+([\w.]+)", source, flags=re.MULTILINE):
        imports.add(target.replace(".", "/"))
    for target in re.findall(r"(?:import|from)\s+['\"]([^'\"]+)['\"]", source):
        imports.add(target.replace(".", "/"))
    return imports


def _git_change_counts(paths: list[str]) -> dict[str, int]:
    """Count commits touching indexed files. Gracefully degrades outside a git checkout."""
    try:
        output = subprocess.run(
            ["git", "log", "--format=", "--name-only", "--", *paths], cwd=_REPO_ROOT,
            capture_output=True, text=True, timeout=4, check=False,
        ).stdout
    except (OSError, subprocess.TimeoutExpired):
        return {}
    counts: dict[str, int] = defaultdict(int)
    wanted = set(paths)
    for path in output.splitlines():
        if path in wanted:
            counts[path] += 1
    return dict(counts)


def _repository_url() -> str | None:
    try:
        remote = subprocess.run(["git", "remote", "get-url", "origin"], cwd=_REPO_ROOT, capture_output=True, text=True, timeout=2, check=False).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return None
    if remote.startswith("git@github.com:"):
        return "https://github.com/" + remote.removeprefix("git@github.com:").removesuffix(".git")
    if remote.startswith("https://github.com/"):
        return remote.removesuffix(".git")
    return None


def _docstrings(paths: list[str]) -> list[str]:
    found: list[str] = []
    for path in paths:
        if not path.endswith(".py"):
            continue
        try:
            tree = ast.parse(_source_for_path(path))
            docstring = ast.get_docstring(tree)
            if docstring:
                found.append(docstring.splitlines()[0].strip())
        except SyntaxError:
            continue
    return found[:4]


def _coverage_for_paths(paths: list[str]) -> float | None:
    """Use coverage.py's optional JSON report when a CI or local run provides one."""
    report = _REPO_ROOT / "coverage.json"
    try:
        files = json.loads(report.read_text(encoding="utf-8")).get("files", {})
    except (OSError, ValueError):
        return None
    values = []
    for path in paths:
        entry = files.get(path) or files.get(str(_REPO_ROOT / path))
        if entry and isinstance(entry.get("summary", {}).get("percent_covered"), (int, float)):
            values.append(entry["summary"]["percent_covered"])
    return round(sum(values) / len(values), 1) if values else None


def _body_for_path(path: str) -> tuple[str, str]:
    for pattern, label, color in _BODY_MAP:
        if re.search(pattern, path):
            return label, color
    return "support", "#887b9a"


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
        module_paths = {module: [member["path"] for member in members] for module, members in modules.items()}
        module_imports: dict[str, set[str]] = defaultdict(set)
        for module, paths in module_paths.items():
            for source_path in paths:
                for target in _imports(_source_for_path(source_path)):
                    target_module = _module_for_path(target)
                    if target_module in modules and target_module != module:
                        module_imports[module].add(target_module)
        inbound: dict[str, set[str]] = defaultdict(set)
        for source, targets in module_imports.items():
            for target in targets:
                inbound[target].add(source)
        change_counts = _git_change_counts([path for paths in module_paths.values() for path in paths])
        for module, members in modules.items():
            system, color = _body_for_path(module)
            body_counts[system] += 1
            groups[system].append(module)
            metrics = _module_metrics(module_paths[module])
            changed = sum(change_counts.get(path, 0) for path in module_paths[module])
            test_files = [path for path in module_paths[module] if "/test" in path or path.startswith("tests/")]
            nodes.append({
                "id": _node_id(module), "module": module, "path": module, "title": f"{len(members)} indexed file{'s' if len(members) != 1 else ''}",
                "file_count": len(members), "system": system, "color": color,
                # Position is computed by the front-end force layout — backend only supplies
                # an optional seed (deterministic hash) so the layout is stable across reloads.
                "x_seed": (hashlib.sha1(module.encode()).digest()[0] / 255),
                "y_seed": (hashlib.sha1(module.encode()).digest()[1] / 255),
                "loc": metrics["loc"], "function_count": metrics["function_count"], "complexity": metrics["complexity"],
                "change_count": changed, "test_file_count": len(test_files), "coverage": _coverage_for_paths(module_paths[module]),
                "dependency_count": len(module_imports[module]), "dependent_count": len(inbound[module]),
            })
        edges = []
        for source, targets in module_imports.items():
            for target in targets:
                edges.append({"source": _node_id(source), "target": _node_id(target), "kind": "dependency", "weight": 1})
        # Light intra-system "neighborhood" edges so dense clusters look like a mesh
        # rather than isolated dots — kept at low weight so dependency edges dominate.
        for system, module_names in groups.items():
            members = list(module_names)
            for i, src in enumerate(members):
                for dst in members[i + 1: i + 3]:
                    edges.append({"source": _node_id(src), "target": _node_id(dst), "kind": "neighbor", "weight": 0.15})
        return {"nodes": nodes, "edges": edges, "meta": {"user_id": uid, "path": str(path), "exists": True, "count": len(nodes), "system_counts": dict(body_counts), "limit": limit}, "systems": {key: {"label": _SYSTEM_LABELS[key]} for key in _SYSTEM_LABELS}}
    finally:
        conn.close()


def markdown_atlas(user_id: str | None = None, limit: int = 400) -> str:
    """Create a portable architecture handout from the same graph data shown in the UI."""
    graph = export_codebase_graph(user_id=user_id, limit=limit)
    lines = ["# Aiko Codebase Atlas", "", "Generated from the indexed Codebase Studio graph.", ""]
    for system in sorted({node["system"] for node in graph["nodes"]}):
        lines.extend([f"## {system.title()} system", ""])
        for node in sorted((node for node in graph["nodes"] if node["system"] == system), key=lambda node: node["module"]):
            lines.append(f"### `{node['module']}`")
            lines.append(f"{node['file_count']} files · {node['loc']} lines · {node['function_count']} functions · complexity estimate {node['complexity']} · {node['change_count']} git changes")
            lines.append("")
    return "\n".join(lines)


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
        metrics = _module_metrics(paths)
        functions = metrics["functions"] or _symbols(text)
        body_part, _ = _body_for_path(module)
        function_phrase = ", ".join(functions[:8]) if functions else "No named functions were found in the indexed excerpts"
        all_paths = [row["path"] for row in conn.execute("SELECT path FROM codebase_docs WHERE user_id=?", (uid,)).fetchall()]
        indexed_modules = {_module_for_path(path) for path in all_paths}
        dependencies = sorted({_module_for_path(target) for path in paths for target in _imports(_source_for_path(path)) if _module_for_path(target) in indexed_modules and _module_for_path(target) != module})
        # Source-based reverse dependencies make this work even when the graph endpoint has not been loaded first.
        dependents = sorted({candidate for candidate in indexed_modules if candidate != module and module in {_module_for_path(target) for path in all_paths if _module_for_path(path) == candidate for target in _imports(_source_for_path(path))}})
        changes = sum(_git_change_counts(paths).values())
        repo_url = _repository_url()
        summary = f"{module} is a {body_part} module in Aiko's codebase. It contains {len(paths)} indexed file{'s' if len(paths) != 1 else ''} and appears to provide {function_phrase}."
        return {
            "module": module, "system": body_part, "summary": summary,
            "functions": functions[:12], "files": paths, "excerpt": re.sub(r"\s+", " ", text)[:360],
            "docstrings": _docstrings(paths), "metrics": {key: metrics[key] for key in ("loc", "function_count", "complexity")},
            "dependencies": dependencies, "dependents": dependents, "change_count": changes,
            "coverage": _coverage_for_paths(paths), "coverage_note": "Coverage is populated from coverage.json when it is available.",
            "source_links": [{"path": path, "github": f"{repo_url}/blob/HEAD/{path}" if repo_url else None, "vscode": f"vscode://file/{_REPO_ROOT / path}"} for path in paths],
        }
    finally:
        conn.close()

"""Read-only repo inspection tools for Aiko architecture work."""

from __future__ import annotations

from itertools import islice
from pathlib import Path

from core.toolkit.common import json_block

REPO_ROOT = Path(__file__).resolve().parents[2]
MAX_REPO_READ_CHARS = 20_000
_ALLOWED_TEXT_SUFFIXES = {".py", ".md", ".toml", ".json", ".yaml", ".yml", ".txt", ".sh", ".html", ".css", ".js", ".ts"}
_SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", "node_modules", "dist", "build"}


def _safe_repo_path(relative_path: str) -> Path:
    cleaned = relative_path.strip().lstrip("/\\")
    path = (REPO_ROOT / cleaned).resolve()
    if path != REPO_ROOT and REPO_ROOT not in path.parents:
        raise ValueError(f"path escapes repository: {relative_path}")
    return path


def _iter_repo_files(root: Path = REPO_ROOT):
    for path in root.rglob("*"):
        if any(part in _SKIP_DIRS for part in path.relative_to(REPO_ROOT).parts):
            continue
        if not path.is_file():
            continue
        resolved = path.resolve()
        if resolved != REPO_ROOT and REPO_ROOT not in resolved.parents:
            continue
        if resolved.suffix.lower() in _ALLOWED_TEXT_SUFFIXES:
            yield resolved


def repo_file_tree(prefix: str = "", limit: int = 200) -> str:
    """List repository text files for architecture/code navigation."""
    try:
        base = _safe_repo_path(prefix) if prefix else REPO_ROOT
        if base.is_file():
            if base.suffix.lower() not in _ALLOWED_TEXT_SUFFIXES:
                return f"[repo tree failed: unsupported file type: {base.suffix}]"
            files = [base]
        else:
            files = list(islice(_iter_repo_files(base), max(1, min(limit, 1000))))
        return json_block("repo file tree", {
            "root": str(REPO_ROOT),
            "prefix": prefix or ".",
            "count": len(files),
            "files": [str(p.relative_to(REPO_ROOT)) for p in files],
        })
    except Exception as e:
        return f"[repo tree failed: {e}]"


def repo_read_file(relative_path: str, max_chars: int = MAX_REPO_READ_CHARS) -> str:
    """Read one repository text file without permitting path traversal."""
    try:
        path = _safe_repo_path(relative_path)
        if not path.exists() or not path.is_file():
            return f"[repo read failed: file not found: {relative_path}]"
        if path.suffix.lower() not in _ALLOWED_TEXT_SUFFIXES:
            return f"[repo read failed: unsupported file type: {path.suffix}]"
        return path.read_text(encoding="utf-8", errors="replace")[: max(1, min(max_chars, 50_000))]
    except Exception as e:
        return f"[repo read failed: {e}]"


def repo_search_text(query: str, prefix: str = "", limit: int = 50) -> str:
    """Search repository text files with simple case-insensitive substring matching."""
    try:
        needle = query.casefold().strip()
        if not needle:
            return "[repo search failed: empty query]"
        base = _safe_repo_path(prefix) if prefix else REPO_ROOT
        if base.is_file():
            if base.suffix.lower() not in _ALLOWED_TEXT_SUFFIXES:
                return f"[repo search failed: unsupported file type: {base.suffix}]"
            files = [base]
        else:
            files = _iter_repo_files(base)
        matches = []
        for path in files:
            try:
                for lineno, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                    if needle in line.casefold():
                        matches.append({
                            "file": str(path.relative_to(REPO_ROOT)),
                            "line": lineno,
                            "text": line.strip()[:240],
                        })
                        break
            except OSError:
                continue
            if len(matches) >= max(1, min(limit, 200)):
                break
        return json_block("repo search", {"query": query, "prefix": prefix or ".", "count": len(matches), "matches": matches})
    except Exception as e:
        return f"[repo search failed: {e}]"

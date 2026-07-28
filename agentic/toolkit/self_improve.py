"""
toolkit/self_improve.py

Read-only repo inspection tools for Aiko architecture work.

This module provides utilities for exploring and analyzing the codebase:

  - repo_file_tree()    — generate a tree view of repository files
  - repo_read_file()    — read file contents with safety checks
  - repo_search_text()  — search for text patterns across repository files

All operations are read-only and respect repository boundaries.
"""

from __future__ import annotations

from itertools import islice
from pathlib import Path

from agentic.toolkit.common import json_block
from agentic.registry import TOOLS, tool

REPO_ROOT = Path(__file__).resolve().parents[2]    # repo directory (the first 2 hierachy directories of this file)
MAX_REPO_READ_CHARS = 20_000                       # the characters allowed to be read in the repo
_ALLOWED_TEXT_SUFFIXES = {".py", ".md", ".toml", ".json", ".yaml", ".yml", ".txt", ".sh", ".html", ".css", ".js", ".ts"}   # file suffixes that allowed access
_SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", "node_modules", "dist", "build"}                    # blacklisted directories


# ── Public API ──────────────────────────────────────────────────────────

@tool(TOOLS["repo_file_tree"])
def repo_file_tree(prefix: str = "", limit: int = 200) -> str:
    """
    List repository text files for architecture/code navigation.

    Walks the repository (or a subdirectory of it) and returns the relative
    paths of files whose extension is in ``_ALLOWED_TEXT_SUFFIXES``,
    skipping directories in ``_SKIP_DIRS`` (e.g. ``.git``, ``venv``,
    ``node_modules``). If ``prefix`` points to a single file instead of a
    directory, that file alone is returned (subject to the same extension
    check).

    Args:
        prefix: Repository-relative path to scope the listing to. Empty
            string (default) lists from the repo root. Resolved and
            confined to the repo via ``_confine_repo_path``; paths that
            escape the repository raise and are reported as a failure.
        limit: Maximum number of files to return. Clamped to [1, 1000].
            Ignored when ``prefix`` resolves to a single file.

    Returns:
        A JSON block (via ``json_block``) containing ``root`` (absolute
        repo root), ``prefix`` (as given, or ``"."``), ``count``, and
        ``files`` (paths relative to the repo root). On failure, returns
        a ``"[repo tree failed: ...]"`` string instead of raising.
    """
    try:
        base = _repo_confine_path(prefix) if prefix else REPO_ROOT
        if base.is_file():
            if base.suffix.lower() not in _ALLOWED_TEXT_SUFFIXES:
                return f"[repo tree failed: unsupported file type: {base.suffix}]"
            files = [base]
        else:
            files = list(islice(_repo_iter_path(base), max(1, min(limit, 1000))))
        return json_block("repo file tree", {
            "root": str(REPO_ROOT),
            "prefix": prefix or ".",
            "count": len(files),
            "files": [str(p.relative_to(REPO_ROOT)) for p in files],
        })
    except Exception as e:
        return f"[repo tree failed: {e}]"


@tool(TOOLS["repo_read_file"])
def repo_read_file(relative_path: str, max_chars: int = MAX_REPO_READ_CHARS) -> str:
    """
    Read one repository text file without permitting path traversal.

    Args:
        relative_path: Repository-relative path to the file to read.
            Resolved and confined to the repo via ``_confine_repo_path``;
            paths that escape the repository, don't exist, aren't a file,
            or have a disallowed extension are reported as a failure
            rather than raising.
        max_chars: Maximum number of characters to return. Clamped to
            [1, 50_000]; defaults to ``MAX_REPO_READ_CHARS`` (20,000).

    Returns:
        The file's text content (UTF-8, invalid bytes replaced),
        truncated to ``max_chars``. On failure, returns a
        ``"[repo read failed: ...]"`` string instead of raising.
    """
    try:
        path = _repo_confine_path(relative_path)
        if not path.exists() or not path.is_file():
            return f"[repo read failed: file not found: {relative_path}]"
        if path.suffix.lower() not in _ALLOWED_TEXT_SUFFIXES:
            return f"[repo read failed: unsupported file type: {path.suffix}]"
        return path.read_text(encoding="utf-8", errors="replace")[: max(1, min(max_chars, 50_000))]
    except Exception as e:
        return f"[repo read failed: {e}]"


@tool(TOOLS["repo_search_text"])
def repo_search_text(query: str, prefix: str = "", limit: int = 50) -> str:
    """
    Search repository text files with simple case-insensitive substring matching.

    Scans allowed text files line by line for ``query`` and records at
    most one match per file (the first matching line). Search scope can
    be narrowed to a subdirectory or a single file via ``prefix``.

    Args:
        query: Substring to search for, case-insensitive. Empty/whitespace
            queries are rejected as a failure.
        prefix: Repository-relative path to scope the search to. Empty
            string (default) searches the whole repo. Resolved and
            confined to the repo via ``_confine_repo_path``.
        limit: Maximum number of matching files to return. Clamped to
            [1, 200].

    Returns:
        A JSON block (via ``json_block``) containing ``query``, ``prefix``
        (as given, or ``"."``), ``count``, and ``matches`` — a list of
        ``{"file", "line", "text"}`` dicts, one per matching file, with
        the matched line truncated to 240 characters. On failure, returns
        a ``"[repo search failed: ...]"`` string instead of raising.
    """
    try:
        needle = query.casefold().strip()
        if not needle:
            return "[repo search failed: empty query]"
        base = _repo_confine_path(prefix) if prefix else REPO_ROOT
        if base.is_file():
            if base.suffix.lower() not in _ALLOWED_TEXT_SUFFIXES:
                return f"[repo search failed: unsupported file type: {base.suffix}]"
            files = [base]
        else:
            files = _repo_iter_path(base)
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


# ── Private Helpers ─────────────────────────────────────────────────────

def _repo_confine_path(jail_path: str) -> Path:
    """Normalize path (resolve symlinks, `..`) and confine within REPO_ROOT; raise if it escapes."""
    cleaned = jail_path.strip().lstrip("/\\")                      # strip out leading slash symbol
    path = (REPO_ROOT / cleaned).resolve()                         # add repo path in front
    if path != REPO_ROOT and REPO_ROOT not in path.parents:        # if resolved path not within repo,
        raise ValueError(f"path escapes repository: {jail_path}")  # raise error
    return path                                                    # return the resolved path


def _repo_iter_path(root: Path = REPO_ROOT):
    """Recursively yield allowed text files, skipping excluded directories."""
    for path in root.rglob("*"):                                                    # recursively walk everything under repo root
        if any(part in _SKIP_DIRS for part in path.relative_to(REPO_ROOT).parts):   # if any part of the path's hierarchy is blacklisted,
            continue                                                                # skip
        if not path.is_file():                                                      # if path is not a file,
            continue                                                                # skip
        resolved = path.resolve()                                                   # grab the resolved path
        if resolved != REPO_ROOT and REPO_ROOT not in resolved.parents:             # safeguard against escaped path
            continue                                                                # skip
        if resolved.suffix.lower() in _ALLOWED_TEXT_SUFFIXES:                       # if path has file suffix that allowed access
            yield resolved                                                          # yield the resolved path
          

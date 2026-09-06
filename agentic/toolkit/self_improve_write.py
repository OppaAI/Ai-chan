"""
agentic/toolkit/self_improve_write.py

Write-enabled repository tools for Aiko self-improvement loop.
These tools allow Aiko to modify her own codebase but operate in a sandboxed manner
or require human review for safety (especially on edge devices).

Provides:
  - repo_write_file()
  - repo_replace_text()
"""

from __future__ import annotations

import os
from pathlib import Path

from agentic.toolkit.common import json_block
from agentic.registry import TOOLS, tool

REPO_ROOT = Path(__file__).resolve().parents[2]
_ALLOWED_WRITE_EXTENSIONS = {".py", ".md", ".json", ".txt", ".sh", ".html", ".css", ".js"}
_SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", "node_modules"}

def _repo_confine_path(jail_path: str) -> Path:
    cleaned = jail_path.strip().lstrip("/\\")
    path = (REPO_ROOT / cleaned).resolve()
    if path != REPO_ROOT and REPO_ROOT not in path.parents:
        raise ValueError(f"Path escapes repository: {jail_path}")
    return path


@tool(
    TOOLS.get("repo_write_file", "repo_write_file"),
    description="Write content to a file in the repository. Restricted to safe text formats. APPROVAL REQUIRED — prefer code_diff_preview + code_apply_patch.",
    graph=True,
    react=True,
    domain="self_improve",
    needs_approval=True,
)
def repo_write_file(relative_path: str, content: str) -> str:
    """Write entire content to a file."""
    try:
        if len(content or "") > 50000:
            return "[repo write failed: content exceeds 50000 chars; use code_apply_patch for small edits]"
        low = (relative_path or "").lower()
        if any(s in low for s in (".env", "secret", "token", "private_key", ".pem", ".key", "credentials")):
            return "[repo write failed: refusing secrets/credentials path]"
        path = _repo_confine_path(relative_path)
        if any(part in _SKIP_DIRS for part in path.relative_to(REPO_ROOT).parts):
            return "[repo write failed: cannot write to restricted directories]"
        if path.suffix.lower() not in _ALLOWED_WRITE_EXTENSIONS:
            return f"[repo write failed: unsupported file type: {path.suffix}]"
        
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        
        return json_block("repo_write_file", {
            "path": str(path.relative_to(REPO_ROOT)),
            "bytes_written": len(content),
            "status": "success"
        })
    except Exception as e:
        return f"[repo write failed: {e}]"


@tool(
    TOOLS.get("repo_replace_text", "repo_replace_text"),
    description="Replace specific text in a repository file. APPROVAL REQUIRED — prefer code_diff_preview + code_apply_patch.",
    graph=True,
    react=True,
    domain="self_improve",
    needs_approval=True,
)
def repo_replace_text(relative_path: str, old_text: str, new_text: str) -> str:
    """Replace occurrences of old_text with new_text in a file."""
    try:
        low = (relative_path or "").lower()
        if any(s in low for s in (".env", "secret", "token", "private_key", ".pem", ".key", "credentials")):
            return "[repo replace failed: refusing secrets/credentials path]"
        path = _repo_confine_path(relative_path)
        if not path.exists() or not path.is_file():
            return f"[repo replace failed: file not found: {relative_path}]"
        if path.suffix.lower() not in _ALLOWED_WRITE_EXTENSIONS:
            return f"[repo replace failed: unsupported file type: {path.suffix}]"
            
        content = path.read_text(encoding="utf-8")
        if old_text not in content:
            return "[repo replace failed: old_text not found in file]"
            
        updated = content.replace(old_text, new_text)
        path.write_text(updated, encoding="utf-8")
        
        return json_block("repo_replace_text", {
            "path": str(path.relative_to(REPO_ROOT)),
            "status": "success"
        })
    except Exception as e:
        return f"[repo replace failed: {e}]"


__all__ = ["repo_write_file", "repo_replace_text"]

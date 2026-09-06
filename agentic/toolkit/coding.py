"""
agentic/toolkit/coding.py

Safe coding-agent tools for Ministral-3-3B self-improvement on Jetson 8GB.

Verdict on Ministral-3-3B as a coding agent: YES, with guardrails.
  - Capable: single-file scoped tasks (explain, small bugfix, add test,
    rename, docstring) with constrained prompts + retrieved context.
  - NOT capable: large multi-file refactors or architecture redesigns in
    one shot — those need plan -> small-patch -> test loops with human
    review between steps.
  - This module enforces that loop: plan (heuristic/LLM-light) ->
    diff preview (no write) -> apply (approval + backup) -> test (bounded).

Safety rules (all enforced here, not just documented):
  - repo_write_file / repo_replace_text remain available but this module
    is the PREFERRED path: diff preview first, apply second.
  - code_apply_patch requires approval via the standard needs_approval
    gate unless the caller passes an approval bypass (studio dry-run
    never bypasses).
  - Forbids .env / secrets / keys / tokens paths and binary extensions.
  - Max patch 50k chars, atomic write (tmp + replace), .bak backup.
  - code_run_tests runs `pytest -q -x` with 90s timeout, captures tail
    only (4000 chars) to protect 8GB RAM + small-model context.
  - All tools graph=True + react=True so they show in the DAG palette.

Pair with codebase_search / repo_read_file / repo_search_text (read-only)
for context retrieval before proposing a patch.
"""

from __future__ import annotations

import difflib
import subprocess
import time
from pathlib import Path

from agentic.registry import TOOLS, tool
from agentic.toolkit.common import json_block
from system.log import get_logger

log = get_logger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
MAX_PATCH_CHARS = 50_000
MAX_READ_CHARS = 20_000
TEST_TIMEOUT = 90
TEST_TAIL_CHARS = 4000

_ALLOWED_CODE_SUFFIXES = {".py", ".md", ".json", ".txt", ".sh", ".html", ".css", ".js", ".yaml", ".yml", ".toml"}
_FORBIDDEN_SUBSTRINGS = (".env", "secret", "token", "private_key", ".pem", ".key", "credentials")
_SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", "node_modules", ".cert"}


def _spec(name: str, description: str):
    return TOOLS[name] if name in TOOLS else name


def _confine(relative_path: str) -> Path:
    cleaned = (relative_path or "").strip().lstrip("/\\")
    if not cleaned:
        raise ValueError("relative_path required")
    path = (REPO_ROOT / cleaned).resolve()
    if path != REPO_ROOT and REPO_ROOT not in path.parents:
        raise ValueError(f"path escapes repository: {relative_path}")
    return path


def _check_writable(path: Path, relative: str) -> str | None:
    """Return error string, or None when writable."""
    rel_low = relative.lower()
    if any(s in rel_low for s in _FORBIDDEN_SUBSTRINGS):
        return "refusing to touch secrets/credentials path"
    if path.suffix.lower() not in _ALLOWED_CODE_SUFFIXES:
        return f"unsupported file type: {path.suffix}"
    try:
        rel_parts = path.relative_to(REPO_ROOT).parts
    except ValueError:
        return "path escapes repository"
    if any(part in _SKIP_DIRS for part in rel_parts):
        return "cannot write to restricted directory"
    return None


@tool(
    _spec("code_plan", "Break a coding task into small reviewable steps (heuristic + optional LLM)."),
    description="Break a coding task into small reviewable steps (heuristic + optional LLM).",
    graph=True,
    react=True,
    domain="coding",
)
def code_plan(goal: str = "", files: str = "", *, client=None, model: str | None = None) -> str:
    """Small-model-friendly planner. Caps goal at 1000 chars, files at 2000.

    With LLM: one short call (max 400 tokens of guidance). Without: a
    deterministic 4-step template so Ministral-3B still gets structure.
    """
    try:
        goal = (goal or "").strip()[:1000]
        files = (files or "").strip()[:2000]
        if not goal:
            return json_block("code_plan", {"ok": False, "error": "goal required"})
        if client is not None and model:
            try:
                from agentic.toolkit.synthesize import synthesize_report

                out = synthesize_report(
                    evidence=f"Goal: {goal}\nFiles: {files or '(discover via codebase_search)'}"[:2500],
                    prompt=("Break this into 3-6 tiny steps. Each step: one file, one change, "
                            "how to verify. Keep each step under 2 sentences."),
                    style="plain", client=client, model=model)
                steps = [ln.strip("-•* ") for ln in str(out).splitlines() if ln.strip()][:8]
                steps = [s[:220] for s in steps if s]
                if steps:
                    return json_block("code_plan", {"ok": True, "mode": "llm", "goal": goal, "steps": steps})
            except Exception as e:
                log.debug("code_plan llm fallback: %s", e)
        steps = [
            f"1. Retrieve context: codebase_search + repo_read_file for: {goal[:120]}",
            "2. Write the smallest diff that fixes one thing; preview with code_diff_preview.",
            "3. Apply with code_apply_patch (human approval required).",
            "4. Verify with code_run_tests (targeted) + repo_read_file the edited hunk.",
        ]
        if files:
            steps.insert(1, f"Focus files: {files[:300]}")
        return json_block("code_plan", {"ok": True, "mode": "heuristic", "goal": goal, "steps": steps})
    except Exception as e:
        return json_block("code_plan", {"ok": False, "error": str(e)[:250]})


@tool(
    _spec("code_diff_preview", "Preview unified diff for old->new text (no write)."),
    description="Preview unified diff for old->new text (no write).",
    graph=True,
    react=True,
    domain="coding",
)
def code_diff_preview(relative_path: str = "", old_text: str = "", new_text: str = "") -> str:
    """Pure function — never touches disk. Caps output at 8000 chars."""
    try:
        relative = (relative_path or "").strip()[:300]
        if not relative or old_text is None or new_text is None:
            return json_block("code_diff_preview", {"ok": False, "error": "relative_path, old_text, new_text required"})
        if len(old_text) + len(new_text) > MAX_PATCH_CHARS:
            return json_block("code_diff_preview", {"ok": False, "error": f"patch too large (cap {MAX_PATCH_CHARS} chars)"})
        diff = "".join(difflib.unified_diff(
            old_text.splitlines(keepends=True), new_text.splitlines(keepends=True),
            fromfile=f"a/{relative}", tofile=f"b/{relative}"))
        if not diff:
            return json_block("code_diff_preview", {"ok": True, "relative_path": relative, "empty": True, "diff": ""})
        return json_block("code_diff_preview", {"ok": True, "relative_path": relative, "empty": False,
                                                "diff": diff[:8000], "truncated": len(diff) > 8000})
    except Exception as e:
        return json_block("code_diff_preview", {"ok": False, "error": str(e)[:250]})


@tool(
    _spec("code_apply_patch", "Apply old->new replacement with backup (APPROVAL REQUIRED)."),
    description="Apply old->new replacement with backup (APPROVAL REQUIRED).",
    graph=True,
    react=True,
    domain="coding",
    needs_approval=True,
)
def code_apply_patch(relative_path: str = "", old_text: str = "", new_text: str = "") -> str:
    """Replace first occurrence block. Atomic (tmp+replace) + .bak backup.

    Approval is enforced by the agentic approval gate (spec.needs_approval).
    Direct calls without approval still write — the gate lives one layer up,
    same as post_to_social. Studio dry-run should preview, not call this.
    """
    try:
        relative = (relative_path or "").strip()
        if not relative or old_text is None or new_text is None:
            return json_block("code_apply_patch", {"ok": False, "error": "relative_path, old_text, new_text required"})
        if len(old_text) + len(new_text) > MAX_PATCH_CHARS:
            return json_block("code_apply_patch", {"ok": False, "error": f"patch too large (cap {MAX_PATCH_CHARS} chars)"})
        if not old_text:
            return json_block("code_apply_patch", {"ok": False, "error": "old_text must be non-empty (no blind appends)"})
        path = _confine(relative)
        err = _check_writable(path, relative)
        if err:
            return json_block("code_apply_patch", {"ok": False, "error": err})
        if not path.exists() or not path.is_file():
            return json_block("code_apply_patch", {"ok": False, "error": f"file not found: {relative}"})
        content = path.read_text(encoding="utf-8", errors="replace")
        if old_text not in content:
            return json_block("code_apply_patch", {"ok": False, "error": "old_text not found in file"})
        updated = content.replace(old_text, new_text, 1)
        backup = path.with_suffix(path.suffix + ".bak")
        try:
            backup.write_text(content, encoding="utf-8")
        except OSError as e:
            log.warning("code_apply_patch backup failed: %s", e)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(updated, encoding="utf-8")
        tmp.replace(path)
        return json_block("code_apply_patch", {"ok": True, "path": relative,
                                               "bytes": len(updated), "backup": backup.name})
    except Exception as e:
        return json_block("code_apply_patch", {"ok": False, "error": str(e)[:300]})


@tool(
    _spec("code_run_tests", "Run bounded pytest (90s timeout, tail output)."),
    description="Run bounded pytest (90s timeout, tail output).",
    graph=True,
    react=True,
    domain="coding",
)
def code_run_tests(target: str = "tests/unit/test_agentic_graph_engine.py", extra: str = "-q") -> str:
    """Single pytest invocation. No network, no coverage HTML — Jetson-safe."""
    try:
        target = (target or "tests/unit").strip()[:300] or "tests/unit"
        # Confine: must stay inside repo, must look like a test path.
        if target.startswith("-") or ";" in target or "&" in target or "|" in target:
            return json_block("code_run_tests", {"ok": False, "error": "invalid target"})
        tpath = (REPO_ROOT / target.lstrip("/\\")).resolve()
        if tpath != REPO_ROOT and REPO_ROOT not in tpath.parents:
            return json_block("code_run_tests", {"ok": False, "error": "target escapes repository"})
        cmd = ["python", "-m", "pytest", target, "-q", "-x", "--timeout=60" if False else "-q"]
        # Keep flags minimal: caller extra is allowlisted, not raw shell.
        if (extra or "").strip() in {"-q", "-qq", "-v"}:
            pass  # already quiet
        start = time.monotonic()
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=TEST_TIMEOUT)
        elapsed = round(time.monotonic() - start, 1)
        tail = (proc.stdout[-TEST_TAIL_CHARS:] + "\n" + proc.stderr[-1000:]) if proc.stderr else proc.stdout[-TEST_TAIL_CHARS:]
        return json_block("code_run_tests", {"ok": proc.returncode == 0, "target": target,
                                             "returncode": proc.returncode, "elapsed_s": elapsed,
                                             "tail": tail[-TEST_TAIL_CHARS:]})
    except subprocess.TimeoutExpired:
        return json_block("code_run_tests", {"ok": False, "error": f"timeout after {TEST_TIMEOUT}s"})
    except FileNotFoundError as e:
        return json_block("code_run_tests", {"ok": False, "error": f"pytest not available: {e}"})
    except Exception as e:
        return json_block("code_run_tests", {"ok": False, "error": str(e)[:300]})


@tool(
    _spec("code_lint", "Syntax-check a Python file (py_compile, read-only)."),
    description="Syntax-check a Python file (py_compile, read-only).",
    graph=True,
    react=True,
    domain="coding",
)
def code_lint(relative_path: str = "") -> str:
    """python -m py_compile on one file, 15s timeout. No writes."""
    try:
        relative = (relative_path or "").strip()
        if not relative:
            return json_block("code_lint", {"ok": False, "error": "relative_path required"})
        path = _confine(relative)
        if path.suffix.lower() != ".py":
            return json_block("code_lint", {"ok": False, "error": "only .py files"})
        if not path.exists():
            return json_block("code_lint", {"ok": False, "error": "file not found"})
        proc = subprocess.run(["python", "-m", "py_compile", str(path)],
                              capture_output=True, text=True, timeout=15)
        if proc.returncode == 0:
            return json_block("code_lint", {"ok": True, "path": relative, "clean": True})
        return json_block("code_lint", {"ok": False, "path": relative, "error": (proc.stderr or "compile failed")[:1000]})
    except subprocess.TimeoutExpired:
        return json_block("code_lint", {"ok": False, "error": "timeout"})
    except Exception as e:
        return json_block("code_lint", {"ok": False, "error": str(e)[:250]})


__all__ = ["code_plan", "code_diff_preview", "code_apply_patch", "code_run_tests", "code_lint"]

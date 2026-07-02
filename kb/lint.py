#!/usr/bin/env python3
"""Lint Aiko's local wiki and skill knowledge files.

This is intentionally dependency-free so it can run before the full runtime
stack is installed. It checks the human-maintained knowledge layer that Aiko
retrieves before agentic work: wiki cards and SKILL.md workflow files.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
WIKI_REQUIRED = ("id", "name", "summary", "status", "owner")
SKILL_REQUIRED = ("id", "name", "summary", "triggers", "tools")
_ID_RE = re.compile(r"^[a-z][a-z0-9_/-]*$")
_LINK_RE = re.compile(r"\[[^\]]+\]\((?!https?://|mailto:|#)([^)]+)\)")


@dataclass(frozen=True)
class LintIssue:
    path: Path
    message: str

    def render(self) -> str:
        return f"{self.path.relative_to(REPO_ROOT)}: {self.message}"


def _front_matter(path: Path) -> tuple[dict[str, str], str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    if not text.startswith("---\n"):
        return {}, text
    _start, _sep, rest = text.partition("---\n")
    meta_text, sep, body = rest.partition("\n---\n")
    if not sep:
        return {}, text
    meta: dict[str, str] = {}
    for line in meta_text.splitlines():
        key, found, value = line.partition(":")
        if found:
            meta[key.strip()] = value.strip().strip('"\'')
    return meta, body


def _check_required(path: Path, meta: dict[str, str], required: tuple[str, ...]) -> list[LintIssue]:
    issues: list[LintIssue] = []
    if not meta:
        return [LintIssue(path, "missing YAML front matter")]
    for key in required:
        if not meta.get(key):
            issues.append(LintIssue(path, f"missing required front matter field: {key}"))
    item_id = meta.get("id", "")
    if item_id and not _ID_RE.match(item_id):
        issues.append(LintIssue(path, f"invalid id {item_id!r}; use lowercase letters, numbers, _, -, /"))
    return issues


def _check_links(path: Path, body: str) -> list[LintIssue]:
    issues: list[LintIssue] = []
    for match in _LINK_RE.finditer(body):
        target = match.group(1).split("#", 1)[0]
        if not target:
            continue
        resolved = (path.parent / target).resolve()
        try:
            resolved.relative_to(REPO_ROOT.resolve())
        except ValueError:
            issues.append(LintIssue(path, f"local link escapes repo: {target}"))
            continue
        if not resolved.exists():
            issues.append(LintIssue(path, f"broken local link: {target}"))
    return issues


def lint_files() -> list[LintIssue]:
    issues: list[LintIssue] = []
    seen_ids: dict[str, Path] = {}
    targets = [
        *((path, WIKI_REQUIRED) for path in sorted((REPO_ROOT / "wiki").glob("*.md"))),
        *((path, SKILL_REQUIRED) for path in sorted((REPO_ROOT / "skills").glob("*/SKILL.md"))),
    ]
    for path, required in targets:
        meta, body = _front_matter(path)
        issues.extend(_check_required(path, meta, required))
        item_id = meta.get("id")
        if item_id:
            previous = seen_ids.get(item_id)
            if previous is not None:
                issues.append(LintIssue(path, f"duplicate id {item_id!r}; first seen in {previous.relative_to(REPO_ROOT)}"))
            else:
                seen_ids[item_id] = path
        issues.extend(_check_links(path, body))
    return issues


def main() -> int:
    issues = lint_files()
    if issues:
        print("KB lint failed:")
        for issue in issues:
            print(f"- {issue.render()}")
        return 1
    print("KB lint passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

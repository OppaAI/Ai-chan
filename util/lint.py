#!/usr/bin/env python3
"""util/lint.py

Lint Aiko's local wiki and skill knowledge files.

This is intentionally dependency-free so it can run before the full runtime
stack is installed. It checks the human-maintained knowledge layer that Aiko
retrieves before agentic work: wiki cards and skill workflow files.
"""

from __future__ import annotations

import re
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
WIKI_REQUIRED = ("id", "name", "summary", "status", "owner")
SKILL_REQUIRED = ("id", "name", "summary", "triggers", "tools")
_ID_RE = re.compile(r"^[a-z][a-z0-9_/-]*$")
_LINK_RE = re.compile(r"\[[^\]]+\]\((?!https?://|mailto:|#)([^)]+)\)")
MetaValue = str | list[str]


@dataclass(frozen=True)
class LintIssue:
    path: Path
    message: str

    def render(self) -> str:
        return f"{self.path.relative_to(REPO_ROOT)}: {self.message}"


def _front_matter(path: Path) -> tuple[dict[str, MetaValue], str]:
    """Parse the small YAML subset used by Aiko knowledge files.

    Supports flat ``key: value`` scalars and block lists:

        triggers:
          - schedule
          - reminder

    The parser is intentionally dependency-free; it does not aim to be a full
    YAML implementation. Unsupported nested structures are ignored rather than
    treated as required-field values.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    if not text.startswith("---\n"):
        return {}, text
    _start, _sep, rest = text.partition("---\n")
    meta_text, sep, body = rest.partition("\n---\n")
    if not sep:
        return {}, text

    meta: dict[str, MetaValue] = {}
    current_list_key: str | None = None
    for raw_line in meta_text.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if current_list_key and raw_line.startswith((" ", "\t")) and stripped.startswith("- "):
            value = stripped[2:].strip().strip('"\'')
            if value:
                existing = meta.setdefault(current_list_key, [])
                if isinstance(existing, list):
                    existing.append(value)
            continue
        current_list_key = None
        key, found, value = raw_line.partition(":")
        if not found or raw_line[:1].isspace():
            continue
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if value:
            meta[key] = value.strip('"\'')
        else:
            meta[key] = []
            current_list_key = key
    return meta, body


def _has_value(value: MetaValue | None) -> bool:
    if isinstance(value, list):
        return any(item.strip() for item in value)
    return bool(value and value.strip())


def _as_scalar(value: MetaValue | None) -> str:
    return value if isinstance(value, str) else ""


def _check_required(path: Path, meta: dict[str, MetaValue], required: tuple[str, ...]) -> list[LintIssue]:
    issues: list[LintIssue] = []
    if not meta:
        return [LintIssue(path, "missing YAML front matter")]
    for key in required:
        if not _has_value(meta.get(key)):
            issues.append(LintIssue(path, f"missing required front matter field: {key}"))
    item_id = _as_scalar(meta.get("id"))
    if item_id and not _ID_RE.match(item_id):
        issues.append(LintIssue(path, f"invalid id {item_id!r}; use lowercase letters, numbers, _, -, /"))
    return issues


def _normalize_link_target(raw_target: str) -> str:
    target = raw_target.strip()
    if target.startswith("<") and ">" in target:
        target = target[1:target.index(">")].strip()
    else:
        try:
            parts = shlex.split(target)
        except ValueError:
            parts = target.split()
        target = parts[0] if parts else ""
    return target.split("#", 1)[0]


def _check_links(path: Path, body: str) -> list[LintIssue]:
    issues: list[LintIssue] = []
    for match in _LINK_RE.finditer(body):
        target = _normalize_link_target(match.group(1))
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
        *((path, SKILL_REQUIRED) for path in sorted((REPO_ROOT / "skills").glob("skillsets/*.md"))),
    ]
    for path, required in targets:
        meta, body = _front_matter(path)
        issues.extend(_check_required(path, meta, required))
        item_id = _as_scalar(meta.get("id"))
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

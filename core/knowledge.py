"""Basic local knowledge retrieval for Aiko.

This module indexes durable, human-maintained project knowledge: wiki cards,
skill docs/defaults, persona docs, selected config, and docs. It deliberately
skips secrets and mutable workspace artifacts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent

_SOURCE_GLOBS: tuple[tuple[str, str], ...] = (
    ("wiki", "wiki/*.md"),
    ("skill", "skills/*/SKILL.md"),
    ("skill_config", "skills/*/*.json"),
    ("persona", "persona/*.md"),
    ("persona_config", "persona/*.json"),
    ("config", "config/*.yaml"),
    ("config", "config/*.json"),
    ("config", "config/*.toml"),
    ("docs", "docs/*.md"),
)

_WORD_RE = re.compile(r"[a-z0-9_./-]+", re.IGNORECASE)
_HEADING_RE = re.compile(r"^#\s+(.+)$", re.MULTILINE)
_FRONT_MATTER_RE = re.compile(r"^---\n(?P<meta>.*?)\n---\n", re.DOTALL)


@dataclass(frozen=True)
class KnowledgeItem:
    item_id: str
    kind: str
    path: Path
    title: str
    text: str
    tags: tuple[str, ...]
    updated_at: str

    def relative_path(self) -> str:
        try:
            return str(self.path.resolve().relative_to(REPO_ROOT.resolve()))
        except ValueError:
            return str(self.path)

    def as_dict(self) -> dict:
        return {
            "id": self.item_id,
            "kind": self.kind,
            "path": self.relative_path(),
            "title": self.title,
            "tags": list(self.tags),
            "updated_at": self.updated_at,
        }


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _title_from_text(path: Path, text: str) -> str:
    front = _FRONT_MATTER_RE.search(text)
    if front:
        for line in front.group("meta").splitlines():
            key, found, value = line.partition(":")
            if found and key.strip() == "name":
                return value.strip().strip("'\"")
    heading = _HEADING_RE.search(text)
    if heading:
        return heading.group(1).strip()
    return path.stem.replace("_", " ").title()


def _tags_for(kind: str, path: Path, text: str) -> tuple[str, ...]:
    tags = {kind, path.stem}
    try:
        rel_parts = path.relative_to(REPO_ROOT).parts
    except ValueError:
        rel_parts = path.parts
    tags.update(part for part in rel_parts[:-1] if part not in {".", ""})
    front = _FRONT_MATTER_RE.search(text)
    if front:
        for line in front.group("meta").splitlines():
            key, found, value = line.partition(":")
            if found and key.strip() in {"id", "triggers", "tools"}:
                tags.update(part.strip() for part in value.split(",") if part.strip())
    return tuple(sorted(tags))


def _updated_at(path: Path) -> str:
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return ""
    return datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()


def discover_knowledge_items(root: str | Path = REPO_ROOT) -> list[KnowledgeItem]:
    base = Path(root)
    items: list[KnowledgeItem] = []
    for kind, pattern in _SOURCE_GLOBS:
        for path in sorted(base.glob(pattern)):
            if not path.is_file():
                continue
            try:
                text = _read_text(path)
            except OSError:
                continue
            rel = path.relative_to(base)
            item_id = str(rel.with_suffix("")).replace("/", ".")
            items.append(KnowledgeItem(
                item_id=item_id,
                kind=kind,
                path=path,
                title=_title_from_text(path, text),
                text=text,
                tags=_tags_for(kind, path, text),
                updated_at=_updated_at(path),
            ))
    return items


def _terms(text: str) -> list[str]:
    return [match.group(0).casefold() for match in _WORD_RE.finditer(text) if match.group(0).strip()]


def _score_item(item: KnowledgeItem, terms: Iterable[str]) -> int:
    title = item.title.casefold()
    item_id = item.item_id.casefold()
    tags = " ".join(item.tags).casefold()
    text = item.text.casefold()
    score = 0
    for term in terms:
        if term in item_id:
            score += 5
        if term in title:
            score += 4
        if term in tags:
            score += 3
        if term in text:
            score += 1
    return score


def search_knowledge(query: str, limit: int = 6, kinds: Iterable[str] | None = None) -> list[KnowledgeItem]:
    wanted = {kind for kind in kinds} if kinds is not None else None
    items = [item for item in discover_knowledge_items() if wanted is None or item.kind in wanted]
    query_terms = _terms(query)
    if not query_terms:
        return items[:limit]

    scored = [(score, item) for item in items if (score := _score_item(item, query_terms)) > 0]
    scored.sort(key=lambda pair: (-pair[0], pair[1].kind, pair[1].item_id))
    return [item for _score, item in scored[:limit]]


def _attr(value: object) -> str:
    return escape(str(value), quote=True)


def knowledge_context_for(query: str, limit: int = 5, max_chars: int = 9000) -> str:
    selected = search_knowledge(query, limit=limit)
    if not selected:
        return "<knowledge_context>\nNo matching local knowledge found.\n</knowledge_context>"

    chunks: list[str] = []
    remaining = max_chars
    for item in selected:
        if remaining <= 0:
            break
        text = item.text.strip()[:remaining]
        meta = item.as_dict()
        attrs = (
            f'id="{_attr(meta["id"])}" '
            f'kind="{_attr(meta["kind"])}" '
            f'path="{_attr(meta["path"])}" '
            f'title="{_attr(meta["title"])}" '
            f'updated_at="{_attr(meta["updated_at"])}"'
        )
        chunks.append(
            f"<knowledge_item {attrs}>\n"
            f"tags: {', '.join(meta['tags'])}\n\n{text}\n</knowledge_item>"
        )
        remaining -= len(text)
    return "<knowledge_context>\n" + "\n\n".join(chunks) + "\n</knowledge_context>"


def wiki_context_for(query: str, limit: int = 2, max_chars: int = 5000) -> str:
    selected = search_knowledge(query, limit=limit, kinds=("wiki",))
    if not selected:
        return "<wiki_context>\nNo operational wiki pages found.\n</wiki_context>"

    chunks: list[str] = []
    remaining = max_chars
    for item in selected:
        if remaining <= 0:
            break
        text = item.text.strip()[:remaining]
        chunks.append(f"<wiki_page id=\"{_attr(item.path.stem)}\">\n{text}\n</wiki_page>")
        remaining -= len(text)
    return "<wiki_context>\n" + "\n\n".join(chunks) + "\n</wiki_context>"

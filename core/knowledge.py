"""Basic local knowledge retrieval for Aiko.

This module indexes durable, human-maintained project knowledge: wiki cards,
skill docs/defaults, persona docs, selected config, and docs. It deliberately
skips secrets and mutable workspace artifacts.
"""

from __future__ import annotations

import os
import re
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Iterable, Protocol

import numpy as np

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
    ("docs", "*.md"),
    ("docs", "docs/*.md"),
)

_WORD_RE = re.compile(r"[a-z0-9_./-]+", re.IGNORECASE)
_HEADING_RE = re.compile(r"^#\s+(.+)$", re.MULTILINE)
_FRONT_MATTER_RE = re.compile(r"^---\n(?P<meta>.*?)\n---\n", re.DOTALL)

# Common words that appear in nearly every document regardless of topic.
# Without filtering these, a long unrelated doc (e.g. an install guide) can
# rack up incidental +1 "text" hits on words like "how"/"make"/"we" and
# qualify for injection even though it has nothing to do with the query.
_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "how", "what", "who", "when", "where", "why",
    "we", "you", "i", "he", "she", "they", "this", "that", "these",
    "those", "some", "any", "all", "each", "can", "could", "will",
    "would", "should", "shall", "may", "might", "must", "to", "of", "in",
    "on", "at", "for", "with", "and", "or", "not", "no", "yes", "make",
    "made", "get", "got", "go", "going", "let", "lets", "want", "wants",
    "just", "so", "up", "down", "out", "about", "if", "then", "than",
})

# Minimum score a knowledge item must reach before it's considered a genuine
# match under the keyword fallback path. Without this floor, search_knowledge
# includes anything with score > 0, which one stray substring hit satisfies.
_MIN_RELEVANCE_SCORE = 3

# ── semantic retrieval (Harrier embeddings) ─────────────────────────────────
#
# knowledge.py has no owner/session object of its own, so it can't reach
# think.py's HarrierEmbedder directly. Callers pass one in explicitly (duck
# typed to embed_query/embed_queries, matching think.py's usage) via the
# `embedder` parameter on search_knowledge/knowledge_context_for/
# wiki_context_for. When no embedder is given, or embedding fails for any
# reason, everything falls back to the keyword scoring above — semantic
# retrieval is strictly additive, never a hard dependency.

_KNOWLEDGE_INSTRUCT = "Which document is most relevant to this request or question?"
_KNOWLEDGE_SEMANTIC_THRESHOLD = float(os.getenv("KNOWLEDGE_SEMANTIC_THRESHOLD", "0.35"))
_EMBED_TEXT_CHARS = 800  # condensed per-item representation, not the full doc


class Embedder(Protocol):
    def embed_query(self, text: str, instruct: str = "") -> object: ...
    def embed_queries(self, texts: list[str], instruct: str = "") -> object: ...


_item_embed_cache: dict[tuple[str, str], np.ndarray] = {}
_item_embed_cache_lock = threading.RLock()


def _normalize(vector: np.ndarray) -> np.ndarray:
    arr = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    return arr / norm if norm > 1e-12 else arr


def _embed_source_text(item: "KnowledgeItem") -> str:
    """Condensed text to embed per item — title + tags + a text excerpt,
    not the full document (keeps embedding calls cheap and keeps the
    vector focused on what the item is about rather than diluted by
    boilerplate deep in a long file)."""
    body = _FRONT_MATTER_RE.sub("", item.text, count=1).strip()
    return f"{item.title}\n{' '.join(item.tags)}\n{body[:_EMBED_TEXT_CHARS]}"


def _get_item_embedding(item: "KnowledgeItem", embedder: Embedder) -> np.ndarray:
    """Cached per (item_id, updated_at) — unchanged files never get
    re-embedded, only new/edited ones."""
    cache_key = (item.item_id, item.updated_at)
    with _item_embed_cache_lock:
        cached = _item_embed_cache.get(cache_key)
        if cached is not None:
            return cached
    vector = _normalize(np.asarray(embedder.embed_query(_embed_source_text(item)), dtype=np.float32))
    with _item_embed_cache_lock:
        _item_embed_cache[cache_key] = vector
    return vector


def _semantic_rank(
    query: str, items: list["KnowledgeItem"], embedder: Embedder, threshold: float,
) -> list["KnowledgeItem"] | None:
    """Return items ranked by cosine similarity to the query, filtered by
    `threshold`, or None if embedding fails (caller falls back to keyword
    scoring in that case)."""
    if not items:
        return []
    try:
        query_vector = _normalize(np.asarray(embedder.embed_query(query, instruct=_KNOWLEDGE_INSTRUCT), dtype=np.float32))
        scored: list[tuple[float, "KnowledgeItem"]] = []
        for item in items:
            item_vector = _get_item_embedding(item, embedder)
            score = float(np.dot(query_vector, item_vector))
            if score >= threshold:
                scored.append((score, item))
        scored.sort(key=lambda pair: -pair[0])
        return [item for _score, item in scored]
    except Exception:
        return None


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
            if found and key.strip() in {"name", "title"}:
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
            if found and key.strip() in {"id", "status", "owner", "related", "triggers", "tools"}:
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
    return [
        match.group(0).casefold()
        for match in _WORD_RE.finditer(text)
        if match.group(0).strip() and match.group(0).casefold() not in _STOPWORDS
    ]


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


def search_knowledge(
    query: str,
    limit: int = 6,
    kinds: Iterable[str] | None = None,
    embedder: Embedder | None = None,
) -> list[KnowledgeItem]:
    wanted = {kind for kind in kinds} if kinds is not None else None
    items = [item for item in discover_knowledge_items() if wanted is None or item.kind in wanted]
    if not query.strip():
        return items[:limit]

    if embedder is not None:
        ranked = _semantic_rank(query, items, embedder, _KNOWLEDGE_SEMANTIC_THRESHOLD)
        if ranked is not None:
            return ranked[:limit]
        # embedding failed for some reason — fall through to keyword scoring

    query_terms = _terms(query)
    if not query_terms:
        # Query had content but it was entirely stopwords (e.g. "how do we
        # do this") — that's not a real topical query, so match nothing
        # rather than silently returning arbitrary items.
        return []

    scored = [
        (score, item)
        for item in items
        if (score := _score_item(item, query_terms)) >= _MIN_RELEVANCE_SCORE
    ]
    scored.sort(key=lambda pair: (-pair[0], pair[1].kind, pair[1].item_id))
    return [item for _score, item in scored[:limit]]


def _attr(value: object) -> str:
    return escape(str(value), quote=True)


def knowledge_context_for(
    query: str, limit: int = 5, max_chars: int = 9000, embedder: Embedder | None = None,
) -> str:
    selected = search_knowledge(query, limit=limit, embedder=embedder)
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


def wiki_context_for(
    query: str, limit: int = 2, max_chars: int = 5000, embedder: Embedder | None = None,
) -> str:
    selected = search_knowledge(query, limit=limit, kinds=("wiki",), embedder=embedder)
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
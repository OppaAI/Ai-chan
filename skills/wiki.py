"""
skills/wiki.py

Basic local wiki retrieval for Aiko.

This module indexes durable, human-maintained project knowledge sources:

  - discover_wiki_items()  — scan wiki and docs
  - search_wiki()          — find relevant items by keyword or semantic search
  - wiki_context_for()     — inject wiki pages via RAG-style chunking
  - wiki_knowledge_context_for() — inject wiki + docs via RAG-style chunking

Supports RAG-style chunking and semantic ranking via reason.select_relevant_chunks.
Designed to avoid injecting secrets and mutable workspace artifacts.
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

from cognition import reason

REPO_ROOT = Path(__file__).resolve().parent.parent

_SOURCE_GLOBS: tuple[tuple[str, str], ...] = (
    ("wiki", "wiki/*.md"),
    ("docs", "*.md"),
    ("docs", "docs/*.md"),
)

_WORD_RE = re.compile(r"[a-z0-9_./-]+", re.IGNORECASE)
_HEADING_RE = re.compile(r"^#\s+(.+)$", re.MULTILINE)
_FRONT_MATTER_RE = re.compile(r"^---\n(?P<meta>.*?)\n---\n", re.DOTALL)

_STOPWORDS = reason.STOPWORDS

# Minimum score for the keyword-fallback item ranking (whole-item, not
# chunk-level — chunk-level thresholds are separate, below).
_MIN_RELEVANCE_SCORE = 3

# ── item-level semantic ranking (which items are candidates at all) ────────
_WIKI_INSTRUCT = "Which document is most relevant to this request or question?"
_WIKI_SEMANTIC_THRESHOLD = float(os.getenv("WIKI_SEMANTIC_THRESHOLD", "0.35"))
_EMBED_TEXT_CHARS = 800  # condensed per-item representation, not the full doc

# ── chunk-level RAG selection (which PART of a selected item gets injected) ─
_WIKI_CHUNK_CHARS = int(os.getenv("WIKI_CHUNK_CHARS", "600"))
_WIKI_CHUNKS_PER_ITEM = int(os.getenv("WIKI_CHUNKS_PER_ITEM", "3"))
_WIKI_CHUNK_MIN_SCORE = float(os.getenv("WIKI_CHUNK_MIN_SCORE", "0.30"))


class Embedder(Protocol):
    def embed_query(self, text: str, instruct: str = "") -> object: ...
    def embed_queries(self, texts: list[str], instruct: str = "") -> object: ...


_item_embed_cache: dict[tuple[str, str], np.ndarray] = {}
_item_embed_cache_lock = threading.RLock()


def _embed_source_text(item: "WikiItem") -> str:
    """Condensed text to embed per item — title + tags + a text excerpt,
    not the full document."""
    body = _FRONT_MATTER_RE.sub("", item.text, count=1).strip()
    return f"{item.title}\n{' '.join(item.tags)}\n{body[:_EMBED_TEXT_CHARS]}"


def _get_item_embedding(item: "WikiItem", embedder: Embedder) -> np.ndarray:
    """Cached per (item_id, updated_at) — unchanged files never get
    re-embedded, only new/edited ones."""
    cache_key = (item.item_id, item.updated_at)
    with _item_embed_cache_lock:
        cached = _item_embed_cache.get(cache_key)
        if cached is not None:
            return cached
    vector = reason.normalize_vec(np.asarray(embedder.embed_query(_embed_source_text(item)), dtype=np.float32))
    with _item_embed_cache_lock:
        _item_embed_cache[cache_key] = vector
    return vector


def _semantic_rank(
    query: str, items: list["WikiItem"], embedder: Embedder, threshold: float,
) -> list["WikiItem"] | None:
    """Rank items by cosine similarity in one batched numpy matmul instead
    of a per-item Python loop. Returns None if embedding fails (caller
    falls back to keyword scoring)."""
    if not items:
        return []
    try:
        query_vec = np.asarray(embedder.embed_query(query, instruct=_WIKI_INSTRUCT), dtype=np.float32)
        item_vecs = np.stack([_get_item_embedding(item, embedder) for item in items])
        scores = reason.batch_cosine_scores(query_vec, item_vecs)
        order = np.argsort(-scores)
        return [items[i] for i in order if scores[i] >= threshold]
    except Exception:
        return None


@dataclass(frozen=True)
class WikiItem:
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


def discover_wiki_items(root: str | Path = REPO_ROOT) -> list[WikiItem]:
    base = Path(root)
    items: list[WikiItem] = []
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
            items.append(WikiItem(
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


def _score_item(item: WikiItem, terms: Iterable[str]) -> int:
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


def search_wiki(
    query: str,
    limit: int = 6,
    kinds: Iterable[str] | None = None,
    embedder: Embedder | None = None,
) -> list[WikiItem]:
    wanted = {kind for kind in kinds} if kinds is not None else None
    items = [item for item in discover_wiki_items() if wanted is None or item.kind in wanted]
    if not query.strip():
        return items[:limit]

    if embedder is not None:
        ranked = _semantic_rank(query, items, embedder, _WIKI_SEMANTIC_THRESHOLD)
        if ranked is not None:
            return ranked[:limit]

    query_terms = _terms(query)
    if not query_terms:
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


def _relevant_excerpt(item: "WikiItem", query: str, embedder: Embedder | None, remaining: int) -> str:
    """Chunk one item's body and return only the query-relevant excerpts,
    bounded by remaining chars. This is the RAG step: previously the whole
    item.text (up to remaining) was injected regardless of which part of a
    long doc actually mattered."""
    body = _FRONT_MATTER_RE.sub("", item.text, count=1).strip()
    pieces = reason.chunk_text(body, _WIKI_CHUNK_CHARS)
    if not pieces:
        return ""
    relevant = reason.select_relevant_chunks(
        query, pieces, embedder, top_k=_WIKI_CHUNKS_PER_ITEM,
        min_score=_WIKI_CHUNK_MIN_SCORE, instruct=_WIKI_INSTRUCT,
    )
    excerpt = "\n...\n".join(c for _score, c in relevant) if relevant else pieces[0]
    return excerpt[:remaining]


def wiki_knowledge_context_for(
    query: str, limit: int = 5, max_chars: int = 9000, embedder: Embedder | None = None,
) -> str:
    selected = search_wiki(query, limit=limit, embedder=embedder)
    if not selected:
        return "<wiki_knowledge_context>\nNo matching local knowledge found.\n</wiki_knowledge_context>"

    chunks: list[str] = []
    remaining = max_chars
    for item in selected:
        if remaining <= 0:
            break
        excerpt = _relevant_excerpt(item, query, embedder, remaining)
        if not excerpt:
            continue
        meta = item.as_dict()
        attrs = (
            f'id="{_attr(meta["id"])}" '
            f'kind="{_attr(meta["kind"])}" '
            f'path="{_attr(meta["path"])}" '
            f'title="{_attr(meta["title"])}" '
            f'updated_at="{_attr(meta["updated_at"])}"'
        )
        chunks.append(
            f"<wiki_knowledge_item {attrs}>\n"
            f"tags: {', '.join(meta['tags'])}\n\n{excerpt}\n</wiki_knowledge_item>"
        )
        remaining -= len(excerpt)

    if not chunks:
        return "<wiki_knowledge_context>\nNo matching local knowledge found.\n</wiki_knowledge_context>"
    return "<wiki_knowledge_context>\n" + "\n\n".join(chunks) + "\n</wiki_knowledge_context>"


def wiki_context_for(
    query: str, limit: int = 2, max_chars: int = 5000, embedder: Embedder | None = None,
) -> str:
    selected = search_wiki(query, limit=limit, kinds=("wiki",), embedder=embedder)
    if not selected:
        return "<wiki_context>\nNo operational wiki pages found.\n</wiki_context>"

    chunks: list[str] = []
    remaining = max_chars
    for item in selected:
        if remaining <= 0:
            break
        excerpt = _relevant_excerpt(item, query, embedder, remaining)
        if not excerpt:
            continue
        chunks.append(f"<wiki_page id=\"{_attr(item.path.stem)}\">\n{excerpt}\n</wiki_page>")
        remaining -= len(excerpt)

    if not chunks:
        return "<wiki_context>\nNo operational wiki pages found.\n</wiki_context>"
    return "<wiki_context>\n" + "\n\n".join(chunks) + "\n</wiki_context>"


def wiki_agentic_contexts_for(
    query: str,
    embedder: "Embedder | None" = None,
    wiki_limit: int = 1,
    knowledge_limit: int = 2,
    wiki_max_chars: int = 1500,
    knowledge_max_chars: int = 2500,
) -> tuple[str, str]:
    """Return BOTH the operational wiki page block and the wiki knowledge
    block from a SINGLE search_wiki call (same corpus, same query).

    Previously _fetch_agentic_only_context fired wiki_context_for
    (kind="wiki") and wiki_knowledge_context_for (all kinds) as two
    separate search_wiki calls against the same store with the same
    query — so the query was embedded twice per agentic turn (plus
    each block's own excerpt pass). One unified search + partition
    avoids the duplicate embedding while keeping the two distinct
    blocks (wiki_context / wiki_knowledge_context) the prompt expects.
    """
    selected = search_wiki(query, limit=wiki_limit + knowledge_limit, embedder=embedder)
    wiki_items = [it for it in selected if it.kind == "wiki"]
    knowledge_items = [it for it in selected if it.kind != "wiki"]

    wiki_chunks: list[str] = []
    remaining = wiki_max_chars
    for item in wiki_items[:wiki_limit]:
        if remaining <= 0:
            break
        excerpt = _relevant_excerpt(item, query, embedder, remaining)
        if not excerpt:
            continue
        wiki_chunks.append(f'<wiki_page id="{_attr(item.path.stem)}">\n{excerpt}\n</wiki_page>')
        remaining -= len(excerpt)

    knowledge_chunks: list[str] = []
    remaining = knowledge_max_chars
    for item in knowledge_items[:knowledge_limit]:
        if remaining <= 0:
            break
        excerpt = _relevant_excerpt(item, query, embedder, remaining)
        if not excerpt:
            continue
        meta = item.as_dict()
        attrs = (
            f'id="{_attr(meta["id"])}" '
            f'kind="{_attr(meta["kind"])}" '
            f'path="{_attr(meta["path"])}" '
            f'title="{_attr(meta["title"])}" '
            f'updated_at="{_attr(meta["updated_at"])}"'
        )
        knowledge_chunks.append(
            f"<wiki_knowledge_item {attrs}>\n"
            f"tags: {', '.join(meta['tags'])}\n\n{excerpt}\n</wiki_knowledge_item>"
        )
        remaining -= len(excerpt)

    wiki_block = (
        "<wiki_context>\n" + "\n\n".join(wiki_chunks) + "\n</wiki_context>"
        if wiki_chunks else
        "<wiki_context>\nNo operational wiki pages found.\n</wiki_context>"
    )
    knowledge_block = (
        "<wiki_knowledge_context>\n" + "\n\n".join(knowledge_chunks) + "\n</wiki_knowledge_context>"
        if knowledge_chunks else
        "<wiki_knowledge_context>\nNo matching local knowledge found.\n</wiki_knowledge_context>"
    )
    return wiki_block, knowledge_block

"""
core/memorize.py
Aiko's persistent memory — custom backend via sqlite-vec + fastembed + llama.cpp.
Abstracts all memory calls so think.py stays clean.

Memory lifecycle:
  - Every search() call increments access_count and updates last_accessed_at
    in the memories table, enabling Ebbinghaus-style exponential decay scoring.
  - dream() runs nightly (00:00) as a consolidation pass — no new vectors
    are written. It boosts salient memories, merges near-duplicates, then
    prunes decayed entries. Order matters: boost before prune so boosted
    memories aren't immediately swept.
  - cleanup() deletes memories below decay threshold, with grace period
    protection for newly created entries.
  - Decay logic lives in core/forget.py (pure math, no I/O).
  - Pinned memories (created via pin()) are permanently immune to decay
    cleanup and dream pruning. The pinned flag lives in the memories table.

Dream pass overview:
  1. Boost  — increment access_count on memories matching salience heuristics
              (keyword signals, high prior access, recency) so they survive decay.
  2. Merge  — cosine-similarity search per memory; near-duplicates above
              threshold are collapsed: keep the higher access_count copy,
              delete the redundant one to stay in sync.
              Pinned memories are never chosen as the loser in a merge.
  3. Prune  — standard cleanup() pass; runs after boost so newly protected
              memories aren't caught in the sweep.
              Pinned memories are skipped entirely.

Storage layout (single .db file):
  memories        — canonical record: id, user_id, memory, metadata
  memories_fts    — FTS5 virtual table for lexical search (BM25)
  memories_vec    — vec0 virtual table for KNN cosine search

Recall strategy — Reciprocal Rank Fusion (RRF):
  score = 1/(k + rank_knn) + 1/(k + rank_fts)
  k=60 (standard RRF constant — dampens outlier ranks)

  KNN catches semantic similarity ("I love cats" ↔ "I adore cats")
  FTS5 catches exact token matches ("Max", "birthday", proper nouns)
  RRF fuses both without weighting either arbitrarily.

Custom backend (replaces Qdrant + mem0):
  - _MemoryBackend handles LLM-based fact extraction, fastembed embeddings,
    and direct sqlite-vec upsert/search/delete/scroll.
  - Extraction prompt is tuned for small models: asks for a JSON array of
    atomic facts, strips <think> blocks for CoT models, skips trivial turns.
  - All schema fields (memory, user_id, created_at, access_count,
    last_accessed_at, pinned) are owned by this module — no hidden schema.

Dependencies:
  pip install sqlite-vec fastembed
"""
from dotenv import load_dotenv
load_dotenv()

import json
import os
from collections import OrderedDict
import threading
import re
import sqlite3
import struct
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import sqlite_vec
from openai import OpenAI
from fastembed import TextEmbedding

from core.forget import compute_weighted_score, should_cleanup, CLEANUP_THRESHOLD
from core.log import get_logger

log = get_logger(__name__)

# ── boot labels ───────────────────────────────────────────────────────────────

BOOT_LABELS = {
    'mem_sqlite':  'Opening sqlite-vec memory store...',
    'mem_cleanup': 'Running memory cleanup...',
    'mem_ready':   'Memory backend ready',
}

# ── constants ─────────────────────────────────────────────────────────────────

EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-base-en-v1.5")
EMBED_DIMS  = 768
RRF_K       = 60          # standard RRF constant — dampens outlier ranks
KNN_LIMIT   = 20          # candidates fetched before RRF re-rank
FTS_LIMIT   = 20          # candidates fetched before RRF re-rank
SEARCH_CACHE_SIZE = int(os.getenv("MEMORY_SEARCH_CACHE_SIZE", 128))
SEARCH_CACHE_TTL  = float(os.getenv("MEMORY_SEARCH_CACHE_TTL", 20.0))
LIFECYCLE_BATCH_SIZE = int(os.getenv("MEMORY_LIFECYCLE_BATCH_SIZE", 500))

USER_ID = os.getenv("USER_ID", "OppaAI")

# Cosine similarity threshold for near-duplicate detection during dream pass
# and dedup-on-write. 0.95 on write is tight (near-identical only).
# 0.92 on dream merge catches slightly more semantic duplicates.
DREAM_MERGE_THRESHOLD = float(os.getenv("DREAM_MERGE_THRESHOLD", 0.92))
WRITE_DEDUP_THRESHOLD = float(os.getenv("WRITE_DEDUP_THRESHOLD", 0.95))

# access_count boost applied to salient memories during dream pass.
DREAM_BOOST_AMOUNT = int(os.getenv("DREAM_BOOST_AMOUNT", 2))

# Salience keywords — memories containing these are boosted during dream pass.
# Matched on word boundaries (see _SALIENCE_RE) so "works" doesn't match
# "networks"/"fireworks" and "lives" doesn't match "olives".
_SALIENCE_KEYWORDS = frozenset([
    "name", "called", "likes", "loves", "hates", "dislikes", "always", "never",
    "important", "remember", "favourite", "favorite", "birthday", "works",
    "lives", "studying", "job", "afraid", "dream", "goal",
    "deadline", "due", "appointment", "event", "hackathon", "wallet",
    "lost", "passport", "license", "meeting", "interview", "project",
])

_SALIENCE_RE = re.compile(
    r'\b(?:' + '|'.join(re.escape(k) for k in _SALIENCE_KEYWORDS) + r')\b',
    re.IGNORECASE,
)

# Minimum conversation size (chars) worth sending to LLM for extraction.
_EXTRACT_MIN_CHARS = int(os.getenv("MEMORY_EXTRACT_MIN_CHARS", 80))

# Language that signals the LLM is guessing rather than stating a known fact.
# Facts containing these signals are dropped before persistence.
# Matched on word/phrase boundaries (see _HEDGE_RE) so e.g. "Oppa believes in
# ghosts" isn't missed and "Oppa said I believe in hard work" isn't wrongly
# dropped just because "believe" overlaps with a substring of "believes".
_HEDGE_SIGNALS = frozenset([
    "might", "probably", "seems", "i think", "perhaps", "maybe",
    "appears", "possibly", "could be", "not sure", "i believe",
    "it sounds like", "it seems like",
])

_HEDGE_RE = re.compile(
    r'\b(?:' + '|'.join(re.escape(h) for h in _HEDGE_SIGNALS) + r')\b',
    re.IGNORECASE,
)

# Extraction prompt — temperature 0.0, explicit only-stated-facts rule.
_EXTRACT_PROMPT = """\
Extract memorable facts about Oppa from this conversation.
Oppa is the user (he/him). You are Aiko, the assistant.

Rules:
- Only include facts Oppa stated explicitly. Never infer or assume.
- Write facts as short, direct statements in third person about Oppa.
- No facts about Aiko's behavior, feelings, or responses.
- No uncertain language: never use might, probably, seems, maybe, perhaps, appears.
- If nothing is worth remembering, return: []

Return ONLY a JSON array of short strings. No markdown. No explanation.

Good examples:
["Oppa's birthday is June 3", "Oppa is building a robot called GRACE", "Oppa joined the Hugging Face Hackathon", "Oppa lost his wallet", "Oppa has a deadline on Friday", "Oppa dislikes mushrooms"]

Bad examples (do not produce these):
["Oppa might like cats", "It seems Oppa is tired", "Aiko should remember this"]

Conversation:
{conversation}"""


def _sanitize_fts_query(query: str) -> str:
    """
    Strip characters that break FTS5 query parsing.
    FTS5 treats , " ( ) * ^ : - ' as syntax tokens — remove them all.
    """
    cleaned = re.sub(r'[^\w\s]', ' ', query)
    cleaned = ' '.join(cleaned.split())
    return cleaned or "*"


# ── schema ────────────────────────────────────────────────────────────────────

_DDL = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS memories (
    id               TEXT PRIMARY KEY,
    user_id          TEXT NOT NULL,
    memory           TEXT NOT NULL,
    created_at       TEXT NOT NULL,
    access_count     INTEGER NOT NULL DEFAULT 0,
    last_accessed_at TEXT NOT NULL DEFAULT 'never',
    pinned           INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_memories_user ON memories(user_id);

CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    memory,
    id UNINDEXED,
    content='memories',
    content_rowid='rowid'
);

CREATE VIRTUAL TABLE IF NOT EXISTS memories_vec USING vec0(
    id TEXT PRIMARY KEY,
    embedding FLOAT[{dims}]
);

CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, memory, id)
    VALUES (new.rowid, new.memory, new.id);
END;

CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, memory, id)
    VALUES ('delete', old.rowid, old.memory, old.id);
END;

CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE OF memory ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, memory, id)
    VALUES ('delete', old.rowid, old.memory, old.id);
    INSERT INTO memories_fts(rowid, memory, id)
    VALUES (new.rowid, new.memory, new.id);
END;
""".format(dims=EMBED_DIMS)


# ── sqlite payload helpers ────────────────────────────────────────────────────

def _sqlite_get_payload(conn: sqlite3.Connection, mem_id: str) -> dict:
    """Fetch the full memories row for a single id. Returns {} if not found."""
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT * FROM memories WHERE id = ?", (mem_id,)
    ).fetchone()
    return dict(row) if row else {}


def _sqlite_set_payload(
    conn: sqlite3.Connection,
    mem_id: str,
    payload: dict,
) -> None:
    """Update arbitrary column subset for a single memory row."""
    if not payload:
        return
    cols = ", ".join(f"{k} = ?" for k in payload)
    vals = list(payload.values()) + [mem_id]
    conn.execute(f"UPDATE memories SET {cols} WHERE id = ?", vals)
    conn.commit()


def _sqlite_batch_get_payloads(
    conn: sqlite3.Connection,
    mem_ids: list[str],
) -> dict:
    """
    Batch-fetch access_count + last_accessed_at in a single query.
    Returns {mem_id: (access_count, last_accessed_at)}.
    """
    if not mem_ids:
        return {}
    conn.row_factory = sqlite3.Row
    placeholders = ",".join("?" * len(mem_ids))
    rows = conn.execute(
        f"SELECT id, access_count, last_accessed_at FROM memories WHERE id IN ({placeholders})",
        mem_ids,
    ).fetchall()
    return {
        r["id"]: (r["access_count"] or 0, r["last_accessed_at"] or "never")
        for r in rows
    }


def _sqlite_get_vector(conn: sqlite3.Connection, mem_id: str) -> list[float]:
    """
    Retrieve the raw embedding for one memory from the vec0 table.
    Returns [] on miss or error.
    """
    row = conn.execute(
        "SELECT embedding FROM memories_vec WHERE id = ?", (mem_id,)
    ).fetchone()
    if row and row[0]:
        raw = row[0]
        n   = len(raw) // 4
        return list(struct.unpack(f"{n}f", raw))
    return []


def _sqlite_is_pinned(conn: sqlite3.Connection, mem_id: str) -> bool:
    """Return True if memories.pinned == 1 for this id. Defaults to False on error."""
    row = conn.execute(
        "SELECT pinned FROM memories WHERE id = ?", (mem_id,)
    ).fetchone()
    return bool(row and row[0])




def _sqlite_pinned_ids(conn: sqlite3.Connection, mem_ids: list[str]) -> set[str]:
    """Batch fetch pinned memory IDs from the canonical table."""
    ids = [str(mem_id) for mem_id in mem_ids if mem_id]
    if not ids:
        return set()
    placeholders = ",".join("?" * len(ids))
    rows = conn.execute(
        f"SELECT id FROM memories WHERE pinned = 1 AND id IN ({placeholders})",
        ids,
    ).fetchall()
    return {str(row["id"]) for row in rows}

def _sqlite_knn_search(
    conn: sqlite3.Connection,
    vector: list[float],
    user_id: str,
    limit: int,
    threshold: Optional[float] = None,
) -> list[sqlite3.Row]:
    """
    KNN cosine search against memories_vec, filtered by user_id.
    When threshold is supplied, only rows with dist <= (1 - threshold) are returned.
    """
    vec_blob = sqlite_vec.serialize_float32(vector)
    if threshold is not None:
        dist_ceil = 1.0 - threshold
        rows = conn.execute(
            """
            SELECT v.id, vec_distance_cosine(v.embedding, ?) AS dist
            FROM memories_vec v
            JOIN memories m ON m.id = v.id
            WHERE m.user_id = ?
              AND vec_distance_cosine(v.embedding, ?) <= ?
            ORDER BY dist ASC
            LIMIT ?
            """,
            (vec_blob, user_id, vec_blob, dist_ceil, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT v.id, vec_distance_cosine(v.embedding, ?) AS dist
            FROM memories_vec v
            JOIN memories m ON m.id = v.id
            WHERE m.user_id = ?
            ORDER BY dist ASC
            LIMIT ?
            """,
            (vec_blob, user_id, limit),
        ).fetchall()
    return rows


# ── memory backend ────────────────────────────────────────────────────────────

class _MemoryBackend:
    """
    sqlite-vec + FTS5 + RRF memory backend.

    Changes from original:
      - Extraction LLM runs at temperature=0.0 for deterministic fact output.
      - _extract_facts() filters hedging language via _HEDGE_RE before
        returning — uncertain facts are never persisted.
      - add() runs a dedup check per fact before insert: if a near-identical
        vector already exists (cosine >= WRITE_DEDUP_THRESHOLD), the fact is
        skipped rather than creating a redundant entry.
    """

    def __init__(
        self,
        db_path:         str,
        llm_base_url:    str,
        model:           str,
        fastembed_cache: Optional[str] = None,
    ) -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db_path  = db_path
        self._llm_base = llm_base_url.rstrip("/")
        self._model    = model
        self._client   = OpenAI(base_url=self._llm_base, api_key="not-needed")
        self._embedder = TextEmbedding(
            model_name=EMBED_MODEL,
            cache_dir=fastembed_cache,
        )
        self._conn = self._connect()
        self._apply_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout = 5000")  # wait up to 5s on lock contention
        conn.execute("PRAGMA journal_mode = WAL")
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        return conn

    def _apply_schema(self) -> None:
        self._conn.executescript(_DDL)
        self._conn.commit()

    # ── embedding ─────────────────────────────────────────────────────────────

    def _embed(self, text: str) -> list[float]:
        """Embed a single string with fastembed. Returns a plain float list."""
        return list(self._embedder.embed([text]))[0].tolist()

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple strings in a single batched fastembed call."""
        return [v.tolist() for v in self._embedder.embed(texts)]

    # ── extraction ────────────────────────────────────────────────────────────

    def _should_extract(self, messages: list[dict]) -> bool:
        """Return False for trivial turns below minimum char threshold."""
        total = sum(
            len(m.get("content") or "")
            for m in messages
            if m.get("role") in ("user", "assistant")
            and (m.get("content") or "").strip()
        )
        return total >= _EXTRACT_MIN_CHARS

    def _extract_facts(self, messages: list[dict]) -> list[str]:
        """
        Send conversation to the OpenAI-compatible local LLM and parse the returned JSON fact array.

        Changes from original:
          - temperature=0.0 for deterministic output — reduces confabulation.
          - Post-parse hedge filter: facts containing uncertain language
            (_HEDGE_RE, word-boundary matched) are dropped before returning.
          - Only user/assistant turns with real content are sent.
        """
        if not self._should_extract(messages):
            return []

        clean_messages = [
            m for m in messages
            if m.get("role") in ("user", "assistant")
            and (m.get("content") or "").strip()
        ]

        while clean_messages and clean_messages[0].get("role") != "user":
            clean_messages.pop(0)

        while clean_messages and clean_messages[-1].get("role") == "assistant":
            if any(m.get("role") == "user" for m in clean_messages[:-1]):
                break
            clean_messages.pop()

        if not clean_messages:
            return []

        total = sum(len(m.get("content") or "") for m in clean_messages)
        if total < _EXTRACT_MIN_CHARS:
            return []

        convo = "\n".join(
            f"{m['role'].upper()}: {m['content'].strip()}"
            for m in clean_messages
        )

        prompt = _EXTRACT_PROMPT.format(conversation=convo)

        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                stream=False,
                max_tokens=512,
                temperature=0.0,  # deterministic — reduces hallucinated facts
                timeout=45,
            )
            raw = (resp.choices[0].message.content or "").strip()
        except Exception as e:
            log.warning(f"Extraction LLM call failed: {e}")
            return []

        # strip CoT think blocks before JSON parsing
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

        # strip accidental markdown fences
        raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()

        try:
            facts = json.loads(raw)
            if isinstance(facts, list):
                facts = [f.strip() for f in facts if isinstance(f, str) and f.strip()]
            else:
                return []
        except json.JSONDecodeError:
            log.warning(f"Failed to parse extraction JSON: {raw[:200]!r}")
            return []

        # drop facts containing hedging/uncertain language (word-boundary match)
        clean_facts = []
        for fact in facts:
            if _HEDGE_RE.search(fact):
                log.debug(f"Dropped hedging fact: {fact!r}")
                continue
            clean_facts.append(fact)

        return clean_facts

    # ── write ─────────────────────────────────────────────────────────────────

    def add(self, messages: list[dict], user_id: str) -> list[str]:
        """
        Extract facts and persist each as a row in memories + memories_vec.

        Dedup-on-write: before inserting each fact, a KNN search checks for
        a near-identical vector already in the store (cosine >= WRITE_DEDUP_THRESHOLD).
        Duplicates are skipped to prevent redundant entries that compound into
        false confidence during recall.

        Embeddings for all extracted facts are computed in a single batched
        call rather than one-by-one.

        Returns list of new memory IDs. Empty list if nothing extracted.
        """
        facts = self._extract_facts(messages)
        if not facts:
            return []

        now = datetime.now(timezone.utc).isoformat()
        ids = []

        try:
            vectors = self._embed_batch(facts)
        except Exception as e:
            log.warning(f"Batch embedding failed, aborting write: {e}")
            return []

        for fact, vector in zip(facts, vectors):
            mem_id = str(uuid.uuid4())
            try:
                # dedup check — skip if near-identical vector already exists
                existing = _sqlite_knn_search(
                    self._conn, vector, user_id,
                    limit=1, threshold=WRITE_DEDUP_THRESHOLD,
                )
                if existing:
                    log.debug(f"Skipping near-duplicate fact: {fact!r}")
                    continue

                # insert canonical record — FTS5 trigger fires automatically
                self._conn.execute(
                    """
                    INSERT INTO memories
                        (id, user_id, memory, created_at, access_count, last_accessed_at, pinned)
                    VALUES (?, ?, ?, ?, 0, 'never', 0)
                    """,
                    (mem_id, user_id, fact, now),
                )

                # insert embedding into vec0 table
                self._conn.execute(
                    "INSERT INTO memories_vec(id, embedding) VALUES (?, ?)",
                    (mem_id, sqlite_vec.serialize_float32(vector)),
                )

                self._conn.commit()
                ids.append(mem_id)
            except Exception as e:
                log.warning(f"Failed to upsert fact {mem_id!r}: {e}")
                self._conn.rollback()

        return ids

    # ── read ──────────────────────────────────────────────────────────────────

    def search(self, query: str, user_id: str, limit: int = 5) -> list[dict]:
        """
        KNN + FTS5 → RRF fusion search.

        1. KNN: top-KNN_LIMIT by cosine distance from memories_vec
        2. FTS5: top-FTS_LIMIT by BM25 from memories_fts
        3. RRF: score = 1/(k+rank_knn) + 1/(k+rank_fts)
        4. Return top `limit` by RRF score as payload dicts
        """
        vector = self._embed(query)

        knn_rows = _sqlite_knn_search(self._conn, vector, user_id, KNN_LIMIT)
        rank_knn = {row["id"]: i + 1 for i, row in enumerate(knn_rows)}

        fts_rows = self._conn.execute(
            """
            SELECT f.id
            FROM memories_fts f
            JOIN memories m ON m.id = f.id
            WHERE memories_fts MATCH ?
            AND m.user_id = ?
            ORDER BY rank
            LIMIT ?
            """,
            (_sanitize_fts_query(query), user_id, FTS_LIMIT),
        ).fetchall()

        rank_fts = {row["id"]: i + 1 for i, row in enumerate(fts_rows)}

        all_ids = set(rank_knn) | set(rank_fts)
        if not all_ids:
            return []

        def rrf(mem_id: str) -> float:
            knn = rank_knn.get(mem_id, 0)
            fts = rank_fts.get(mem_id, 0)
            score = 0.0
            if knn:
                score += 1.0 / (RRF_K + knn)
            if fts:
                score += 1.0 / (RRF_K + fts)
            return score

        ranked = sorted(all_ids, key=rrf, reverse=True)[:limit]

        placeholders = ",".join("?" * len(ranked))
        rows = self._conn.execute(
            f"SELECT * FROM memories WHERE id IN ({placeholders})",
            ranked,
        ).fetchall()

        order       = {mid: i for i, mid in enumerate(ranked)}
        rows_sorted = sorted(rows, key=lambda r: order.get(r["id"], 999))

        return [dict(r) for r in rows_sorted]

    def iter_all(self, user_id: str, batch_size: int = LIFECYCLE_BATCH_SIZE):
        """Yield memory records for a user in rowid order without one giant list."""
        last_rowid = 0
        while True:
            rows = self._conn.execute(
                """
                SELECT rowid, id, memory, created_at
                FROM memories
                WHERE user_id = ? AND rowid > ?
                ORDER BY rowid ASC
                LIMIT ?
                """,
                (user_id, last_rowid, batch_size),
            ).fetchall()
            if not rows:
                break
            for row in rows:
                last_rowid = row["rowid"]
                yield {"id": row["id"], "memory": row["memory"], "created_at": row["created_at"]}

    def get_all(self, user_id: str) -> list[dict]:
        """
        Return memory records for a user.

        Projected to (id, memory, created_at) only — the three fields
        actually read off get_all() results anywhere in this codebase.
        access_count/last_accessed_at/pinned are intentionally NOT included
        here; callers that need fresh values for those fetch them via
        _sqlite_batch_get_payloads()/_sqlite_is_pinned() instead, since
        get_all() snapshots can be stale by the time those checks run.
        """
        return list(self.iter_all(user_id=user_id))

    def get_since(self, since: datetime, user_id: str = USER_ID) -> list[dict]:
        """Return memories created on or after `since`, newest first."""
        rows = self._conn.execute(
            """
            SELECT * FROM memories
            WHERE user_id = ? AND created_at >= ?
            ORDER BY created_at DESC
            """,
            (user_id, since.isoformat()),
        ).fetchall()
        return [dict(r) for r in rows]

    # ── delete ────────────────────────────────────────────────────────────────

    def delete(self, memory_id: str) -> None:
        """Delete a memory from all three tables."""
        self._conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        self._conn.execute("DELETE FROM memories_vec WHERE id = ?", (memory_id,))
        self._conn.commit()

    def delete_all(self, user_id: str) -> None:
        """Delete every memory for a user from all three tables."""
        ids = [
            r["id"] for r in self._conn.execute(
                "SELECT id FROM memories WHERE user_id = ?", (user_id,)
            ).fetchall()
        ]
        if not ids:
            return
        placeholders = ",".join("?" * len(ids))
        self._conn.execute(
            f"DELETE FROM memories WHERE id IN ({placeholders})", ids
        )
        self._conn.execute(
            f"DELETE FROM memories_vec WHERE id IN ({placeholders})", ids
        )
        self._conn.commit()


# ── memorize ──────────────────────────────────────────────────────────────────

class AikoMemorize:
    """
    Persistent memory with Ebbinghaus decay lifecycle and nightly dream() pass.

    Boot sequence (called by wakeup.py in order):
        memorize = AikoMemorize()
        memorize.cleanup()

    Access tracking:
        Every search() call updates the memories table (access_count,
        last_accessed_at) so the decay formula has fresh data.

    Pinned memories:
        Created via pin() — the pinned=1 column flag makes them
        immune to cleanup(), dream prune, and dream merge (as the loser).

    Dream pass (call nightly at 00:00):
        1. Boost salient memories' access_count so they survive decay.
        2. Merge near-duplicate vectors — keeps higher-access copy.
        3. Prune decayed memories via cleanup().
    """

    def __init__(self, silent: bool = False) -> None:
        db_path = os.getenv(
            "SQLITE_MEMORY_PATH",
            str(Path.home() / ".aiko" / "memory.db"),
        )

        if not silent:
            log.info("Opening sqlite-vec memory store...")

        self._mem = _MemoryBackend(
            db_path=db_path,
            llm_base_url=os.getenv("LLM_BASE_URL", "http://localhost:8080/v1"),
            model=os.getenv("EXTRACT_MODEL") or os.getenv("LLM_MODEL", "ministral"),
            fastembed_cache=os.getenv("FASTEMBED_CACHE_PATH"),
        )
        self._conn = self._mem._conn
        self._search_cache: OrderedDict[tuple[str, str, int], tuple[float, list[dict]]] = OrderedDict()
        self._search_cache_lock = threading.RLock()

        if not silent:
            log.info("Ready.")

    # ── write ─────────────────────────────────────────────────────────────────

    def add(self, messages: list[dict], user_id: str = USER_ID) -> bool:
        """
        Store a conversation turn into long-term memory.
        Returns True on success, False on failure.
        """
        try:
            t       = time.perf_counter()
            ids     = self._mem.add(messages, user_id=user_id)
            elapsed = time.perf_counter() - t
            if ids:
                self._clear_search_cache()
                log.info(f"Saved {len(ids)} memories in {elapsed:.2f}s")
            else:
                log.debug(f"No facts extracted ({elapsed:.2f}s) — nothing saved.")
            return True
        except Exception as e:
            log.error(f"Save failed: {e}")
            return False

    def pin(self, messages: list[dict], user_id: str = USER_ID) -> bool:
        """
        Store messages and immediately mark all resulting memories as pinned.
        Pinned memories are immune to cleanup, dream pruning, and merge losses.
        Returns True on success, False on any failure.
        """
        try:
            ids = self._mem.add(messages, user_id=user_id)

            if not ids:
                query = "\n".join(
                    (m.get("content") or "").strip()
                    for m in messages
                    if (m.get("content") or "").strip()
                )
                ids = [
                    str(m.get("id"))
                    for m in self.search(query, user_id=user_id, limit=3)
                    if m.get("id")
                ]

            if not ids:
                log.warning("pin(): add succeeded but no memory IDs were found to pin.")
                return False

            for mem_id in ids:
                _sqlite_set_payload(self._conn, mem_id, {"pinned": 1})

            self._clear_search_cache()
            log.info(f"Pinned {len(ids)} memories: {ids}")
            return True
        except Exception as e:
            log.error(f"Pin failed: {e}")
            return False

    # ── read ──────────────────────────────────────────────────────────────────

    def search(self, query: str, user_id: str = USER_ID, limit: int = 5) -> list[dict]:
        """
        Retrieve top-k memories relevant to the current query.
        Side-effect: increments access_count and updates last_accessed_at
        for all returned memories in a single batched UPDATE.
        """
        cache_key = (user_id, " ".join((query or "").lower().split()), int(limit))
        now_s = time.monotonic()

        with self._search_cache_lock:
            cached = self._search_cache.get(cache_key)
            if cached and now_s - cached[0] <= SEARCH_CACHE_TTL:
                self._search_cache.move_to_end(cache_key)
                results = [dict(r) for r in cached[1]]
                self._touch_memories(results)
                return results
            if cached:
                self._search_cache.pop(cache_key, None)

        results = self._mem.search(query, user_id=user_id, limit=limit)
        self._touch_memories(results)

        with self._search_cache_lock:
            self._search_cache[cache_key] = (now_s, [dict(r) for r in results])
            while len(self._search_cache) > SEARCH_CACHE_SIZE:
                self._search_cache.popitem(last=False)

        return results

    def _touch_memories(self, results: list[dict]) -> None:
        """Update decay access metadata for a search result set."""
        if not results:
            return
        now = datetime.now(timezone.utc).isoformat()
        mem_ids = [str(r.get("id", "")) for r in results if r.get("id")]
        if not mem_ids:
            return
        try:
            placeholders = ",".join("?" * len(mem_ids))
            self._conn.execute(
                f"""
                UPDATE memories
                SET access_count = MIN(access_count + 1, 255),
                    last_accessed_at = ?
                WHERE id IN ({placeholders})
                """,
                [now] + mem_ids,
            )
            self._conn.commit()
        except Exception as e:
            log.warning(f"Access tracking failed for {mem_ids}: {e}")

    def _clear_search_cache(self) -> None:
        with self._search_cache_lock:
            self._search_cache.clear()

    def format_for_context(self, memories: list[dict]) -> Optional[str]:
        """
        Format retrieved memories into a compact string for injection
        into the conversation context. Returns None if nothing to inject.
        """
        if not memories:
            return None

        now   = datetime.now(timezone.utc)
        lines = [
            "<memory_context>",
            "Background facts about Oppa. Use silently. Never quote or reference this block directly.",
            "",
        ]
        for m in memories:
            text       = m.get("memory") or m.get("text")
            if not text:
                continue
            created_at = m.get("created_at")
            if created_at:
                try:
                    ts    = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                    delta = now - ts
                    days  = delta.days
                    if days == 0:
                        age = "today"
                    elif days == 1:
                        age = "yesterday"
                    else:
                        age = f"{days} days ago"
                    lines.append(f"  - [{age}] {text}")
                except Exception:
                    lines.append(f"  - {text}")
            else:
                lines.append(f"  - {text}")

        lines.append("</memory_context>")
        return "\n".join(lines)

    # ── dream pass ────────────────────────────────────────────────────────────

    def dream(
        self,
        user_id:   str   = USER_ID,
        dry_run:   bool  = False,
        threshold: float = DREAM_MERGE_THRESHOLD,
    ) -> dict:
        """
        Nightly memory consolidation pass.

        Stages (in order):
          1. Boost  — salient memories get +DREAM_BOOST_AMOUNT access_count.
          2. Merge  — near-duplicate pairs (cosine >= threshold) are collapsed.
          3. Prune  — standard decay cleanup runs last.

        all_mems is fetched once and passed through to cleanup() so the
        prune stage doesn't re-scan the table from scratch.

        Returns dict: {boosted, merged, pruned, duration_s}
        """
        t_start = time.perf_counter()
        log.info(f"{'(dry-run) ' if dry_run else ''}Starting consolidation pass...")

        mem_ids: list[str] = []
        boosted = 0

        for batch in self._iter_memory_batches(user_id):
            batch_ids = [str(m.get("id", "")) for m in batch if m.get("id")]
            if not batch_ids:
                continue
            mem_ids.extend(batch_ids)
            payload_map = self._batch_get_payloads(batch_ids)
            pinned_ids = _sqlite_pinned_ids(self._conn, batch_ids)
            boosted += self._dream_boost(batch, payload_map, pinned_ids=pinned_ids, dry_run=dry_run)

        if not mem_ids:
            log.info("No memories found — nothing to do.")
            return {"boosted": 0, "merged": 0, "pruned": 0, "duration_s": 0.0}

        pinned_ids = _sqlite_pinned_ids(self._conn, mem_ids)
        merged = self._dream_merge(mem_ids, user_id=user_id, threshold=threshold, pinned_ids=pinned_ids, dry_run=dry_run)
        prune_result = self.cleanup(user_id=user_id, dry_run=dry_run)
        pruned = prune_result.get("deleted", 0)

        duration = round(time.perf_counter() - t_start, 2)
        log.info(
            f"{'(dry-run) ' if dry_run else ''}"
            f"Done — boosted={boosted}, merged={merged}, pruned={pruned}, "
            f"duration={duration}s"
        )
        return {"boosted": boosted, "merged": merged, "pruned": pruned, "duration_s": duration}

    def _dream_boost(
        self,
        all_mems:    list[dict],
        payload_map: dict,
        pinned_ids:  set[str] | None = None,
        dry_run:     bool = False,
    ) -> int:
        """
        Increment access_count on memories matching salience heuristics.
        Pinned memories pass through unchanged.
        Returns count of memories boosted.
        """
        now     = datetime.now(timezone.utc)
        boost_ids: list[str] = []
        pinned_ids = pinned_ids or set()

        for m in all_mems:
            mem_id = str(m.get("id", ""))
            if not mem_id:
                continue
            if mem_id in pinned_ids:
                continue

            text     = m.get("memory") or ""
            ac, _la  = payload_map.get(mem_id, (0, "never"))

            is_recent  = False
            created_at = m.get("created_at", "")
            if created_at:
                try:
                    ts        = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                    is_recent = (now - ts).days <= 7
                except Exception:
                    pass

            is_salient = (
                bool(_SALIENCE_RE.search(text))
                or ac >= 3
                or is_recent
            )

            if not is_salient:
                continue

            boost_ids.append(mem_id)

        if boost_ids and not dry_run:
            try:
                placeholders = ",".join("?" * len(boost_ids))
                self._conn.execute(
                    f"""
                    UPDATE memories
                    SET access_count = MIN(access_count + ?, 255)
                    WHERE id IN ({placeholders})
                    """,
                    [DREAM_BOOST_AMOUNT] + boost_ids,
                )
                self._conn.commit()
            except Exception as e:
                log.warning(f"Batch boost failed for {len(boost_ids)} memories: {e}")
                self._conn.rollback()
                return 0

        boosted = len(boost_ids)
        if boosted:
            log.info(f"{'(dry-run) ' if dry_run else ''}Boosted {boosted} memories.")
        return boosted

    def _dream_merge(
        self,
        mem_ids:   list[str],
        user_id:   str,
        threshold: float = DREAM_MERGE_THRESHOLD,
        pinned_ids: set[str] | None = None,
        dry_run:   bool  = False,
    ) -> int:
        """
        Detect and collapse near-duplicate memory vectors.
        Pinned memories are never chosen as the loser.
        Returns count of memories deleted as duplicates.
        """
        deleted_ids: set[str] = set()
        pinned_ids = pinned_ids or set()
        merged = 0

        for mem_id in mem_ids:
            if mem_id in deleted_ids:
                continue
            if mem_id in pinned_ids:
                continue

            vector = _sqlite_get_vector(self._conn, mem_id)
            if not vector:
                continue

            try:
                neighbor_rows = _sqlite_knn_search(
                    self._conn, vector, user_id, limit=4, threshold=threshold
                )
            except Exception as e:
                log.warning(f"Similarity search failed for {mem_id}: {e}")
                continue

            for row in neighbor_rows:
                neighbor_id = row["id"]
                if neighbor_id == mem_id:
                    continue
                if neighbor_id in deleted_ids:
                    continue

                similarity = 1.0 - row["dist"]
                n_merged = self._resolve_duplicate(
                    mem_id, neighbor_id, similarity, pinned_ids=pinned_ids, dry_run=dry_run
                )
                if n_merged:
                    deleted_ids.add(neighbor_id)
                    merged += 1

        if merged:
            log.info(f"{'(dry-run) ' if dry_run else ''}Merged {merged} duplicate memories.")
        return merged

    def _resolve_duplicate(
        self,
        id_a:    str,
        id_b:    str,
        score:   float,
        pinned_ids: set[str] | None = None,
        dry_run: bool = False,
    ) -> bool:
        """
        Compare two near-duplicate memories and delete the weaker one.
        Pinned memories are never deleted. Tie goes to id_a (query origin).
        Returns True if a deletion occurred.
        """
        pinned_ids = pinned_ids or set()
        if id_a in pinned_ids or id_b in pinned_ids:
            log.info(f"Skipping merge: one or both of ({id_a}, {id_b}) is pinned.")
            return False

        payload_map = self._batch_get_payloads([id_a, id_b])
        ac_a, _     = payload_map.get(id_a, (0, "never"))
        ac_b, _     = payload_map.get(id_b, (0, "never"))
        row_map = {
            row["id"]: row["created_at"]
            for row in self._conn.execute(
                "SELECT id, created_at FROM memories WHERE id IN (?, ?)", (id_a, id_b)
            ).fetchall()
        }
        if ac_a == ac_b:
            loser = id_b if row_map.get(id_a, "") >= row_map.get(id_b, "") else id_a
        else:
            loser = id_b if ac_a > ac_b else id_a

        if dry_run:
            log.info(
                f"(dry-run) Would merge: score={score:.3f} "
                f"ac_a={ac_a} ac_b={ac_b} → delete {loser}"
            )
            return True

        try:
            self._mem.delete(memory_id=loser)
            log.info(
                f"Merged duplicate (score={score:.3f}, "
                f"ac_a={ac_a}, ac_b={ac_b}) → deleted {loser}"
            )
            return True
        except Exception as e:
            log.warning(f"Merge delete failed for {loser}: {e}")
            return False

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def cleanup(
        self,
        user_id:   str   = USER_ID,
        threshold: float = CLEANUP_THRESHOLD,
        dry_run:   bool  = False,
        _all_mems: Optional[list[dict]] = None,
        _pinned_ids: Optional[set[str]] = None,
    ) -> dict:
        """
        Prune decayed memories below threshold score.
        Grace period (14 days) protects newly created memories.
        Pinned memories are unconditionally kept.

        _all_mems: internal — when called from dream(), the already-fetched
        memory list is passed through here to avoid a redundant get_all() scan.

        Returns dict: {deleted, kept, failed, candidates (dry_run only)}.
        """
        source = [_all_mems] if _all_mems is not None else self._iter_memory_batches(user_id)

        kept = 0
        deleted: list[str] = []
        failed: list[dict] = []
        dry_candidates: list[dict] = []
        saw_any = False

        for batch in source:
            if not batch:
                continue
            saw_any = True
            batch_kept, candidates = self._cleanup_candidates(
                batch,
                _pinned_ids=_pinned_ids,
            )
            kept += batch_kept

            if dry_run:
                dry_candidates.extend(candidates)
                continue

            for c in candidates:
                try:
                    self._mem.delete(memory_id=c["id"])
                    deleted.append(c["id"])
                except Exception as e:
                    failed.append({"id": c["id"], "error": str(e)})

        if not saw_any:
            return {"deleted": 0, "kept": 0, "failed": 0}

        if dry_run:
            dry_candidates.sort(key=lambda x: x["weighted_score"])
            log.info(f"Dry run: {len(dry_candidates)} candidates for deletion, {kept} kept.")
            return {"deleted": 0, "kept": kept, "failed": 0, "candidates": dry_candidates}

        if deleted:
            self._clear_search_cache()
            self.optimize()

        log.info(f"Cleanup: deleted={len(deleted)}, kept={kept}, failed={len(failed)}")
        return {"deleted": len(deleted), "kept": kept, "failed": len(failed)}

    def _iter_memory_batches(self, user_id: str, batch_size: int = LIFECYCLE_BATCH_SIZE):
        """Yield lifecycle scan batches without retaining the full table."""
        batch: list[dict] = []
        for mem in self._mem.iter_all(user_id=user_id, batch_size=batch_size):
            batch.append(mem)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    def _cleanup_candidates(
        self,
        all_mems: list[dict],
        _pinned_ids: Optional[set[str]] = None,
    ) -> tuple[int, list[dict]]:
        mem_ids     = [str(m.get("id", "")) for m in all_mems if m.get("id")]
        payload_map = self._batch_get_payloads(mem_ids)
        pinned_ids  = _pinned_ids if _pinned_ids is not None else _sqlite_pinned_ids(self._conn, mem_ids)

        candidates = []
        kept       = 0

        for m in all_mems:
            mem_id     = str(m.get("id", ""))
            ac, la     = payload_map.get(mem_id, (0, "never"))
            created_at = m.get("created_at", "")

            if mem_id in pinned_ids:
                kept += 1
                continue

            if should_cleanup(ac, la, created_at):
                w = compute_weighted_score(ac, la)
                candidates.append({
                    "id":               mem_id,
                    "memory":           m.get("memory", "")[:120],
                    "access_count":     ac,
                    "weighted_score":   round(w, 4),
                    "last_accessed_at": la,
                })
            else:
                kept += 1

        candidates.sort(key=lambda x: x["weighted_score"])
        return kept, candidates

    def optimize(self) -> None:
        """Run SQLite's lightweight planner/index maintenance hook."""
        try:
            self._conn.execute("PRAGMA optimize")
            self._conn.commit()
        except Exception as e:
            log.debug(f"SQLite optimize skipped: {e}")

    # ── debug ─────────────────────────────────────────────────────────────────

    def get_all(self, user_id: str = USER_ID) -> list[dict]:
        """Return all stored memories for a user."""
        return self._mem.get_all(user_id=user_id)

    def get_since(self, since: datetime, user_id: str = USER_ID) -> list[dict]:
        """Return memories created on or after `since`, newest first."""
        return self._mem.get_since(since, user_id=user_id)

    def clear(self, user_id: str = USER_ID) -> None:
        """Wipe all memories for a user. Use carefully."""
        self._mem.delete_all(user_id=user_id)
        self._clear_search_cache()
        log.info(f"Cleared all memories for user '{user_id}'.")

    # ── internal ──────────────────────────────────────────────────────────────

    def _batch_get_payloads(self, mem_ids: list[str]) -> dict:
        """Batch retrieve access_count + last_accessed_at in a single query."""
        return _sqlite_batch_get_payloads(self._conn, mem_ids)

    def _get_vector(self, mem_id: str) -> list[float]:
        """Retrieve the raw embedding vector for a single memory."""
        return _sqlite_get_vector(self._conn, mem_id)

    def _is_pinned(self, mem_id: str) -> bool:
        """Return True if memories.pinned == 1 for this id."""
        return _sqlite_is_pinned(self._conn, mem_id)

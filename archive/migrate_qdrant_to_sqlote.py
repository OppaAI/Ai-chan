"""
migrate_qdrant_to_sqlite.py
One-shot migration: Qdrant aiko_memory → sqlite-vec memory.db

Bypasses LLM extraction — inserts pre-extracted facts directly,
preserving access_count, last_accessed_at, pinned, and created_at.

Usage:
    cd ~/Aiko-chan
    python migrate_qdrant_to_sqlite.py

    # dry run first (no writes):
    python migrate_qdrant_to_sqlite.py --dry-run

    # if Qdrant is on a non-default URL:
    QDRANT_URL=http://localhost:6333 python migrate_qdrant_to_sqlite.py
"""
from dotenv import load_dotenv
load_dotenv()

import argparse
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

import sqlite_vec
from fastembed import TextEmbedding
from qdrant_client import QdrantClient

# ── config ────────────────────────────────────────────────────────────────────

QDRANT_URL       = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME  = "aiko_memory"
EMBED_MODEL      = "BAAI/bge-base-en-v1.5"
EMBED_DIMS       = 768
USER_ID          = os.getenv("USER_ID", "OppaAI")
DB_PATH          = os.getenv(
    "SQLITE_MEMORY_PATH",
    str(Path.home() / ".aiko" / "memory.db"),
)
FASTEMBED_CACHE  = os.getenv("FASTEMBED_CACHE_PATH")
SCROLL_BATCH     = 100   # points per Qdrant scroll page

# ── schema (mirrors memorize.py _DDL exactly) ─────────────────────────────────

_DDL = f"""
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
    embedding FLOAT[{EMBED_DIMS}]
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
"""

# ── helpers ───────────────────────────────────────────────────────────────────

def open_db(db_path: str) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    sqlite_vec.load(conn)
    conn.executescript(_DDL)
    conn.commit()
    return conn


def already_exists(conn: sqlite3.Connection, memory_text: str, user_id: str) -> bool:
    """Skip exact duplicates already in the sqlite store."""
    row = conn.execute(
        "SELECT id FROM memories WHERE memory = ? AND user_id = ?",
        (memory_text, user_id),
    ).fetchone()
    return row is not None


def extract_memory_text(payload: dict) -> str:
    """
    Extract the memory text from a Qdrant payload.

    mem0 stores the actual text under the 'data' key in Qdrant.
    The 'memory' key appears in mem0's Python API response but not
    necessarily in the raw Qdrant payload — so we check 'data' first.
    Falls back through common alternatives for safety.
    """
    return (
        payload.get("data")       # mem0's actual Qdrant payload key
        or payload.get("memory")  # mem0 API response key (may appear in older versions)
        or payload.get("text")    # generic fallback
        or payload.get("content") # generic fallback
        or ""
    ).strip()


def scroll_all_qdrant(client: QdrantClient) -> list[dict]:
    """
    Page through the entire Qdrant collection and return all points
    as plain dicts with keys: id, memory, access_count,
    last_accessed_at, created_at, pinned.
    """
    points  = []
    offset  = None
    page    = 0

    while True:
        result, next_offset = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=None,
            limit=SCROLL_BATCH,
            offset=offset,
            with_payload=True,
            with_vectors=False,   # re-embed from scratch for clean vectors
        )

        if not result:
            break

        for p in result:
            payload = p.payload or {}

            memory_text = extract_memory_text(payload)

            if not memory_text:
                # dump payload keys to help diagnose unexpected schemas
                keys = list(payload.keys()) if payload else []
                print(f"  [skip] Point {p.id} has no memory text — payload keys: {keys}")
                continue

            # created_at: mem0 stores it in the payload under 'created_at'
            # fall back to hash or now if missing
            created_at = (
                payload.get("created_at")
                or payload.get("timestamp")
                or datetime.now(timezone.utc).isoformat()
            )

            points.append({
                "id":               str(p.id),
                "memory":           memory_text,
                "access_count":     int(payload.get("access_count", 0)),
                "last_accessed_at": payload.get("last_accessed_at", "never") or "never",
                "created_at":       created_at,
                "pinned":           int(bool(payload.get("pinned", False))),
            })

        page += 1
        print(f"  Scrolled page {page}: {len(result)} points (total so far: {len(points)})")

        if next_offset is None:
            break
        offset = next_offset

    return points


def migrate(dry_run: bool = False, debug: bool = False) -> None:
    print(f"\n{'='*60}")
    print(f"  Aiko Memory Migration: Qdrant → sqlite-vec")
    print(f"  Collection : {COLLECTION_NAME}")
    print(f"  Target DB  : {DB_PATH}")
    print(f"  Dry run    : {dry_run}")
    print(f"  Debug      : {debug}")
    print(f"{'='*60}\n")

    # ── connect to Qdrant ─────────────────────────────────────────────────────
    print(f"Connecting to Qdrant at {QDRANT_URL}...")
    try:
        client = QdrantClient(url=QDRANT_URL)
        info   = client.get_collection(COLLECTION_NAME)
        total  = info.points_count
        print(f"  Collection found — {total} points.\n")
    except Exception as e:
        print(f"  ERROR: Could not connect to Qdrant: {e}")
        print("  Is Qdrant running? Try: docker start <qdrant_container>")
        return

    # ── debug: inspect raw payload of first 3 points ──────────────────────────
    if debug:
        print("DEBUG — raw Qdrant payloads (first 3 points):")
        sample, _ = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=3,
            with_payload=True,
            with_vectors=False,
        )
        for p in sample:
            print(f"\n  --- Point {p.id} ---")
            print(f"  Payload keys : {list(p.payload.keys()) if p.payload else 'EMPTY'}")
            print(f"  Payload      : {p.payload}")
        print()

    # ── scroll all points ─────────────────────────────────────────────────────
    print("Scrolling all points from Qdrant...")
    points = scroll_all_qdrant(client)
    print(f"\n  Fetched {len(points)} valid points.\n")

    if not points:
        print("Nothing to migrate.")
        print("Tip: run with --debug to inspect raw Qdrant payloads and find the right key.")
        return

    if dry_run:
        print("DRY RUN — no writes. Sample of what would be migrated:\n")
        for p in points[:5]:
            pinned_tag = " [PINNED]" if p["pinned"] else ""
            print(f"  [{p['access_count']:>3} hits]{pinned_tag} {p['memory'][:80]}")
        if len(points) > 5:
            print(f"  ... and {len(points) - 5} more.")
        print("\nRe-run without --dry-run to perform migration.")
        return

    # ── load embedder ─────────────────────────────────────────────────────────
    print(f"Loading fastembed model ({EMBED_MODEL})...")
    embedder = TextEmbedding(model_name=EMBED_MODEL, cache_dir=FASTEMBED_CACHE)
    print("  Embedder ready.\n")

    # ── open sqlite-vec store ─────────────────────────────────────────────────
    print(f"Opening sqlite-vec store at {DB_PATH}...")
    conn = open_db(DB_PATH)
    print("  Schema applied.\n")

    # ── migrate ───────────────────────────────────────────────────────────────
    inserted  = 0
    skipped   = 0
    failed    = 0
    now_iso   = datetime.now(timezone.utc).isoformat()

    print(f"Migrating {len(points)} memories...\n")

    for i, p in enumerate(points, 1):
        memory_text = p["memory"]

        # skip exact duplicates already in sqlite
        if already_exists(conn, memory_text, USER_ID):
            print(f"  [{i:>3}] SKIP (duplicate): {memory_text[:60]}")
            skipped += 1
            continue

        try:
            # re-embed from scratch — clean float32 vectors
            vector = list(embedder.embed([memory_text]))[0].tolist()
            mem_id = str(uuid.uuid4())

            conn.execute(
                """
                INSERT INTO memories
                    (id, user_id, memory, created_at, access_count, last_accessed_at, pinned)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    mem_id,
                    USER_ID,
                    memory_text,
                    p["created_at"] or now_iso,
                    p["access_count"],
                    p["last_accessed_at"],
                    p["pinned"],
                ),
            )

            conn.execute(
                "INSERT INTO memories_vec(id, embedding) VALUES (?, ?)",
                (mem_id, sqlite_vec.serialize_float32(vector)),
            )

            conn.commit()

            pinned_tag = " [PINNED]" if p["pinned"] else ""
            print(f"  [{i:>3}] OK{pinned_tag}: {memory_text[:60]}")
            inserted += 1

        except Exception as e:
            conn.rollback()
            print(f"  [{i:>3}] FAIL: {memory_text[:60]}")
            print(f"         Error: {e}")
            failed += 1

    # ── summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Migration complete.")
    print(f"  Inserted : {inserted}")
    print(f"  Skipped  : {skipped}  (duplicates already in sqlite)")
    print(f"  Failed   : {failed}")
    print(f"{'='*60}\n")

    if failed:
        print("  Some entries failed — check logs above.")
    else:
        print("  All memories migrated successfully. Aiko remembers. 🌸")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate Aiko memories from Qdrant to sqlite-vec")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing")
    parser.add_argument("--debug",   action="store_true", help="Dump raw Qdrant payloads before migrating")
    args = parser.parse_args()
    migrate(dry_run=args.dry_run, debug=args.debug)

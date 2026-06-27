#!/usr/bin/env python3
"""
migrate_embeddings.py
─────────────────────
Rebuild Aiko's memories_vec table with a new embedding model.

Run this once when changing EMBED_MODEL or EMBED_DIMS.
The memories and memories_fts tables are untouched — only the
vec0 table is dropped and rebuilt with fresh vectors.

NOTE on instruction prefixes: stored memories are DOCUMENTS, not queries.
For decoder-only instruct-style embedding models (e.g. harrier-oss-v1),
the model card specifies query-side text needs an "Instruct: ...\\nQuery: "
prefix while document-side text gets NO prefix. This script intentionally
embeds raw `memory` text with no prefix — correct for documents. Only the
live search query embedded at runtime in memorize.py needs the prefix.

Usage:
    python3 migrate_embeddings.py [--db PATH] [--model MODEL_ID] [--dims N] [--dry-run]

Defaults pulled from environment (.env if python-dotenv is installed):
    DB:    $SQLITE_MEMORY_PATH  (~/.aiko/memory.db)
    MODEL: $EMBED_MODEL         (ferrisS/harrier-oss-v1-270m-fastembed in .env.example)
    DIMS:  $EMBED_DIMS          (640 for harrier-oss-v1-270m)
"""

import argparse
import os
import sqlite3
import struct
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── optional dotenv ───────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── optional progress bar ─────────────────────────────────────────────────────
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(it, **kw):  # noqa: F811
        total = kw.get("total", "?")
        print(f"  (install tqdm for a progress bar — processing {total} memories)")
        return it

# ── defaults ──────────────────────────────────────────────────────────────────
DEFAULT_DB    = os.getenv("SQLITE_MEMORY_PATH", str(Path.home() / ".aiko" / "memory.db"))
DEFAULT_MODEL = os.getenv("EMBED_MODEL", "ferrisS/harrier-oss-v1-270m-fastembed")
DEFAULT_DIMS  = int(os.getenv("EMBED_DIMS", "640"))
BATCH_SIZE    = int(os.getenv("EMBED_BATCH_SIZE", "64"))   # memories per embedding batch — tune down if OOM on Jetson


def parse_args():
    p = argparse.ArgumentParser(description="Rebuild Aiko memories_vec with a new embedding model.")
    p.add_argument("--db",      default=DEFAULT_DB,    help=f"Path to memory.db (default: {DEFAULT_DB})")
    p.add_argument("--model",   default=DEFAULT_MODEL, help=f"Harrier ONNX model ID (default: {DEFAULT_MODEL})")
    p.add_argument("--dims",    default=DEFAULT_DIMS,  type=int, help=f"Embedding dimensions (default: {DEFAULT_DIMS})")
    p.add_argument("--dry-run", action="store_true",   help="Load model and count memories, but don't write anything")
    p.add_argument("--batch",   default=BATCH_SIZE,    type=int, help=f"Embedding batch size (default: {BATCH_SIZE})")
    return p.parse_args()


def connect(db_path: str) -> sqlite3.Connection:
    import sqlite_vec
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA busy_timeout = 10000")
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn


def fetch_all_memories(conn: sqlite3.Connection) -> list[dict]:
    """Pull id + memory text from canonical table — no vec table needed."""
    rows = conn.execute("SELECT id, memory FROM memories ORDER BY rowid ASC").fetchall()
    return [{"id": row["id"], "memory": row["memory"]} for row in rows]


def drop_vec_table(conn: sqlite3.Connection) -> None:
    conn.execute("DROP TABLE IF EXISTS memories_vec")
    conn.commit()
    print("  ✓ Dropped memories_vec")


def create_vec_table(conn: sqlite3.Connection, dims: int) -> None:
    conn.execute(f"""
        CREATE VIRTUAL TABLE memories_vec USING vec0(
            id TEXT PRIMARY KEY,
            embedding FLOAT[{dims}]
        )
    """)
    conn.commit()
    print(f"  ✓ Created memories_vec (dims={dims})")


def serialize(vector: list[float]) -> bytes:
    return struct.pack(f"{len(vector)}f", *vector)


def insert_vectors(
    conn:     sqlite3.Connection,
    memories: list[dict],
    embedder,
    batch_size: int,
) -> tuple[int, int]:
    """
    Embed and insert in batches. Memory text is embedded RAW — no instruction
    prefix — since stored memories are documents, not queries.
    Returns (inserted, failed).
    """
    import sqlite_vec

    inserted = 0
    failed   = 0
    total    = len(memories)

    for start in tqdm(range(0, total, batch_size), total=(total + batch_size - 1) // batch_size, desc="  Embedding"):
        batch = memories[start : start + batch_size]
        texts = [m["memory"] for m in batch]  # document-side: no prefix

        try:
            vectors = [v.tolist() for v in embedder.embed(texts)]
        except Exception as e:
            print(f"\n  ✗ Embedding batch {start}–{start+len(batch)} failed: {e}", file=sys.stderr)
            failed += len(batch)
            continue

        for mem, vector in zip(batch, vectors):
            try:
                conn.execute(
                    "INSERT INTO memories_vec(id, embedding) VALUES (?, ?)",
                    (mem["id"], sqlite_vec.serialize_float32(vector)),
                )
                inserted += 1
            except Exception as e:
                print(f"\n  ✗ Insert failed for {mem['id']!r}: {e}", file=sys.stderr)
                failed += 1

        conn.commit()

    return inserted, failed


def verify_counts(conn: sqlite3.Connection) -> tuple[int, int]:
    """Return (memories_count, vec_count)."""
    mem_n = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    vec_n = conn.execute("SELECT COUNT(*) FROM memories_vec").fetchone()[0]
    return mem_n, vec_n


def main():
    args = parse_args()

    db_path = args.db
    model   = args.model
    dims    = args.dims

    print(f"\nAiko Embedding Migration")
    print(f"{'─'*48}")
    print(f"  DB:      {db_path}")
    print(f"  Model:   {model}")
    print(f"  Dims:    {dims}")
    print(f"  Dry run: {args.dry_run}")
    print()

    # ── sanity checks ─────────────────────────────────────────────────────────
    if not Path(db_path).exists():
        sys.exit(f"✗ Database not found: {db_path}")

    try:
        import sqlite_vec  # noqa: F401
    except ImportError:
        sys.exit("✗ sqlite_vec not installed — pip install sqlite-vec")

    # ── load model ────────────────────────────────────────────────────────────
    print("Loading embedding model (may download on first run)...")
    t0 = time.perf_counter()
    try:
        from core.embed import HarrierEmbedder
        embedder = HarrierEmbedder(model_id=model, dims=dims, batch_size=args.batch)
        # warm up — triggers ONNX session load now, not mid-migration
        list(embedder.embed(["warmup"]))
    except Exception as e:
        sys.exit(f"✗ Failed to load model: {e}")
    print(f"  ✓ Model loaded in {time.perf_counter()-t0:.1f}s\n")

    # ── connect ───────────────────────────────────────────────────────────────
    conn = connect(db_path)
    memories = fetch_all_memories(conn)
    print(f"  Found {len(memories)} memories to re-embed\n")

    if args.dry_run:
        print("Dry run — no changes written.")
        return

    if not memories:
        print("Nothing to migrate.")
        return

    # ── backup reminder ───────────────────────────────────────────────────────
    print("⚠  This will DROP and rebuild memories_vec.")
    print(f"   Backup recommended: cp {db_path} {db_path}.bak")
    ans = input("   Continue? [y/N] ").strip().lower()
    if ans not in ("y", "yes"):
        print("Aborted.")
        return
    print()

    # ── migrate ───────────────────────────────────────────────────────────────
    t1 = time.perf_counter()

    print("Step 1/3 — Dropping old vec table...")
    drop_vec_table(conn)

    print("Step 2/3 — Creating new vec table...")
    create_vec_table(conn, dims)

    print("Step 3/3 — Embedding and inserting...")
    inserted, failed = insert_vectors(conn, memories, embedder, args.batch)

    elapsed = time.perf_counter() - t1
    print(f"\n{'─'*48}")
    print(f"  ✓ Inserted: {inserted}")
    if failed:
        print(f"  ✗ Failed:   {failed}  ← check logs above")
    print(f"  ✓ Duration: {elapsed:.1f}s")

    # ── verify ────────────────────────────────────────────────────────────────
    mem_n, vec_n = verify_counts(conn)
    print(f"\nVerification:")
    print(f"  memories     : {mem_n}")
    print(f"  memories_vec : {vec_n}")
    if mem_n == vec_n:
        print("  ✓ Counts match — migration complete.\n")
    else:
        print(f"  ✗ Count mismatch ({mem_n - vec_n} missing vectors) — check failed rows above.\n")

    conn.close()


if __name__ == "__main__":
    main()

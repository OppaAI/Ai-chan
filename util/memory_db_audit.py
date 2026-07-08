#!/usr/bin/env python3
"""
check_memory_integrity.py
Audits Aiko's memory.db for orphaned and inconsistent records.

Checks:
  1. memories rows with no matching memories_vec entry (orphaned canonical)
  2. memories_vec rows with no matching memories entry (orphaned vector)
  3. memories rows with no matching memories_fts entry (orphaned FTS)
  4. memories_vec entries with a NULL / zero-length embedding blob
  5. memories rows with NULL or empty 'memory' text
  6. memories rows with NULL or missing created_at
  7. Duplicate UUIDs within any single table (sanity)
  8. memories_fts content/rowid sync check (fts rowid → memories rowid mismatch)

Usage:
  python check_memory_integrity.py [--db PATH] [--fix]

  --db    Path to memory.db  (default: ~/.aiko/<user_id>/memory/memory.db)
  --fix   Auto-delete confirmed orphans (backs up db first)
""

import argparse
import shutil
import sqlite3
import struct
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    import sqlite_vec
    HAS_VEC = True
except ImportError:
    HAS_VEC = False
    print("[WARN] sqlite_vec not importable — vec0 extension won't load. "
          "Vector checks will be skipped.\n")

# ── helpers ───────────────────────────────────────────────────────────────────

RESET  = "\033[0m"
RED    = "\033[91m"
YELLOW = "\033[93m"
GREEN  = "\033[92m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"

def ok(msg):    print(f"  {GREEN}✓{RESET} {msg}")
def warn(msg):  print(f"  {YELLOW}⚠{RESET} {msg}")
def err(msg):   print(f"  {RED}✗{RESET} {msg}")
def info(msg):  print(f"  {CYAN}·{RESET} {msg}")
def header(msg): print(f"\n{BOLD}{msg}{RESET}")

def connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    if HAS_VEC:
        sqlite_vec.load(conn)
    return conn

def table_exists(conn, name: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' OR type='shadow' AND name=?",
        (name,)
    ).fetchone()
    if row:
        return True
    # also check virtual tables
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE name=?", (name,)
    ).fetchone()
    return row is not None

def count(conn, table: str) -> int:
    try:
        return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    except Exception:
        return -1

# ── checks ────────────────────────────────────────────────────────────────────

def check_table_sizes(conn):
    header("Table sizes")
    for t in ("memories", "memories_vec", "memories_fts"):
        n = count(conn, t)
        if n < 0:
            warn(f"{t}: could not query (table may not exist)")
        else:
            info(f"{t}: {n} rows")

def check_orphaned_vec(conn):
    """memories rows with no memories_vec entry."""
    header("Check 1 — memories without a vector (orphaned canonical)")
    if not HAS_VEC:
        warn("Skipped — sqlite_vec not loaded.")
        return []

    rows = conn.execute("""
        SELECT m.id, m.memory, m.created_at
        FROM memories m
        LEFT JOIN memories_vec v ON v.id = m.id
        WHERE v.id IS NULL
    """).fetchall()

    if not rows:
        ok("All memories have a corresponding vector.")
    else:
        err(f"{len(rows)} memories have NO vector entry:")
        for r in rows:
            print(f"      id={r['id']}  created={r['created_at']}  text={r['memory'][:80]!r}")
    return [r['id'] for r in rows]

def check_orphaned_canonical(conn):
    """memories_vec rows with no memories entry."""
    header("Check 2 — vectors without a canonical memory (orphaned vector)")
    if not HAS_VEC:
        warn("Skipped — sqlite_vec not loaded.")
        return []

    rows = conn.execute("""
        SELECT v.id
        FROM memories_vec v
        LEFT JOIN memories m ON m.id = v.id
        WHERE m.id IS NULL
    """).fetchall()

    if not rows:
        ok("All vectors have a corresponding memories row.")
    else:
        err(f"{len(rows)} vectors have NO canonical memory row:")
        for r in rows:
            print(f"      vec id={r['id']}")
    return [r['id'] for r in rows]

def check_orphaned_fts(conn):
    """memories rows not present in memories_fts."""
    header("Check 3 — memories without FTS entry (orphaned FTS)")
    try:
        rows = conn.execute("""
            SELECT m.id, m.memory
            FROM memories m
            LEFT JOIN memories_fts f ON f.id = m.id
            WHERE f.id IS NULL
        """).fetchall()
    except Exception as e:
        warn(f"FTS check skipped: {e}")
        return []

    if not rows:
        ok("All memories have an FTS entry.")
    else:
        err(f"{len(rows)} memories have NO FTS entry:")
        for r in rows:
            print(f"      id={r['id']}  text={r['memory'][:80]!r}")
    return [r['id'] for r in rows]

def check_null_embeddings(conn):
    """memories_vec rows with null or zero-byte embedding."""
    header("Check 4 — NULL or empty embeddings in memories_vec")
    if not HAS_VEC:
        warn("Skipped — sqlite_vec not loaded.")
        return []

    # vec0 doesn't support IS NULL nicely — fetch all and check length
    try:
        rows = conn.execute("SELECT id, embedding FROM memories_vec").fetchall()
    except Exception as e:
        warn(f"Could not scan memories_vec: {e}")
        return []

    bad = []
    for r in rows:
        emb = r['embedding']
        if emb is None or len(emb) == 0:
            bad.append(r['id'])
        else:
            # validate it's divisible into float32s
            if len(emb) % 4 != 0:
                bad.append(r['id'])

    if not bad:
        ok("All embeddings look well-formed.")
    else:
        err(f"{len(bad)} embeddings are NULL or malformed:")
        for vid in bad:
            print(f"      vec id={vid}")
    return bad

def check_empty_memory_text(conn):
    """memories rows with NULL or empty memory text."""
    header("Check 5 — NULL or empty memory text")
    rows = conn.execute("""
        SELECT id, memory, created_at
        FROM memories
        WHERE memory IS NULL OR TRIM(memory) = ''
    """).fetchall()

    if not rows:
        ok("All memories have non-empty text.")
    else:
        err(f"{len(rows)} memories have empty/NULL text:")
        for r in rows:
            print(f"      id={r['id']}  created={r['created_at']}")
    return [r['id'] for r in rows]

def check_null_created_at(conn):
    """memories rows with NULL created_at."""
    header("Check 6 — NULL or missing created_at")
    rows = conn.execute("""
        SELECT id, memory
        FROM memories
        WHERE created_at IS NULL OR TRIM(created_at) = ''
    """).fetchall()

    if not rows:
        ok("All memories have a created_at timestamp.")
    else:
        warn(f"{len(rows)} memories missing created_at (won't affect search but breaks decay scoring):")
        for r in rows:
            print(f"      id={r['id']}  text={r['memory'][:60]!r}")
    return [r['id'] for r in rows]

def check_duplicate_ids(conn):
    """Duplicate UUIDs within memories or memories_vec."""
    header("Check 7 — Duplicate IDs")
    for table in ("memories", "memories_vec"):
        if not HAS_VEC and table == "memories_vec":
            continue
        try:
            rows = conn.execute(f"""
                SELECT id, COUNT(*) AS cnt
                FROM {table}
                GROUP BY id HAVING cnt > 1
            """).fetchall()
            if not rows:
                ok(f"{table}: no duplicate IDs.")
            else:
                err(f"{table}: {len(rows)} duplicate IDs found:")
                for r in rows:
                    print(f"      id={r['id']}  count={r['cnt']}")
        except Exception as e:
            warn(f"Skipped {table}: {e}")

def check_fts_rowid_sync(conn):
    """
    Spot-check that memories_fts rowids align with memories rowids.
    Grabs 20 memories, verifies their rowid matches via fts content table.
    """
    header("Check 8 — FTS5 rowid sync")
    try:
        sample = conn.execute(
            "SELECT rowid, id, memory FROM memories ORDER BY rowid LIMIT 20"
        ).fetchall()
    except Exception as e:
        warn(f"Skipped: {e}")
        return

    mismatches = []
    for r in sample:
        try:
            fts_row = conn.execute(
                "SELECT rowid, id FROM memories_fts WHERE id = ?", (r['id'],)
            ).fetchone()
            if fts_row is None:
                mismatches.append((r['id'], "missing from FTS"))
            elif fts_row['rowid'] != r['rowid']:
                mismatches.append((r['id'], f"rowid mismatch: memories={r['rowid']} fts={fts_row['rowid']}"))
        except Exception as e:
            mismatches.append((r['id'], str(e)))

    if not mismatches:
        ok(f"FTS rowid sync OK on sampled {len(sample)} rows.")
    else:
        err(f"{len(mismatches)} FTS rowid mismatches:")
        for mid, reason in mismatches:
            print(f"      id={mid}  reason={reason}")

# ── fix ───────────────────────────────────────────────────────────────────────

def fix_orphans(conn, db_path: str, orphan_mem_ids: list, orphan_vec_ids: list):
    """Back up then delete confirmed orphans."""
    if not orphan_mem_ids and not orphan_vec_ids:
        print("\nNothing to fix.")
        return

    backup = db_path + f".bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(db_path, backup)
    print(f"\n{BOLD}Backup written to:{RESET} {backup}")

    deleted = 0
    if orphan_mem_ids:
        # delete from memories (FTS trigger fires) + memories_vec
        ph = ",".join("?" * len(orphan_mem_ids))
        conn.execute(f"DELETE FROM memories WHERE id IN ({ph})", orphan_mem_ids)
        try:
            conn.execute(f"DELETE FROM memories_vec WHERE id IN ({ph})", orphan_mem_ids)
        except Exception:
            pass
        deleted += len(orphan_mem_ids)
        print(f"  {RED}Deleted{RESET} {len(orphan_mem_ids)} orphaned memories rows.")

    if orphan_vec_ids:
        ph = ",".join("?" * len(orphan_vec_ids))
        try:
            conn.execute(f"DELETE FROM memories_vec WHERE id IN ({ph})", orphan_vec_ids)
            deleted += len(orphan_vec_ids)
            print(f"  {RED}Deleted{RESET} {len(orphan_vec_ids)} orphaned vector rows.")
        except Exception as e:
            print(f"  {YELLOW}Could not delete orphaned vectors: {e}{RESET}")

    conn.commit()
    print(f"\n{GREEN}Done. Removed {deleted} orphaned records.{RESET}")

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Aiko memory.db integrity checker")
    parser.add_argument("--db", default=None,
                        help="Path to memory.db (default: ~/.aiko/<user_id>/memory/memory.db)")
    parser.add_argument("--fix", action="store_true",
                        help="Auto-delete orphans after backing up the db")
    args = parser.parse_args()

    db_path = args.db
    if db_path is None:
        # Build path from USER_STATE_ROOT
        from pathlib import Path
        import os
        user_state_root = Path(os.getenv("USER_STATE_ROOT", str(Path.home() / ".aiko"))).expanduser()
        db_path = str(user_state_root / "memory" / "memory.db")
    if not Path(db_path).exists():
        print(f"{RED}Error:{RESET} Database not found at: {db_path}")
        print("Use --db to specify the correct path.")
        sys.exit(1)

    print(f"{BOLD}Aiko Memory Integrity Check{RESET}")
    print(f"Database: {db_path}")
    print(f"sqlite_vec: {'loaded' if HAS_VEC else 'NOT loaded — vector checks skipped'}")

    conn = connect(db_path)

    check_table_sizes(conn)
    orphan_no_vec  = check_orphaned_vec(conn)
    orphan_no_mem  = check_orphaned_canonical(conn)
    check_orphaned_fts(conn)
    check_null_embeddings(conn)
    check_empty_memory_text(conn)
    check_null_created_at(conn)
    check_duplicate_ids(conn)
    check_fts_rowid_sync(conn)

    # summary
    header("Summary")
    total_issues = len(orphan_no_vec) + len(orphan_no_mem)
    if total_issues == 0:
        ok("No critical orphan issues found.")
    else:
        err(f"{total_issues} orphaned records detected.")
        if not args.fix:
            print(f"\n  Run with {BOLD}--fix{RESET} to auto-delete orphans (db is backed up first).")

    if args.fix:
        fix_orphans(conn, db_path, orphan_no_vec, orphan_no_mem)

    conn.close()

if __name__ == "__main__":
    main()
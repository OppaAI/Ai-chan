#!/usr/bin/env python3
"""
migrate_memory_phase_b.py

Optional offline backfill for Phase B entity tags on legacy personal memories.
Does not re-embed. Safe to re-run (only_empty by default).

Usage:
  uv run python -m util.migrate_memory_phase_b --dry-run
  uv run python -m util.migrate_memory_phase_b
  uv run python -m util.migrate_memory_phase_b --limit 500 --user-id <id>
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Backfill Phase B entities/kind on memory.db")
    parser.add_argument("--db", type=str, default="", help="Path to memory.db")
    parser.add_argument("--user-id", type=str, default="", help="User id filter / path resolve")
    parser.add_argument("--limit", type=int, default=0, help="Max rows to update (0 = all)")
    parser.add_argument("--all", action="store_true", help="Re-tag even rows that already have entities")
    parser.add_argument("--dry-run", action="store_true", help="Count candidates only")
    args = parser.parse_args(argv)

    from cognition.memory.memorize import backfill_entities, ensure_phase_a_schema, existing_columns
    from cognition.memory.vecstore import initialize_store_db, resolve_user_db_path
    from system.userspace import current_user_id

    uid = (args.user_id or "").strip() or current_user_id()
    if args.db.strip():
        db_path = Path(args.db).expanduser()
    else:
        env = os.getenv("SQLITE_MEMORY_PATH", "").strip()
        db_path = Path(env).expanduser() if env else resolve_user_db_path("memory/memory.db", user_id=uid)

    print(f"user_id={uid}")
    print(f"db={db_path}")
    if not db_path.exists() and str(db_path) != ":memory:":
        print("DB does not exist — nothing to backfill.")
        return 0

    conn = initialize_store_db(str(db_path), "PRAGMA journal_mode = WAL;", user_id=uid, vector=True)
    try:
        ensure_phase_a_schema(conn)
        cols = existing_columns(conn)
        if "entities" not in cols:
            print("entities column missing — run Phase A migrate first.")
            return 1

        if args.dry_run:
            sql = "SELECT COUNT(*) AS n FROM memories WHERE 1=1"
            params: list = []
            if args.user_id:
                sql += " AND user_id = ?"
                params.append(uid)
            if not args.all:
                sql += " AND (entities IS NULL OR entities = '' OR entities = '[]')"
            n = conn.execute(sql, params).fetchone()["n"]
            print(f"Dry-run candidates: {n}")
            return 0

        n = backfill_entities(
            conn,
            user_id=uid if args.user_id else None,
            limit=args.limit,
            only_empty=not args.all,
        )
        print(f"Updated {n} rows (no vectors rebuilt).")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())

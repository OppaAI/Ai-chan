#!/usr/bin/env python3
"""
migrate_memory_phase_a.py

Idempotent Phase A migration for Aiko personal memory DBs.

Adds columns (no data loss, no re-embed):
  status, supersedes_id, kind, source, entities

Usage:
  # Migrate the active user's DB (respects USER_ID / userspace):
  uv run python -m util.migrate_memory_phase_a

  # Explicit path:
  uv run python -m util.migrate_memory_phase_a --db ~/.aiko/<user>/memory/memory.db

  # Dry-run (report only):
  uv run python -m util.migrate_memory_phase_a --dry-run

Safe to re-run. Boot path also calls ensure_phase_a_schema() so this CLI is
optional — useful for offline/batch migration before upgrading the process.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Migrate Aiko memory.db to Phase A schema")
    parser.add_argument(
        "--db",
        type=str,
        default="",
        help="Path to memory.db (default: resolve via SQLITE_MEMORY_PATH / userspace)",
    )
    parser.add_argument(
        "--user-id",
        type=str,
        default="",
        help="User id for path resolution when --db is omitted",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print missing columns only; do not ALTER",
    )
    args = parser.parse_args(argv)

    # Local imports after path setup so `uv run python -m util...` works from repo root.
    from cognition.memory.memorize import _PHASE_A_COLUMNS, ensure_phase_a_schema, existing_columns
    from cognition.memory.vecstore import initialize_store_db, resolve_user_db_path
    from system.userspace import current_user_id

    uid = (args.user_id or "").strip() or current_user_id()
    if args.db.strip():
        db_path = Path(args.db).expanduser()
    else:
        env = os.getenv("SQLITE_MEMORY_PATH", "").strip()
        if env:
            db_path = Path(env).expanduser()
        else:
            db_path = resolve_user_db_path("memory/memory.db", user_id=uid)

    print(f"user_id={uid}")
    print(f"db={db_path}")

    if not db_path.exists() and str(db_path) != ":memory:":
        print("DB does not exist yet — nothing to migrate (fresh CREATE TABLE will include Phase A columns).")
        return 0

    # Minimal DDL so initialize_store_db can open legacy files; migration is ALTER-based.
    ddl = "PRAGMA journal_mode = WAL;"
    conn = initialize_store_db(str(db_path), ddl, user_id=uid, vector=True)
    try:
        cols = existing_columns(conn)
        if "id" not in cols and "memory" not in cols:
            # table missing — open via full memorize DDL would create it; report only
            print("No memories table found — open Aiko once or run with a real memory.db.")
            return 1

        missing = [name for name, _ in _PHASE_A_COLUMNS if name not in cols]
        if not missing:
            print("Already migrated: all Phase A columns present.")
            print(f"columns={sorted(cols)}")
            return 0

        print(f"Missing columns: {missing}")
        if args.dry_run:
            print("Dry-run: no changes written.")
            return 0

        added = ensure_phase_a_schema(conn)
        cols_after = existing_columns(conn)
        print(f"Added: {added or '(none — race/already present)'}")
        print(f"columns_now={sorted(cols_after)}")
        print("Done. No vectors were rebuilt.")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())

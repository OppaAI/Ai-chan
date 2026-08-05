#!/usr/bin/env python3
"""
Rebuild Phase D entity_relations from personal memory entity tags.

Requires Phase A schema (entities column). Best after Phase B backfill.

Usage:
  uv run python -m util.migrate_memory_phase_d --dry-run
  uv run python -m util.migrate_memory_phase_d
  uv run python -m util.migrate_memory_phase_d --user-id <id>
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rebuild entity_relations (Phase D)")
    parser.add_argument("--db", type=str, default="")
    parser.add_argument("--user-id", type=str, default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    from cognition.memory.memorize import ensure_entity_relations_schema, ensure_phase_a_schema, rebuild_entity_relations
    from cognition.memory.vecstore import initialize_store_db, resolve_user_db_path
    from system.userspace import current_user_id

    uid = (args.user_id or "").strip() or current_user_id()
    if args.db.strip():
        db_path = Path(args.db).expanduser()
    else:
        env = os.getenv("SQLITE_MEMORY_PATH", "").strip()
        db_path = Path(env).expanduser() if env else resolve_user_db_path("memory/memory.db", user_id=uid)

    print(f"user_id={uid}\ndb={db_path}")
    if not db_path.exists() and str(db_path) != ":memory:":
        print("DB missing — nothing to do.")
        return 0

    conn = initialize_store_db(str(db_path), "PRAGMA journal_mode = WAL;", user_id=uid, vector=True)
    try:
        ensure_phase_a_schema(conn)
        ensure_entity_relations_schema(conn)
        if args.dry_run:
            n = conn.execute(
                "SELECT COUNT(*) AS n FROM memories WHERE user_id=? AND entities IS NOT NULL AND entities != '[]'",
                (uid,),
            ).fetchone()["n"]
            print(f"Dry-run: {n} memories with entity tags")
            return 0
        stats = rebuild_entity_relations(conn, user_id=uid, clear=True)
        print(f"Rebuilt: {stats}")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())

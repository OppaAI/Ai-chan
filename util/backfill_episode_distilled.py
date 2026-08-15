"""Backfill distilled_into for episodes distilled before EMC-5 added the link.

The distilled_into JSON column (EM→SM link, written by distill_episodes) was
added in EMC-5. Episodes distilled by older dream() runs have distilled_at set
but distilled_into empty, so the LTM/ITM studios show them as distilled without
the edge to their facts.

Strategy (best-effort, never deletes anything):
  1. For each episode with distilled_at set and distilled_into empty, look for
     semantic memories whose entity set overlaps the episode's entities.
  2. Only link when the overlap is strong (>= 2 shared entities, or 1 shared
     entity with high entity importance).
  3. Dry-run by default; pass --apply to write. Prints a report either way.

Usage:
    python util/backfill_episode_distilled.py [--apply] [--db PATH] [--user USER]
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cognition.memory.vecstore import initialize_store_db

MIN_SHARED_ENTITIES = 2
STRONG_ENTITY_THRESHOLD = 0.5


def _entities(raw) -> set[str]:
    if not raw:
        return set()
    try:
        data = json.loads(raw) if isinstance(raw, str) else raw
        if isinstance(data, list):
            return {str(x).strip().casefold() for x in data if str(x).strip()}
    except (TypeError, json.JSONDecodeError):
        pass
    return set()


def backfill(db_path: str, user_id: str | None, dry_run: bool) -> dict:
    conn = initialize_store_db(db_path, "PRAGMA journal_mode = WAL;", user_id=user_id or "", vector=True)
    result = {
        "db": str(db_path),
        "user": user_id,
        "candidates": 0,
        "linked": 0,
        "skipped_weak": 0,
        "errors": 0,
    }
    try:
        # The distilled_at/distilled_into columns are only added by
        # ensure_distilled_column during a dream() run. Add them here so the
        # script works on DBs that never ran EMC-4, and so --apply actually
        # writes the link.
        from cognition.memory.episode_dream import ensure_distilled_column
        ensure_distilled_column(conn)

        cols = {r[1] for r in conn.execute("PRAGMA table_info(emc_storage)").fetchall()}
        if "distilled_into" not in cols or "distilled_at" not in cols:
            result["errors"] = 1
            result["note"] = "emc_storage lacks distilled_at/distilled_into columns"
            return result

        # episodes needing a link: distilled but distilled_into empty
        ep_rows = conn.execute(
            """
            SELECT id, user_id, entities
            FROM emc_storage
            WHERE distilled_at IS NOT NULL
              AND (distilled_into IS NULL OR distilled_into = '' OR distilled_into = '[]')
            """,
        ).fetchall()
        ep_candidates = [r for r in ep_rows if user_id is None or r["user_id"] == user_id]
        result["candidates"] = len(ep_candidates)

        mem_rows = conn.execute(
            """
            SELECT id, user_id, memory, entities
            FROM memories
            WHERE user_id = ?
              AND (status IS NULL OR status = 'active')
            """,
            (user_id or ""),
        ).fetchall() if not user_id else conn.execute(
            """
            SELECT id, user_id, memory, entities
            FROM memories
            WHERE user_id = ?
              AND (status IS NULL OR status = 'active')
            """,
            (user_id,),
        ).fetchall()

        mem_by_id: dict[str, dict] = {}
        for mr in mem_rows:
            mem_by_id[str(mr["id"])] = {
                "text": mr["memory"],
                "entities": _entities(mr["entities"]),
            }

        updates: list[tuple[str, int]] = []
        for er in ep_candidates:
            ep_ents = _entities(er["entities"])
            if not ep_ents:
                continue
            # rank facts by entity overlap
            scored: list[tuple[int, set[str]]] = []
            for mid, mrec in mem_by_id.items():
                overlap = ep_ents & mrec["entities"]
                if overlap:
                    scored.append((len(overlap), mid))
            if not scored:
                continue
            best_count, best_mid = max(scored, key=lambda t: t[0])
            if best_count < MIN_SHARED_ENTITIES:
                result["skipped_weak"] += 1
                continue
            updates.append((json.dumps([best_mid], ensure_ascii=False), er["id"]))

        if dry_run:
            result["linked"] = len(updates)
            return result

        for into_json, eid in updates:
            conn.execute(
                "UPDATE emc_storage SET distilled_into = ? WHERE id = ?",
                (into_json, eid),
            )
        conn.commit()
        result["linked"] = len(updates)
        return result
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _resolve_db_paths(user: str | None) -> list[tuple[Path, str | None]]:
    """Resolve the DB file(s) to process: per-user dir, or a single --db path."""
    # The memory DB lives at <USER_SPACE_ROOT>/<user_id>/memory/memory.db, or
    # SQLITE_MEMORY_PATH when set. Reuse schema's resolution.
    import os

    env = os.getenv("SQLITE_MEMORY_PATH", "").strip()
    if env:
        return [(Path(env).expanduser(), user)]

    if user:
        from cognition.memory.schema import _memory_db_path_for_user
        return [(Path(_memory_db_path_for_user(user)), user)]

    from system.userspace import _user_state_root_value
    root = Path(_user_state_root_value())
    dbs: list[tuple[Path, str | None]] = []
    if root.is_dir():
        for mem_db in root.glob("*/memory/memory.db"):
            dbs.append((mem_db, mem_db.parent.parent.name))
    return dbs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true", help="write links (default: dry-run)")
    ap.add_argument("--dry-run", dest="dry_run_flag", action="store_true", help="explicit dry-run (default)")
    ap.add_argument("--db", help="specific memory DB path")
    ap.add_argument("--user", help="only process this user (else scan all users)")
    args = ap.parse_args()

    targets = [(Path(args.db), args.user)] if args.db else _resolve_db_paths(args.user)
    if not targets:
        print("No memory DBs found (and --db not given).")
        return 1

    for db_path, user in targets:
        if not db_path.exists():
            print(f"skip {db_path} (missing)")
            continue
        report = backfill(str(db_path), user, dry_run=not args.apply)
        mode = "DRY-RUN" if not args.apply else "APPLIED"
        print(
            f"[{mode}] {report['db']} user={report['user'] or 'all'} "
            f"candidates={report['candidates']} linked={report['linked']} "
            f"weak={report['skipped_weak']} note={report.get('note', '-')}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())

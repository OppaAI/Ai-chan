"""Delete frozen date/time facts from per-user memory DBs.

Stale temporal facts ("Oppa today is Monday, August 10, 2026", "Aiko checks
the date of July 3 (day after tomorrow)", "[2026-08-11] OppaAI asks Aiko to
verify the date of July 3") contaminate Aiko's sense of "now" when recalled.
The write-path extraction prompt (cognition/memory/imprint.py) now avoids
creating them and format_for_context drops them from context at render time,
but old rows already in the DB should be removed so they don't occupy recall
slots or leak through other context blocks (persona_context, scene_context).

Dry-run by default; pass --apply to delete. Always backs up each DB file
beside it before deleting rows.

Usage:
    python util/scrub_stale_memory_dates.py [--apply] [--db PATH] [--verbose]
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cognition.memory.imprint import _is_stale_temporal_fact
from system.userspace import _user_state_root_value


def _discover_user_dbs(root: Path) -> list[Path]:
    """Find <root>/<uid>/memory/memory.db and legacy <root>/<uid>/memory.db."""
    found: list[Path] = []
    if not root.exists():
        return found
    for uid_dir in root.iterdir():
        if not uid_dir.is_dir():
            continue
        for rel in ("memory/memory.db", "memory.db"):
            p = uid_dir / rel
            if p.is_file():
                found.append(p)
    return sorted(found)


def _scrub(db_path: Path, *, apply: bool, verbose: bool) -> int:
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT id, memory, created_at FROM memories ORDER BY created_at DESC"
        ).fetchall()
    except sqlite3.OperationalError as exc:
        if verbose:
            print(f"  [skip] {db_path}: {exc}")
        return 0

    stale = [r for r in rows if _is_stale_temporal_fact(r[1] or "")]
    if not stale:
        if verbose:
            print(f"  [ok] {db_path}: no stale temporal facts ({len(rows)} rows)")
        return 0

    print(f"  [found] {db_path}: {len(stale)}/{len(rows)} stale temporal facts")
    for rid, text, created_at in stale:
        print(f"    - {rid[:8]} [{created_at[:10]}] {text[:100]}")

    if not apply:
        return len(stale)

    backup = db_path.with_name(f"{db_path.name}.bak-{int(time.time())}")
    backup.write_bytes(db_path.read_bytes())
    print(f"  [backup] {backup}")

    ids = [r[0] for r in stale]
    conn.executemany("DELETE FROM memories WHERE id = ?", [(i,) for i in ids])
    conn.commit()
    print(f"  [deleted] {len(ids)} rows from {db_path}")
    return len(stale)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Actually delete; default is dry-run")
    parser.add_argument("--db", type=Path, default=None, help="Single memory DB to scrub (default: all users)")
    parser.add_argument("--verbose", action="store_true", help="Report DBs with nothing to delete")
    args = parser.parse_args()

    if args.db is not None:
        dbs = [args.db.expanduser()] if args.db.exists() else []
    else:
        root = Path(_user_state_root_value()).expanduser()
        dbs = _discover_user_dbs(root)
        print(f"Scanning {root}")

    total = 0
    for db in dbs:
        total += _scrub(db, apply=args.apply, verbose=args.verbose)

    print(f"\n{len(dbs)} DB(s) scanned, {total} stale temporal fact(s) "
          + ("deleted" if args.apply else "would be deleted (dry-run; use --apply)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
#!/usr/bin/env python3
"""
migrate_daily_journal.py

One-time migration: moves pinned "Day record for {date}:" memories out of
memory.db into a separate journal.db (created alongside memory.db), and
renames the tag from "Day record for" to "Daily journal of" in the process.

Leaves atomic "[YYYY-MM-DD] fact" pins untouched in memory.db — those keep
working with the existing search()/dream()/consolidate() flow. journal.db
is a plain archival table, not wired into recall, so these blocks stop
competing in RRF scoring and stop risking oversized context injections or
lopsided consolidation chunk budgets.

Encryption: memory.db is SQLCipher-encrypted (SQLITE_ENCRYPTION=1). This
script opens it via core.secure.connect_sqlite() -- same key derivation
(HMAC-SHA256 over DATA_KEY_SECRET/SECRET_KEY + user_id) and same
legacy-passphrase-to-raw-key migration path memorize.py already relies on.
journal.db is created with the SAME per-user key via the same helper.

Before running, make sure the environment this script runs in has the same
DATA_KEY_SECRET (or SECRET_KEY) that the running app uses -- otherwise the
derived key won't match and you'll get a "file is not a database" error
(wrong key), not a crash.

IMPORTANT: stop Aiko (or anything holding memory.db open) before running
this, to avoid writing alongside a live process.

Usage:
    python migrate_daily_journal.py /path/to/memory.db --user-id <user_id>
    python migrate_daily_journal.py /path/to/memory.db --user-id <user_id> --dry-run
    python migrate_daily_journal.py /path/to/memory.db --user-id <user_id> --journal-db /path/to/journal.db
    python migrate_daily_journal.py /path/to/memory.db --user-id <user_id> --yes

Requires: pip install sqlite-vec pysqlcipher3
Run from (or with PYTHONPATH including) the Aiko-chan project root so
`core.secure` and `core.userspace` are importable.
"""
import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

import sqlite_vec

# Make sure the project root (parent of this script's directory, e.g.
# util/../ == Aiko-chan/) is importable so `core.secure` resolves even when
# this script is run directly rather than as part of the package.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from system.secure import connect_sqlite, sqlite_encryption_enabled  # noqa: E402

try:
    from system.userspace import current_user_id  # noqa: E402
except Exception:
    current_user_id = None  # fall back to requiring --user-id explicitly

OLD_PREFIX = "Day record for "
NEW_PREFIX = "Daily journal of "

JOURNAL_DDL = """
CREATE TABLE IF NOT EXISTS journal (
    id               TEXT PRIMARY KEY,
    user_id          TEXT NOT NULL,
    memory           TEXT NOT NULL,
    created_at       TEXT NOT NULL,
    access_count     INTEGER NOT NULL DEFAULT 0,
    last_accessed_at TEXT NOT NULL DEFAULT 'never',
    pinned           INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_journal_user ON journal(user_id);
CREATE INDEX IF NOT EXISTS idx_journal_created ON journal(created_at);
"""


def load_vec_extension(conn) -> None:
    """Load sqlite-vec into an already-open (possibly SQLCipher) connection.
    Needed to read/delete rows from the memories_vec virtual table."""
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)


def rewrite_tag(text: str) -> str:
    if text.startswith(OLD_PREFIX):
        return NEW_PREFIX + text[len(OLD_PREFIX):]
    return text


def resolve_user_id(cli_value):
    if cli_value:
        return cli_value
    if current_user_id is not None:
        try:
            return current_user_id()
        except Exception:
            pass
    sys.exit("Could not resolve user_id automatically -- pass --user-id explicitly.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("memory_db", help="Path to existing memory.db")
    ap.add_argument("--user-id", default=None,
                    help="User id whose per-user SQLCipher key should be derived "
                         "(required unless core.userspace.current_user_id() can resolve it)")
    ap.add_argument("--journal-db", default=None,
                    help="Path to journal.db (default: journal.db next to memory.db)")
    ap.add_argument("--dry-run", action="store_true", help="Show what would move, change nothing")
    ap.add_argument("--yes", action="store_true", help="Skip confirmation prompt")
    ap.add_argument("--no-backup", action="store_true",
                    help="Skip the automatic memory.db backup (not recommended)")
    args = ap.parse_args()

    memory_path = Path(args.memory_db).expanduser().resolve()
    if not memory_path.exists():
        sys.exit(f"memory.db not found: {memory_path}")

    journal_path = (
        Path(args.journal_db).expanduser().resolve()
        if args.journal_db
        else memory_path.parent / "journal.db"
    )

    user_id = resolve_user_id(args.user_id)
    print(f"Using user_id: {user_id}  (SQLCipher encryption {'ON' if sqlite_encryption_enabled() else 'OFF'})")

    src = connect_sqlite(memory_path, user_id=user_id)
    load_vec_extension(src)

    rows = src.execute(
        "SELECT * FROM memories WHERE pinned = 1 AND memory LIKE ?",
        (OLD_PREFIX + "%",),
    ).fetchall()

    if not rows:
        print("No 'Day record for ...' pinned memories found. Nothing to do.")
        src.close()
        return

    print(f"Found {len(rows)} daily-record memories to move:")
    for r in rows:
        preview = r["memory"][:70].replace("\n", " ")
        print(f"  {r['id']}  {r['created_at']}  {preview!r}...")

    if args.dry_run:
        print(f"\n--dry-run: would write to {journal_path}, delete {len(rows)} row(s) from {memory_path}. No changes made.")
        src.close()
        return

    if not args.yes:
        resp = input(f"\nMove {len(rows)} row(s) to {journal_path} and delete from {memory_path}? [y/N] ")
        if resp.strip().lower() != "y":
            print("Aborted.")
            src.close()
            return

    if not args.no_backup:
        # Checkpoint WAL into the main db file first, so the backup copy is
        # self-contained (a raw file copy while WAL-mode writes are pending
        # in -wal/-shm sidecars would otherwise miss recent data). Works
        # fine against an unlocked SQLCipher connection just like plain
        # SQLite -- the pragma operates above the encryption layer.
        try:
            src.execute("PRAGMA wal_checkpoint(FULL)")
        except Exception as e:
            print(f"  ! wal_checkpoint failed (continuing anyway): {e}")
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        backup_path = memory_path.with_name(f"{memory_path.name}.bak-{timestamp}")
        shutil.copy2(memory_path, backup_path)
        # WAL/SHM sidecars, if present, also need copying for a fully
        # consistent standalone backup.
        for suffix in ("-wal", "-shm"):
            side = memory_path.with_name(memory_path.name + suffix)
            if side.exists():
                shutil.copy2(side, backup_path.with_name(backup_path.name + suffix))
        print(f"Backed up memory.db -> {backup_path}")
    else:
        print("--no-backup set: skipping memory.db backup.")

    # journal.db gets the SAME per-user key via the same helper, so it's
    # encrypted at rest exactly like memory.db.
    dst = connect_sqlite(journal_path, user_id=user_id)
    dst.executescript(JOURNAL_DDL)

    moved, skipped = 0, 0
    moved_ids = []

    for r in rows:
        new_text = rewrite_tag(r["memory"])
        cur = dst.execute(
            """
            INSERT OR IGNORE INTO journal
                (id, user_id, memory, created_at, access_count, last_accessed_at, pinned)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                r["id"], r["user_id"], new_text, r["created_at"],
                r["access_count"], r["last_accessed_at"], r["pinned"],
            ),
        )
        if cur.rowcount:
            moved += 1
            moved_ids.append(r["id"])
        else:
            # id already present in journal.db from a prior run -- leave the
            # source row alone rather than silently deleting unmigrated data.
            print(f"  ! {r['id']} already exists in journal.db -- skipping (not deleted from source).")
            skipped += 1

    dst.commit()
    dst.close()

    # Delete from source: memories table delete auto-cleans memories_fts via
    # trigger; memories_vec is a separate virtual table and needs an
    # explicit delete.
    deleted = 0
    for mem_id in moved_ids:
        try:
            src.execute("DELETE FROM memories WHERE id = ?", (mem_id,))
            src.execute("DELETE FROM memories_vec WHERE id = ?", (mem_id,))
            deleted += 1
        except Exception as e:
            print(f"  ! failed to delete {mem_id} from memory.db: {e}")
    src.commit()
    try:
        src.execute("PRAGMA optimize")
    except Exception:
        pass
    src.close()

    print(f"\nDone. Moved {moved} row(s) to {journal_path}, deleted {deleted} from {memory_path}.")
    if skipped:
        print(f"Skipped {skipped} row(s) already present in journal.db (left untouched in memory.db).")


if __name__ == "__main__":
    main()
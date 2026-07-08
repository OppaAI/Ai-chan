#!/usr/bin/env python3
"""
encrypt_memory.py

One-time migration: encrypt an existing plaintext Aiko memory.db into a
SQLCipher-encrypted database, using the *exact* key derivation and PRAGMA
settings that core/secure.py's connect_sqlite() expects at runtime. This
avoids any drift between how the file is created here and how Aiko will
try to open it later.

Safety:
  - Never modifies the original plaintext file directly.
  - Copies it to a .bak first (if a .bak doesn't already exist).
  - Writes the encrypted output to a *new* file.
  - Verifies row counts in every user table match between old and new
    before telling you it's safe to swap the file into place.
  - You do the final swap manually — this script never overwrites the
    live path itself.

Usage:
    cd ~/Aiko-chan   # run from repo root so `core.secure` imports cleanly
    export DATA_KEY_SECRET="your-high-entropy-secret"   # must match what
                                                          # Aiko will use at runtime
    uv run python encrypt_memory.py \\
        --user-id OppaAI \\
        --input /home/oppa-ai/.aiko/OppaAI/memory/memory.db

Then, after it reports success:
    mv memory.db memory.db.plaintext-backup   # keep for a while, don't delete yet
    mv memory.db.encrypted memory.db
    # set SQLITE_ENCRYPTION=1 in your .env
    # run your normal boot smoke test before trusting it in production
"""

import argparse
import shutil
import sqlite3
import sys
from pathlib import Path

# Ensure the repo root is importable regardless of where this script lives
# (e.g. util/encrypt_memory.py) or what directory it's invoked from — Python
# only auto-adds the *script's own* directory to sys.path, not its parent,
# so `import core.secure` fails if this script sits in a subfolder like util/.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _fail(msg: str) -> None:
    print(f"\n❌ {msg}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to existing plaintext memory.db")
    parser.add_argument("--user-id", required=True, help="Aiko user_id (e.g. OppaAI) — must match core.userspace.current_user_id()")
    parser.add_argument("--output", default=None, help="Output path for encrypted db (default: <input>.encrypted)")
    parser.add_argument("--page-size", type=int, default=4096, help="cipher_page_size — must match core/secure.py (default: 4096)")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve() if args.output else input_path.with_suffix(input_path.suffix + ".encrypted")
    backup_path = input_path.with_suffix(input_path.suffix + ".bak-premigration")

    if not input_path.exists():
        _fail(f"Input db not found: {input_path}")

    if output_path.exists():
        _fail(f"Output path already exists, refusing to overwrite: {output_path}")

    # ── import the real key derivation from your codebase, not a copy ──────
    # Run this script from your repo root (or add it to sys.path) so this
    # import resolves to the actual core/secure.py, guaranteeing the key
    # matches what connect_sqlite() will derive at runtime.
    try:
        from core.secure import derive_user_sqlite_key
    except ImportError as e:
        _fail(
            f"Could not import core.secure (repo root guessed as {_REPO_ROOT}): {e}\n"
            "If that path looks wrong, move this script or adjust _REPO_ROOT above."
        )

    try:
        from pysqlcipher3 import dbapi2 as sqlcipher  # type: ignore
    except ImportError:
        _fail(
            "pysqlcipher3 is not installed in this environment. Install libsqlcipher-dev "
            "system package first, then `uv add pysqlcipher3`, before running this migration."
        )

    try:
        raw_key = derive_user_sqlite_key(args.user_id)
    except ValueError as e:
        _fail(str(e))  # raised when DATA_KEY_SECRET/SECRET_KEY isn't set

    # ── step 1: backup the plaintext original ──────────────────────────────
    if not backup_path.exists():
        print(f"Backing up plaintext db → {backup_path}")
        shutil.copy2(input_path, backup_path)
    else:
        print(f"Backup already exists at {backup_path}, leaving it as-is.")

    # ── step 2: get pre-migration row counts for verification later ────────
    plain_counts = _table_row_counts(input_path)
    print(f"Source tables: {plain_counts}")

    # ── step 3: run the actual SQLCipher export ─────────────────────────────
    # Pattern: open the plaintext db with the SQLCipher driver but WITHOUT
    # setting PRAGMA key (so it's read as an ordinary unencrypted sqlite3
    # file — SQLCipher's driver can do this transparently). Then ATTACH a
    # brand-new encrypted db, set its cipher pragmas, and export everything
    # from main -> the attached encrypted schema in one shot.
    print(f"\nEncrypting → {output_path}")
    conn = sqlcipher.connect(str(input_path))
    try:
        conn.execute("ATTACH DATABASE ? AS encrypted KEY ?", (str(output_path), f"x'{raw_key}'"))
        conn.execute(f"PRAGMA encrypted.cipher_page_size = {int(args.page_size)}")
        conn.execute("SELECT sqlcipher_export('encrypted')")
        conn.execute("DETACH DATABASE encrypted")
        conn.commit()
    finally:
        conn.close()

    # ── step 4: verify the encrypted copy opens and matches row counts ─────
    print("Verifying encrypted copy...")
    verify_conn = sqlcipher.connect(str(output_path))
    try:
        verify_conn.execute(f"PRAGMA key = \"x'{raw_key}'\"")
        verify_conn.execute(f"PRAGMA cipher_page_size = {int(args.page_size)}")
        verify_conn.execute("SELECT count(*) FROM sqlite_master")  # forces key validation
        encrypted_counts = _table_row_counts_conn(verify_conn)
    finally:
        verify_conn.close()

    print(f"Encrypted tables: {encrypted_counts}")

    if plain_counts != encrypted_counts:
        _fail(
            f"Row count mismatch after migration!\n"
            f"  plaintext:  {plain_counts}\n"
            f"  encrypted:  {encrypted_counts}\n"
            f"Do NOT swap the file. Original plaintext db and its backup are untouched."
        )

    print("\n✅ Migration verified — row counts match across all tables.")
    print(f"\nOriginal plaintext db untouched at: {input_path}")
    print(f"Backup copy also at:                {backup_path}")
    print(f"New encrypted db ready at:           {output_path}")
    print(
        "\nNext steps (manual, not done by this script):\n"
        f"  mv {input_path} {input_path}.plaintext-retired\n"
        f"  mv {output_path} {input_path}\n"
        "  # then set SQLITE_ENCRYPTION=1 in your .env\n"
        "  # and run your normal boot smoke test before trusting it in production"
    )


def _table_row_counts(db_path: Path) -> dict:
    """Row counts per user table in a plain (unencrypted) sqlite3 file."""
    conn = sqlite3.connect(str(db_path))
    try:
        return _table_row_counts_conn(conn)
    finally:
        conn.close()


def _table_row_counts_conn(conn) -> dict:
    """Row counts per user table (excludes sqlite/fts/vec internal tables)
    for any already-open connection (plain or SQLCipher)."""
    tables = [
        row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' "
            "AND name NOT LIKE '%_fts%' "
            "AND name NOT LIKE '%_vec%'"
        ).fetchall()
    ]
    counts = {}
    for t in tables:
        counts[t] = conn.execute(f"SELECT count(*) FROM \"{t}\"").fetchone()[0]
    return counts


if __name__ == "__main__":
    main()
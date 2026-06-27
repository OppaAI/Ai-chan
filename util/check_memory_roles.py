#!/usr/bin/env python3
"""
check_memory_roles.py

Scans Aiko's memory.db for consecutive same-role message patterns
stored in the memories table, then optionally removes violations.

Usage:
    python check_memory_roles.py              # dry run — report only
    python check_memory_roles.py --fix        # delete offending rows
    python check_memory_roles.py --db /path/to/memory.db
"""

import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path

DEFAULT_DB = Path.home() / ".aiko" / "memory.db"

# Memories are stored as plain text strings like:
#   "user: hello\nassistant: hi there"
# or sometimes JSON. We try both.

def parse_roles(memory: str) -> list[str]:
    """
    Extract ordered list of roles from a stored memory string.
    Tries JSON first, then plain 'role: content' line format.
    Returns e.g. ['user', 'assistant'] or [] if unparseable.
    """
    # Try JSON array of {role, content} dicts
    try:
        data = json.loads(memory)
        if isinstance(data, list):
            return [m["role"] for m in data if "role" in m]
    except (json.JSONDecodeError, TypeError):
        pass

    # Try plain text line format: "user: ..." / "assistant: ..."
    roles = []
    for line in memory.splitlines():
        m = re.match(r"^(user|assistant|system)\s*:", line.strip(), re.IGNORECASE)
        if m:
            roles.append(m.group(1).lower())
    return roles


def has_consecutive_same_role(roles: list[str]) -> bool:
    """Return True if any two consecutive roles are identical."""
    return any(roles[i] == roles[i + 1] for i in range(len(roles) - 1))


def scan(db_path: Path, fix: bool = False) -> None:
    if not db_path.exists():
        print(f"[error] DB not found: {db_path}")
        sys.exit(1)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute("SELECT id, user_id, memory, created_at FROM memories ORDER BY created_at")
    rows = cur.fetchall()

    violations = []

    for row in rows:
        roles = parse_roles(row["memory"])
        if not roles:
            # Can't parse roles — flag for review but don't auto-delete
            print(f"[warn] unparseable memory id={row['id']} user={row['user_id']}")
            continue
        if has_consecutive_same_role(roles):
            violations.append(dict(row))
            violations[-1]["roles"] = roles

    print(f"\n{'='*60}")
    print(f"  Scanned : {len(rows)} memories")
    print(f"  Violations: {len(violations)}")
    print(f"{'='*60}\n")

    if not violations:
        print("✓ No consecutive same-role violations found.")
        conn.close()
        return

    for v in violations:
        print(f"  id       : {v['id']}")
        print(f"  user_id  : {v['user_id']}")
        print(f"  created  : {v['created_at']}")
        print(f"  roles    : {' → '.join(v['roles'])}")
        print(f"  memory   : {v['memory'][:120].strip()!r}{'...' if len(v['memory']) > 120 else ''}")
        print()

    if fix:
        ids = [v["id"] for v in violations]
        placeholders = ",".join("?" * len(ids))
        cur.execute(f"DELETE FROM memories WHERE id IN ({placeholders})", ids)
        conn.commit()
        print(f"[fix] Deleted {cur.rowcount} violating rows.")
    else:
        print("[dry-run] No changes made. Run with --fix to delete violations.")

    conn.close()


def main():
    parser = argparse.ArgumentParser(description="Check Aiko memory.db for role alternation violations.")
    parser.add_argument("--db",  type=Path, default=DEFAULT_DB, help="Path to memory.db")
    parser.add_argument("--fix", action="store_true",           help="Delete violating rows (default: dry run)")
    args = parser.parse_args()

    mode = "FIX MODE" if args.fix else "DRY RUN"
    print(f"\nAiko memory role checker — {mode}")
    print(f"DB: {args.db}")

    scan(args.db, fix=args.fix)


if __name__ == "__main__":
    main()
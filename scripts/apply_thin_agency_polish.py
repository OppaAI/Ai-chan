#!/usr/bin/env python3
"""Apply thin agency polish patch to edge_state.py and think.py."""
from __future__ import annotations
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATCH = Path(__file__).resolve().parent / "thin_agency_polish.patch"


def main() -> int:
    if not PATCH.exists():
        print(f"missing {PATCH}", file=sys.stderr)
        return 1
    r = subprocess.run(
        ["git", "apply", "--check", str(PATCH)],
        cwd=ROOT, capture_output=True, text=True,
    )
    if r.returncode != 0:
        print(r.stderr or r.stdout, file=sys.stderr)
        return 1
    r = subprocess.run(["git", "apply", str(PATCH)], cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr or r.stdout, file=sys.stderr)
        return 1
    print("Applied thin_agency_polish.patch to edge_state.py and think.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

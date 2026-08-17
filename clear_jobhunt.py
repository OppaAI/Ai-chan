#!/usr/bin/env python3
from pathlib import Path

# Clear the dedup ledger
ledger = Path("/home/oppa-ai/.aiko/github_205369547/agentic/workflows/job_hunt/ledger.json")
if ledger.exists():
    ledger.write_text("{}")
    print("Ledger cleared")

# Also clear old cache files to force fresh fetch
cache_dir = Path("/home/oppa-ai/.aiko/github_205369547/agentic/workflows/job_hunt/cache")
for f in cache_dir.glob("fetch_2026-08-1*_*.jsonl"):
    f.unlink()
    print(f"Removed: {f.name}")

print("Done - next run will fetch fresh")
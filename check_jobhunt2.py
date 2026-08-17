#!/usr/bin/env python3
import sqlite3
from pathlib import Path

db = Path("/home/oppa-ai/.aiko/github_205369547/memory/memory.db")
conn = sqlite3.connect(db)
conn.row_factory = sqlite3.Row

# Check job_hunt cache directory
import os
cache_dir = Path("/home/oppa-ai/.aiko/github_205369547/agentic/workflows/job_hunt/cache")
print(f"Cache dir exists: {cache_dir.exists()}")
if cache_dir.exists():
    for f in cache_dir.glob("*"):
        print(f"  {f.name} ({f.stat().st_size} bytes)")

# Check dedup ledger
ledger = Path("/home/oppa-ai/.aiko/github_205369547/agentic/workflows/job_hunt/ledger.json")
print(f"\nLedger exists: {ledger.exists()}")
if ledger.exists():
    import json
    data = json.loads(ledger.read_text())
    print(f"  Entries: {len(data)}")
    for k, v in list(data.items())[:5]:
        print(f"  {k}: {v}")

# Check email cache
email_cache = list(Path("/home/oppa-ai/.aiko/github_205369547/agentic/workflows/job_hunt/cache").glob("fetch_*_email_*.jsonl"))
print(f"\nEmail cache files: {len(email_cache)}")
for f in email_cache[:3]:
    try:
        lines = f.read_text().strip().split('\n')
        print(f"  {f.name}: {len(lines)} lines")
        for line in lines[:2]:
            d = json.loads(line)
            print(f"    subject: {d.get('subject', '')[:80]}")
            print(f"    matched: {d.get('matched')}")
    except Exception as e:
        print(f"    error: {e}")

conn.close()
#!/usr/bin/env python3
import json
from pathlib import Path

cache_dir = Path("/home/oppa-ai/.aiko/github_205369547/agentic/workflows/job_hunt/cache")

for f in sorted(cache_dir.glob("fetch_2026-08-1*_rss_*.jsonl")):
    lines = f.read_text().strip().split("\n")
    print(f"\n{f.name}: {len(lines)} entries")
    for line in lines[:3]:
        if line.strip():
            d = json.loads(line)
            print(f"  {d.get('title', '')[:80]} | {d.get('source_feed', '')[:60]}")
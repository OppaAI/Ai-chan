#!/usr/bin/env python3
import json
from pathlib import Path

cache_dir = Path("/home/oppa-ai/.aiko/github_205369547/agentic/workflows/job_hunt/cache")

for f in sorted(cache_dir.glob("fetch_2026-08-1*_email_*.jsonl"))[-5:]:
    lines = f.read_text().strip().split("\n")
    for line in lines:
        d = json.loads(line)
        print(f"{f.name} | subject={d.get('subject', '')[:60]} | matched={d.get('matched')} | from={d.get('from', '')[:40]}")
#!/usr/bin/env python3
import json
from pathlib import Path

ledger = Path("/home/oppa-ai/.aiko/github_205369547/agentic/workflows/job_hunt/ledger.json")
data = json.loads(ledger.read_text())
print(f"Total ledger entries: {len(data)}")
for k, v in list(data.items())[:20]:
    print(f"  {k}: {v}")
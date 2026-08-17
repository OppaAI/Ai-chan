#!/usr/bin/env python3
import json
from pathlib import Path

config_path = Path("/home/oppa-ai/jetson/agentic/workflows/job_hunt/config.json")
config = json.load(config_path.open())

from cognition.workflows.job_hunt.toolset import fetch_today_jobs_from_rss, fetch_today_jobs_from_email

# Test RSS fetch
rss = fetch_today_jobs_from_rss(config)
print(f"RSS postings: {len(rss)}")
for p in rss[:3]:
    print(f"  {p['title'][:60]} | {p['source_feed'][:60]}")

# Test email fetch  
email, raw = fetch_today_jobs_from_email(config)
print(f"Email postings: {len(email)} (raw msgs: {raw})")
for p in email[:3]:
    print(f"  {p['title'][:60]} | {p['organization']}")
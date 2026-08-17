#!/usr/bin/env python3
import json
from pathlib import Path

# Check job_hunt config
config = Path("/home/oppa-ai/jetson/agentic/workflows/job_hunt/config.json")
data = json.loads(config.read_text())
print("job_keywords:", data.get("job_keywords", []))
print("dedup_days:", data.get("dedup_days"))
print("include_email:", data.get("include_email"))
print("email_source_domains:", data.get("email_source_domains"))
print("rss_feeds:", data.get("rss_feeds", []))
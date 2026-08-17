#!/usr/bin/env python3
import sqlite3
from pathlib import Path

db = Path("/home/oppa-ai/.aiko/github_205369547/memory/memory.db")
conn = sqlite3.connect(db)
conn.row_factory = sqlite3.Row

rows = conn.execute("""
    SELECT id, created_at, memory, kind
    FROM memories 
    WHERE user_id = ? AND kind = 'fact'
    AND (memory LIKE '%job_hunt%' OR memory LIKE '%job hunt%')
    ORDER BY created_at DESC 
    LIMIT 10
""", ("github_205369547",)).fetchall()

for r in rows:
    print(f"{r['id'][:8]} | {r['created_at'][:19]} | {r['memory'][:150]}")
    print()

conn.close()
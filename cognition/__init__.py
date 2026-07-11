"""
cognition/__init__.py

Shared thread pool for concurrent context-fetch calls used across
cognition.think and skills.agentic.

Two fetch groups use this pool:
  1. Memory + KB (cognition.think._fetch_memory_and_knowledge) — fired from
     route() BEFORE intent is known, since every path (localchat/webchat/
     agentic) needs them regardless of which one gets chosen. Concurrent
     with intent classification itself, not just with each other.
  2. Wiki + agentic-policy + skill + experience
     (skills.agentic._fetch_agentic_only_context) — fired only once intent
     has resolved to "agentic", since these blocks are agentic-only.

All of these are independent reads against separate backing stores
(memory.db, knowledge.db, wiki store, skills store, experience store,
persona/*.md files) keyed only on (user_input, embedder). No fetch
depends on another's output, so completion order never matters — callers
just wait for the ones they need and join the results into the prompt
afterward.

Sized for the busiest caller (agentic's up-to-4-way post-intent fetch)
plus headroom for an overlapping second request's smaller 2-way
pre-intent fetch.
"""
from concurrent.futures import ThreadPoolExecutor

CONTEXT_POOL = ThreadPoolExecutor(max_workers=10, thread_name_prefix="ctx-fetch")

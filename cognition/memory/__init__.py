"""Aiko memory package — personal memory backend and shared recall helpers.

The personal memory engine and the stable public API live in this package.
External consumers import from the hub:

    from cognition.memory.memorize import AikoMemorize, ...

The unified recall facade (memory + KB interleaved) lives in the LTM Studio
backend (interface/webui/studio/memory/ltm/backend/search_memory.py) — it is
user-facing UI tooling, not part of Aiko's own recall path.

Module layout (backend split):
  memorize    — engine classes (_MemoryBackend, AikoMemorize) + re-export hub
  schema      — memory-domain DDL, migrations, env constants, low-level access
  entity      — entity extraction/classification, valence/arousal/salience,
                entity relations, importance (I_e)
  imprint     — write-path extraction/persistence helpers
  search      — recall-time pure helpers (trivial skip, FTS/normalize)
  lifecycle   — dream-pass tunables
  vecstore    — shared SQLite/sqlite-vec access + HarrierEmbedder
"""
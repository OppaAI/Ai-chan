# EMC-2 wire-up (buffer + eviction)

Core logic is in `cognition/memory/episode.py`:
- `ingest_turn(user, assistant)` → staging (skips trivial)
- `maybe_flush()` → storage when size or turn threshold hit
- `flush_all()` on session end

## Required call-site hooks

### 1. `AikoMemorize` facade (`cognition/memory/memorize.py`)

```python
def queue_episode(self, user_input: str, response_text: str) -> None:
    """EMC-2: stage one turn into episodic memory (best-effort)."""
    try:
        store = self._get_episode_store()
        if store is None:
            return
        store.ingest_turn(user_input, response_text, user_id=self.get_user_id())
    except Exception as e:
        log.debug("queue_episode skipped: %s", e)

def _get_episode_store(self):
    if getattr(self, "_episode_store", None) is not None:
        return self._episode_store
    try:
        from cognition.memory.episode import EpisodicStore, EMC_ENABLED
        if not EMC_ENABLED:
            return None
        from cognition.memory.schema import _memory_db_path_for_user
        uid = self.get_user_id()
        self._episode_store = EpisodicStore(
            _memory_db_path_for_user(uid),
            user_id=uid,
            embedder=self._mem._embedder,
        )
        return self._episode_store
    except Exception as e:
        log.debug("episode store init failed: %s", e)
        return None
```

On `switch_user` / shutdown, flush and clear:

```python
if getattr(self, "_episode_store", None) is not None:
    try:
        self._episode_store.flush_all()
        self._episode_store.close()
    except Exception:
        pass
    self._episode_store = None
```

### 2. `AikoThink._store_async` (`cognition/think.py`)

After `queue_write` for facts:

```python
try:
    mem.queue_episode(user_input, response_text)
except Exception:
    pass
```

This also covers the agentic path (it calls `_store_async`).

## Config (`config/memory.yaml`)

```yaml
EMC_EVICT_ENABLED: "1"
EMC_EVICT_MIN_CHARS: "40"
EMC_FLUSH_EVERY_TURNS: "8"
EMC_FLUSH_ON_STAGING: "24"
```

## Out of scope (EMC-3+)

- Injecting episodes into the prompt / joint token budget
- Dream distillation of episodes → facts

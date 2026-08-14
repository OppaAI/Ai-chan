# EMC-1: memorize.py wire-up

Apply these two edits in `cognition/memory/memorize.py`:

## 1. Import

In the `from .schema import (` block, add:

```python
    ensure_episode_schema,
```

next to the other `ensure_*` imports.

## 2. Boot path

In `_MemoryBackend.__init__`, change:

```python
        with self._db_lock:
            ensure_phase_a_schema(self._conn)
            ensure_l2_scene_schema(self._conn)
            ensure_entity_relations_schema(self._conn)
```

to:

```python
        with self._db_lock:
            ensure_phase_a_schema(self._conn)
            ensure_l2_scene_schema(self._conn)
            ensure_entity_relations_schema(self._conn)
            ensure_episode_schema(self._conn)
```

## 3. Optional `__all__`

Add `"ensure_episode_schema"` next to the other ensure exports.

`schema.py` already re-exports `ensure_episode_schema` and defines `KIND_EPISODE`.

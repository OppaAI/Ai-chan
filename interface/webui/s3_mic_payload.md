In `get_voice_input` mic broadcast, include:

```python
"echo_guard_ms": int(os.getenv("BARGE_IN_ECHO_GUARD_MS", "450")),
```

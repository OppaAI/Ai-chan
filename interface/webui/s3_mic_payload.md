S3 mic payload is now in `webui.py` `get_voice_input`:

```python
"echo_guard_ms": _echo_guard_ms(),  # BARGE_IN_ECHO_GUARD_MS, default 450
```

Browser reads it in `webui.js` mic:start:

```js
window.AIKO_BARGE_ECHO_GUARD_MS = msg.echo_guard_ms ?? 450;
```

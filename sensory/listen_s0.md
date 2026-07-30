# S0 listen wiring (manual if auto-hooks miss)

`voice_gates.py` holds flags. `listen.py` should:

1. `from sensory.voice_gates import barge_in_enabled, barge_in_always_on, speaker_verify_gate`
2. `trigger_barge_in`: return immediately if not `barge_in_enabled()`
3. `_barge_in_loop`: treat as disabled when not `barge_in_enabled()`; use `barge_in_always_on()` instead of `BARGE_IN_ALWAYS_ON`
4. After speaker verify in `listen()`: if `speaker_verify_gate()` and `info["verified"] is False`: return `"", info`

WebUI already ignores `barge_in` WS when disabled and sets `window.AIKO_BARGE_IN_ENABLED` via mic start payload.

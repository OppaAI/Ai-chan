# S4 — Dual VAD policy

Locked architecture for Aiko voice input. **No code change required** — this documents what S0–S3 already implement.

## Authority split

| Mode | Client | Server (Jetson) |
|------|--------|------------------|
| **WebUI** | Energy gate in `vad.js` (filter uplink) | **Silero** = authority on every **received** chunk |
| **Local robot (`parec`)** | — | **Silero** only |

**Rule:** browser decides *worth sending*; Jetson Silero decides *speech vs silence* for audio that actually arrives (frames dropped by the energy gate are never scored).

## Keep energy + Silero (not double Silero)

| Choice | Status |
|--------|--------|
| Client **energy** + server **Silero** | **Canonical** — current setup (`vad.js` is energy-gate only) |
| Client Silero + server Silero | **Not adopted / not supported** — no browser Silero path |
| Client-only VAD (no server Silero) | **Rejected** — phones/tabs/throttle produce bad transcripts |
| Unload server Silero “to save RAM” | **Rejected** — Silero is tiny; SenseVoice is the real cost |

If browser energy is noisy on a bad device, tune `ENERGY_START_RMS` / min frames first. Do **not** add client Silero; never remove server Silero.

## Why dual (energy + Silero)

1. **Uplink filter** — energy drops silence frames so the Jetson is not flooded.
2. **Server authority** — Silero is consistent across devices; browser energy is not.
3. **Barge-in** — local Silero monitor (when `BARGE_IN_ENABLED=1`) uses the same family of thresholds as listen VAD, independent of the browser gate. Outside TTS wait, that monitor runs only when **`BARGE_IN_ALWAYS_ON=1`** as well; with `ALWAYS_ON=0` it runs only while TTS wait is armed.

## Related knobs

See `config/sensory.yaml` and `docs/DEBUG_AUDIO.md`.

| Area | Keys / files |
|------|----------------|
| Server VAD | `LISTEN_VAD_*`, `LISTEN_SILENCE_CHUNKS`, … |
| Browser gate | `interface/webui/static/vad.js` (`ENERGY_*`, `PRE_SPEECH_BUFS`) — energy only |
| Barge | `BARGE_IN_ENABLED`, `BARGE_IN_ALWAYS_ON`, `BARGE_IN_ECHO_GUARD_MS`, … |
| Web gate flag | `WEBUI_BROWSER_VAD_GATE` in `webui.py` |

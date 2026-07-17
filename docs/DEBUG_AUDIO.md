# Aiko-chan Voice Input Debug Checklist

Work top to bottom. Stop at the first ❌ — that's your bug. Each step says
exactly which file to open and what to add/check.

---

## Step 0 — Environment & connectivity sanity

**Files:** none yet — just DevTools

- [ ] Page loads with no console errors on load.
- [ ] `#auth-overlay` is NOT covering the screen (check visually, or
      `document.getElementById('auth-overlay').classList` in console).
      If OAuth isn't configured, this can sit on top and swallow clicks on
      `micBtn` even though the WS connects fine underneath.
      → `webui.js`, bottom: `checkAuth()` chain.
- [ ] `wsDot` / `wsLabel` show **"ws connected"**, not "ws offline".
- [ ] You are accessing the page via `https://` or `localhost` — **not**
      plain `http://` over Tailscale/DuckDNS/tunnel. Browsers block mic
      access on insecure origins.
      → confirmed by the exact `sys` chat message wired in
      `webui.js` → `startMic()`:
      `'Microphone blocked — browsers only allow mic access on localhost or HTTPS...'`

If any of these fail, fix here first — nothing downstream matters yet.

---

## Step 1 — Auth overlay double-check

**File:** `webui.js` — `checkAuth()` / `authOverlay` block (bottom of file)

- [ ] `fetch('/api/auth/me')` succeeds → overlay hidden → `connectWS()` runs.
- ❌ If overlay stays visible: fix your OAuth config, or temporarily hardcode
  `hideAuthOverlay(); connectWS();` to bypass for local testing.

---

## Step 2 — Secure context check

**File:** `webui.js` — `startMic()`, very first lines

```js
if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
```

- [ ] Look for the `sys` message in chat: *"Microphone blocked..."*
- ❌ If present → restart backend with `WEBUI_HTTPS=1` (see `webui.py` config
  block, `SSL_CERT`/`SSL_KEY`) and reload via `https://`.

---

## Step 3 — Static asset routing (worklet + VAD files reachable)

**File:** `webui.js` — `startMic()`:
```js
await micContext.audioWorklet.addModule('./pcm-worklet.js');
```
**File:** `webui.js` — `loadOptionalOrt()`:
```js
const required = [
  './ort.min.js', './silero_vad.onnx',
  './ort-wasm-simd-threaded.jsep.mjs', './ort-wasm-simd-threaded.jsep.wasm',
];
```

- [ ] DevTools → Network tab → filter each filename above → all `200 OK`.
- ❌ 404 on `pcm-worklet.js` → tunnel/proxy isn't routing static paths correctly
  (check whatever fronts `webui.py`'s HTTP server, e.g. Cloudflare Tunnel/DuckDNS config).
- ❌ 404 on any ORT/onnx file → files missing from `interface/webui/static/` on the
  Jetson — copy them there.

---

## Step 4 — Worklet is emitting frames

**File:** `pcm-worklet.js` — inside `process()`

Add temporarily:
```js
if (this._fill === this._FRAME_SAMPLES) {
    const out = this._buf.slice(0);
    console.log('[worklet] frame emitted, first sample:', out[0]); // TEMP
    this.port.postMessage(out.buffer, [out.buffer]);
    this._fill = 0;
}
```

Also check the existing log in **`webui.js`** → `startMic()`:
```js
if (!micFirstFrameSeen) {
    micFirstFrameSeen = true;
    console.log('[mic] AudioWorklet is sending PCM frames');
}
```

- [ ] `[mic] AudioWorklet is sending PCM frames` appears in console after mic opens.
- ❌ Never appears → OS-level mic permission issue (common on Linux/Wayland +
  Zen Browser), or wrong input device selected as OS default.

---

## Step 5 — `micStreamingEnabled` gate (the real switch)

**File:** `webui.js` — `onmessage` handler for `micWorklet.port`:
```js
if (wsReady() && micStreamingEnabled) processVADFrame(new Float32Array(e.data), ws, browserVadGate);
```
**File:** `webui.js` — WS `case 'mic':` handler, where the flag gets set:
```js
if (msg.action === 'start') {
    ...
    micStreamingEnabled = true;  // ← only set here
```

Add temporarily right after that line:
```js
console.log('[mic] streaming enabled, gate=', browserVadGate);
```

- [ ] This log fires shortly after your Python backend actually starts a
      voice turn.
- ❌ Never fires → the server never sent `{"type":"mic","action":"start"}`.
  This means `main.py`'s agentic loop isn't calling
  `listen.get_voice_input()` / the ASR toggle isn't wired up. **Go check
  `main.py` and `agentic/agentic.py`, not the browser**, if this fails.

---

## Step 6 — VAD model loaded (Silero vs. fallback)

**File:** `webui.js` — `loadOptionalOrt().then(() => initVAD())...` block near top

- [ ] Console/UI shows `vad ready` (Silero loaded), not `vad fallback`.
- ❌ Fallback unexpectedly → re-check Step 3 (missing ORT/onnx assets).

---

## Step 7 — VAD is actually detecting your speech

**File:** `vad.js` — inside `processVADFrame` (Silero path), right after:
```js
const prob = out.output.data[0];
```
add:
```js
console.log('[vad] prob=', prob, '_speaking=', _speaking);
```

**File:** `vad.js` — inside `processEnergyVADFrame` (fallback path), right after:
```js
const rms = _rms(frame);
```
add:
```js
console.log('[vad] rms=', rms, '_speaking=', _speaking);
```

- [ ] Value clearly crosses `VAD_THRESHOLD = 0.5` (Silero) or
      `ENERGY_START_RMS = 0.018` (fallback) when you talk.
- ❌ Never crosses → mic gain too low. Try setting `autoGainControl: false`
  in **`webui.js`** → `startMic()`:
  ```js
  micStream = await navigator.mediaDevices.getUserMedia({
    audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: false },
  });
  ```

---

## Step 8 — WebSocket transport (no code change — DevTools only)

**Where:** DevTools → Network tab → WS → Messages

- [ ] See `{"type":"vad","event":"start"}`
- [ ] See a stream of binary frames
- [ ] See `{"type":"vad","event":"end"}` (~1.2s after you stop talking —
      this is `SILENCE_TIMEOUT` in `vad.js`, not a bug)
- ❌ `start`/`end` present but zero binary frames between them → check
  `browserVadGate` value matches what you expect
  (`WEBUI_BROWSER_VAD_GATE` env var, read in `webui.py`).

---

## Step 9 — Server receives the frames

**File:** `webui.py` — `_ws_handler()`, binary branch:
```python
if isinstance(raw, bytes):
    if self._mic_active.is_set():
        self._audio_q.put(raw)
    continue
```
Add temporarily:
```python
if isinstance(raw, bytes):
    if self._mic_active.is_set():
        log.info("[aiko-web] frame recv %d bytes, q=%d", len(raw), self._audio_q.qsize())
        self._audio_q.put(raw)
    else:
        log.warning("[aiko-web] frame DROPPED — mic not active")
    continue
```

**File:** `webui.py` — `_ws_handler()`, `vad` message branch:
```python
elif mtype == "vad":
    event = msg.get("event")
```
Add temporarily:
```python
    log.info("[aiko-web] vad event: %s (mic_active=%s)", event, self._mic_active.is_set())
```

- ❌ "DROPPED — mic not active" → timing race between the server's
  `mic:start` broadcast and `self._mic_active.set()` in
  **`webui.py`** → `get_voice_input()`. Check ordering there.
- ❌ Nothing logs at all → WS not reachable — check port/TLS mismatch
  between `WEBUI_HTTPS` and the `wss://` URL built in
  **`webui.js`** → `websocketURL()`.

---

## Step 10 — `listen.py` buffer contents

**File:** `listen.py` — `_record()`, right before `return` at the bottom:
```python
return np.concatenate(audio_chunks).astype(np.float32)
```
Add just above it:
```python
log.info("[listen] chunks=%d speech_count=%d vad_presegmented=%s",
          len(audio_chunks), speech_count, vad_presegmented)
```

**File:** `listen.py` — `listen()` method, right before `text = self._transcribe(audio)` (both call sites):
```python
import soundfile as sf
sf.write("/tmp/debug_utterance.wav", audio, SAMPLE_RATE)
log.info("[listen] wrote debug wav, %d samples", len(audio))
```

- [ ] `chunks=0` → `_chunk_source` returned `None` immediately —
  `_audio_q` was empty, `FRAME_TIMEOUT_S` (5s) expired. Trace back to Step 9.
- [ ] Play `/tmp/debug_utterance.wav` back:
  - Clear, correctly-paced speech → pipeline is fine up to here, problem
    is ASR itself (Step 11).
  - Silence/hiss → mic gain or wrong input device (back to Step 7).

---

## Step 11 — ASR itself (isolated test, no browser needed)

**File:** run standalone on the Jetson, referencing `sensory/listen.py`:
```bash
python3 -c "
import soundfile as sf
from sensory.listen import _load_sense_voice_recognizer, SAMPLE_RATE
audio, sr = sf.read('/tmp/debug_utterance.wav', dtype='float32')
assert sr == SAMPLE_RATE
model = _load_sense_voice_recognizer()
stream = model.create_stream()
stream.accept_waveform(sr, audio)
model.decode_stream(stream)
print(repr(stream.result.text))
"
```

- ❌ Empty/garbled text on clearly-good audio → check `ASR_LANGUAGE` env var,
  or delete the HF cache for `ASR_MODEL` (`sensory/listen.py` config block) and
  let it re-download — cache may be corrupted/stale.

---

## Quick-start priority order

Given a "no response at all" symptom, check in this order first —
fastest signal per minute spent:

1. Step 0 (auth overlay + secure context) — 1 min, DevTools only
2. Step 8 (WS Messages tab) — 1 min, no code changes
3. Step 5 (`micStreamingEnabled` log) — pinpoints browser vs. backend
4. Step 9 (server-side frame receipt log)
5. Step 10 (WAV dump + playback) — definitive proof of what reached ASR

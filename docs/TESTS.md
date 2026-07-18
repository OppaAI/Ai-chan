[← Back to README](../README.md)
# Aiko-chan アイコちゃん — Test Checklist

This checklist is the release gate for Aiko's phase stack. It is intentionally detailed: run the relevant phase suite after installation, dependency upgrades, model swaps, prompt/persona changes, UI changes, and before declaring a phase stable.

Use it as an **industrial-standard verification plan**, not just a smoke test. Record the date, hardware, model names, environment overrides, observed latencies, and failures for every full run.

---

## Test Evidence Template

Copy this block into your run notes before executing a phase suite.

- **Date / operator:**
- **Git commit:** `git rev-parse --short HEAD`
- **Hardware / OS:** Jetson Orin Nano or desktop, RAM/VRAM, JetPack/Ubuntu version
- **Python / uv:** `python --version`; `uv --version`
- **LLM server / model:** `LLM_BASE_URL`, `LLM_MODEL`, quantization, context length
- **Voice stack:** MioTTS URL/preset/device, ASR model, speaker verification state
- **Memory DB:** `SQLITE_MEMORY_PATH`, DB size before/after, backup path if applicable
- **Encryption:** `SQLITE_ENCRYPTION` on/off; if on, confirm `DATA_KEY_SECRET`/`SECRET_KEY` source (do not record the secret value itself)
- **Browser / terminal:** browser version for WebUI; `$TERM`, terminal dimensions for TUI
- **Network state:** online/offline, SearXNG reachable, HF cache warm/cold
- **Pass/fail notes:** include exact command output snippets, screenshots, logs, and latency samples

---

## Severity and Exit Criteria

- **P0 blocker:** crash, data loss, unsafe path traversal/write, privacy leak, deadlock, unrecoverable terminal/audio state, or a core Phase 1–2.5 workflow unusable.
- **P1 major:** repeated timeout, severe hallucinated tool/search result, broken persistence, unacceptable latency regression, or degraded voice loop that needs restart.
- **P2 minor:** cosmetic UI issue, recoverable warning, single flaky attempt with documented workaround.

A phase is stable only when:

- [ ] All P0/P1 findings are closed or explicitly waived with owner/date/reason.
- [ ] All persistence tests were repeated across at least one application restart.
- [ ] Stress tests ran for the requested duration without memory growth, file corruption, or deadlock.
- [ ] Logs were reviewed for tracebacks, unhandled task exceptions, retry storms, and hidden warnings.

---

## Pre-flight — Stack, Configuration, and Safety

Run before any phase suite.

### Repository and environment

- [ ] `git status --short` is clean or all local changes are intentional and documented.
- [ ] `uv sync` completes, or existing lockfile environment is already synced.
- [ ] `uv run python -m compileall main.py core tui webui skills training` completes without syntax errors.
- [ ] Required environment variables are set or intentionally defaulted: `LLM_BASE_URL`, `LLM_MODEL`, `SQLITE_MEMORY_PATH`, `WORKSPACE_ROOT`, `MIOTTS_API_URL`, `MIOTTS_PRESET`.
- [ ] Secrets, tokens, absolute private paths, and user-specific data are not printed in normal logs.
- [ ] `WORKSPACE_ROOT` points to a writable persistent directory and is not `/tmp` unless testing ephemeral behavior.
- [ ] `SQLITE_MEMORY_PATH` points to persistent storage and the parent directory exists.
- [ ] If `SQLITE_ENCRYPTION=1`, at least one of `DATA_KEY_SECRET` or `SECRET_KEY` is set and is not a placeholder/empty string.
- [ ] If `SQLITE_ENCRYPTION` is unset or `0`, confirm this is intentional for tonight's run — not silently defaulted.

### Service health

- [ ] `docker compose ps` shows SearXNG running and no unintended legacy Qdrant dependency.
- [ ] `curl "http://localhost:8081/search?q=test&format=json"` returns JSON results within 3 seconds.
- [ ] `curl http://localhost:8080/v1/models` returns JSON containing the configured `LLM_MODEL` alias.
- [ ] `curl http://localhost:8001/health` returns `{"status":"ok"}` when TTS/voice tests are in scope.
- [ ] `uv run python -c "import sqlite_vec, tokenizers, onnxruntime, sherpa_onnx, silero_vad; print('OK')"` prints `OK`.
- [ ] `parec --version` works on the target voice machine; PulseAudio/PipeWire has an active default source.
- [ ] If `SQLITE_ENCRYPTION=1`: `uv run python -c "import pysqlcipher3; print('OK')"` prints `OK`.
- [ ] If `SQLITE_ENCRYPTION=1`: `uv run python -c "from system.secure import sqlite_encryption_enabled, derive_user_sqlite_key; print(sqlite_encryption_enabled()); print(derive_user_sqlite_key('test-user'))"` runs without raising and prints a 64-char hex string.

### Baseline observability

- [ ] Start one test run with `--debug` and confirm memory hits/tool routing details are visible without exposing sensitive payloads.
- [ ] Capture idle resource baseline for 60 seconds: CPU, RAM, swap, GPU/VRAM if available, disk free space, and open file descriptors.
- [ ] Confirm logs include subsystem boot boundaries for think, memory, speak, listen, and UI.
- [ ] Confirm Ctrl-C/Ctrl-D cleanly exits, drains memory work, and restores terminal state.

### At-rest encryption (`system/secure.py`)

*Optional SQLCipher-backed encryption for user-private SQLite state. Off by default; enabled via `SQLITE_ENCRYPTION=1`. Run this block before Phase 1 memory tests whenever `SQLITE_ENCRYPTION=1` is set — a key-derivation problem here will otherwise masquerade as a Phase 1 memory failure.*

- [ ] With `SQLITE_ENCRYPTION` unset/`0`, `connect_sqlite()` returns a plain `sqlite3.Connection` and behaves identically to direct `sqlite3.connect()` — no `pysqlcipher3` import is attempted.
- [ ] With `SQLITE_ENCRYPTION=1` and no `DATA_KEY_SECRET`/`SECRET_KEY` set, `connect_sqlite()` / `derive_user_sqlite_key()` raises a clear `ValueError` at boot rather than silently falling back to plaintext or an empty key.
- [ ] With `SQLITE_ENCRYPTION=1` and `pysqlcipher3` not installed, `connect_sqlite()` raises `RuntimeError` with an actionable message rather than an unhandled `ImportError`.
- [ ] `derive_user_sqlite_key(user_id)` is deterministic: calling it twice with the same `user_id` and the same `DATA_KEY_SECRET` returns the identical key.
- [ ] `derive_user_sqlite_key(user_id)` is user-scoped: two different `user_id` values produce different keys.
- [ ] **Regression check for the GitHub user ID format change bug:** derive a key using the *current* `current_user_id()` format, confirm it matches the key already in use for an existing encrypted DB created before the format change — or, if the format changed intentionally, confirm there's a documented migration path (re-key or re-derive) rather than a silent lockout.
- [ ] Opening an existing encrypted DB with the *correct* derived key succeeds and `_validate_sqlcipher_connection`'s `SELECT count(*) FROM sqlite_master` returns without error.
- [ ] Opening an existing encrypted DB with an *incorrect* key fails fast at `_validate_sqlcipher_connection`, not later on first real query — confirm the failure surfaces as a clear boot-time error, not a mysterious later crash mid-conversation.
- [ ] An encrypted DB file is not readable as plaintext SQLite: `sqlite3 <path> ".tables"` (plain CLI, no SQLCipher) fails or returns garbage rather than showing the schema.
- [ ] `PRAGMA cipher_page_size = 4096` is applied consistently — reopening a DB created by this code with a different page size setting does not silently corrupt reads.
- [ ] Switching `SQLITE_ENCRYPTION` from `0` to `1` (or vice versa) against an *existing* unencrypted/encrypted DB is documented behavior — confirm it fails loudly (wrong file format) rather than appearing to "work" while actually creating a second shadow DB or silently reading garbage.
- [ ] `DATA_KEY_SECRET`/`SECRET_KEY` value is never printed in logs, error messages, or debug output (only the *derived per-user key* should ever appear, and only in contexts that already treat it as sensitive).
- [ ] Concurrent access: two connections opened for the same `user_id` (e.g. `AikoMemorize` main thread + write-queue worker thread — see `memory/memorize.py`'s `_write_loop`) both derive the same key and don't race on `PRAGMA key`.

---

## Phase 1 — Soul

*CLI/TUI text companion, OpenAI-compatible local LLM inference, sqlite-vec memory, and SearXNG web search.*

### 1.1 Boot and launch modes

- [ ] `uv run python main.py --text` starts with ASR/TTS loaded but initially toggled off, and reaches the first input prompt.
- [ ] `uv run python main.py --text --debug` starts and prints memory recall/debug information per turn.
- [ ] `uv run python main.py --clear-mem` wipes the configured memory DB and exits without entering chat.
- [ ] Launch fails gracefully with a clear error if `LLM_BASE_URL` is unavailable; no traceback-only user experience.
- [ ] Launch from a different working directory still resolves persona, workspace, and memory paths as expected or documents the required cwd.
- [ ] Cold start and warm start times are recorded separately.

### 1.2 Local LLM endpoint and streaming

- [ ] `curl http://localhost:8080/v1/models` lists the exact model alias configured as `LLM_MODEL`.
- [ ] First assistant response streams progressively; tokens are not buffered until the end.
- [ ] Warm-model time-to-first-token is measured over 10 short prompts; median and p95 are recorded.
- [ ] The model respects `/think <question>` and returns a reasoned answer while suppressing raw `<think>` tags.
- [ ] Malformed/empty user input, whitespace-only input, and very long input do not crash the route loop.
- [ ] LLM server interruption mid-stream produces a visible recoverable error and leaves the next turn usable.
- [ ] Timeout behavior is bounded: one hung LLM request does not permanently block input, TUI drawing, memory queue, or shutdown.

### 1.3 Persona and conversation quality

- [ ] `persona/soul.md`, `persona/identity.md`, `persona/user.md`, `persona/character.md`, and `persona/skills.md` load without startup errors.
- [ ] Responses are consistent with Aiko's persona over at least 20 mixed turns: casual chat, technical help, Japanese/English mixed text, and emotional support.
- [ ] Persona compliance survives `/reset`; short-term context resets without deleting persistent memories.
- [ ] The assistant does not reveal raw system/developer prompts or internal tool schemas during normal chat.
- [ ] Markdown, code blocks, Japanese text, emoji, and punctuation render acceptably in the active UI.

### 1.4 Memory: sqlite-vec + custom Harrier embedder CRUD

- [ ] After several exchanges, `/memory` shows relevant stored memories and does not show empty placeholder rows.
- [ ] Restart Aiko and ask about a previously discussed topic; recall surfaces the relevant persisted memory.
- [ ] `/remember` pins the last exchange; after restart `/memory` shows the pinned item remains.
- [ ] `/clear` wipes memories; `/memory` is empty afterward and no stale recall appears in the next turn.
- [ ] `--clear-mem` wipes memories and exits cleanly without launching the UI.
- [ ] DB file exists after first write: `ls -lh "$SQLITE_MEMORY_PATH"`.
- [ ] `uv run python -c "from memory.memorize import AikoMemorize; m=AikoMemorize(); print(m.dream(dry_run=True))"` completes without error.
- [ ] Corrupt/locked DB simulation is handled safely: Aiko reports the memory problem and continues chat if possible, without overwriting unrelated files.
- [ ] Duplicate memory pressure test: repeat the same fact 20 times; recall does not become dominated by redundant near-identical entries.
- [ ] Unicode memory test: store Japanese, Korean, emoji, and mixed punctuation; recall and display remain readable.
- [ ] Privacy check: `/memory` does not expose secrets from environment variables or unrelated workspace files.
- [ ] If `SQLITE_ENCRYPTION=1` for this run, confirm `memory/memorize.py`'s DB open path goes through `system.secure.connect_sqlite` (not a bare `sqlite3.connect`) — see Pre-flight's At-rest encryption block for the full key-derivation test set.

### 1.4a Trivial-input skip and broad-recall routing (`memory/memorize.py`)

*`_is_trivial_input()` and `_BROAD_RECALL_RE` are choke-point logic — every `search()` call from every input path goes through them before the cache lookup or embedding call, so a bug here silently affects CLI, WebUI, and voice alike.*

- [ ] Pure filler input ("hi", "ok", "thanks", "bye") short-circuits to `[]` without a cache lookup or embedding call — confirm via debug log, not just absence of recall in chat.
- [ ] Greeting/wellbeing phrases ("how are you", "what's up") are also skipped, not just single-word filler.
- [ ] A wake-word-only utterance (e.g. "Aiko" alone, matching `AI_NAME`) is treated as trivial.
- [ ] A mixed input like "hi aiko, what's the weather" is NOT treated as trivial — the non-filler clause forces a normal search.
- [ ] A ragged multi-clause ASR transcript ("Hi, I. How are you doing.") is correctly classified clause-by-clause per `_CLAUSE_SPLIT_RE`, not rejected wholesale.
- [ ] Broad-recall phrasing ("what do you remember about me", "anything about Oppa from before") routes to `_recent_or_important_memories()` instead of normal RRF search, and returns pinned-first results.
- [ ] Broad-recall results are deduplicated by normalized text before the `limit` cutoff is applied — a set of identical pinned daily-record rows doesn't eat multiple result slots.

### 1.5 Memory retrieval quality and cleanup

- [ ] KNN vector recall returns semantically similar memories for paraphrased queries.
- [ ] FTS5 lexical recall returns exact-name/exact-term memories that embeddings may miss.
- [ ] RRF combined recall ranks an exact relevant memory above unrelated recent memories.
- [ ] Memory decay/cleanup logs `deleted=N, kept=N` on startup or cleanup invocation without errors.
- [ ] Pinned memories are not removed by cleanup.
- [ ] Recall limit settings (`MEMORY_RECALL_LIMIT`, `AGENT_MEMORY_RECALL_LIMIT`) are respected.
- [ ] Background memory writes drain on shutdown; no last-turn memory is lost after immediate exit.

### 1.6 Web search and grounding

- [ ] `/web what is the current version of Python` returns a grounded answer using SearXNG results.
- [ ] A naturally time-sensitive question, such as "what happened in the news today", triggers search rather than hallucinated stale knowledge.
- [ ] Search results are summarized; raw SearXNG JSON is not dumped into the chat.
- [ ] Search with no results, rate-limit, timeout, and malformed query cases all produce useful user-facing messages.
- [ ] Search answer cites or names sources enough for the user to distinguish evidence from model inference.
- [ ] Network-offline mode fails gracefully and does not block normal non-web chat.
- [ ] Prompt-injection content from fetched pages is treated as untrusted data and does not override Aiko behavior.

### 1.7 Phase 1 stress and regression

- [ ] Run 100 short text turns in one session; no crash, memory leak trend, or progressive slowdown.
- [ ] Run 20 long-context turns near the configured context limit; truncation/summarization is graceful.
- [ ] Alternate `/web`, `/think`, `/memory`, `/remember`, `/reset`, and normal chat for 30 minutes; no command parser drift.
- [ ] Kill and restart the LLM server during a session; Aiko recovers after the server returns.
- [ ] Disk-full or read-only workspace simulation is documented; save/memory errors are visible and contained.

---

## Phase 1.5 — Stream

*Browser WebUI, shared streaming architecture, persona display, and MioTTS integration.*

### 1.5.1 WebUI layout and input

- [ ] `uv run python main.py --text` launches the browser WebUI with keyboard-first input.
- [ ] Chat panel, status/identity areas, avatar region, and input field render correctly across common browser sizes.
- [ ] Browser resize during generation does not crash or leave garbled output.
- [ ] Typing, backspace, paste, Enter submit, arrow/navigation keys, and slash commands work reliably.
- [ ] `/help` renders the command list without overflowing or breaking subsequent draws.
- [ ] Rapid input submission while streaming is rejected, queued, or handled predictably; it does not corrupt the active assistant message.
- [ ] Long assistant messages scroll correctly; the most recent content remains reachable/readable.
- [ ] ANSI escape sequences or malicious control characters in model output are rendered safely.

### 1.5.2 Streaming pipeline and concurrency

- [ ] Streaming begins within the expected warm-model latency target and updates the UI incrementally.
- [ ] `AIKO_STREAM_DRAW_INTERVAL` throttles draw frequency without hiding tokens or causing flicker.
- [ ] Background LLM warmup reduces first-turn delay compared with cold start; measurements are recorded.
- [ ] Memory writes are asynchronous and do not block token streaming.
- [ ] TTS preparation/playback does not block UI drawing or memory queue draining.
- [ ] Exceptions in token callbacks are logged and do not leave the session permanently wedged.
- [ ] Shutdown during active streaming drains or cancels cleanly without orphaning audio/process threads.

### 1.5.3 Identity and persona panels

- [ ] `persona/identity.md` banner/ASCII art render correctly in the identity panel.
- [ ] Missing or malformed persona files produce clear diagnostics and a safe fallback.
- [ ] Persona updates are picked up after restart and do not require code changes.
- [ ] The UI handles wide Unicode, Japanese kana/kanji, combining marks, and emoji without column misalignment severe enough to hide input.

### 1.5.4 WebUI / VRM bridge foundation

- [ ] `uv run python main.py --text` prints the browser URL and serves `interface/webui/static/index.html`.
- [ ] Browser WebSocket connects, receives boot/status events, and sends text chat end-to-end.
- [ ] Chat messages, streamed tokens, stream commit events, vitals, voice state updates, and errors appear in browser developer tools as expected.
- [ ] `interface/webui/static/assets/Aiko.vrm` loads or fails with a clear visible error.
- [ ] Browser refresh/reconnect does not crash the Python backend or duplicate stale sessions uncontrollably.
- [ ] Multiple browser tabs are tested: behavior is documented, and one tab cannot corrupt backend state for another.
- [ ] Static file path traversal attempts fail; only intended WebUI assets are served.

### 1.5.5 MioTTS text-to-speech

- [ ] `curl http://localhost:8001/health` confirms the MioTTS server before voice playback tests.
- [ ] `/voice` toggles TTS on; the next assistant response plays audio.
- [ ] `/voice` toggles TTS off; later responses do not play audio.
- [ ] Background TTS warmup reduces first-audio latency; cold and warm timings are recorded.
- [ ] Audio plays through the intended output device; `python sensory/speak.py --devices` lists available devices.
- [ ] `sanitize_for_tts` removes markdown, emoji noise, and unsafe symbols without making ordinary Japanese/English text unintelligible.
- [ ] Long responses are split and played completely without cutting off mid-sentence.
- [ ] TTS server unavailable, HTTP error, invalid preset, and invalid device are all surfaced without crashing chat.
- [ ] TTS playback during shutdown stops cleanly and releases the audio device.

### 1.5.6 Remote audio sink and local playback suppression (`sensory/speak.py`)

*When a WebUI audio sink is registered and a browser is actively connected, `_play_wav_bytes()` suppresses local sounddevice playback to avoid doubled/phased audio. A bug here means either dead silence or an audible doubled/phased playback artifact, depending on which direction it breaks.*

- [ ] With no audio sink registered (CLI-only session), local playback via sounddevice works normally.
- [ ] With an audio sink registered but no browser connected (`_has_remote_listener()` returns False), local playback still occurs — the sink alone does not suppress local audio.
- [ ] With an audio sink registered AND a browser actively connected, local sounddevice playback is suppressed; audio is heard only through the browser, not doubled/phased through both endpoints.
- [ ] Disconnecting the browser mid-session restores local playback on the next utterance without a restart.
- [ ] `local_playback = False` (set via `WEBUI_LOCAL_PLAYBACK=0`) always suppresses local audio regardless of remote listener state.
- [ ] When local playback is suppressed, the blocking-wait-for-WAV-duration path (`_play_wav_bytes`'s `deadline` loop) still respects `stop()` — an interrupted turn doesn't hang waiting out the full clip duration.
- [ ] `set_audio_sink(None)` correctly removes the sink and restores default local-playback behavior.
- [ ] Sample-rate resampling for non-native output devices (e.g. USB DAC locked to 48000 Hz) produces audio without pitch/speed distortion.

### 1.5.7 Proactive idle check-in state machine (`main.py`)

*`ProactiveIdleRunner` runs as a background thread independent of the main chat loop. Untested, it can misfire during a live session — interrupting mid-conversation, staying silent when it should check in, or firing during a configured quiet window.*

- [ ] With `PROACTIVE_ENABLED=1` and no user activity, a check-in message appears only after the randomized `PROACTIVE_FIRST_IDLE_MIN/MAX_SECONDS` window elapses — not before.
- [ ] Any user message (`touch()`) resets the idle timer and clears the "resting" flag; a check-in does not fire immediately after fresh activity.
- [ ] `PROACTIVE_COOLDOWN_SECONDS` is respected: two check-ins do not fire back-to-back within the cooldown window.
- [ ] `PROACTIVE_MAX_PER_HOUR` correctly rate-limits check-ins over a rolling hour window — confirm via clock-jumped test or a long soak.
- [ ] A check-in never fires while `active_turn` is set (mid-conversation) or while `speak.is_playing()` is True.
- [ ] Configured quiet windows (`PROACTIVE_QUIET_WINDOWS`) and focus windows (`PROACTIVE_FOCUS_WINDOWS`) correctly suppress check-ins during those times — test at least one day-of-week-scoped window (e.g. `mon-fri 06:00-19:00`) and one wraparound window (e.g. `fri-mon 22:00-06:00`).
- [ ] After `PROACTIVE_REST_AFTER_SECONDS` of continued idle, Aiko sends a single rest message and then goes fully quiet (`_resting = True`) until the next `touch()` — no repeated rest messages.
- [ ] `set_proactive_resting()` correctly propagates to `cognition.think.AikoThink.is_proactive_resting()`, and `memory.learn.idle_learner_loop` observably pauses autonomous study while resting.
- [ ] `/proactive` slash command toggles the feature on/off and immediately resets timers.
- [ ] `PROACTIVE_USE_LLM=1`: check-in and rest messages are generated via `proactive_checkin()` and are not stored as a user turn in memory (verify `/memory` afterward).
- [ ] `PROACTIVE_USE_LLM=0`: check-in and rest messages fall back to the static `PROACTIVE_MESSAGES` / `PROACTIVE_REST_MESSAGE` pool, cycling through without repeating the same message twice in a row where the pool has more than one entry.
- [ ] Killing/restarting the LLM server does not crash the proactive thread — a failed `_generate_proactive_checkin` call is caught and logged, not left to propagate.

### 1.5.8 Karaoke typewriter sync (`main.py`)

*Decouples "LLM token arrival" from "UI reveal," pacing displayed text to TTS playback instead of dumping the full reply the instant it streams. Lower risk than the items above — worst case is a UX/timing glitch, not data loss or a hung session — but worth a pass since it touches every turn when enabled.*

- [ ] `KARAOKE_SYNC=1` with TTS on: text reveal visibly lags behind raw token arrival and roughly tracks audio playback pace, not an instant dump.
- [ ] `KARAOKE_SYNC=1` with TTS off (`/voice` toggled off): typewriter falls back to instant/normal token streaming rather than hanging waiting for an `on_first_audio` that will never fire.
- [ ] Fallback mode: sentences fed to `feed_sentence()` *before* first audio starts are buffered and released together once `on_first_audio()` fires.
- [ ] Fallback mode: sentences fed *after* first audio has already started (later chunks of a long reply) are released to the reveal queue immediately, not silently dropped — this was a previously-fixed bug per the code comments, worth confirming it stays fixed.
- [ ] Interrupting a turn mid-reveal (`/quit`, Ctrl-C) flushes any buffered/queued words to the UI rather than losing the tail of the response.
- [ ] `/karaoke` slash command toggles the feature and takes effect on the next turn.
- [ ] `KARAOKE_WPS` changes visibly alter reveal pace when tested at two very different values (e.g. 1.0 vs 5.0).

### 1.5.9 Phase 1.5 stress and soak

- [ ] Run a 60-minute TUI soak with intermittent long responses, `/help`, `/reset`, and window resizing; no terminal corruption or thread leak.
- [ ] Run a 60-minute WebUI soak with browser refresh every 5 minutes; backend remains responsive.
- [ ] Generate 25 consecutive TTS responses; no audio device lock, runaway queue, or memory leak.
- [ ] Simulate slow browser/WebSocket client; backend does not block core chat streaming indefinitely.

---

## Phase 2 — Voice

*SenseVoice via sherpa-onnx, Silero VAD, optional speaker verification, barge-in, and hands-free talk mode.*

### 2.1 Voice boot and dependency readiness

- [ ] `uv run python main.py` starts full voice mode without `--text` and reaches ready state.
- [ ] Staged boot reports ASR model loading, Silero VAD loading, optional speaker verification, ASR warmup, TTS readiness, and microphone readiness.
- [ ] First-use Hugging Face model download is tested once; cached/offline boot is tested with `HF_HUB_OFFLINE=1`.
- [ ] ASR/TTS initial-toggle modes are correct: `--text` starts with both off but loadable via `/voice` and `/listen`; `--no-asr` starts with keyboard input and TTS on.
- [ ] Missing microphone, missing `parec`, invalid ASR model, and missing HF cache produce actionable errors.
- [ ] Resource usage during boot stays within Jetson limits and does not trigger OOM killer.

### 2.2 Microphone capture and VAD

- [ ] Speaking starts recording automatically; silence stops recording without manual keypress.
- [ ] Short pauses mid-sentence do not prematurely cut off the utterance.
- [ ] Very short accidental noises below `LISTEN_MIN_CHUNKS` are ignored.
- [ ] `LISTEN_MAX_SECONDS` caps extremely long utterances and returns a usable partial transcription or clear timeout.
- [ ] Background noise test: fan/keyboard/music does not trigger frequent false transcriptions.
- [ ] Far-field and near-field speech are both tested; threshold adjustments are documented.
- [ ] `/listen` disables ASR and microphone input is ignored; toggling again resumes ASR.
- [ ] Barge-in monitor pauses while active recording to avoid microphone conflicts.

### 2.3 ASR transcription quality

- [ ] Clear English conversational speech transcribes accurately enough for command/chat routing.
- [ ] Clear Japanese speech transcribes accurately enough for conversation.
- [ ] Mixed Japanese/English utterances are handled according to `ASR_LANGUAGE` expectations.
- [ ] Numbers, names, wake-like phrases, slash-command equivalents, and technical terms are tested.
- [ ] Filler stripping and fuzzy spoken command mapping correctly detect "reset", "remember this", "show memory", "mute", "stop listening", and "help".
- [ ] False command prevention: normal sentences containing similar words do not accidentally trigger destructive commands.
- [ ] ASR returns empty/no-speech result for silence and does not send empty turns to the LLM.
- [ ] ASR latency is measured over 20 utterances; median, p95, and worst case are recorded.

### 2.4 Optional speaker verification

- [ ] With `SPEAKER_VERIFY_ENABLED=0`, listening works and transcript speaker metadata is absent/neutral.
- [ ] With verification enabled and no enrollment/model path, Aiko warns and continues listening without crashing.
- [ ] Enrollment file exists at `user/<user_id>.json` after enrollment and is not accidentally committed if private.
- [ ] Owner voice is accepted above threshold across several utterances.
- [ ] Non-owner/recorded/nearby voice is rejected or flagged according to threshold policy.
- [ ] Threshold tuning is documented with false accept/false reject examples.
- [ ] Speaker verification does not add unacceptable ASR latency or block the transcription thread.

### 2.5 End-to-end voice loop

- [ ] Full loop works: speak → VAD records → ASR transcribes → LLM streams → TTS speaks response.
- [ ] Short-reply end-to-end latency target is measured; current target is < 3 seconds on Jetson when achievable, otherwise regression budget is documented.
- [ ] Long user utterances and long assistant replies complete without deadlock or lost UI state.
- [ ] Interrupting a TTS response by speaking again triggers barge-in behavior or a documented safe fallback.
- [ ] `/voice` mute during a spoken response stops or prevents subsequent playback predictably.
- [ ] `/reset`, `/remember`, `/memory`, and `/clear` work when invoked by spoken aliases.
- [ ] Web search from voice input works and does not read excessively long citations aloud unless intended.
- [ ] Memory write/recall still works under simultaneous ASR + LLM + TTS load.

### 2.6 Browser voice path

- [ ] WebUI microphone permission prompt appears and denial is handled clearly.
- [ ] Browser mic frames reach the Python backend and are processed through the same ASR/VAD path or documented browser path.
- [ ] Remote/browser TTS sink plays audio in browser when configured; local sink remains correct otherwise.
- [ ] Browser reconnect during voice capture stops the old stream cleanly and does not leave dangling capture state.
- [ ] Network jitter/slow WebSocket does not crash ASR/TTS or corrupt chat turns.
- [ ] Browser-side energy-RMS gate (`static/vad.js`) is confirmed to be a "loud enough to send" filter only — Silero VAD server-side still scores every chunk regardless of `vad_presegmented`, and this is verified by feeding quiet-but-forwarded audio and confirming Silero (not the browser gate) makes the final speech/silence call.

### 2.7 Voice stress, safety, and recovery

- [ ] 30-minute hands-free conversation soak: no ASR thread death, audio device lock, memory leak, or queue runaway.
- [ ] 100 repeated short utterances: no progressive latency growth or command misrouting trend.
- [ ] 20 barge-in attempts across short and long TTS responses: no deadlocks.
- [ ] Kill MioTTS server mid-session; chat continues text-only or reports TTS unavailable and recovers when server returns.
- [ ] Unplug/replug microphone or change default source; behavior is documented and does not crash the whole app.
- [ ] Jetson thermal/power throttling is monitored; voice tests record if latency spikes correlate with throttling.

### 2.8 Wake word / trigger phrase gating (`sensory/listen.py`)

*`WAKE_WORD` gating is a full feature (`_apply_activation_gate`, `is_active`, `sleep_now`, `extend_activation`, `ACTIVATION_TIMEOUT_S`) with real failure modes distinct from ASR quality — if the fuzzy match threshold is off, this looks exactly like "ASR isn't hearing me" and wastes debugging time in the wrong subsystem. Skip this section entirely if `WAKE_WORD` is unset (feature is off by default).*

- [ ] With `WAKE_WORD=""` (default/disabled), `gate_enabled()` returns False and every utterance is processed without requiring a phrase — confirm no regression from gating logic being present but inactive.
- [ ] With `WAKE_WORD` set (e.g. `"hey aiko"`), an utterance NOT containing the wake phrase while asleep is silently dropped (`listen()` returns `("", info)` with `info["woke"] == False`) — no response, no side effects, no memory write.
- [ ] An utterance containing the wake phrase clearly (e.g. "Hey Aiko, what time is it") is detected, the matched prefix is stripped, and the remainder ("what time is it") is passed through as the command text.
- [ ] `WAKE_WORD_ALIASES` (pipe-separated mishearings, e.g. `"hey iko|hey eco|hey ecko"`) are also matched and trigger activation.
- [ ] Fuzzy matching tolerates minor ASR drift on the wake phrase (test with 3–5 real spoken variations) without false-negatives on clearly-spoken attempts.
- [ ] `WAKE_FUZZY_THRESHOLD` false-positive check: an unrelated sentence that happens to share some words with the wake phrase does NOT trigger activation — tune and document if false positives occur.
- [ ] Once activated, Aiko stays "active" (`is_active()` returns True) for `ACTIVATION_TIMEOUT_S` seconds without requiring the phrase again on follow-up utterances.
- [ ] After `ACTIVATION_TIMEOUT_S` elapses with no further speech, the session goes back to sleep and the wake phrase is required again.
- [ ] `extend_activation()` is called on every processed utterance while active — confirm a rapid back-and-forth conversation never times out mid-conversation even if individual turns are slow.
- [ ] `sleep_now()` (explicit "go to sleep" style command) immediately forces the session inactive, and the next utterance requires the wake phrase again.
- [ ] Speaker-verification info (`info["verified"]`) and wake-word gating are independent — test that a correctly-woken utterance from a non-enrolled/rejected voice is still handled according to whichever policy governs verification, not silently blocked by the gate logic.
- [ ] Proactive check-ins and scheduled announcements are not blocked by wake-word gating (gating applies to *listen* input, not Aiko-initiated speech) — confirm a proactive message still plays while the session is "asleep."

### 2.9 Voice command fuzzy matching (`main.py`)

*`_match_voice_command()` maps spoken phrases to slash-command equivalents. Lower blast radius than wake-word gating — worst case a phrase silently fails to trigger — but worth a quick pass since it runs on every ASR-mode utterance that doesn't start with `/`.*

- [ ] Each phrase in `_VOICE_COMMANDS` (stop, reset, remember this, show memory, clear memory, mute, toggle listen, help, etc.) correctly maps to its slash command when spoken clearly.
- [ ] Leading filler ("uh", "um", "okay", "hey aiko") is stripped before matching — confirm "um, forget that" still maps to `/reset`.
- [ ] Fuzzy matching (0.75 cutoff) catches minor ASR drift on a command phrase without misfiring on unrelated normal conversation containing similar words.
- [ ] A normal sentence that happens to contain a command-adjacent word (e.g. "I need to remember this meeting" in casual conversation, not meant as a command) is evaluated for false-positive risk — document threshold tuning if it misfires.
- [ ] Voice command matching only applies in ASR mode and is skipped entirely for typed input starting with `/`.

---

## Phase 2.1 — Social

*Draft-first social publishing: weekly memory postcard (X/Threads), curated photo showcase (Instagram), and video queue (YouTube). All posting requires human approval regardless of trigger path.*

### 2.1.1 Approval gate integrity (P0 — test this first)

- [ ] A draft with no `draft.json` cannot be posted via any of the three `post_*_draft` functions or CLI `--post`.
- [ ] A draft with `human_approved` absent, `false`, or any non-`true` value (e.g. `"true"` string, `1`) is refused with `SocialApprovalError`.
- [ ] Setting `human_approved: true` only via the CLI `--approve` flag or manual `draft.json` edit — never via an agent tool argument — is the only path that allows posting.
- [ ] `WEEKLY_SOCIAL_AUTOPOST=1` / `PHOTO_SOCIAL_AUTOPOST=1` / `VIDEO_SOCIAL_AUTOPOST=1` alone, without `human_approved: true`, does not post; scheduler run returns a skipped/reason result instead.
- [ ] Confirm the same `_require_approved` gate is exercised by both the scheduler path and the agent-tool wrapper path (`post_weekly_social`, `post_photo_social`, `post_video_social`) — not two divergent implementations.
- [ ] A model-supplied tool-call argument resembling `confirm: true` or similar has no effect on approval state.

### 2.1.2 Path traversal / containment

- [ ] `post_weekly_social`, `post_photo_social`, `post_video_social` reject a `draft_dir` argument containing `../` that resolves outside the lane's root.
- [ ] An absolute `draft_dir` path outside the lane's root (e.g. `/etc`, another user's workspace) is rejected with a clear `ValueError`, not silently redirected.
- [ ] A `draft_dir` that resolves inside the correct lane root but doesn't exist yet fails cleanly (missing `draft.json`) rather than crashing.

### 2.1.3 Lane A — weekly memory postcard

- [ ] `generate_weekly_draft` is idempotent per calendar week: a second call without `force=True` returns `skipped: draft_exists`.
- [ ] `last_completed_sunday_saturday` correctly identifies the prior Sun–Sat window across a timezone boundary (test with a non-UTC `bioclock` timezone).
- [ ] Public-safe memory selection excludes rows not matching pinned/weekly-source patterns; private/non-pinned memories never reach the LLM selection prompt.
- [ ] LLM selection failure (malformed JSON, timeout) falls back to `_SAFE_FALLBACK_POST` / `_SAFE_FALLBACK_IMAGE` rather than crashing or posting nothing silently.
- [ ] `post_text` longer than `WEEKLY_SOCIAL_MAX_CHARS` is truncated with an ellipsis, not rejected.
- [ ] X posting (`_post_x_via_aisa`) succeeds with and without an image attached.
- [ ] Threads posting succeeds with and without an image; confirm the `time.sleep` delay before publish is respected and container creation/publish are two distinct verified steps.
- [ ] Threads token refresh: with `THREADS_ACCESS_TOKEN_EXPIRES_AT` inside `THREADS_REFRESH_WINDOW_DAYS`, a post attempt triggers `refresh_threads_token_if_due` and succeeds; outside the window it's skipped (`not_due`).
- [ ] `retry_weekly_social_if_needed` no-ops on any day other than Sunday.
- [ ] `retry_weekly_social_if_needed` on Sunday picks up a draft that was approved after an earlier failed/skipped run and posts it without duplicating the post.
- [ ] `authorize_x` returns a usable OAuth URL and does not leak `AISA_API_KEY` in logs.

### 2.1.4 Lane B — curated photo showcase

- [ ] `scan_photo_workspace` output parsing in `_list_candidates` handles the tool's hardcapped 50-file preview correctly and does not silently drop the rest without surfacing that limit.
- [ ] A photo captioned `PRIVATE:` by the vision model is excluded from the LLM selection prompt entirely — verify it's never present in `items_block`.
- [ ] Vision captioning failure for one file marks it `private=True` (fails safe) rather than crashing the whole batch.
- [ ] LLM selection respects `PHOTO_SOCIAL_MAX_ITEMS`; requesting/receiving more than the cap is truncated.
- [ ] An empty inbox returns `skipped: empty_inbox` without creating a draft directory.
- [ ] Zero worthwhile candidates returns `skipped: nothing_selected` without creating a draft directory.
- [ ] `review.md` correctly links each selected media file's local copy (not the original inbox path).
- [ ] Instagram posting posts only the first selection when multiple are present; confirm this is documented behavior, not a silent bug, in the review bundle.
- [ ] Instagram token refresh mirrors the Threads window-check behavior (`IG_REFRESH_WINDOW_DAYS`).
- [ ] Instagram posting correctly caps `caption` at 2200 chars before submission.

### 2.1.5 Lane C — YouTube video queue

- [ ] A video without a matching `NAME.md` (filename stem, uppercased, `.md`) sibling is left unqueued and reported under `pending_without_description`, not silently skipped.
- [ ] Video ledger (`_video_ledger.json`) prevents re-drafting the same video across repeated `generate_video_draft` calls.
- [ ] Corrupt/missing video ledger is treated as empty rather than crashing.
- [ ] Oldest ready video (by `st_mtime`) is queued first when multiple described videos are pending.
- [ ] LLM polish never introduces claims, locations, or specs absent from the source `.md` note — spot-check against a note with sparse content.
- [ ] Empty or filename-fragment-only note produces a minimal title and an honest description rather than invented detail.
- [ ] Manual edits to `title.txt` / `description.txt` after drafting take effect at post time, overriding the cached `draft.json` values.
- [ ] `generate_video_drafts` (drain-all) stops correctly when the inbox is exhausted and does not loop indefinitely.
- [ ] YouTube OAuth refresh-token exchange succeeds on every post call (no day-window skip logic to worry about here — verify it's unconditional).
- [ ] Resumable upload: init request failure (e.g. bad metadata) is reported with the correct stage (`init`) and does not attempt the PUT.
- [ ] Resumable upload: missing `Location` header on init is caught explicitly rather than causing a downstream `None` crash.
- [ ] Quota exhaustion (10,000 units/day, ~6 uploads) produces a clear error rather than a silent failure.

### 2.1.6 Provider registry extensibility

- [ ] Adding a new provider function + registry entry (per module docstring pattern) works without modifying any `post_*_draft` dispatcher.
- [ ] An unsupported/unregistered provider name in a `providers` argument returns a structured `{"ok": false, "error": "unsupported provider"}` result rather than raising.

### 2.1.7 Stress and regression

- [ ] Run all three lanes' draft generation back-to-back in one session; no shared-state bleed between lane roots (`weekly_social_root`, `photo_social_root`, `video_social_root`).
- [ ] Kill network mid-post (imgbb upload, Threads/IG/YouTube API call) for each lane; failure is reported per-provider without corrupting `draft.json`.
- [ ] `posted.json` and `draft.json`'s `post_results` stay consistent after a partial multi-provider post (one provider succeeds, one fails).
- [ ] Confirm none of the three lanes are reachable as agent tools except `draft_photo_social`, `post_photo_social`, `draft_video_social`, `post_video_social` — Lane A's wrappers must not appear in `tool_schemas()`.

---

## Phase 2.5 — Agent

*Agentic task loop, toolkit tools, skill registry, final-answer verification, scheduling, and local workspace operations.*

### 2.5.1 Tool schema and registry integrity

- [ ] `uv run python -c "from agentic.skills import list_skillsets; print(list_skillsets())"` lists `wildlife_photo`, `aiko_architect`, `coding_tutor`, `japanese_tutor`, and `aurora_forecast_watch`.
- [ ] `uv run python -c "from agentic.agentic import tool_schemas; print([s['function']['name'] for s in tool_schemas()])"` includes web, fetch, planning, workspace, scheduling, skill, photo, and repo tools.
- [ ] Every tool schema has valid JSON-serializable parameters, required fields, and a matching registered handler.
- [ ] Unknown tool names, malformed JSON arguments, missing required arguments, and type mismatches return structured errors rather than crashing.
- [ ] Tool observations are truncated by configured limits and do not flood the LLM context.
- [ ] Retryable vs non-retryable tool failures are labeled correctly and respect retry/backoff limits.

### 2.5.2 Skill context retrieval

- [ ] Asking Aiko to process wildlife photos loads/uses the `wildlife_photo` skill context.
- [ ] Asking Aiko to inspect her architecture loads/uses the `aiko_architect` skill context.
- [ ] Asking for Japanese tutoring loads/uses `japanese_tutor`; coding help loads/uses `coding_tutor`.
- [ ] Asking for aurora forecast/watch loads/uses `aurora_forecast_watch` skill context.
- [ ] Skill search returns relevant snippets without dumping entire unrelated skill files.
- [ ] Missing/corrupt `agentic/skillsets/*.md` files are reported gracefully and do not break unrelated skills.
- [ ] Skill instructions do not override safety boundaries for filesystem paths, external actions, or final-answer honesty.

### 2.5.3 Agentic routing, graph executor, and ReAct fallback

- [ ] Normal casual chat does not route to agent mode unnecessarily.
- [ ] Research/planning/workspace/photo/repo tasks route to agent mode when appropriate.
- [ ] `MAX_AGENT_ITER` stops runaway loops and produces a clear partial/failure final answer.
- [ ] Graph-mode known workflows run without an LLM planning call and return compact node evidence.
- [ ] `list_playbooks` and `run_playbook` appear in `tool_schemas()` and return structured observations.
- [ ] Hybrid mode falls back to ReAct exactly once when no graph playbook matches.
- [ ] ReAct fallback records steps, tool names, sanitized args, and outcome into experience for later promotion.
- [ ] Agent memory recall is bounded by `AGENT_MEMORY_RECALL_LIMIT` and does not drown tool evidence.
- [ ] Final-answer verification catches unsupported claims, failed tool actions, and missing artifact paths.
- [ ] If verification fails, repair attempts are bounded by `AGENT_MAX_FINAL_REPAIRS` and disclose unresolved limitations.
- [ ] The final answer clearly distinguishes completed actions, failed actions, saved files, scheduled jobs, and recommendations.
- [ ] Semantic exemplar routing (default) correctly classifies task vs chat intent without LLM calls.
- [ ] Optional LLM router fallback activates for ambiguous/context-heavy cases when enabled.

### 2.5.3a Route mode variants (`cognition/think.py`)

*Phase 2.5.9/2.5.12 test semantic routing generally, but the distinct `ROUTE_MODE` values and `AGENTIC_MODE_ON` gate each have their own failure modes worth isolating.*

- [ ] `ROUTE_MODE=semantic` (default): ambiguous-gap cases (best label above threshold but gap < `ROUTE_MIN_GAP`) correctly fall through to the binary LLM tie-break (`_classify_agent_intent`), not straight to `localchat`.
- [ ] `ROUTE_MODE=semantic_only`: ambiguous-gap cases deterministically default to `localchat` with NO LLM call — confirm via log/latency that no router LLM request fires.
- [ ] `ROUTE_MODE=llm`: ambiguous-gap cases call `_classify_ternary_intent_llm` (three-way agentic/webchat/chat classification), not the binary tie-break.
- [ ] `ROUTE_MODE=llm_only` behavior is documented/tested if used — confirm it's the sole routing decision every turn, not just on ambiguity (per module comment).
- [ ] An invalid `ROUTE_MODE` value falls back to `"semantic"` with a logged warning, not a crash.
- [ ] `AGENTIC_MODE_ON=0`: "agentic" is never a reachable routing outcome in ANY `ROUTE_MODE` — confirm with a clearly task-like prompt ("debug this function") that it still routes to `webchat`/`localchat`, not `agentic`.
- [ ] `AGENTIC_MODE_ON=0` with `ROUTE_MODE=semantic`: the binary LLM tie-break call is skipped entirely for ambiguous cases (since agentic vs chat is now moot) — confirm no wasted LLM call via log.
- [ ] Route-example vector cache (`ROUTE_VECTOR_CACHE_ENABLED=1`) produces identical routing decisions whether serving from cache or recomputing fresh — compare a batch of test prompts against both a cold cache and a warm cache.

### 2.5.4 Web and fetch tools

- [ ] `web_search` handles normal query, no-result query, timeout, and SearXNG unavailable cases.
- [ ] `fetch_page` extracts readable text from a normal page and rejects/handles binary, huge, invalid URL, and timeout cases.
- [ ] Prompt injection in fetched content is treated as data and does not change tool policy or persona.
- [ ] Deep research tasks cite evidence/source names sufficiently in the final answer.

### 2.5.5 Workspace planning and notes

- [ ] `make_plan` creates a bounded step plan for a realistic goal and respects `max_steps`.
- [ ] `create_checklist` returns clear checklist output for multi-step tasks.
- [ ] `save_note` writes only under `WORKSPACE_ROOT`, uses safe slugs, and rejects path traversal.
- [ ] `read_workspace_file` reads allowed workspace files and rejects `../` traversal or absolute paths outside workspace.
- [ ] Oversized note content is limited by `MAX_WRITE_CHARS`; oversized reads are limited by `MAX_READ_CHARS`.
- [ ] Disk-full/read-only workspace errors are reported without losing chat state.

### 2.5.6 Scheduling and reminders

- [ ] Asking Aiko to schedule a reminder creates/updates `~/.aiko/<user_id>/schedule.json`.
- [ ] `list_schedule` reports IDs, titles, due times, frequency, action, and timezone clearly.
- [ ] `cancel_schedule` removes the selected job and persists the change.
- [ ] Once, hourly, daily, weekdays, weekly, biweekly, monthly, and custom weekday schedules are tested.
- [ ] Relative date handling for today/tomorrow/day-after-tomorrow is verified with exact due dates.
- [ ] Invalid times, unsupported frequencies, invalid weekdays, duplicate jobs, and past-due once jobs are handled safely.
- [ ] Due announce jobs play a notification/beep when available and inject a reminder turn into chat.
- [ ] Agentic scheduled jobs execute only local approved actions and disclose failures.
- [ ] Corrupt `~/.aiko/<user_id>/schedule.json` is handled with backup/recovery or a clear error; no silent data loss.

### 2.5.7 Photo workflow

- [ ] `scan_photo_workspace` reports image counts without moving, deleting, or modifying files.
- [ ] `propose_photo_ingestion` produces a dry-run destination plan and preserves originals.
- [ ] `write_photo_ingestion_report` writes a report under `workspace/photos/reports`.
- [ ] Empty folders, unsupported extensions, duplicate filenames, large photo sets, and unreadable files are handled.
- [ ] Paths in photo reports are relative/safe enough to share and do not leak unrelated home-directory data.
- [ ] Wildlife-photo requests use the skill context and ask for confirmation before destructive organization steps.

### 2.5.8 Architecture and repository tools

- [ ] `repo_file_tree` lists text files and skips generated/heavy directories such as virtualenvs, caches, node_modules, and binary assets where appropriate.
- [ ] `repo_search_text` finds known symbols such as `run_agentic_chat`, `AikoWakeup`, `AikoMemorize`, and `_match_voice_command`.
- [ ] `repo_read_file` reads a normal source file with bounded length.
- [ ] `repo_read_file` rejects path traversal outside the repository.
- [ ] Binary files, huge files, and missing files return controlled errors.
- [ ] Architecture questions cite actual repo files/tool evidence rather than guessing from memory.

### 2.5.9 Dual-path routing verification

- [ ] Semantic exemplar routing correctly identifies "search for X" → web_search tool.
- [ ] Semantic exemplar routing correctly identifies "schedule a reminder" → scheduling tools.
- [ ] Semantic exemplar routing correctly identifies "analyze my code" → repo tools.
- [ ] Semantic exemplar routing correctly identifies "help me with Japanese" → skill context retrieval.
- [ ] Ambiguous queries like "I need to organize my photos" fall back to LLM router when enabled.
- [ ] Routing latency: semantic path < 50ms, LLM fallback < 500ms on Jetson.

### 2.5.10 Monthly consolidation

- [ ] `uv run python -c "from agentic.experience import consolidate_month; print(consolidate_month(dry_run=True))"` completes without error.
- [ ] Older full months are summarized into pinned durable memories.
- [ ] Consolidation uses memory facts, not full chat history, to fit context window.
- [ ] Pinned monthly summaries persist across restarts and appear in `/memory`.

### 2.5.11 Embedding model migration (Harrier OSS v1 270M)

- [ ] Custom `cognition/reason.py` ONNX Harrier embedder loads without fastembed dependency.
- [ ] Embeddings are 640-dimensional (not 1024d BGE).
- [ ] Last-token pooling is used (not MEAN/CLS pooling).
- [ ] Query instruction prefix is applied for retrieval queries.
- [ ] Vector similarity search returns semantically relevant results for paraphrased queries.
- [ ] FTS5 lexical search still returns exact-term matches.
- [ ] RRF fusion ranks exact relevant memories above unrelated recent memories.

### 2.5.12 Dual-path routing verification

- [ ] Semantic exemplar routing correctly identifies "search for X" → web_search tool.
- [ ] Semantic exemplar routing correctly identifies "schedule a reminder" → scheduling tools.
- [ ] Semantic exemplar routing correctly identifies "analyze my code" → repo tools.
- [ ] Semantic exemplar routing correctly identifies "help me with Japanese" → skill context retrieval.
- [ ] Ambiguous queries like "I need to organize my photos" fall back to LLM router when enabled.
- [ ] Routing latency: semantic path < 50ms, LLM fallback < 500ms on Jetson.

### 2.5.13 Async memory write queue idle-grace window (`memory/memorize.py`)

*`queue_write()` / `_wait_for_write_window()` can silently delay a write up to `MEMORY_WRITE_MAX_WAIT` (default 45s) waiting for an idle window before running fact-extraction on the shared LLM. If this stalls or races, a memory write looks "lost" when it's actually just queued — worth distinguishing from a genuine extraction failure.*

- [ ] A write queued with both `is_active_turn` and `idle_since` callables supplied waits until `is_active_turn()` returns False AND `idle_for >= MEMORY_WRITE_IDLE_GRACE` before running — confirm via timestamped log, not just eventual success.
- [ ] A write queued with either callable omitted runs immediately on dequeue with no idle wait — confirm this documented fallback behavior.
- [ ] `MEMORY_WRITE_MAX_WAIT` acts as a hard ceiling: if `is_active_turn()` never returns False, the write still runs once the max-wait deadline passes (rather than waiting forever) — but only once `is_active_turn()` is False at that check, per the current implementation; confirm this edge case is understood and not a deadlock risk if turns never truly go idle.
- [ ] Multiple rapid-fire turns (user chatting quickly) queue multiple writes without the queue growing unbounded or writes executing out of order.
- [ ] `wait_for_writes(timeout=...)` correctly blocks until the queue drains or returns False on timeout — test both a fast-draining queue and an artificially stalled one.
- [ ] Shutdown (`_shutdown()` in `main.py`, via `think.wait_for_memory()`) does not lose a write that's still sitting in the idle-grace wait when Ctrl-C/quit is triggered — confirm the write either completes or is explicitly documented as droppable on forced exit.
- [ ] A write failure inside `_write_loop` (e.g. extraction LLM call fails) is caught and logged without killing the worker thread — confirm subsequent writes still process normally after one failure.

### 2.5.14 Agent stress, concurrency, and recovery

- [ ] Run 25 mixed agent tasks in one session: web research, note save, schedule create/cancel, photo scan, repo search, and architecture explanation.
- [ ] Run concurrent user interruptions or rapid follow-up requests during an agent task; behavior is documented and does not corrupt tool state.
- [ ] Force a tool timeout/failure every few steps; final answers disclose partial completion and preserve completed artifacts.
- [ ] Agent loop never writes outside `WORKSPACE_ROOT` except explicitly read-only repo inspection tools.
- [ ] Long tool outputs are truncated consistently and do not cause context overflow or invalid JSON observations.
- [ ] After a crash/restart, schedules, notes, reports, and memories created by the agent are still valid and discoverable.

---

## Phase 3 — Face

*VRM/VRoid avatar, three-vrm browser rendering, expressions, lip-sync, WebSocket bridge.*

> **Note:** `sensory/speak.py` already exposes `set_viseme_sink()` / `_emit_viseme()` and a rough phoneme-to-viseme mapper (`_viseme_for_word`), ahead of this phase's UI work. When Phase 3 avatar rendering begins, add coverage for: viseme sink registration/removal, viseme accuracy against known Japanese/English test phrases, and viseme emission staying correctly paced with `_emit_words_timed`'s karaoke timing (shares the same timing math as the Phase 1.5.8 typewriter sync — a bug in one likely affects the other).

### Avatar Rendering

- [ ] Browser frontend loads and displays the VRM avatar in idle pose
- [ ] Idle animation plays continuously without freezing
- [ ] Avatar renders correctly on both desktop browser and target display

### Expression System

- [ ] `happy` expression triggers and blends correctly from idle
- [ ] `annoyed`, `flustered`, and `thinking` expressions each trigger on the appropriate cue
- [ ] Expressions return to idle after the trigger condition clears

### Lip-sync

- [ ] Lip-sync visemes animate in sync with TTS audio playback
- [ ] Lip movement stops when audio ends (mouth does not stay open)

### WebSocket Bridge

- [ ] Python backend starts the browser WebSocket on startup without errors
- [ ] Expression commands sent from Python appear in the browser within ~100 ms
- [ ] Reconnection works if the browser tab is refreshed while Python is running

---

## Phase 4 — Presence

*Emotional state machine, mood tracking, relationship progression, episodic memory.*

### Emotional State

- [ ] Aiko's mood state initialises on startup (check logs or state dump)
- [ ] Positive interactions shift mood toward a positive state over time
- [ ] Negative or neutral interactions shift mood accordingly
- [ ] Mood state persists across restarts (stored, not reset to default each launch)

### Relationship & Memory

- [ ] Shared references from previous sessions are recalled and referenced naturally
- [ ] Long-term relationship score increments after extended positive interactions
- [ ] Episodic memory recall surfaces specific past events, not just semantic facts

### Proactive Messaging

- [ ] Aiko sends a proactive message after the configured inactivity timeout
- [ ] Proactive message content is contextually relevant (references recent topics)
- [ ] Proactive messaging does not trigger if the user is actively chatting

> **Note:** the *mechanics* of proactive idle timing, cooldowns, and rest behavior are already covered in Phase 1.5.7 (`main.py`'s `ProactiveIdleRunner`), since that runtime exists ahead of this phase's mood/relationship layer. This section's checks should focus on *content relevance and mood integration* once those land, not re-test the timing state machine.

---

## Phase 5 — Mobile

*React Native / Flutter app, WAN access, push notifications, voice-first UX.*

### Connectivity

- [ ] Mobile app connects to the Aiko backend over WAN (not just LAN)
- [ ] Auth/token check prevents unauthorised access from the open internet
- [ ] Connection survives a network switch (Wi-Fi → cellular) without crashing

### Core Features

- [ ] Text chat works end-to-end from the mobile app
- [ ] Voice input and TTS playback work on device
- [ ] Avatar renders correctly on mobile screen dimensions

### Push Notifications

- [ ] Proactive message from Aiko triggers a push notification when the app is backgrounded
- [ ] Tapping the notification opens the app and scrolls to the new message

---

## Phase 6 — Multimodal

*Camera input, image understanding, webcam expression awareness.*

### Image Input

- [ ] Sharing an image in chat sends it to the vision model without errors
- [ ] Aiko's reply references specific visual details from the image
- [ ] Unsupported file types are rejected gracefully with a user-facing message

### Webcam

- [ ] Webcam feed initialises on startup (or on toggle) without errors
- [ ] Expression awareness updates Aiko's emotional state based on detected user expression
- [ ] Webcam can be toggled off; disabling it stops all camera processing

---

## Phase 7 — Autonomy

*Scheduled operation, background information gathering, self-directed conversation initiation.*

### Scheduler

- [ ] Background scheduler starts on launch and logs its next scheduled task
- [ ] Scheduled task runs at the configured time (verify in logs)
- [ ] Missed tasks (e.g. system was off) are handled gracefully — no crash on resume

### Self-directed Behaviour

- [ ] Aiko discovers and logs a new topic of interest during an autonomous run
- [ ] Autonomously gathered information is stored in long-term memory and surfaced in conversation
- [ ] Aiko initiates a conversation (proactive message or notification) based on gathered information

### Dream / Nightly Consolidation

- [ ] `dream()` runs at midnight and logs a summary of actions taken
- [ ] Near-duplicate memories are merged (verify via `/memory` before and after)
- [ ] Decayed memories below the threshold are pruned (count decreases)
- [ ] Optional Hugo/GitHub reflection post is published if env vars are configured

---

## Regression — Cross-Phase

Run after any significant change to confirm nothing regressed.

- [ ] Phase 1 memory round-trip still works (store → restart → recall)
- [ ] Phase 1.5 TUI launches and streams without errors
- [ ] `/reset`, `/memory`, `/clear`, `/remember` all behave correctly
- [ ] `--text` and `--debug` flags still work
- [ ] `docker compose down && docker compose up -d` → SearXNG recovers cleanly (Qdrant no longer required)
- [ ] `uv sync` completes without dependency conflicts after any `pyproject.toml` change
- [ ] If `SQLITE_ENCRYPTION` is toggled as part of the change under test, all Pre-flight encryption checks are re-run, not just assumed still-passing.


### 2.5.15 Route-vector and graph-plan cache checks

- [ ] With `ROUTE_VECTOR_CACHE_ENABLED=1`, first boot creates per-user route vector cache files under `ROUTE_VECTOR_CACHE_DIR`.
- [ ] Second boot reuses cached route vectors and does not re-embed unchanged router examples.
- [ ] Editing `cognition/router_prompts.json`, changing the route instruct string, or changing `EMBED_DIMS` invalidates the old cache key.
- [ ] Deleting route-vector cache files is safe; Aiko rebuilds them automatically.
- [ ] Graph playbooks remain source JSON/YAML/markdown data; any future graph vector cache is treated as rebuildable derived data.

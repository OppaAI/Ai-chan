[← Back to README](../README.md)
# Aiko-chan 愛子ちゃん — Industrial Test Checklist

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

### Service health

- [ ] `docker compose ps` shows SearXNG running and no unintended legacy Qdrant dependency.
- [ ] `curl "http://localhost:8081/search?q=test&format=json"` returns JSON results within 3 seconds.
- [ ] `curl http://localhost:8080/v1/models` returns JSON containing the configured `LLM_MODEL` alias.
- [ ] `curl http://localhost:8001/health` returns `{"status":"ok"}` when TTS/voice tests are in scope.
- [ ] `uv run python -c "import sqlite_vec, fastembed, sherpa_onnx, silero_vad; print('OK')"` prints `OK`.
- [ ] `parec --version` works on the target voice machine; PulseAudio/PipeWire has an active default source.

### Baseline observability

- [ ] Start one test run with `--debug` and confirm memory hits/tool routing details are visible without exposing sensitive payloads.
- [ ] Capture idle resource baseline for 60 seconds: CPU, RAM, swap, GPU/VRAM if available, disk free space, and open file descriptors.
- [ ] Confirm logs include subsystem boot boundaries for think, memory, speak, listen, and UI.
- [ ] Confirm Ctrl-C/Ctrl-D cleanly exits, drains memory work, and restores terminal state.

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

### 1.4 Memory: sqlite-vec + fastembed CRUD

- [ ] After several exchanges, `/memory` shows relevant stored memories and does not show empty placeholder rows.
- [ ] Restart Aiko and ask about a previously discussed topic; recall surfaces the relevant persisted memory.
- [ ] `/remember` pins the last exchange; after restart `/memory` shows the pinned item remains.
- [ ] `/clear` wipes memories; `/memory` is empty afterward and no stale recall appears in the next turn.
- [ ] `--clear-mem` wipes memories and exits cleanly without launching the UI.
- [ ] DB file exists after first write: `ls -lh "$SQLITE_MEMORY_PATH"`.
- [ ] `uv run python -c "from core.memorize import AikoMemorize; m=AikoMemorize(); print(m.dream(dry_run=True))"` completes without error.
- [ ] Corrupt/locked DB simulation is handled safely: Aiko reports the memory problem and continues chat if possible, without overwriting unrelated files.
- [ ] Duplicate memory pressure test: repeat the same fact 20 times; recall does not become dominated by redundant near-identical entries.
- [ ] Unicode memory test: store Japanese, Korean, emoji, and mixed punctuation; recall and display remain readable.
- [ ] Privacy check: `/memory` does not expose secrets from environment variables or unrelated workspace files.

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

*Curses TUI, shared streaming architecture, browser WebUI bridge, persona display, and MioTTS integration.*

### 1.5.1 Curses TUI layout and input

- [ ] `uv run python main.py --tui --text` launches the full-screen curses UI and restores the terminal after exit.
- [ ] Chat panel, architecture/status panels, identity area, and input field render in correct positions at 80x24, 120x40, and a large terminal size.
- [ ] Terminal resize during generation does not crash or leave garbled output.
- [ ] Typing, backspace, paste, Enter submit, arrow/navigation keys, and slash commands work reliably.
- [ ] `/help` renders the command list without overflowing or breaking subsequent draws.
- [ ] Rapid input submission while streaming is rejected, queued, or handled predictably; it does not corrupt the active assistant message.
- [ ] Long assistant messages scroll correctly; the most recent content remains reachable/readable.
- [ ] ANSI escape sequences or malicious terminal control characters in model output do not damage the terminal.

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

- [ ] `uv run python main.py --webui --text` prints the browser URL and serves `webui/static/index.html`.
- [ ] Browser WebSocket connects, receives boot/status events, and sends text chat end-to-end.
- [ ] Chat messages, streamed tokens, stream commit events, vitals, voice state updates, and errors appear in browser developer tools as expected.
- [ ] `webui/static/assets/Aiko.vrm` loads or fails with a clear visible error.
- [ ] Browser refresh/reconnect does not crash the Python backend or duplicate stale sessions uncontrollably.
- [ ] Multiple browser tabs are tested: behavior is documented, and one tab cannot corrupt backend state for another.
- [ ] Static file path traversal attempts fail; only intended WebUI assets are served.

### 1.5.5 MioTTS text-to-speech

- [ ] `curl http://localhost:8001/health` confirms the MioTTS server before voice playback tests.
- [ ] `/voice` toggles TTS on; the next assistant response plays audio.
- [ ] `/voice` toggles TTS off; later responses do not play audio.
- [ ] Background TTS warmup reduces first-audio latency; cold and warm timings are recorded.
- [ ] Audio plays through the intended output device; `python core/speak.py --devices` lists available devices.
- [ ] `sanitize_for_tts` removes markdown, emoji noise, and unsafe symbols without making ordinary Japanese/English text unintelligible.
- [ ] Long responses are split and played completely without cutting off mid-sentence.
- [ ] TTS server unavailable, HTTP error, invalid preset, and invalid device are all surfaced without crashing chat.
- [ ] TTS playback during shutdown stops cleanly and releases the audio device.

### 1.5.6 Phase 1.5 stress and soak

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

### 2.7 Voice stress, safety, and recovery

- [ ] 30-minute hands-free conversation soak: no ASR thread death, audio device lock, memory leak, or queue runaway.
- [ ] 100 repeated short utterances: no progressive latency growth or command misrouting trend.
- [ ] 20 barge-in attempts across short and long TTS responses: no deadlocks.
- [ ] Kill MioTTS server mid-session; chat continues text-only or reports TTS unavailable and recovers when server returns.
- [ ] Unplug/replug microphone or change default source; behavior is documented and does not crash the whole app.
- [ ] Jetson thermal/power throttling is monitored; voice tests record if latency spikes correlate with throttling.

---

## Phase 2.5 — Agent

*Agentic task loop, toolkit tools, skill registry, final-answer verification, scheduling, and local workspace operations.*

### 2.5.1 Tool schema and registry integrity

- [ ] `uv run python -c "from core.skills import list_skillsets; print(list_skillsets())"` lists `wildlife_photo`, `aiko_architect`, `coding_tutor`, `japanese_tutor`, and `aurora_forecast_watch`.
- [ ] `uv run python -c "from core.agentic import tool_schemas; print([s['function']['name'] for s in tool_schemas()])"` includes web, fetch, planning, workspace, scheduling, skill, photo, and repo tools.
- [ ] Every tool schema has valid JSON-serializable parameters, required fields, and a matching registered handler.
- [ ] Unknown tool names, malformed JSON arguments, missing required arguments, and type mismatches return structured errors rather than crashing.
- [ ] Tool observations are truncated by configured limits and do not flood the LLM context.
- [ ] Retryable vs non-retryable tool failures are labeled correctly and respect retry/backoff limits.

### 2.5.2 Skill context retrieval

- [ ] Asking Aiko to process wildlife photos loads/uses the `wildlife_photo` skill context.
- [ ] Asking Aiko to inspect her architecture loads/uses the `aiko_architect` skill context.
- [ ] Asking for Japanese tutoring loads/uses `japanese_tutor`; coding help loads/uses `coding_tutor`.
- [ ] Skill search returns relevant snippets without dumping entire unrelated skill files.
- [ ] Missing/corrupt `SKILL.md` files are reported gracefully and do not break unrelated skills.
- [ ] Skill instructions do not override safety boundaries for filesystem paths, external actions, or final-answer honesty.

### 2.5.3 Agentic routing and ReAct loop

- [ ] Normal casual chat does not route to agent mode unnecessarily.
- [ ] Research/planning/workspace/photo/repo tasks route to agent mode when appropriate.
- [ ] `MAX_AGENT_ITER` stops runaway loops and produces a clear partial/failure final answer.
- [ ] Agent memory recall is bounded by `AGENT_MEMORY_RECALL_LIMIT` and does not drown tool evidence.
- [ ] Final-answer verification catches unsupported claims, failed tool actions, and missing artifact paths.
- [ ] If verification fails, repair attempts are bounded by `AGENT_MAX_FINAL_REPAIRS` and disclose unresolved limitations.
- [ ] The final answer clearly distinguishes completed actions, failed actions, saved files, scheduled jobs, and recommendations.

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

- [ ] Asking Aiko to schedule a reminder creates/updates `workspace/schedule.json`.
- [ ] `list_schedule` reports IDs, titles, due times, frequency, action, and timezone clearly.
- [ ] `cancel_schedule` removes the selected job and persists the change.
- [ ] Once, hourly, daily, weekdays, weekly, biweekly, monthly, and custom weekday schedules are tested.
- [ ] Relative date handling for today/tomorrow/day-after-tomorrow is verified with exact due dates.
- [ ] Invalid times, unsupported frequencies, invalid weekdays, duplicate jobs, and past-due once jobs are handled safely.
- [ ] Due announce jobs play a notification/beep when available and inject a reminder turn into chat.
- [ ] Agentic scheduled jobs execute only local approved actions and disclose failures.
- [ ] Corrupt `workspace/schedule.json` is handled with backup/recovery or a clear error; no silent data loss.

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

### 2.5.9 Agent stress, concurrency, and recovery

- [ ] Run 25 mixed agent tasks in one session: web research, note save, schedule create/cancel, photo scan, repo search, and architecture explanation.
- [ ] Run concurrent user interruptions or rapid follow-up requests during an agent task; behavior is documented and does not corrupt tool state.
- [ ] Force a tool timeout/failure every few steps; final answers disclose partial completion and preserve completed artifacts.
- [ ] Agent loop never writes outside `WORKSPACE_ROOT` except explicitly read-only repo inspection tools.
- [ ] Long tool outputs are truncated consistently and do not cause context overflow or invalid JSON observations.
- [ ] After a crash/restart, schedules, notes, reports, and memories created by the agent are still valid and discoverable.

---

## Phase 3 — Face

*VRM/VRoid avatar, three-vrm browser rendering, expressions, lip-sync, WebSocket bridge.*

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

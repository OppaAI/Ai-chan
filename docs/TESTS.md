[← Back to README](../README.md)
# Aiko-chan 愛子ちゃん — Test Checklist

Manual smoke-test checklist for each phase. Run the relevant section after installing or after making changes to that phase's components. Check off each item before marking a phase stable.

---

## Pre-flight — Stack Health

Run before any phase tests. All items must pass.

- [ ] `curl "http://localhost:8081/search?q=test&format=json"` returns JSON results
- [ ] `curl http://localhost:11434/api/tags` returns a JSON list containing your configured model
- [ ] `curl http://localhost:8001/health` returns `{"status":"ok"}` (voice mode only)
- [ ] `uv run python -c "import sqlite_vec; import fastembed; print('OK')"` prints `OK`
- [ ] `docker compose ps` shows `searxng` as `running`
- [ ] SQLite memory DB path exists and is on persistent storage (not `/tmp`): `ls -lh $SQLITE_MEMORY_PATH`

---

## Phase 1 — Soul

*CLI chatbot, Ollama inference, mem0 + Qdrant memory, web search.*

### Ollama / LLM

- [ ] Aiko launches without errors in `--text` mode
- [ ] First message receives a streamed response (tokens appear progressively, not all at once)
- [ ] Response is coherent and matches the persona defined in `persona/soul.md`
- [ ] `/reset` clears short-term context; next reply has no memory of the previous exchange
- [ ] `/think <question>` returns a higher-quality reasoning response and suppresses raw `<think>` tags from output

### Memory (sqlite-vec + fastembed)

> Memory backend was rewritten in Phase 2 — these tests verify the current sqlite-vec implementation,
> not the original mem0 + Qdrant stack.

- [ ] After a few exchanges, `/memory` prints stored memories (not empty)
- [ ] Restarting Aiko and asking about a previously discussed topic surfaces a relevant memory
- [ ] `/remember` pins the last exchange; restarting and running `/memory` shows it is still present
- [ ] `/clear` wipes all memories; `/memory` returns empty afterward
- [ ] `--clear-mem` flag wipes memories and exits cleanly without launching the TUI
- [ ] DB file exists at `SQLITE_MEMORY_PATH` after first memory write: `ls -lh ~/.aiko/memory.db`
- [ ] `dream()` dry-run completes without error: `uv run python -c "from core.memorize import AikoMemorize; m = AikoMemorize(); print(m.dream(dry_run=True))"`

### Web Search

- [ ] `/web what is the current version of Python` returns a grounded answer citing search results
- [ ] A question that naturally triggers search (e.g. "what happened in the news today") receives a search-grounded reply rather than a hallucinated one
- [ ] Search results are filtered — Aiko does not dump raw SearXNG JSON into the reply

---

## Phase 1.5 — Stream

*Curses TUI, streaming architecture, Kokoro/MioTTS TTS, persona system.*

### TUI

- [ ] Full-screen curses UI launches and fills the terminal without layout errors
- [ ] Chat panel, architecture panel, and status areas render in the correct positions
- [ ] Typing input appears in the input field; submitting with Enter sends the message
- [ ] Streamed LLM tokens appear in the chat panel incrementally as they arrive
- [ ] TUI resizes gracefully when the terminal window is resized (no crash, no garbled output)
- [ ] `/help` renders the command list inside the TUI without breaking layout
- [ ] Exiting with `/quit` or `/exit` restores the terminal cleanly (no leftover curses artifacts)

### Persona & Identity

- [ ] `persona/soul.md` loads without error on startup (check logs)
- [ ] `persona/identity.md` banner and ASCII art render correctly in the identity panel
- [ ] Aiko's tone and personality match `soul.md` across multiple turns

### Streaming Pipeline

- [ ] LLM response streaming begins within ~1 second of submitting a message (warm model)
- [ ] Background LLM warmup on startup eliminates cold-start delay on the first real message
- [ ] Memory writes do not block the streaming response (non-blocking queue worker)

### TTS — MioTTS

- [ ] `/voice` toggle enables TTS; spoken audio plays for the next assistant response
- [ ] `/voice` toggle again disables TTS; no audio plays for subsequent responses
- [ ] Background TTS warmup on startup eliminates cold-start audio delay
- [ ] Audio plays through the correct output device (check PulseAudio default sink on Jetson)
- [ ] Long responses play completely without cutting off mid-sentence

---

## Phase 2 — Voice

*ReazonSpeech K2 ASR, Silero VAD, hands-free talk mode.*

### Memory Backend Migration (sqlite-vec)

- [ ] No Qdrant container is required — `docker compose ps` shows only `searxng`
- [ ] Memory writes complete without OOM errors under concurrent ASR + LLM load (`jtop` VRAM stays stable)
- [ ] KNN + FTS5 RRF recall returns relevant results: run `--debug` and confirm memory hits appear each turn
- [ ] `cleanup()` runs on startup and logs `deleted=N, kept=N` without error

### ASR — ReazonSpeech K2

- [ ] Aiko launches in full voice mode (no `--text` flag) without errors
- [ ] Speaking clearly into the microphone produces a transcription in the chat panel
- [ ] Transcription accuracy is acceptable for normal conversational speech
- [ ] Non-speech background noise does not trigger spurious transcriptions (VAD is active)

### VAD — Silero

- [ ] Speaking starts recording; silence stops recording without manual key press
- [ ] Short pauses mid-sentence do not prematurely cut off the utterance
- [ ] `/listen` toggle disables ASR; microphone input is ignored
- [ ] `/listen` toggle again re-enables ASR; microphone input resumes

### End-to-End Voice Loop

- [ ] Full loop works: speak → transcribe → LLM response streams → TTS speaks the reply
- [ ] Loop completes in an acceptable latency (target: < 3 s on Jetson for short replies)
- [ ] Interrupting a TTS response and speaking again does not deadlock the pipeline

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

- [ ] Python backend connects to the browser WebSocket on startup without errors
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

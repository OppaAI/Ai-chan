[← Back to README](../README.md)
# Aiko-chan 愛子ちゃん — Roadmap

Aiko-chan is built in phases. Each phase is a self-contained capability layer that runs on top of the previous one. Phases are shipped when stable, not on a fixed schedule.

---

## Phase 1 — Soul ✅
![Phase 1](../assets/phase-1.0.png)

*The foundation. A working local AI companion you can have a real conversation with.*

**Goal:** Prove the full memory + inference + search loop works on constrained hardware.

| Feature | Status |
|---|---|
| CLI chatbot architecture | ✅ Done |
| Local inference via Ollama | ✅ Done |
| Persistent memory — mem0 + Qdrant | ✅ Done |
| Async (non-blocking) memory writes | ✅ Done |
| Web search integration via SearXNG | ✅ Done |

---

## Phase 1.5 — Stream ✅
![Phase 1.5](../assets/phase-1.5.png)

*Make the experience feel alive. Replace the CLI with a full curses TUI and overhaul the streaming pipeline.*

**Goal:** Real-time token streaming, decoupled TTS, and a cyberpunk terminal interface.

| Feature | Status |
|---|---|
| Full-screen curses TUI with cyberpunk ASCII interface | ✅ Done |
| Streaming inference architecture overhaul | ✅ Done |
| Decoupled LLM → TTS pipeline | ✅ Done |
| Callback-based response streaming | ✅ Done |
| Realtime speech synthesis via MioTTS | ✅ Done |
| Background LLM warmup — eliminates cold-start latency | ✅ Done |
| Background TTS warmup — eliminates cold-start latency | ✅ Done |
| Soul persona system (`persona/soul.md`) | ✅ Done |
| Identity metadata and character framework (`persona/identity.md`) | ✅ Done |
| Architectural renaming (`brain → think`, `memory → memorize`) | ✅ Done |
| Non-blocking memory queue worker | ✅ Done |
| Search output filtering and instruction refinement | ✅ Done |
| Jetson AI Lab dependency migration | ✅ Done |

---

## Phase 2 — Voice 🔲
![Phase 2](../assets/phase-2.0.png)

*Go fully hands-free. Aiko listens and speaks without any keyboard involvement.*

**Goal:** A complete voice loop — speak, transcribe, respond, synthesise — running in real time on the Jetson.

> **Memory backend migration:** mem0 + Qdrant were replaced in this phase with a custom
> sqlite-vec backend (Ollama extraction + fastembed embeddings + RRF dual-path retrieval).
> Qdrant caused OOM crashes on the Jetson Orin Nano under concurrent ASR + LLM + memory load.
> The new backend is fully serverless — no Docker containers required for memory.

| Feature | Status |
|---|---|
| **Memory backend rewrite — sqlite-vec + fastembed (custom, no server)** | ✅ Done |
| Microphone capture with SenseVoice via sherpa-onnx | 🔲 Planned |
| Voice Activity Detection via Silero VAD | 🔲 Planned |
| Interactive Talk mode (hands-free conversation) | 🔲 Planned |
| Interrupt handling — speak over Aiko mid-response | 🔲 Planned |
| Latency target: < 3 s end-to-end on Jetson Orin Nano | 🔲 Planned |
| TTS voice/runtime decision — MioTTS active; Kokoro/RealtimeTTS removed | ✅ Done |


### Voice backend trial ledger

| Area | Tried | Current decision | Reason |
|---|---|---|---|
| TTS | Kokoro, RealtimeTTS | Removed | OOM/latency/quality tradeoffs on Jetson; kept archived experiments only |
| TTS | MioTTS | Active | Best current voice runtime for Aiko's local stack |
| ASR | faster-whisper prototype | Removed from active runtime | Useful prototype, but heavier and less aligned with current Jetson constraints |
| ASR | ReazonSpeech K2 | Removed from active runtime | Kept as trial/archive; current listener moved to SenseVoice via sherpa-onnx |
| ASR | SenseVoice via sherpa-onnx + Silero VAD | Active | CPU-friendly int8 ONNX path with multilingual support and stable VAD gating |

---

## Phase 2.5 — Agent 🔲

*Give Aiko a real task layer. Skills describe repeatable workflows; toolkit modules provide safe executable actions.*

**Goal:** Let Aiko use predefined skillsets, local tools, schedules, and workspace artifacts to complete repeatable tasks without re-explaining every step.

| Feature | Status |
|---|---|
| ReAct-style task loop with tool dispatch | ✅ Done |
| Persistent schedule jobs with `action=agentic` | ✅ Done |
| `core/toolkit/` focused tool modules | ✅ Done |
| `skills/<id>/SKILL.md` workflow registry | ✅ Done |
| Skill-context retrieval in agentic mode | ✅ Done |
| Initial wildlife-photo and Aiko-architecture skills | ✅ Done |
| Safer code-edit/review workflow for self-improvement | 🔲 Planned |
| Optional MCP wrappers for stable long-running tools | 🔲 Planned |
| 24/7 worker/watchers for queues and folders | 🔲 Planned |

---

## Phase 3 — Face 🔲

*Give Aiko a visible presence. A VRM avatar that reacts to the conversation in real time.*

**Goal:** A browser-rendered anime avatar with expressions and lip-sync driven by live TTS audio.

| Feature | Status |
|---|---|
| VRM / VRoid avatar support | 🔲 Planned |
| Browser-based rendering via `@pixiv/three-vrm` | 🔲 Planned |
| WebSocket bridge — Python backend ↔ browser frontend | 🔲 Planned |
| Expression system — idle, happy, annoyed, flustered, thinking | 🔲 Planned |
| Lip-sync driven by generated speech audio | 🔲 Planned |
| Smooth expression blending and transition | 🔲 Planned |
| Idle animation loop | 🔲 Planned |
| Real-time avatar interaction (expression reacts to conversation tone) | 🔲 Planned |

---

## Phase 4 — Presence 🔲

*Make the relationship feel real. Aiko has a persistent emotional state and remembers the arc of your history together.*

**Goal:** Mood that carries across sessions, relationship depth that grows over time, and a companion that reaches out to you — not just the other way around.

| Feature | Status |
|---|---|
| Persistent emotional state machine | 🔲 Planned |
| Mood tracking across conversations | 🔲 Planned |
| Mood influences tone, word choice, and expression | 🔲 Planned |
| Long-term relationship progression score | 🔲 Planned |
| Shared references and inside jokes | 🔲 Planned |
| Episodic memory recall — specific past events, not just facts | 🔲 Planned |
| Context-aware personality evolution over time | 🔲 Planned |
| Proactive messaging — Aiko reaches out after inactivity | 🔲 Planned |

---

## Phase 5 — Mobile 🔲

*Take Aiko off the desk. Full companion experience from anywhere on your phone.*

**Goal:** A polished mobile app with voice-first UX, WAN connectivity, and push notifications for proactive messages.

| Feature | Status |
|---|---|
| Mobile application (React Native or Flutter) | 🔲 Planned |
| WAN access — connect from anywhere, not just LAN | 🔲 Planned |
| Auth layer — token-based, prevents open-internet abuse | 🔲 Planned |
| Push notifications for proactive messages | 🔲 Planned |
| Voice-first user experience on device | 🔲 Planned |
| Avatar integration on mobile screen | 🔲 Planned |
| Network resilience — survives Wi-Fi ↔ cellular handoff | 🔲 Planned |

---

## Phase 6 — Multimodal 🔲

*Give Aiko eyes. She can see what you share with her and read your expression in real time.*

**Goal:** Image input for shared context, webcam for expression-awareness, visual understanding woven into conversation.

| Feature | Status |
|---|---|
| Image input — share images directly in chat | 🔲 Planned |
| Vision model integration for image understanding | 🔲 Planned |
| Webcam feed for real-time user expression detection | 🔲 Planned |
| Expression-aware emotional state updates | 🔲 Planned |
| Visual context woven naturally into responses | 🔲 Planned |
| Webcam toggle — on/off without restarting | 🔲 Planned |
| Graceful handling of unsupported file types | 🔲 Planned |

---

## Phase 7 — Autonomy 🔲

*Let Aiko run on her own. She gathers information, develops interests, and starts conversations — you don't have to.*

**Goal:** A companion that operates independently on a schedule, builds her own knowledge base, and initiates contact based on what she has discovered.

| Feature | Status |
|---|---|
| Scheduled independent operation | 🔲 Planned |
| Background information gathering and topic discovery | 🔲 Planned |
| Self-directed exploration of persistent interests | 🔲 Planned |
| Autonomous conversation initiation | 🔲 Planned |
| Develops and evolves opinions over time | 🔲 Planned |
| Optional social media presence | 🔲 Planned |
| Autonomous content posting | 🔲 Planned |

---

## Nightly Dream Pipeline — Ongoing

The `dream()` consolidation system runs across all phases and improves continuously.

| Feature | Status |
|---|---|
| Midnight scheduler | ✅ Done |
| Salient memory boost | ✅ Done |
| Near-duplicate merging by vector similarity | ✅ Done |
| Decayed memory pruning | ✅ Done |
| Hugo + GitHub daily reflection publishing | ✅ Done |
| Cross-session memory coherence improvements | 🔲 Ongoing |
| Dream-driven personality drift (subtle, long-term) | 🔲 Phase 4+ |

---

## Notes

- Phases are additive — each one requires the previous to be stable.
- Phase numbering is mostly fixed; half-phases such as 1.5 and 2.5 are used when a major enabling layer belongs between two visible product phases.
- Hardware target throughout: **Jetson Orin Nano** (8 GB), with x86 as secondary.
- This roadmap reflects the Aiko-chan standalone project. The broader cognitive architecture lives in [GRACE / AuRoRA](https://github.com/OppaAI/AGi).

[← Back to README](../README.md)
# Aiko-chan アイコちゃん — Roadmap

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

## Phase 2 — Voice ✅
![Phase 2](../assets/phase-2.0.png)

*Make Aiko usable as a voice companion, not just a typed chatbot.*

**Goal:** A local-first voice loop — listen, detect speech, transcribe, respond, and speak back — that runs on the Jetson with stable memory under ASR + LLM + TTS load.

> **Memory backend migration:** Phase 2 replaced mem0 + Qdrant with a custom
> sqlite-vec + llama.cpp backend using local SQLite storage, KNN vector search,
> FTS5 lexical search, and Reciprocal Rank Fusion retrieval.
> Qdrant caused OOM crashes on the Jetson Orin Nano when ASR, LLM inference,
> TTS, and memory were active together.
> The new backend is serverless and keeps memory local — no Qdrant container,
> no mem0 runtime, and no Docker dependency for memory.
>
> **Current embedding note:** Aiko now uses a custom `cognition/reason.py` ONNX
> llama.cpp embedder instead of fastembed. Harrier OSS v1 270M is decoder-only
> and needs last-token pooling, while fastembed custom registration only
> exposed `PoolingType.MEAN`/CLS-style pooling for this path.

| Feature | Status |
|---|---|
| **Memory backend rewrite — sqlite-vec + fastembed (custom, no server)** | ✅ Done |
| Embedding model migration — BGE v1.5 → Harrier OSS v1 270M for newer 640d vectors and better expected semantic separation | ✅ Done |
| fastembed removal — custom Harrier ONNX embedder with last-token pooling | ✅ Done |
| KNN + FTS5 + RRF memory retrieval | ✅ Done |
| Monthly memory consolidation — older full months summarized into pinned durable memories | ✅ Done |
| Microphone capture via PulseAudio `parec` | ✅ Done |
| ASR via SenseVoice + sherpa-onnx | ✅ Done |
| Voice Activity Detection via Silero VAD | ✅ Done |
| Interactive Talk mode — local/TUI hands-free conversation | ✅ Done |
| Spoken command aliases in ASR mode | ✅ Done |
| Interrupt handling / barge-in — speak over Aiko mid-response | 🟡 Implemented — testing ongoing |
| Optional owner voice verification via sherpa-onnx speaker embeddings | ✅ Done |
| TTS runtime decision — MioTTS active; Kokoro/RealtimeTTS removed | ✅ Done |
| MioTTS HTTP client + local sounddevice playback | ✅ Done (*OOM issue)|
| Remote/browser TTS audio sink for WebUI playback | ✅ Done |
| Browser/WebUI microphone streaming into ASR/VAD pipeline | ✅ Done |
| Staged TTS/ASR/VAD warmup during boot | ✅ Done |
| Latency target: ~3s end-to-end on Jetson Orin Nano |  ✅ Done (*3-4s for short normal chat)| |
| Wake word to activate system from Idle Mode |  🔲 Proposed |
| Trigger phrase (on top of Speaker Verification) to increase security |  🔲 Proposed |

### Voice backend trial ledger

| Area | Tried | Current decision | Reason |
|---|---|---|---|
| TTS | MioTTS | ✅Active | Best current voice runtime for Aiko's local stack; Able to speak Japanese and English in a same sentence |
| TTS | XTTSv2 via CoquiTTS, RealtimeTTS | ❌Removed | A bit obsolete due to depreciation in dependencies; Much slower to inference |
| TTS | Kokoro, RealtimeTTS | ❌Removed | Slightly robotic voice quality; Japanese voice speaking English non-understandable |
| TTS | PocketTTS, RealtimeTTS | ❌Removed | A bit too heavy in RAM usage in Jetson Orin Nano |
| ASR | SenseVoice via sherpa-onnx + Silero VAD | ✅Active | CPU-friendly int8 ONNX path with multilingual support and stable VAD gating |
| ASR | faster-whisper prototype | ❌Removed | Useful prototype, but heavier and less aligned with current Jetson constraints |
| ASR | ReazonSpeech K2 | ❌Removed | Good for Japanese only; Cannot understand English too well |

---

## Phase 2.1 — Social 🔲

*Let Aiko introduce herself carefully. Social posting starts supervised, low-volume, and platform-aware before any autonomous public presence.*

**Goal:** Give Aiko a safe outbound social layer for publishing introductions, status updates, workspace photos, and longer reflections with explicit owner approval, audit logs, and per-platform tone/risk controls.

| Feature | Status |
|---|---|
| Social account connector registry — X, Threads, Instagram, Discord, Reddit, Bluesky, Mastodon, Pixelfed | 🔲 Planned |
| Draft-first posting workflow — Aiko prepares posts, owner approves before publish | 🔲 Planned |
| Platform policy and community-fit guardrails before posting | 🔲 Planned |
| Social identity/persona card for public introductions | 🔲 Planned |
| Workspace photo picker for future Instagram/Pixelfed posts | 🔲 Planned |
| Discord and Reddit introduction posting with rate limits and community rules checks | 🔲 Planned |
| Post history archive with links, timestamps, captions, and used media | 🔲 Planned |
| Abuse/spam prevention — cooldowns, blocklists, and no unsolicited mass outreach | 🔲 Planned |
| Human handoff mode for replies and moderation | 🔲 Planned |

### Social platform stance

| Platform | Roadmap stance | Reason |
|---|---|---|
| X | ✅ Primary experiment | Large public reach, already added, useful for short updates despite higher moderation/reputation risk |
| Threads | ✅ Primary experiment | Already added, friendlier mainstream short-post channel and a natural Instagram-adjacent path |
| Instagram | 🔲 Later, photo-gated | Best fit once Aiko can select approved workspace photos and write captions |
| Discord | ✅ Community introduction channel | Good for controlled server-by-server introductions and future bot-style presence |
| Reddit | 🟡 Careful experiment | Useful for long introductions, but subreddit rules and anti-promotion norms require strict approval |
| Bluesky | 🟡 Optional experiment | Good replacement for users avoiding X; likely better for open-web identity if AI disclosure is clear |
| Mastodon | 🟡 Optional experiment | Federation culture can be skeptical of bots/AI, so use only opt-in instances and explicit labeling |
| Pixelfed | 🟡 Later, photo-gated | Instagram-like fediverse option, but AI-generated or AI-curated media needs careful labeling and instance fit |
| Facebook | ❌ Not prioritized | Older social graph and lower value for Aiko's public identity experiments |
| Flickr | ❌ Not prioritized | Mostly archival/photo-community use; not a strong discovery channel for Aiko |

---

## Phase 2.2 — Message 🔲

*Let trusted people message Aiko through everyday apps without exposing her core runtime directly to the internet.*

**Goal:** Add inbound and outbound private-message gateways for Telegram, Slack, Discord DMs, and email, with authentication, consent, memory boundaries, and clear separation between private conversation and public posting.

| Feature | Status |
|---|---|
| Gateway abstraction for chat/email adapters | 🔲 Planned |
| Telegram bot adapter for owner-approved private chat | 🔲 Planned |
| Slack app adapter for workspace/team chat | 🔲 Planned |
| Discord bot DM and server mention adapter | 🔲 Planned |
| Email adapter — receive, summarize, draft, and send with approval | 🔲 Planned |
| Identity and allowlist controls per channel/user | 🔲 Planned |
| Per-channel memory policy — what can be remembered, ignored, or pinned | 🔲 Planned |
| Notification routing into the TUI/WebUI/mobile app | 🔲 Planned |
| Reply approval modes — manual, trusted-contact, and fully autonomous later | 🔲 Planned |
| Attachment handling path into Phase 6 multimodal vision | 🔲 Planned |

---

## Phase 2.5 — Agent ⏳
![Phase 2.5](../assets/phase-2.5.png)

*Give Aiko a real task layer. Skills describe repeatable workflows; toolkit modules provide safe executable actions.*

**Goal:** Let Aiko use predefined skillsets, local tools, schedules, and workspace artifacts to complete repeatable tasks without re-explaining every step.

| Feature | Status |
|---|---|
| ReAct-style task loop with tool dispatch | ✅ Done |
| Semantic Intent Routing for fast delegation | ✅ Done |
| Graph-first master-plan executor with model-free DAG tool nodes | ✅ Done |
| Autonomous sub-agent worker runtime with queues/leases/cancellation | 🔲 Planned |
| Implement goal-based DAG HTN One-step LLM Agentic architecture | ✅ Done |
| Persistent schedule jobs with reminder | ✅ Done |
| Toolkit set Agentic-focused tool modules | ✅ Done |
| Agentic skill workflow registry | ✅ Done |
| Skill-context retrieval in agentic mode | ✅ Done |
| Dual-path agentic routing — semantic exemplar route by default, optional LLM router/fallback instead of keyword-only dispatch | ✅ Done |
| Initial wildlife-photo and Aiko-architecture skills | ✅ Done |
| Safer code-edit/review workflow for self-improvement | 🔲 Planned |
| Optional MCP wrappers for stable long-running tools | 🔲 Planned |
| 24/7 worker/watchers for queues and folders | 🔲 Planned |

### Architecture changes in Phase 2.5

| Component | Before | After |
|---|---|---|
| Routing | Keyword-only (`/web`, `/think`, task keywords) | **Dual-path: fast semantic exemplar routing by default, optional LLM router/fallback for context-heavy cases** |
| Embeddings | BGE v1.5 (fastembed, 1024d, MEAN pooling) | **Harrier OSS v1 270M (custom ONNX, 640d, last-token pooling, query instructions)** |
| Embedder | `fastembed` library | **Custom `cognition/reason.py` ONNX Harrier embedder** (fastembed only exposed MEAN/CLS pooling) |
| Tools | Scattered functions | **Focused `toolkit/` modules: web, planning, scheduling, photo, architecture** |
| Skills | N/A | **`skills/skillsets/*.md` workflow registry loaded by `skills/skills.py`** |
| Agentic facade | Direct calls | **`toolkit/tools.py` compatibility facade + graph-first executor + ReAct fallback loop** |

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

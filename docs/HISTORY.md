# Aiko-chan 愛子ちゃん — Development History

Aiko-chan did not begin as a polished AI companion.

It began as a simple question:

> How much of a complete AI companion can run locally on an 8 GB Jetson Orin Nano?

The project became an experiment in balancing capability, personality, privacy, and hardware limitations while remaining entirely local-first.

This document records the major architectural decisions, successes, failures, rewrites, and lessons learned throughout development.

---

# Before Phase 1

The original concept was much smaller.

The first goal was simply to create a local chatbot that could:

- run through Ollama
- remember previous conversations
- search the web when necessary
- avoid any cloud dependency

At the time, the architecture was intentionally simple.

Conversations started feeling less like interactions with a model and more like interactions with a character.

That observation shaped every phase that followed.

---

# Phase 1 — Soul

Goal:

> Prove that a local AI companion with memory and web search is viable on constrained hardware.

Major accomplishments:

- persistent long-term memory
- memory retrieval during conversation
- asynchronous memory writes
- web-grounded responses
- fully local deployment

Lessons learned:

- Memory mattered more than model size.
- Retrieval quality mattered more than retrieval quantity.
- Local search was essential for factual grounding.
- Character consistency mattered more than expected.

---

# Phase 1.5 — Stream

Goal:

> Make Aiko feel alive.

Major additions:

- full-screen curses TUI
- token streaming
- callback-based architecture
- decoupled TTS pipeline
- Kokoro TTS integration
- persona framework
- identity framework

Lessons learned:

- Streaming is more important than raw speed.
- Personality is more important than prompt complexity.
- Users perceive responsiveness more strongly than benchmark numbers.
- A companion needs presence, not just intelligence.

---

# Phase 2 — Voice

Goal:

> Remove the keyboard.

Major additions:

- faster-whisper
- Silero VAD
- microphone capture
- hands-free interaction
- barge-in interruption

Major architectural change:

- mem0 removed
- Qdrant removed
- sqlite-vec adopted
- fastembed adopted
- custom retrieval pipeline implemented

Lessons learned:

- Simpler systems are often more reliable.
- Removing dependencies can be more valuable than adding features.
- Local-first design requires ruthless resource discipline.
- Memory management is harder than memory storage.

---

# Looking Ahead

Future phases introduce embodiment, emotional persistence, mobility, multimodal perception, and autonomy.

The core philosophy remains unchanged:

> Privacy first.
>
> Local first.
>
> Personality before polish.
>
> Simplicity over unnecessary complexity.
>
> Build for constrained hardware and everything else becomes easier.

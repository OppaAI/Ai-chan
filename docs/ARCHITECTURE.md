# Aiko Runtime Architecture

## Current Split

Aiko's runtime is already partly separated:

- `core/tools.py` is the compatibility facade for pure callable tools and should stay as the stable import surface; focused implementations live under `core/toolkit/` (web, planning, scheduling, photo, architecture). These functions do not own the ReAct loop.
- `core/think.py` owns the public chat facade: routing, normal chat, TTS/history glue, scheduled job callbacks, and background memory writes.
- `core/agentic.py` owns task-mode tool schemas, ReAct loop execution, and tool dispatch.
- `core/skills.py` owns local skill document CRUD/search helpers and the `skills/<skill_id>/SKILL.md` registry used by task mode.
- `core/memorize.py` owns persistent memory CRUD, recall, pinning, decay, cleanup, and nightly consolidation.
- `core/experience.py` owns the daily JSONL chat-turn log used by factual daily summaries.
- `core/reflect.py` owns factual daily summary generation, blog publishing, and pinning the generated daily summary.

## Module Boundaries

The runtime split should stay close to this shape:

```text
core/tools.py      compatibility facade for pure callables
core/toolkit/      focused tool implementations, no LLM loop, no conversational state
core/agentic.py    ReAct loop, tool schemas, tool dispatch
core/skills.py     skill CRUD and retrieval: load, append, prune, search, skill registry
skills/<id>/       human-readable repeatable workflow documents
core/think.py      public chat facade: normal chat, agentic handoff, TTS/history glue
```

Keep memory separate from all three: `core/memorize.py` should remain the single owner of persistent memory, including `pin()`.

## Memory Use Rules

- Normal chat should retrieve relevant memories before generation.
- Task mode should also retrieve relevant memories before tool choice, so tools and final answers can use user preferences and prior context.
- Tool functions should not read memory directly. The agent loop should retrieve memory and pass relevant context into the LLM.
- Daily summaries should use both the daily chat-turn log and persistent memory snippets, then pin the factual summary as permanent memory.
- Daily summaries should preserve important facts such as dates, deadlines, commitments, projects, events, losses, incidents, and goals. Mundane details should be downweighted unless they imply a pattern, risk, or follow-up.

---
id: runtime_state
name: Runtime State
summary: Directory and state-file rules that keep mutable runtime artifacts separate from static config.
status: active
owner: human
related: agentic_workflow, schedule
---
# Runtime State

Purpose: keep Aiko from mixing user settings, runtime state, and generated work.

## Directory Meanings

- `config/`: human-maintained defaults and settings that shape Aiko's behavior.
- `skills/`: reusable workflows plus skill-specific defaults.
- `wiki/`: operational routing cards and examples Aiko can retrieve before acting.
- `workspace/`: Aiko's working area for generated notes, reports, schedules, reminders, and task artifacts.
- `logs/`: runtime logs and diagnostics.

## Schedule Files

Keep scheduler defaults in `config/schedule.yaml`.

Keep user-created scheduled jobs in `~/.aiko/<user_id>/schedule.json`. This file is runtime state: Aiko and the scheduler update it while running. It should stay in the per-user state directory, not in config or shared workspace directories.

Do not move `~/.aiko/<user_id>/schedule.json` into `config/` just because it looks like settings. It contains mutable jobs, not static defaults.

## When To Use Runtime

Use a runtime directory only for cache-like files that can be deleted and regenerated safely. User-created schedules, reminders, reports, and notes are not cache; keep them in workspace.

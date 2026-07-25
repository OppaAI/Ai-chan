---
id: directory_map
name: Directory Map
summary: Directory and state-file rules that keep mutable runtime artifacts separate from static config.
status: active
owner: human
related: operating_procedure, schedule
---
# Runtime State

Purpose: keep Aiko from mixing user settings, runtime state, and generated work.

## Directory Meanings

- `config/`: human-maintained defaults and settings that shape Aiko's behavior.
- `agentic/`: reusable workflows plus skill-specific defaults.
- `wiki/`: operational routing cards and examples Aiko can retrieve before acting.
- `workspace/`: Aiko's working area for generated notes, reports, and task artifacts.
- `logs/`: runtime logs and diagnostics.

## Schedule Files

Keep scheduler defaults in `config/schedule.yaml`.

Keep user-created scheduled jobs in `~/.aiko/<user_id>/tasks/schedule.json`. This file is runtime state: Aiko and the scheduler update it while running. It should stay in the per-user state directory, not in config or shared workspace directories.

Do not move `~/.aiko/<user_id>/tasks/schedule.json` into `config/` just because it looks like settings. It contains mutable jobs, not static defaults.

A direct tool job is data-only and may select any registered agentic tool:

```json
{
  "title": "daily_job_post_social",
  "time_of_day": "09:00",
  "frequency": "daily",
  "action": "tool",
  "tool_call": {"name": "draft_job_post_social", "arguments": {}}
}
```

For an `agentic` job, the optional `skill` field can contain custom `SKILL.md`-style Markdown instructions. The scheduler supplies it with the task when the job fires. Tool names are restricted to the existing agentic tool registry; schedule files cannot run arbitrary Python.

## When To Use Runtime

Use a runtime directory only for cache-like files that can be deleted and regenerated safely. User-created schedules, reminders, reports, and notes are not cache; keep them in workspace.

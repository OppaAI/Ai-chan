---
id: agentic_workflow
name: Agentic Workflow
summary: Compact operating rules Aiko retrieves before task-mode tool selection.
status: active
owner: human
related: runtime_state, skills
---
# Agentic Workflow

Purpose: give Aiko compact operating rules before she chooses tools.

## Default Task Loop

Use this loop for research, coding, scheduling, writing artifacts, workspace work, skill workflows, and multi-step requests.

1. Identify the concrete goal.
2. Load matching skill, wiki, learned knowledge, and similar experience context when available.
3. Pick the next useful tool call. In task mode, use `deep_search` for web snippets and `deep_research` for fetched-page research; `web_search`/`web_fetch` are chat-mode primitives, not skill tools.
4. Read the tool result before deciding the next step.
5. Save or schedule only through tools.
6. Finish with a natural final answer that says what was done and what remains uncertain.

## Anti-Confusion Rule

If Aiko feels unsure, she should not stop at "I'm confused." She should do one of these:

- Ask one short blocking question when a required detail is missing.
- Use `make_plan` or `summarize_task_state` when the task has many steps.
- Use `search_skillsets` or `load_skillset` when the task sounds like a repeatable workflow.
- Use similar `<experience_context>` as a hint for proven/failed tool sequences, but follow skill/wiki policy first.
- Use repository or workspace read/search tools when the answer depends on local files.
- State the safest assumption and continue when the missing detail is not dangerous.

## Tool Choice Examples

- "Find jobs for me" -> load `job_hunt`, use configured default location unless the user gives another, then call `search_jobs`.
- "Schedule this every morning" -> call `schedule_job` or `schedule_reminder`.
- "Inspect Aiko's code" -> load `aiko_architect`, then use repo file/search tools.
- "Write/save a note/report" -> do the work, then call `save_note`.
- "Remember this document/knowledge for RAG" -> call `learn_knowledge` with pasted text or a workspace-relative document path.
- "What should I do next?" -> make a short plan/checklist; save only if requested.

## Final Answer

Final answers should be concise but complete:

- Name the artifact path when something was saved.
- Name the schedule/reminder id when something was scheduled.
- Say when a web search failed or was not run.
- Do not claim external actions happened unless the tool succeeded.

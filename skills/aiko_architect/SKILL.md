---
id: aiko_architect
name: Aiko Architecture Research and Improvement
summary: Help research, inspect, plan, and safely improve Aiko's own codebase and architecture with repo-reading tools, web research, notes, and explicit review gates.
triggers: improve Aiko, architecture, refactor, optimize, debug Aiko, implement feature, codebase, tools, skills, memory, agentic
tools: repo_file_tree, repo_search_text, repo_read_file, web_search, fetch_page, make_plan, create_checklist, save_note, summarize_task_state
---
# Aiko Architecture Research and Improvement

Use this skill when Oppa asks Aiko to research, inspect, design, optimize, refactor, or improve her own architecture/code.

## Workflow

1. Classify the task:
   - research only;
   - bug investigation;
   - implementation plan;
   - safe code edit request;
   - performance/architecture review.
2. Inspect the repository before making claims:
   - use `repo_file_tree` to locate likely files;
   - use `repo_search_text` for symbols/concepts;
   - use `repo_read_file` for relevant files.
3. If current external facts are needed, use `web_search`/`fetch_page`; prefer official docs or primary sources.
4. Produce a concise implementation plan before any risky action.
5. Use notes/checklists/task-state tools to preserve research and decisions under the workspace.
6. Clearly distinguish:
   - what was verified from files/tool outputs;
   - what is inference;
   - what still needs tests or human approval.

## Current safety boundary

The available architecture tools are read-only repository inspection plus planning/research/note tools. They do not edit files, run shell commands, install packages, or commit changes. If Oppa wants code changes, Aiko should propose the patch plan and ask for an implementation path/tooling.

## Rules

- Never pretend code was changed unless a real code-editing tool performed it.
- Prefer small reversible changes.
- Do not modify persona, memory, or skill files without explicit instruction.
- Do not expose broad shell execution as a casual tool.
- When optimizing for AuRoRA/AIVA/hardware-specific constraints, verify versions and specs from repo context or fetched sources.

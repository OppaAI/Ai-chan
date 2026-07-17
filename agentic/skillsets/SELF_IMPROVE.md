---
id: SELF_IMPROVE
name: Self-Improvement — Architecture Research and Code Improvement
summary: Help research, inspect, plan, and safely improve Aiko's own codebase and architecture with repo-reading tools, web research, notes, and explicit review gates.
triggers: improve Aiko, architecture, refactor, optimize, debug Aiko, implement feature, codebase, tools, skills, memory, agentic
tools: repo_file_tree, repo_search_text, repo_read_file, deep_search, deep_research, make_plan, create_checklist, save_note, summarize_task_state
---
# Self-Improvement — Architecture Research and Code Improvement

Use this skill when Oppa asks Aiko to research, inspect, design, optimize, refactor, or improve her own architecture/code.

## Workflow

0. **Orient first**: always call `repo_file_tree` immediately before any other step.
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
3. If current external facts are needed, use `deep_search` for a single well-scoped question. If the research spans multiple unclear angles, use `deep_research` instead. Prefer official docs or primary sources either way.
4. Produce a concise implementation plan before any risky action.
5. Use notes/checklists/task-state tools to preserve research and decisions under the workspace.
6. Clearly distinguish:
   - what was verified from files/tool outputs;
   - what is inference;
   - what still needs tests or human approval.

## Current safety boundary
`repo_file_tree`, `repo_read_file`, and `repo_search_text` are **fully available and must be used freely** — they are read-only and safe to call at any time without restriction.

The boundary applies only to write/execute operations: do not edit files, run shell commands, install packages, or commit changes. If Oppa wants code changes, propose the patch plan and ask for an implementation path/tooling.

## Rules

- Never pretend code was changed unless a real code-editing tool performed it.
- Prefer small reversible changes.
- Do not modify persona, memory, or skill files without explicit instruction.
- Do not expose broad shell execution as a casual tool.
- When optimizing for hardware-specific constraints, verify versions and specs from repo context or fetched sources.

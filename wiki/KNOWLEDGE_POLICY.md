---
id: knowledge_policy
name: Knowledge Policy
summary: Rules for separating trusted wiki/skills, learned knowledge, memory, and experience.
status: active
owner: human
related: operating_procedure, directory_map
---
# Knowledge Base Governance

Purpose: make Aiko's wiki, learned knowledge, memory, and experience grow without mixing trust levels.

## Trust Levels

- **wiki/skills (human):** markdown/json files written or approved by OppaAI or a maintainer. Highest trust for operating policy.
- **learned knowledge (vector store):** durable facts, document/PDF excerpts, and self-learning notes stored with `learn_knowledge` in `core/knowledge.py`. Retrieve as RAG evidence, not policy.
- **experience (vector store):** sanitized task traces written automatically by `core/experience.py`; useful for similar workflows and known failures, not authoritative facts.
- **memory:** private/personal user facts managed by `core/memorize.py`; do not duplicate into learned knowledge unless the user explicitly asks.
- **proposed:** generated wiki/skill changes from failures, research, or repeated corrections. Must be reviewed before becoming policy.
- **runtime:** generated state or reports in `workspace/`; useful as evidence, not standing policy.

## Update Rule

Aiko should not silently rewrite trusted wiki or skill files during normal work. Learned knowledge and experience may be written by tools, but they must not overwrite human policy. When she discovers a missing rule, stale instruction, repeated failure, or useful new workflow, she should draft a proposal under `workspace/kb_proposals/` with:

1. the problem or failure that triggered the proposal,
2. source evidence or file paths,
3. the suggested wiki/skill change,
4. confidence and freshness notes,
5. whether human approval is needed.

## Lint Rule

Run `python -m kb.lint` after editing wiki or skill documents. Wiki cards need front matter with `id`, `name`, `summary`, `status`, and `owner`. Skill documents need `id`, `name`, `summary`, `triggers`, and `tools`.

## Retrieval Rule

Normal chat may retrieve wiki and learned knowledge for Aiko/self-knowledge questions. Agentic mode should retrieve relevant policy, wiki, skills, memory, learned knowledge, and similar experience after task intent is confirmed and before choosing tools. Priority order is: explicit user request > trusted skill/wiki policy > learned knowledge evidence > similar experience hints > general model knowledge.

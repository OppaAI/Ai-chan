# n8n-style DAG Studio + IT/Everyday Nodes + Coding Agent + Needle Team

Jetson Orin Nano 8GB companion doc. Everything here is bounded for
Ministral-3-3B + 8GB unified RAM: max 2 parallel graph workers, 60s studio
dry-runs, capped tool outputs, approval gates on every write.

## 1. DAG Studio (n8n-style) — `/studio/dag`

What you can do now (was view-only + drag-edge before):

| Action | How |
|---|---|
| New workflow | `+ New` button → id + name → starts with a `make_plan` node |
| Add node | Click a tool in the **Node palette** (grouped by domain, 🔒 = needs approval), then **Add Node** (double-click adds instantly) |
| Connect | Drag from a node's **right port** to another's **left port** |
| Change edge type | Click edge → switch `depends_on` / `loop_to` / `fallback_to` → Apply |
| Delete | Select node/edge → **Delete**; whole graph → **Delete graph** |
| Clone | **Duplicate** (works on built-ins too — built-ins themselves are delete-protected) |
| Validate | **Validate** → cycles, unknown deps/tools, entry points, size warnings |
| Dry-run | **▶ Run** → bounded execute (60s), per-node ok/fail + final answer |
| Import/Export | JSON file round-trip |

Backend (`studio/dag/backend/api.py`):
`GET /api/tools` (palette), `POST /api/playbooks` (create),
`DELETE` (user graphs only), `POST …/duplicate|validate|run`.
`$prompt` in node args is substituted at run time, same as the engine.

## 2. IT + everyday nodes

`agentic/toolkit/it_ops.py` — stdlib + psutil only, read-only, JSON-block
outputs. All are graph+react enabled so the palette picks them up.

IT: `sys_health`, `process_top`, `disk_large_files` (preview, never deletes),
`log_triage` (defaults to `logs/aiko.log`), `service_status`, `docker_ps`,
`port_check`, `http_check`, `dns_lookup`, `ssl_expiry`, `file_find`, `net_info`.
Everyday: `text_summarize`, `text_translate` (LLM when available, heuristic
fallback otherwise), `unit_convert` (offline), `pass_gen` (`secrets`).

Ready workflows (`_default_playbooks` in `agentic/graph_engine.py`):
`it_syscheck`, `it_log_triage`, `it_endpoint_check`, `it_disk_cleanup_preview`,
`everyday_summarize`, `everyday_translate_save`, `everyday_morning_brief`.

## 3. Coding agent — can Ministral-3-3B self-improve?

**Yes, for small scoped tasks.** Single-file fixes (<200 lines, clear goal)
with retrieved context work well. Large multi-file refactors in one shot do
not — keep patches small and verify each step.

`agentic/toolkit/coding.py` enforces the safe loop:

```
codebase_search / repo_read_file (context)
  → code_plan (tiny steps, LLM-light or heuristic)
  → code_diff_preview (NO write — review this)
  → code_apply_patch (APPROVAL REQUIRED, .bak backup, secrets blocked, 50k cap)
  → code_lint → code_run_tests (90s timeout, tail output only)
```

`repo_write_file` / `repo_replace_text` still exist but now require approval
too (`CODE_WRITE_APPROVAL_TOOLS` in `agentic/registry.py` + `needs_approval`
in `config/tools.yaml`). Studio dry-runs should preview diffs, never apply.
Workflows: `code_explain` (read-only), `code_small_fix` (full loop, approval
node included).

## 4. Needle2 as multi-agent spawner — yes

`agentic/toolkit/needle_team.py`: `needle_team_status` (config probe, never
leaks URLs) + `needle_team_run` (fan-out, merge proposals). Rule unchanged:
**Needle proposes, Aiko disposes** — every call is validated against the
capability-filtered subset and executed only via Aiko's registry + approvals.

Setup (2 workers max on 8GB):
```bash
NEEDLE_WORKERS='[{"id":"research","role":"researcher","base_url":"http://127.0.0.1:8082","allowed_tools":["adaptive_search","deep_read"]},
 {"id":"coder","role":"code reviewer","base_url":"http://127.0.0.1:8083","allowed_tools":["codebase_search","repo_read_file","repo_search_text"]}]'
AGENT_REACT_BACKEND=needle_multi   # or keep openai and call needle_team_run per-graph
```
Low-confidence/unavailable workers fall back to the main LLM with a clear
note. Workflow: `needle_team_research`. Compat shim
`needle_multi_agent_delegate` still works (delegates to `needle_team_run`).

## 5. Jetson budgets

- `GRAPH_MAX_WORKERS=2` (`config/agentic.yaml`); studio dry-runs cap at 60s.
- Playbooks stay ≤6 nodes, ≤2 parallel LLM calls.
- Tool outputs truncated (`GRAPH_NODE_RESULT_MAX_CHARS`, per-tool caps).
- `code_run_tests`: one `pytest -q -x`, 90s timeout, 4k tail.
- Tests: `tests/unit/test_n8n_it_coding_needle.py` (22 tests, hermetic).
  `test_parallel_execution` + ~30 others fail identically on clean HEAD
  (pre-existing, unrelated).

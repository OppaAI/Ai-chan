# Bright Data Scraper Studio MCP (optional)

Isolated stdio MCP server so Aiko can call **Scraper Studio** collectors and **Self-Healing** without baking Bright Data into core research code.

**Remove later:** delete this directory and any client spawn that points at `interface.mcp_server.brightdata.server`.

## How self-healing works (Bright Data)

Self-healing is **not** local selector repair inside Aiko. Bright Data runs an AI refactor on *their* infrastructure:

1. Site layout changes → your collector returns empty / null fields.
2. You (or the agent) call **`bd_self_heal`** with a plain-language prompt
   (e.g. `"price is undefined; it is now in span.price-now"`).
3. API: `POST /dca/collectors/{c_*}/refactor_template` → poll progress.
4. Job pauses at **`pending_answer` / awaiting approval** (HITL by default).
5. **`bd_self_heal_approve`** only **submits** the decision (`phase=decision_submitted`).
   It does **not** mean the heal job finished.
6. Poll **`bd_self_heal_progress`** until `completed=true` (or `failed=true`).
7. Same **`c_*` collector id** keeps working; re-run **`bd_run_collect`**.

CLI equivalent: `bdata scraper heal <id> "..."` then `bdata scraper approve`.

## Env

```bash
export BRIGHT_DATA_API_TOKEN=...          # required for live calls
export BRIGHT_DATA_COLLECTOR_ID=c_...     # default collector
# optional:
# export BRIGHT_DATA_API_BASE=https://api.brightdata.com
```

Hackathon credits: Bright Data promo code `wemakedevs` (see Scrape-Verse).

## Run standalone

```bash
python -m interface.mcp_server.brightdata.server
```

## Tools

| Tool | Purpose |
|------|---------|
| `bd_trigger_collect` | Queue batch run → `collection_id` |
| `bd_get_results` | Poll / download rows |
| `bd_run_collect` | Trigger + wait for rows |
| `bd_self_heal` | Start self-heal; optional `auto_approve` (waits for completion) |
| `bd_self_heal_progress` | Poll heal job (`completed` / `failed` flags) |
| `bd_self_heal_approve` | Submit approve/reject only (`decision_submitted`) |

## Aiko client (manual for now)

Point a stdio MCP client at:

```text
python -m interface.mcp_server.brightdata.server
```

(with the same env as Aiko). Core `agentic/mcp_client` currently boots the social server only; wire a second process or temporary swap when testing. Prefer **not** merging permanent research graph changes until the experiment is kept.

## Typical agent flow

1. `bd_run_collect` with URLs → structured rows for deep research / `learn_report`.
2. If rows empty or fields null → `bd_self_heal` with a fix prompt + sample URL.
3. Review → `bd_self_heal_approve` → expect `phase=decision_submitted` (not completed).
4. Poll `bd_self_heal_progress` until `completed=true`.
5. `bd_run_collect` again with the **same** collector id.

With `bd_self_heal(..., auto_approve=true)`, steps 3–4 are handled inside the tool and the result uses `phase=completed` only after the job finishes.

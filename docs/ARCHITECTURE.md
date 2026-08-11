# Aiko Runtime Architecture

## Current Split

Aiko is organized by runtime responsibility rather than by one monolithic `core/` package:

- `main.py` is the launch orchestrator. The browser WebUI is the primary human surface; adapters are optional extra I/O channels.
- `system/` owns process, config, identity, and shared infrastructure concerns.
- `cognition/` owns model-facing reasoning helpers and related orchestration utilities.
- `memory/` owns STM/LTM, knowledge, and memory-side persistence/query surfaces.
- `agentic/` owns tool calling, MCP client bridges, capability routing, and agentic toolkit code.
- `interface/` owns human and external I/O: WebUI, adapters, and MCP servers.
- `util/` remains shared helpers that do not own a runtime boundary of their own.

## Social publishing lanes

Social publishing is draft-first and approval-gated:

- Lane A1 is weekly Patreon dev-post syndication.
- Lane B is the curated photo pipeline for Pixelfed posting through the social MCP server.
- Lane C is the YouTube video queue.
- Lane D is the nightly RSS-only tech-jobs draft path.

All posting requires human approval regardless of trigger path.

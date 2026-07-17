# Aiko Runtime Architecture

## Current Split

Aiko is organized by runtime responsibility rather than by one monolithic `core/` package:

- `main.py` is the launch orchestrator. The browser WebUI is the default UI; `--cli` runs a simple local CLI for testing.
- `interface/webui/webui.py` serves the browser frontend, owns the WebSocket bridge, accepts browser microphone frames, and broadcasts chat, vitals, voice, expression, and viseme events.
- `interface/cli/` contains authentication and helper code for the simple CLI path.
- `system/wakeup.py` owns parallel subsystem startup and returns a `BootResult` containing thinking, memory, speech, and listening modules.
- `cognition/think.py` owns the public chat facade: OpenAI-compatible LLM setup, semantic/LLM routing, normal chat, TTS/history glue, scheduled job callbacks, idle learner handoff, and background memory writes.
- `agentic/agentic.py` owns task-mode tool schemas, ReAct loop execution, final-answer verification, and tool dispatch.
- `agentic/schema.py` owns the graph-first playbook DAG executor.
- `agentic/tools.py` is the compatibility facade for executable tools; focused implementations live under `agentic/toolkit/`.
- `agentic/skills.py` owns skillset CRUD/search helpers and the `agentic/skillsets/` workflow registry used by task mode.
- `memory/memorize.py` owns persistent memory CRUD, recall, pinning, decay, cleanup, and consolidation hooks.
- `memory/reflect.py` owns factual daily summary publishing and pinning of generated daily summaries.

## Module Boundaries

```text
main.py               CLI flags, WebUI default launch, simple CLI option, shared session loop
interface/webui/      browser adapter, HTTP static server, WebSocket bridge, UI API
interface/cli/        CLI auth and local testing helpers
system/wakeup.py      boot orchestration and BootResult assembly
cognition/think.py    chat facade, routing, scheduled callbacks, TTS/history glue
memory/               memory, journal, consolidation, reflection, sqlite-vec helpers
sensory/              speech and listening adapters
agentic/tools.py      stable facade for pure callable tools
agentic/toolkit/              focused tool implementations; no LLM loop or conversation state
agentic/agentic.py     ReAct loop, tool schemas, dispatch, verification, experience recording
agentic/schema.py      graph-first playbook DAG executor
agentic/skillsets/     human-readable repeatable workflow documents
wiki/                 trusted local knowledge cards
```

Memory stays separate from chat routing, tool execution, and UI rendering. Tool functions should not read memory directly; the chat/agent layer retrieves memory and passes relevant context into prompts or tool arguments.

## High-Level Runtime Flow

```mermaid
flowchart TD
    User[User] --> Entry[main.py]
    Entry --> Choice{Runtime mode?}
    Choice -->|default| WebUI[AikoWeb\ninterface/webui]
    Choice -->|--cli| CLI[Simple CLI\ninterface/cli]

    WebUI --> Session[Shared session loop]
    CLI --> Session

    Session --> Wakeup[AikoWakeup.boot]
    Wakeup --> Think[AikoThink\nOpenAI-compatible client]
    Wakeup --> Memory[AikoMemorize\nsqlite-vec + embeddings]
    Wakeup --> Speak[AikoSpeak\nMioTTS]
    Wakeup --> Listen[AikoListen\nSenseVoice + Silero VAD]

    Listen -->|voice transcript| Session
    User -->|typed message| Session
    Session -->|turn text| Think
    Think <-->|recall + async writes| Memory
    Think -->|task route| Agentic[agentic.agentic\nReAct task loop]
    Agentic --> Schema[agentic.schema\nplaybook DAG executor]
    Agentic --> Tools[agentic.tools facade]
    Tools --> Toolkit[toolkit modules]
    Think -->|streamed tokens| Session
    Session -->|draw events| WebUI
    Session -->|print events| CLI
    Think -->|optional TTS| Speak
    Speak --> User
```

## Boot Sequence

The second Mermaid diagram is intentionally conservative: it avoids characters that older Mermaid sequence parsers often misread inside message labels.

```mermaid
sequenceDiagram
    autonumber
    participant U as User
    participant M as main.py
    participant UI as WebUI or CLI
    participant W as system.wakeup.AikoWakeup
    participant T as cognition.think.AikoThink
    participant Mem as memory.memorize.AikoMemorize
    participant S as sensory.speak.AikoSpeak
    participant L as sensory.listen.AikoListen

    U->>M: start Aiko with optional runtime flags
    M->>UI: construct selected interface adapter
    M->>UI: start initialization display
    M->>W: boot callbacks
    par cognition boot
        W->>T: construct client and warm routing cache
        T-->>W: ready
    and memory boot
        W->>Mem: open sqlite vec store and embedder
        Mem-->>W: ready
    end
    W->>T: attach memory reference
    W->>S: warm TTS client
    S-->>W: ready
    W->>L: initialize ASR VAD and barge-in monitor
    L-->>W: ready
    W-->>M: BootResult
    M->>UI: finish initialization display
    M->>M: enter shared input loop
```

## Conversation Turn Sequence

```mermaid
sequenceDiagram
    autonumber
    participant U as User
    participant UI as WebUI or CLI
    participant M as main.py session loop
    participant T as cognition.think.AikoThink
    participant Mem as memory.memorize.AikoMemorize
    participant A as agentic.agentic
    participant G as agentic.schema
    participant Tools as toolkit tools
    participant S as sensory.speak.AikoSpeak

    U->>UI: type message or speak audio
    UI-->>M: provide input text
    M->>UI: show user message and start turn
    M->>T: route text with token callback
    T->>Mem: start memory and knowledge lookup
    T->>T: classify chat web or task intent
    alt normal chat or web chat
        Mem-->>T: context block
        T->>T: generate response
    else agentic task mode
        T->>A: run_agentic_chat
        A->>G: try playbook graph when enabled
        alt graph matched
            G->>Tools: execute ready DAG nodes
            Tools-->>G: node results
            G-->>A: deterministic graph answer
        else no graph match in hybrid mode
            A->>Mem: use memory and task context
            loop ReAct iterations
                A->>Tools: validated tool call
                Tools-->>A: structured observation
            end
            A-->>T: final answer
        end
    end
    loop streamed tokens
        T-->>M: token callback
        M->>UI: stream token and redraw
    end
    M->>UI: commit assistant message
    T->>Mem: enqueue background memory write
    opt voice enabled
        T->>S: speak response
        S-->>U: audio
    end
```

## Routing and Execution Flow

```mermaid
flowchart TD
    Input[User input] --> Slash{Slash command?}
    Slash -->|yes| Command[Handle slash command]
    Command --> Draw[Update UI]

    Slash -->|no| StartTurn[Start chat turn]
    StartTurn --> PreFetch[Start memory + KB lookup]
    PreFetch --> Route[AikoThink.route]
    Route --> Intent{Intent classifier}
    Intent -->|chat| Chat[AikoThink chat]
    Intent -->|web| Web[Web-grounded chat]
    Intent -->|task| Task[AikoThink.agentic_chat]

    Chat --> LLM[Normal LLM generation]
    Web --> Search[SearXNG context]
    Search --> LLM

    Task --> GraphGate{Graph mode?}
    GraphGate -->|graph or hybrid| Master[agentic.schema plan_from_master]
    Master -->|matched| DAG[Execute PlanGraph DAG]
    DAG --> GraphFinal[Deterministic graph answer]
    Master -->|no match and graph| NoMatch[No-playbook message]
    Master -->|no match and hybrid| React[agentic.agentic ReAct loop]
    GraphGate -->|react| React
    React --> ToolCall[Tool call through toolkit]
    ToolCall --> Observation[Structured observation]
    Observation --> Continue{More work?}
    Continue -->|yes| React
    Continue -->|no| Final[final_answer]

    LLM --> Stream[Token stream to UI]
    GraphFinal --> Stream
    NoMatch --> Stream
    Final --> Stream
    Stream --> Commit[Commit assistant message]
    Commit --> AsyncMem[Background memory write]
```

## Agentic Task Flow

```mermaid
flowchart TD
    TaskInput[Task-like user request] --> Think[AikoThink.agentic_chat]
    Think --> Agent[agentic.agentic.run_agentic_chat]
    Agent --> Caps[Capability matching]
    Caps --> Tools[Filtered tool schemas]
    Caps --> Graph{AGENT_EXECUTOR_MODE}
    Graph -->|graph or hybrid| Schema[agentic.schema]
    Schema -->|playbook matched| DAG[PlanGraph DAG execution]
    DAG --> Ready[Run independent ready nodes in parallel]
    Ready --> NodeResults[NodeResult records]
    NodeResults --> GraphAnswer[Deterministic answer]
    GraphAnswer --> Experience[Record procedural experience]
    Schema -->|no match and graph| GraphMiss[No-match response]
    Schema -->|no match and hybrid| Context[Fetch task-only context]
    Graph -->|react| Context
    Context --> Memory[Memory and knowledge]
    Context --> Policy[Skill policy]
    Context --> Wiki[Wiki cards]
    Context --> Past[Experience recall]
    Memory --> Prompt[Task-mode prompt]
    Policy --> Prompt
    Wiki --> Prompt
    Past --> Prompt
    Prompt --> Model[LLM chooses tool call]
    Model --> Validate[Validate arguments]
    Validate --> Dispatch[Dispatch via toolkit]
    Dispatch --> Obs[Structured observation]
    Obs --> Done{Goal complete?}
    Done -->|no| Model
    Done -->|yes| Final[final_answer]
    Final --> Experience
```

## Autonomous Sub-Agent Status

Aiko can already run **autonomous graph nodes inside one orchestrated agentic turn**: `agentic.schema.execute_graph` finds nodes whose dependencies are satisfied, runs independent ready nodes through a thread pool, marks nodes with failed dependencies as `dependency_failed`, and records the resulting trace. That is useful for deterministic playbook workflows.

Aiko does **not** yet have a fully independent, long-running autonomous sub-agent runtime. A true sub-agent layer would add durable queues, leases or heartbeats, cancellation, per-agent workspace/artifact boundaries, retry policy, permissions, and observability. The current architecture is compatible with that future layer because graph nodes are already explicit units of work, but today they are lightweight in-process tool tasks rather than separately managed workers.

## Current Runtime Configuration

- `LLM_BASE_URL` and `LLM_MODEL` select the local OpenAI-compatible LLM endpoint.
- `EMBED_MODEL`, `EMBED_DIMS`, `EMBED_CACHE_PATH`, and `SQLITE_MEMORY_PATH` configure local sqlite-vec memory and the Harrier ONNX embedder.
- `ROUTE_ENABLED`, `ROUTE_MODE`, and route thresholds are read by `cognition/think.py`.
- `AGENT_EXECUTOR_MODE` selects `graph`, `hybrid`, or `react` task execution.
- `GRAPH_playbook_PATH` and `GRAPH_MAX_WORKERS` configure `agentic.schema`.
- `ROUTE_VECTOR_CACHE_ENABLED` enables safe `.npz` route-vector cache files.
- `MIOTTS_API_URL`, `MIOTTS_PRESET`, and `MIOTTS_DEVICE` configure voice output.
- `ASR_*`, `LISTEN_*`, and `SPEAKER_*` configure SenseVoice, Silero VAD, speaker verification, and barge-in.
- `WORKSPACE_ROOT`, `SCHEDULE_PATH`, and `SCHEDULE_POLL_SECONDS` configure local workspace and scheduled work.

## Knowledge Governance

- Wiki cards and skill workflow files are treated as trusted local knowledge only when they include required front matter.
- Run `python -m util.lint` after changing `wiki/*.md` or `agentic/skillsets/*.md` where applicable.
- Aiko should draft proposed knowledge updates under `workspace/kb_proposals/` instead of silently rewriting trusted wiki or skill policy.

## Semantic Vector Cache

Intent-routing examples are authored as text in `cognition/router_prompts.json`, but their embedding matrix can be cached on disk with `ROUTE_VECTOR_CACHE_ENABLED=1`. The cache is keyed by the examples, instruct string, embedding backend metadata, and `EMBED_DIMS`, and is stored as a NumPy `.npz` archive loaded with `allow_pickle=False`.

Graph playbooks are currently matched by trigger and capability metadata, so there are no graph vectors to precompute yet. If graph matching becomes semantic, the same pattern should be used: stable JSON/YAML plan specs as the source of truth, plus generated vector-cache artifacts that are safe to delete and rebuild.

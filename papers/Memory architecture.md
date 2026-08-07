# Memory architecture

> Explicit, local-first long-term memory for Aiko-chan.
> Edge-friendly (Jetson / 8GB class). Precursor patterns for **Grace / AuRoRA**.

Related: [[Memory-Phases]] · [[Memory-Deferred]] · [[Memory-Papers]]

---

## Design goals

| Goal | Approach | Reviewer check |
|------|----------|----------------|
| **Edge-first** | In-process **sqlite-vec** + FTS5; Harrier GGUF/ONNX embeddings; no Qdrant/mem0 server | Can boot locally without external memory service |
| **Explicit memory** | Rows with clear write/read semantics, not only weights or KV cache | Inspectable DB rows and deterministic migrations |
| **Belief revision** | `supersedes_id` + `status` — update without silent overwrite | Old facts remain auditable through lineage |
| **Affect-aware** | **Valence** (pleasant↔unpleasant) and **arousal** (calm↔activated) | Recall can distinguish “happy calm” from “urgent scary” |
| **Human-feel recall** | Soft neg avoid + hard filter; session topic anchor; state tags | Avoids unsolicited bad memories while allowing direct queries |
| **Lifecycle** | Access tracking, decay, nightly dream, monthly retention gate | Important memories survive without unbounded growth |
| **Inspectable** | Memory Graph Studio + lineage API | Debuggable by humans before changing ranking weights |

Design weights for monthly consolidation are cited in-repo as **Paper I** §5.2 (see [[Memory-Papers]]).

---

## System overview

```mermaid
flowchart TB
  subgraph Chat["Chat / agent"]
    U[User turn]
    T[cognition/think.py]
    U --> T
  end

  subgraph Stores["Explicit stores"]
    M[(memories<br/>sqlite-vec + FTS)]
    K[(knowledge)]
    E[(experience)]
    G[(entity_relations)]
  end

  subgraph Write["Write path"]
    Q[Async write queue]
    X[LLM fact extract]
    INS[_insert_row]
    Q --> X --> INS
    INS --> M
    INS --> G
  end

  subgraph Recall["Recall path"]
    S[search / RRF]
    R[Rank bonuses]
    F[Neg soft avoid + hard filter]
    C[format_for_context]
    S --> R --> F --> C
  end

  T -->|enqueue| Q
  T -->|search| S
  C --> T
  M --> S
  G --> S
  K -.->|cross-store| C
  E -.->|cross-store| C

  subgraph Lifecycle["Offline"]
    D[dream / cleanup]
    MC[monthly consolidation]
  end
  D --> M
  MC --> M
```

### Store responsibilities

```mermaid
erDiagram
  memories ||--o{ memories : supersedes
  memories ||--o{ memories_vec : embeds
  memories ||--o{ entity_relations : mentions
  memories ||--o{ knowledge : enriches
  memories ||--o{ experience : contextualizes

  memories {
    integer id PK
    text user_id
    text memory
    text status
    integer supersedes_id FK
    integer scene_id
    integer valence_score
    integer arousal_score
    text state_json
  }
  entity_relations {
    text user_id
    text entity_a
    text entity_b
    real strength
  }
  knowledge {
    integer id PK
    text text
    text entities
  }
  experience {
    integer id PK
    text trace
    text entities
  }
```

### Primary code

| Role | Path |
|------|------|
| Facade | `cognition/memory/memorize.py` → `AikoMemorize` |
| Backend | `cognition/memory/backend.py` → `_MemoryBackend` |
| Lineage | `cognition/memory/lineage.py` |
| Session anchor | `cognition/memory/session_anchor.py` |
| Config | `config/memory.yaml` |
| Studio | `interface/webui/studio/memory/` |

---

## Data model (core columns)

| Column | Role | Notes |
|--------|------|-------|
| `id`, `user_id`, `memory`, `created_at` | Identity + text | User-scoped, local-first memory rows |
| `access_count`, `last_accessed_at`, `access_day_count` | Practice / spacing | Used by rank and monthly retention |
| `pinned` | Keep signal | Immune to ordinary decay / prune |
| `status`, `supersedes_id` | Active vs superseded; belief chain | Enables explicit revision history |
| `kind`, `source`, `entities` | Fact/scene/…; origin; entity JSON | Drives graph and Studio grouping |
| `scene_id` | L2 episode membership | Lets scene recall expand related facts |
| `valence_tag`, `valence_score` | Polarity (−2…+2) | Negative filters and emotion-modified decay |
| `salience_hit` | Cheap importance flag | Monthly consolidation feature |
| `arousal_score` | Activation (−2…+2 schema) | P19 rank bonus uses magnitude |
| `state_json` | Encode-time context | Example: local hour or other state tags |

Vectors live in `memories_vec`. Co-mentions live in `entity_relations`.

**Arousal note:** Schema and docs target a 5-point scale (−2…+2). The P19 lexicon currently emits **{−1, 0, +1, +2}** only (`−2` reserved). Rank uses `|arousal|`, so sign does not change ordering yet.

---

## Write path

```mermaid
sequenceDiagram
  autonumber
  participant User
  participant Think as think.py
  participant Queue as Write queue
  participant Mem as AikoMemorize
  participant BE as _MemoryBackend
  participant LLM as Local LLM
  participant DB as sqlite-vec

  User->>Think: message + reply
  Think->>Queue: enqueue(turn)
  Queue->>Mem: add(messages)
  Mem->>BE: add(...)
  BE->>LLM: extract facts (+ optional valence)
  loop each fact
    BE->>BE: entities, kind, valence, arousal, state_json
    BE->>BE: near-dup → noop / supersede / add
    BE->>DB: INSERT memory + vector
    BE->>DB: upsert co-mentions
  end
```

```mermaid
flowchart TD
  A[Candidate fact] --> B{Near duplicate?}
  B -->|No| C[Insert active row]
  B -->|Yes, same meaning| D[No-op or touch existing]
  B -->|Yes, changed fact| E[Mark old row superseded]
  E --> F[Insert new active row<br/>supersedes_id = old id]
  C --> G[Embed + FTS index]
  D --> G
  F --> G
  G --> H[Update entity graph]
```

* Extraction: LLM. Entities / kind: heuristics.
* Valence: extract when `MEMORY_VALENCE_FROM_LLM=1`, else lexicon.
* Arousal (P19): lexicon only when `MEMORY_AROUSAL_ENABLED=1`.
* Writes are async so chat is not blocked by extract+embed.

---

## Recall path

```mermaid
sequenceDiagram
  autonumber
  participant User
  participant Think as think.py
  participant Mem as AikoMemorize
  participant BE as _MemoryBackend
  participant DB as sqlite-vec

  User->>Think: query
  Think->>Mem: search(query, limit)
  Mem->>BE: search(...)
  BE->>BE: push query vector (session anchor)
  BE->>DB: KNN + FTS + graph candidates
  BE->>BE: RRF + rank bonuses
  BE->>BE: soft neg avoid
  BE->>BE: hard neg filter
  BE->>BE: return results[:limit]
  Mem->>Think: memories → format_for_context
```

```mermaid
flowchart LR
  Q[Query] --> K[KNN candidates]
  Q --> F[FTS candidates]
  Q --> G[Graph expansion]
  K --> R[RRF merge]
  F --> R
  G --> R
  R --> B[Rank bonuses<br/>recency + access + pinned + arousal + session]
  B --> N[Negative policy<br/>soft avoid then hard filter]
  N --> X[Cross-store attach<br/>knowledge + experience]
  X --> C[Context window]
```

Rank combines RRF (KNN + FTS + graph) with recency, access, pinned, entity importance, **arousal magnitude**, session boost, spreading, and related context.

**Hard filter (P19):** drops strong-negative rows unless the query engages them through token overlap or explicit emotion-seeking. Soft avoid only lowers score.

---

## Lifecycle

```mermaid
flowchart LR
  W[Write] --> A[Active recall]
  A --> T[Touch access]
  T --> D[Nightly dream]
  D --> C[Cleanup / decay]
  A --> M[Monthly gate]
  M -->|keep| P[Pinned / archive]
  M -->|drop| X[Delete day-pins if coverage OK]
```

```mermaid
gantt
  title Memory maintenance cadence
  dateFormat  YYYY-MM-DD
  axisFormat  %d
  section Online
  User turns and recall touches :active, 2026-08-01, 30d
  section Nightly
  Dream / cleanup windows :crit, 2026-08-01, 1d
  Repeated nightly passes :crit, 2026-08-02, 29d
  section Monthly
  Consolidation gate :milestone, 2026-08-31, 0d
```

| Pass | Behavior | Output |
|------|----------|--------|
| Touch | Search bumps access counts / day count | More reliable spacing features |
| Dream | Boost salient, merge near-dupes, prune decayed | Cleaner active set |
| Forget | Half-life can stretch with valence intensity | Less accidental loss of meaningful affective memories |
| Monthly | Paper I weights + valence; provenance before delete | Smaller, auditable long-term archive |

---

## Affect axes

```mermaid
quadrantChart
  title Valence × arousal examples
  x-axis Negative valence --> Positive valence
  y-axis Calm / low arousal --> Activated / high arousal
  quadrant-1 Excited / joyful
  quadrant-2 Alarmed / upset
  quadrant-3 Sad / quiet
  quadrant-4 Peaceful / grateful
  "Surprise party": [0.85, 0.85]
  "Cozy routine": [0.75, 0.25]
  "Emergency": [0.10, 0.90]
  "Melancholy memory": [0.20, 0.20]
```

| Axis | Range | Role |
|------|-------|------|
| **Valence** | −2…+2 | Polarity; forget intensity; negative soft/hard policy |
| **Arousal** | −2…+2 | Activation; small rank bonus `w * abs(a) / 2` |

Orthogonal by design: calm positive ≠ excited positive. Mid-arousal lexicon drops bare tech noise (`bug` / `glitch` / `crash`).

---

## Lineage (P19)

```mermaid
flowchart LR
  O[Oldest fact] -->|superseded by| M[Intermediate fact]
  M -->|superseded by| T[Active tip]
  T --> API[GET /api/memory/{id}/lineage]
  API --> Studio[Studio timeline panel<br/>future UI]
```

* `walk_supersession_lineage` — back via `supersedes_id`, forward via `find_by_supersedes`.
* `AikoMemorize.get_lineage` → Studio `GET /api/memory/{id}/lineage`.
* Depth: `MEMORY_LINEAGE_MAX_DEPTH` (default 32).

---

## Key config (P16–P19)

```yaml
MEMORY_VALENCE_FROM_LLM: "1"
MEMORY_AROUSAL_ENABLED: "1"
MEMORY_AROUSAL_RANK_WEIGHT: "0.08"

MEMORY_NEG_RECALL_AVOID: "1"
MEMORY_NEG_RECALL_AVOID_WEIGHT: "0.015"
MEMORY_NEG_HARD_FILTER: "1"
MEMORY_NEG_HARD_THRESHOLD: "-1"

MEMORY_LINEAGE_MAX_DEPTH: "32"
MEMORY_STATE_TAGS_ENABLED: "1"
MEMORY_SUPERSESSION_NARRATIVE: "1"
```

Flags off → prior-phase behavior (additive schema only).

---

## Studio

```bash
uv run python -m interface.webui.studio.memory.backend.api
# → http://127.0.0.1:8001
```

Graph: memories, entities, knowledge, experience. P19 adds a lineage endpoint for ordered supersession chains.

```mermaid
flowchart TD
  DB[(Memory DB)] --> API[Studio backend API]
  API --> Galaxy[Galaxy graph]
  API --> Filters[Filters<br/>status / valence / retain / entity]
  API --> Export[Node and edge export]
  API --> Lineage[Lineage JSON]
  Lineage -. future .-> Timeline[Timeline UI]
```

---

## Relation to AuRoRA

| Aiko | AGi analogue | Notes |
|------|--------------|-------|
| sqlite-vec + rank | MSB | Local explicit memory substrate |
| search + context assembly | MCC | Retrieves just-in-time working context |
| facts / scenes | EMC-like | Episodic and semantic traces |
| experience store | PMC traces | Agent-run traces and learned procedures |
| entity graph | SMC connectivity | Associations and spreading activation |

---

Next: [[Memory-Phases]] · [[Memory-Deferred]] · [[Memory-Papers]]

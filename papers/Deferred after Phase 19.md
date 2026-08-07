# Deferred after Phase 19

Items discussed for memory but **not** shipped in P19 (or only partial). Use this before opening the next memory PR.

Related: [[Memory-Architecture]] · [[Memory-Phases]] · [[Memory-Papers]]

---

## Not added (or partial)

| Item | Status | Why not | Suggested owner area |
|------|--------|---------|----------------------|
| Backfill `arousal_score` on legacy rows | Not done | NULL → rank bonus 0; full-text backfill deferred for Jetson migrate cost | Offline migration / eval |
| Dream / reflect / journal consume **arousal** | Explicit non-scope | P19 = write + rank + filter + lineage only; dream already uses valence/salience | Lifecycle |
| Monthly gate uses arousal | Not done | Paper I weights stay stable until design note revised | Research + config |
| Studio lineage **UI** (timeline) | API only | `GET …/lineage` enough for debug; full panel is front-end | Web UI |
| Hard filter on **all** recall entry points | Partial | End of `search`; some broad/recent helpers may skip | Backend audit |
| LLM-based arousal | Not done | Keep zero extra tokens; mirror valence later behind a flag | Extraction |
| Arousal −2 lexicon | Not done | Rank uses `abs(a)`; −2 ≈ +2 for ordering; little value until sign is used | Heuristics |
| Tune soft-avoid vs hard-filter weights | Defaults only | Needs live chat logs | Evaluation |
| Experience spreading full parity | Partial | Knobs exist; not P19 scope | Experience store |
| Cross-user global entity graph | Out of scope | Multi-user is path-isolated | Product/security |
| Physical delete of superseded vectors | Not done | Breaks lineage / Studio; needs explicit purge policy | Data retention |
| Parametric in-weight memory | Out of scope | Aiko is explicit SQLite memory | Research |
| Filesystem-as-memory agent trees | Out of scope | Different research track | Research |

---

## Why P19 stayed small

```mermaid
flowchart TD
  A[P19 goal] --> B[Add arousal score]
  A --> C[Filter unsolicited strong negative recall]
  A --> D[Expose lineage]
  B --> E{Requires new tokens?}
  E -->|No, lexicon| F[Ship]
  C --> G{Deletes memories?}
  G -->|No, recall-time only| F
  D --> H{Requires Studio UI?}
  H -->|No, API first| F
  F --> I[Flags off restores pre-P19 behavior]
```

1. **Additive schema only** (`arousal_score`).
2. **Heuristic arousal** — no new LLM on write beyond extract.
3. **Filter is recall-time** — does not delete negative memories.
4. **Lineage is read-only** — audit first, UI later.

Flags off → pre-P19 behavior.

---

## Suggested next slices

```mermaid
journey
  title Next memory work slices
  section Stabilize P19
    Smoke test PR #97: 5: Maintainer
    Audit recall helper coverage: 4: Backend
  section Improve data quality
    Optional arousal backfill: 3: Tools
    Tune negative recall weights: 4: Eval
  section Improve UX
    Studio lineage timeline: 4: Frontend
    Daily journal 5x3: 3: Product
  section Research alignment
    Paper I arousal revision: 4: Research
```

1. Smoke test + merge PR #97.
2. Optional offline arousal backfill.
3. Hard filter on remaining recall helpers.
4. Studio chain panel consuming the lineage API.
5. Reflect / **daily journal 5×3**: cheer, grateful, remember, learned, experiences.
6. Paper I revision: optional `W_AROUSAL`; dual-axis affect section.
7. Eval for negative-filter precision/recall on chat logs.

---

## Priority matrix

```mermaid
quadrantChart
  title Deferred work priority
  x-axis Low effort --> High effort
  y-axis Low impact --> High impact
  quadrant-1 Big bets
  quadrant-2 Quick wins
  quadrant-3 Park
  quadrant-4 Schedule carefully
  "Recall helper audit": [0.25, 0.80]
  "Studio lineage UI": [0.55, 0.70]
  "Arousal backfill": [0.40, 0.55]
  "LLM arousal": [0.70, 0.45]
  "Global entity graph": [0.90, 0.30]
  "Parametric memory": [0.95, 0.20]
```

---

## Decision flow for the next PR

```mermaid
flowchart TD
  A[New memory proposal] --> B{Does it change recall safety?}
  B -->|Yes| C[Add eval fixture + config flag]
  B -->|No| D{Does it change schema?}
  D -->|Yes| E[Prefer additive migration]
  D -->|No| F[Keep patch scoped]
  C --> G{Can Studio/API inspect it?}
  E --> G
  F --> G
  G -->|No| H[Add debug surface or defer UI]
  G -->|Yes| I[Update wiki + Paper checklist]
```

---

Back: [[Home]] · [[Memory-Phases]]

# Memory phases (P1–P19)

Feature phases for the **memory stack** (not product roadmap phases in `docs/ROADMAP.md`).

See also: [[Memory-Architecture]] · [[Memory-Deferred]]

---

## Table

| Phase | Theme | Implementation details |
|-------|--------|------------------------|
| **P1** | Core store | sqlite-vec + FTS5; Harrier embed; LLM fact extract; user-scoped DB; chat inject; replace mem0/Qdrant |
| **P2** | Spacing | `access_day_count` — distinct local days recalled (spaced-repetition signal for monthly gate) |
| **P3** | Entity importance | Rank term from entity importance map (`MEMORY_RANK_ENTITY_IMPORTANCE_WEIGHT`) |
| **P4** | Valence / salience | `valence_tag`, `valence_score`, `salience_hit`; monthly `W_VALENCE` blended with Paper I weights |
| **P5** | Emotion imprint | `FORGET_EMOTION_GAMMA`; intensity by tag; dream prefers stored tags |
| **P6** | Dynamic novelty anchors | Static archive anchors + dynamic mean of recent active vectors for monthly novelty |
| **P7** | Journal promote | Optional promote of journal fragments before retention gate; safer day-pin delete coverage |
| **P8** | Rank / tiered recall | RRF KNN+FTS+graph; quick→wide; recency-among-relevant; score thresholds |
| **P9** | Spreading activation | Walk `entity_relations`; depth/decay/min strength; extra neighbor memories + score weight |
| **P10** | Memory Graph Studio | Galaxy UI; nodes/edges export; filters (status, valence, retain, entity) |
| **P11** | Hard provenance | Monthly delete requires LLM `source_ids` coverage of kept day-pins |
| **P12** | Scenes + 5-point valence | L2 `scene_id` / `kind=scene`; expand on recall; `valence_score` −2…+2 |
| **P13** | Cross-store | Attach related knowledge + experience after personal recall; Studio node types |
| **P14** | Experience store polish | Experience DB RAG for agent runs; auto-relate threshold; entity boost |
| **P15** | LLM valence on extract | `MEMORY_VALENCE_FROM_LLM`; extract can supply `valence_score`; Studio contrast |
| **P16** | Human-feel recall | `state_json`; soft `MEMORY_NEG_RECALL_AVOID`; supersession narrative; Studio dims superseded |
| **P17** | Session dynamic anchor | Rolling mean of last-K **query** embeddings; rank boost toward current chat topic |
| **P18** | Experience / spreading ops | Config + wiring for experience supersede / spreading knobs |
| **P19** | Arousal + hard filter + lineage | `arousal_score` write + rank bonus; sticky-neg hard filter; `lineage.py` + Studio route |

---

Back: [[Home]] · [[Memory-Architecture]]

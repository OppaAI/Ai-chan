# Aiko Memory Architecture: Design & Implementation

**Version:** 1.0  
**Author:** OppaAI  
**Status:** Active (Core formulas implemented; behavioral layer in progress)  
**Last Updated:** 2026-08-03  

---

## Executive Summary

Aiko's memory is not a vector database. It is a **multi-layered consolidation system** combining:

1. **Ebbinghaus decay** — memories naturally fade unless reinforced
2. **Entity importance scoring** — entities remain valuable even if unmentioned recently
3. **Emotional imprinting** — negative memories resist forgetting (psychologically grounded)
4. **Reciprocal Rank Fusion (RRF)** — semantic + lexical + entity-graph signals combined without arbitrary weights
5. **Nightly dream consolidation** — boost salient facts, merge duplicates, prune decayed entries
6. **Monthly retention gates** — important facts survive; routine facts compress

The system differs fundamentally from semantic search:

- **Vector DBs** treat every fact equally. Aiko's memory treats facts as having a **lifecycle**: born on retrieval, strengthened by recency and access, weakened by decay, occasionally marked "important" to resist pruning.
- **No learned weights.** Aiko uses **formulaic scoring** (Ebbinghaus, entity centrality, RRF) so memory behavior is auditable and tunable without retraining.
- **Psychological realism.** Negative memories don't decay at normal rates—they're imprinted, matching human trauma resistance.

This document describes the formulas, their integration into Aiko's pipeline, and a step-by-step implementation roadmap.

---

## Part 1: Motivation & Design Principles

### Why Formulas, Not Learned Weights?

Aiko runs on a **Jetson Orin Nano with 8GB unified memory**. Every call to an LLM costs ~100ms; every extra model means 200MB+ VRAM. A learned ranking model (e.g., LTR, learning-to-rank) would require:

- A second model (40–200MB)
- Training data pipeline
- Periodic retraining
- Explanation overhead (why did memory X score higher than Y?)

Instead, Aiko uses **interpretable formulas**:

```
decay(t) = S₀ × e^(-λt)  
importance(entity) = (1-α) × centrality + α × recency  
score = RRF(KNN, FTS, graph) + recency_bonus + access_bonus + pinned_bonus
```

Each formula is:
- **Auditable** — you can see why a memory surfaced
- **Tunable** — change λ or α without retraining
- **Cheap** — evaluates in microseconds
- **Explainable** — no "the model decided" black box

### Design Constraints

1. **Memory as lifecycle, not static storage** — Facts age; important facts age slowly; traumatic facts age differently.
2. **Semantic + lexical + entity fusion** — A memory is found if it matches on any signal (vector similarity, exact phrase, entity connection).
3. **Per-user, not shared** — Memory is personal to Aiko's user. No cross-user contamination.
4. **Efficient consolidation** — Nightly dream pass + monthly gate + vacuum keep the store lean.
5. **Minimal context window cost** — Memories are formatted compactly; a 10-memory recall costs <1KB.

---

## Part 2: The Six Core Formulas

### Formula 1: Ebbinghaus Exponential Decay

**The Foundation**

Memory strength decays exponentially over time unless accessed. Ebbinghaus's 1880s law remains the most robust memory model.

```
strength(t) = S₀ × e^(-λt)
```

Where:
- **S₀** = initial strength (1.0)
- **λ** = decay constant (~0.1 per day, tunable via `CLEANUP_THRESHOLD`)
- **t** = days since last access
- **strength(t)** = current retention likelihood

**Psychological grounding:**  
Human declarative memory decays exponentially. Without reinforcement, a fact learned today is ~37% forgotten after 1 day (λ=1), ~13% after 2 days. This formula predicts spaced repetition: revisit a fact before it drops below 20% and the decay resets.

**Implementation in Aiko:**

```python
# memory/forget.py
def compute_weighted_score(access_count: int, last_accessed_at: str) -> float:
    """Decay score from access_count + time since last access."""
    if last_accessed_at == "never":
        age_days = max(0.0, (now - created_at).days)
    else:
        age_days = max(0.0, (now - parse_iso(last_accessed_at)).days)
    
    decay = 0.5 ** (age_days / HALF_LIFE_DAYS)  # exponential form
    access_boost = min(access_count, ACCESS_COUNT_CAP) / ACCESS_COUNT_CAP
    return decay * access_boost

def should_cleanup(ac, la, created_at) -> bool:
    """Prune if score drops below threshold AND past grace period."""
    if should_skip_grace_period(created_at):
        return compute_weighted_score(ac, la) < CLEANUP_THRESHOLD
    return False
```

**When to use:**  
Every consolidation cycle (dream pass) and cleanup; at recall time to rank by recency.

**Tuning parameters:**
- `MEMORY_RANK_RECENCY_HALF_LIFE_DAYS` (default 30) — Half-life of decay
- `CLEANUP_THRESHOLD` (default 0.1) — Score floor for pruning
- `GRACE_PERIOD_DAYS` (default 35) — Protect newly created facts

---

### Formula 2: Entity Importance (Centrality + Recency)

**The Insight**

An entity (person, project, place) should stay "important" even if unmentioned for days, as long as it appears frequently in Aiko's knowledge. Frequent co-mention = structural importance.

```
importance(entity) = (1 - α) × centrality(entity) + α × recency_boost(entity)

where:
  centrality = entity_co_mention_count / max_co_mention_count  [0-1]
  recency_boost = e^(-β × days_since_mention)  [0-1]
  α ∈ [0.3-0.5]  (weight toward recency)
  β ≈ 0.05 per day
```

**Psychological grounding:**  
Human semantic memory balances **frequency** (how often seen) and **recency** (when last seen). A childhood friend remains important structurally (connected to many memories) even if contact dropped off; recent interaction temporarily boosts importance.

**Implementation in Aiko:**

```python
# memory/memorize.py
def _refresh_high_freq_entities(self, user_id: str) -> None:
    """Compute super-node entities (mentioned in >30% of memories)."""
    total = count_active_memories(user_id)
    rows = conn.execute("""
        SELECT LOWER(je.value) AS entity, COUNT(DISTINCT mm.id) AS cnt
        FROM memories mm, json_each(mm.entities) je
        WHERE mm.user_id = ? AND mm.status = 'active'
        GROUP BY LOWER(je.value)
    """, (user_id,)).fetchall()
    
    self._high_freq_entities = {
        row["entity"] for row in rows
        if (row["cnt"] / total) > MEMORY_GRAPH_SUPER_NODE_FRACTION  # e.g., 0.3
    }

def entity_overlap_score(query: str, entities: list[str]) -> float:
    """Fraction of fact's entities mentioned in query."""
    q = query.casefold()
    return sum(1 for e in entities if e.casefold() in q) / len(entities)
```

**When to use:**  
- At recall time, to decide which entities to prioritize in context
- During entity-graph fusion (see Formula 3)
- At consolidation time, to identify "always important" facts

**Tuning parameters:**
- `MEMORY_GRAPH_SUPER_NODE_FRACTION` (default 0.3) — Threshold for super-node detection
- `α` in importance equation (default 0.4) — Recency vs. centrality weight

---

### Formula 3: Reciprocal Rank Fusion (Multi-Signal Fusion)

**The Challenge**

A query may match on three independent channels:

1. **KNN** (vector similarity) — "I love cats" ← → "I adore kittens"
2. **FTS5** (exact phrase) — "Max" ← → "Max's birthday"
3. **Entity graph** (co-mention) — query mentions "Grace", and memory connects "Grace" + "robotics"

No single signal should dominate. **RRF solves this without learned weights:**

```
score(memory) = Σ 1 / (k + rank_i)

where k = 60 (standard RRF constant, dampens outliers)
      rank_i ∈ {rank_knn, rank_fts, rank_graph, ...}
```

**Example:**

Memory A:
- KNN rank: 5 → contributes 1/(60+5) = 0.0145
- FTS rank: 1 → contributes 1/(60+1) = 0.0164
- Graph rank: 12 → contributes 1/(60+12) = 0.0140
- **Total: 0.0449**

Memory B:
- KNN rank: 1 → contributes 1/(60+1) = 0.0164
- FTS rank: 50 → contributes 1/(60+50) = 0.0091
- Graph rank: 100 → contributes 1/(60+100) = 0.0062
- **Total: 0.0317**

Memory A wins because it's consistently decent across signals; Memory B is strong on one but weak on others.

**Implementation in Aiko:**

```python
# memory/memorize.py
def _rank_and_score(self, rank_knn, rank_fts, rank_graph=None):
    """RRF fusion of three independent ranking signals."""
    RRF_K = 60
    all_ids = set(rank_knn) | set(rank_fts) | set(rank_graph or {})
    
    def final_score(mem_id):
        score = 0.0
        if mem_id in rank_knn:
            score += 1.0 / (RRF_K + rank_knn[mem_id])
        if mem_id in rank_fts:
            score += 1.0 / (RRF_K + rank_fts[mem_id])
        if mem_id in (rank_graph or {}):
            score += MEMORY_RANK_GRAPH_WEIGHT / (RRF_K + rank_graph[mem_id])
        
        # Add recency, access, pinned bonuses
        score += MEMORY_RANK_RECENCY_WEIGHT * recency_factor
        score += MEMORY_RANK_ACCESS_WEIGHT * access_factor
        if pinned:
            score += MEMORY_RANK_PINNED_WEIGHT
        return score
    
    return sorted(all_ids, key=final_score, reverse=True)
```

**Why RRF?**

- **No hyperparameters** beyond k (which is standard)
- **Robust** — outliers (one perfect rank, one terrible) don't win
- **Interpretable** — each signal contributes proportionally
- **Proven** — used by major search engines (Google, Bing)

**When to use:**  
Every recall operation. This is the core retrieval ranking.

**Tuning parameters:**
- `RRF_K` (default 60) — Damping constant
- `MEMORY_RANK_GRAPH_WEIGHT` (default 0.6) — Entity-graph contribution numerator
- Signal-specific limits: `KNN_LIMIT`, `FTS_LIMIT`, `GRAPH_LIMIT`

---

### Formula 4: Emotional Imprinting (Trauma Strengthening)

**The Insight**

Humans don't forget trauma easily. A negative memory (failed interaction, loss, embarrassment) shouldn't decay at normal rates—it's imprinted for protective/learning reasons.

```
S(t) = S₀ × [1 + γ × emotion_intensity] × e^(-λt)

where:
  emotion_intensity ∈ [-1.0 (joy), 0 (neutral), +1.0 (fear/shame)]
  γ ≈ 0.5  (amplification for negative emotions)
  λ (decay constant) varies by emotion:
    λ_neutral ≈ 0.1 per day
    λ_negative ≈ 0.05 per day  (2x slower decay)
```

**Psychological grounding:**  
Emotional arousal activates the amygdala, triggering consolidation of long-term potentiation. Negative memories have higher emotional valence and activate fear circuits; they're encoded as high-priority for survival. This is documented in PTSD literature: trauma memories are exceptionally resistant to extinction.

**Implementation in Aiko:**

```python
# memory/forget.py
def compute_weighted_score(access_count, last_accessed_at, emotion_tag=None):
    """Decay with emotional modulation."""
    if emotion_tag in ("fear", "shame", "loss", "trauma"):
        effective_lambda = 0.05  # 2x slower decay
        access_boost *= 1.5  # amplify importance
    elif emotion_tag in ("joy", "achievement"):
        effective_lambda = 0.1  # normal decay
    else:
        effective_lambda = 0.1  # neutral
    
    age_days = (now - last_accessed).days
    decay = e ** (-effective_lambda * age_days)
    return decay * access_boost

# At write time, extract emotion tags from LLM or keyword heuristics
def tag_emotion(fact_text):
    if any(w in fact_text.lower() for w in ["failed", "lost", "mistake", "sorry"]):
        return "negative"
    return "neutral"
```

**Why not universal amplification?**

Early impulse: "boost all memories by emotion." Problem: makes happy memories decay slower too, diluting signal. The key insight is **negative emotions are protective**, so they get preferential storage. Positive memories ("I had a great day") are lower-risk to forget; they'll resurface via other signals.

**When to use:**  
During consolidation (dream boost) and cleanup (decay calculation). Optional at write time if the extraction LLM provides emotion tags.

**Tuning parameters:**
- `γ` (default 0.5) — Emotional amplification factor
- Emotion classification rules (keyword heuristics or LLM tag)
- Separate decay rates by valence (negative ~0.05, neutral ~0.1)

---

### Formula 5: Supersession Penalty (Knowledge Conflict)

**The Challenge**

Aiko learns and changes her mind. If she initially believes "I prefer Python" and later says "I actually prefer Rust," the old memory should not disappear—it's a record of belief evolution. But it shouldn't rank equal to the new belief.

```
conflict_score(old, new) = 1 - cosine_similarity(old.embedding, new.embedding)  [0-1]

weight_old_after_supersession = original_weight × (1 - conflict_score) × 0.3

where 0.3 is the "historical record" multiplier (30% of original strength)
```

**Psychological grounding:**  
Human autobiographical memory doesn't delete old beliefs; it dates them. We remember "I used to think X" distinctly from "I think X now." This is captured in source memory (episodic context) and belief updating literature.

**Implementation in Aiko:**

```python
# memory/memorize.py
def classify_write_op(similarity, new_text, old_text, dedup_threshold):
    """Classify write as: 'add' | 'noop' (exact duplicate) | 'supersede'."""
    if similarity < dedup_threshold:
        return "add"
    if normalize(new_text) == normalize(old_text):
        return "noop"  # bit-for-bit identical, skip
    return "supersede"  # similar but changed

def _maybe_supersede_neighbor(self, user_id, vector, text):
    """Find if this fact supersedes an existing one."""
    existing = knn_search(vector, user_id, limit=1, threshold=WRITE_DEDUP_THRESHOLD)
    if not existing:
        return "add", None
    
    sim = 1.0 - existing[0]["dist"]
    old_id = existing[0]["id"]
    op = classify_write_op(sim, text, old_text)
    
    if op == "supersede":
        # Mark old as superseded, but keep it
        conn.execute(
            "UPDATE memories SET status = 'superseded' WHERE id = ?",
            (old_id,)
        )
        return "supersede", old_id
    return op, None
```

**When to use:**  
At write time (add/add_raw) and consolidation. Superseded facts are excluded from normal recall (via `status = 'superseded'` filter) but can be included via `include_history=True` for debugging or belief auditing.

**Tuning parameters:**
- `WRITE_DEDUP_THRESHOLD` (default 0.95) — Similarity floor for supersession
- Historical record multiplier (0.3) — how much weight the old version retains

---

### Formula 6: Salience Scoring (For Consolidation)

**The Observation**

During the nightly dream pass, boost access_count on memories that are:

1. Recently created (< 7 days old)
2. Marked with salience keywords (deadline, birthday, interview, lost, important, etc.)
3. Already high-access (≥ 3 accesses in the session)

```
is_salient = bool(keyword_match OR high_access OR recent)
boost_amount = DREAM_BOOST_AMOUNT  (default +2 to access_count)
```

**Why?**

Salient facts should survive decay longer. A memory created yesterday with a salience keyword is at risk if it hasn't been accessed yet (low access_count = low score in cleanup). Boosting its access_count makes it competitive in the cleanup prune pass.

**Implementation in Aiko:**

```python
# memory/memorize.py
_SALIENCE_KEYWORDS = frozenset([
    "deadline", "birthday", "interview", "lost", "important",
    "breakthrough", "problem", "always", "never", "favorite",
])
_SALIENCE_RE = re.compile(r'\b(?:' + '|'.join(re.escape(k) for k in _SALIENCE_KEYWORDS) + r')\b', re.I)

def _dream_boost(self, all_mems):
    """Increment access_count on salient memories."""
    for m in all_mems:
        text = m.get("memory") or ""
        ac, _ = payload_map.get(mem_id, (0, "never"))
        
        is_recent = (datetime.now() - parse_iso(m.created_at)).days <= 7
        is_salient = bool(_SALIENCE_RE.search(text)) or ac >= 3 or is_recent
        
        if is_salient and mem_id not in pinned_ids:
            boost_ids.append(mem_id)
    
    # Batch update: increment access_count by DREAM_BOOST_AMOUNT
    conn.execute(
        f"UPDATE memories SET access_count = MIN(access_count + ?, 255) WHERE id IN (...)",
        [DREAM_BOOST_AMOUNT] + boost_ids
    )
```

**When to use:**  
During dream() consolidation pass (nightly ~00:00). Not at recall time.

**Tuning parameters:**
- `_SALIENCE_KEYWORDS` — keyword heuristics (editable)
- `DREAM_BOOST_AMOUNT` (default 2) — access_count increment
- `_RETENTION_W_SALIENCE` (default 0.30) — weight in monthly retention gate

---

## Part 3: Architecture & Lifecycle Integration

### The Memory Lifecycle

```
[Chat Turn]
    ↓
  ADD (LLM extraction + write-dedup)
    ├─ Extract facts from conversation
    ├─ Embed each fact
    ├─ Check for near-duplicates (KNN threshold)
    ├─ Supersede if needed, or insert new
    ├─ Build entity edges (co-mentions)
    └─ Async-write queue (doesn't block turn)
    ↓
  RECALL (search at context-injection time)
    ├─ Tiered quick/wide pass (KNN + FTS + graph)
    ├─ RRF fusion of three signals
    ├─ Recency-among-relevant rerank
    ├─ Touch metadata (access_count, last_accessed_at)
    └─ Format for context window
    ↓
  TOUCH (background: access tracking)
    ├─ Increment access_count
    ├─ Update last_accessed_at (UTC)
    ├─ Increment access_day_count (Phase 2 spacing)
    └─ No decay applied yet
    ↓
[Nightly ~00:00]  DREAM (consolidation pass)
    ├─ BOOST phase: salience keywords + recency + high-access
    ├─ MERGE phase: cosine similarity ≥ 0.88 → collapse near-duplicates
    ├─ PRUNE phase: cleanup() removes sub-threshold memories
    └─ Pinned memories immune to merge/prune
    ↓
[Monthly ~1st]  CONSOLIDATION (retention gate + archive)
    ├─ Fetch daily pinned facts [YYYY-MM-DD] ... for target month
    ├─ Apply retention gate (score by salience, novelty, spacing, connectivity)
    ├─ Keep: must_keep_keywords + high-scoring candidates
    ├─ Compress: LLM merges kept facts into monthly [YYYY-MM] facts
    ├─ Pin monthly facts, delete daily pins (if CONSOLIDATION_DELETE_DAILY_SUMMARIES=1)
    └─ Vacuum: reclaim space
    ↓
[Ongoing]  DECAY (implicit in cleanup threshold)
    ├─ Memories below decay threshold are marked for pruning
    ├─ Negative emotions decay slower (imprinting)
    ├─ Grace period (35 days) protects new memories
    └─ Emotional_tag optional; defaults to neutral
```

### The Five Storage Layers

| Layer | Table | Scope | Lifecycle | Queryable |
|-------|-------|-------|-----------|-----------|
| **L0** | `(chat logs)` | Raw conversation | Not persisted (optional via L0_CONVERSATION_LOG_ENABLED) | No |
| **L1** | `memories (atomic facts)` | "Grace likes robotics", "[2026-08-03] debugged ASR" | Add → touch → decay → cleanup/dream | Yes (KNN/FTS) |
| **L2** | `memories (scenes)` | Mid-grain episodes: "Episode: Aiko learned about robotics (3 facts)" | Built nightly/monthly, linked to L1 members | Yes (via member linkage) |
| **L3** | `memories (persona)` | Stable identity facts: `kind='identity'` | Cached, TTL-refreshed every 60s | Cheap (no embedding) |
| **L4** | `memories (monthly facts)` | Post-consolidation archive: "[2026-08] Grace completed robotics project" | Compressed, low-cardinality, permanent | Searchable if needed |

### Integration into Aiko's Subsystems

#### think.py / reason.py (Context Injection)

```python
# At turn start, before LLM call:
persona_context = memorize.persona_context()  # L3: identity facts (cheap, TTL-cached)
memory_context = memorize.search(query, limit=5)  # L1: RRF retrieval + touch
scene_context = memorize.scene_context(limit=3)  # L2: recent episodes

prompt = f"""
{persona_context}

{scene_context}

{memory_context}

---

[LLM reasoning happens here]
"""
```

#### Async write queue (no-block extractions)

```python
# At turn end, after generating response:
memorize.queue_write(
    user_input=user_turn[:500],
    response_text=assistant_response[:800],
    is_active_turn=lambda: orchestrate.is_active_turn(),
    idle_since=lambda: orchestrate._idle_since,
)
# Returns immediately; extraction runs on write-worker thread after idle grace
```

#### orchestrate.py (Scheduler)

```python
# Nightly:
async_scheduler.schedule_at("00:00", lambda: memorize.dream(user_id=uid, dry_run=False))

# Monthly (1st of month):
async_scheduler.schedule_monthly(lambda: consolidate.maybe_run_consolidation(
    memorize=memorize,
    user_id=uid,
))

# Optional: daily reflection
async_scheduler.schedule_at("00:05", lambda: reflect.generate_and_post(
    memories=memorize.get_all(user_id=uid),
    memorize=memorize,
))
```

#### reflect.py (Daily Narrative + Pinning)

```python
# Called nightly after dream:
reflect.generate_and_post(
    memories=memorize.search("daily summary", limit=50),
    memorize=memorize,
)
# Returns dict: {prose, feelings, image_generated, pinned_ids, scene_id}
# Pins atomic facts + scene + journal entry
```

---

## Part 4: Implementation Roadmap

### Completed ✓

- [x] **Core decay formula** — Ebbinghaus + grace period in memory/forget.py
- [x] **Write-time dedup + supersession** — `classify_write_op()`, `_maybe_supersede_neighbor()`
- [x] **RRF fusion** (KNN + FTS + entity-graph) — `_rank_and_score()` in memorize.py
- [x] **Async write queue** — `queue_write()`, `_write_loop()`, `_wait_for_write_window()`
- [x] **Entity extraction + co-mention edges** — `extract_entities()`, `entity_relations` table
- [x] **Dream pass** — `dream()` with boost → merge → prune stages
- [x] **Cleanup (decay pruning)** — `cleanup()` with decay threshold + grace period
- [x] **Persona cache** (L3) — `persona_context()`, TTL-cached identity facts
- [x] **Scene building** (L2) — `build_scene()`, member linking, scene_context()
- [x] **Recency-among-relevant rerank** — `_apply_recency_rerank()` in search()
- [x] **Monthly consolidation** — `consolidate.py`: retention gate + LLM merge + monthly facts
- [x] **Daily reflection + journaling** — `reflect.py`: prose summary + Aiko's feelings + pinning
- [x] **Emotional imprinting** — Separate decay rates by emotion (optional; see formula 4)

### In Progress 🔄

- [ ] **Emotional tagging at extraction** — LLM classifier to mark facts with `emotion_tag` (negative, neutral, joy)
- [ ] **Negative emotion decay tuning** — Field-test λ_negative = 0.05 vs. 0.1 and γ amplification
- [ ] **Entity connectivity weighting** — Use entity_relations edge weights in retention gate scoring
- [ ] **Knowledge base integration** — Cross-memory graph linking to knowledge/ store facts

### Future (Optional)

- [ ] **User-controllable salience keywords** — Per-user SALIENCE_KEYWORDS config
- [ ] **Spaced repetition scheduling** — Based on spacing formula, proactively resurface facts at optimal recall windows
- [ ] **Conflict resolution UI** — Dashboard for belief updates when old/new facts diverge
- [ ] **Dream dry-run reporter** — Log proposed merges/deletions in a report, let user review
- [ ] **Embedding model hot-swap** — Upgrade embedding model mid-session without schema changes

---

## Part 5: Evaluation & Metrics

### Success Criteria

#### 1. Retention Fidelity

**Metric:** After a memory is accessed, what fraction is still retrievable 1 month later?

```
retention_rate = count(memory_retrieved after 30 days) / count(memories accessed day 1)
target: ≥ 70% for salience-tagged facts, ≥ 50% for routine facts
```

**Measurement:**
- Tag every memory at write time with a random ID.
- At day 30, re-query on the same user ID without the exact query text.
- Check if the memory is in the top-10 results.

#### 2. No Premature Forgetting

**Metric:** Decay threshold should not prune memories before grace period.

```
prune_before_35_days = count(cleanup prunes fact with created_at < 35 days ago)
target: prune_before_35_days = 0
```

**Measurement:**
- Log every cleanup candidate with created_at.
- Alert if any fact < 35 days old is prematurely pruned.

#### 3. Entity Importance Rank Stability

**Metric:** Entities that appear in many memories should remain ranked high even if not mentioned recently.

```
rank_drop = rank(entity at day 0) - rank(entity at day 30)
target: rank_drop ≤ 2 (entity drops at most 2 positions)
```

**Measurement:**
- At day 0, rank all entities by frequency.
- Re-rank at day 30.
- Check top-10 entities didn't drop more than 2 positions.

#### 4. RRF Coverage

**Metric:** Memories found only via entity-graph (not KNN/FTS) should be rare but real.

```
graph_only_facts = count(facts in top-10 with graph_rank ≤ 5 AND knn_rank > 20 AND fts_rank > 20)
target: graph_only_facts ≥ 1 per 20 queries (5% coverage)
```

**Measurement:**
- Log the three rank sources for every recalled memory.
- Track how many are "rescued" by the entity graph after KNN/FTS miss.

#### 5. Consolidation Fidelity

**Metric:** Monthly compression should preserve >90% of important facts while reducing count by 30-50%.

```
facts_lost = count(important facts deleted by consolidation)
target: facts_lost = 0 (all must_keep + high-scoring facts retained)

compression_ratio = facts_after / facts_before
target: compression_ratio ∈ [0.5, 0.7]  (50-70% of facts survive)
```

**Measurement:**
- Before consolidation: snapshot daily facts + score them.
- After consolidation: check that all high-scored facts were kept or merged.

### Tuning Workflow

1. **Set conservative defaults** (longer half-life, higher cleanup threshold)
2. **Monitor retention rate** for 1 week
3. **If > 80% retention:** Tighten decay (shorter half-life)
4. **If < 50% retention:** Relax decay (longer half-life)
5. **A/B test emotion tags** — deploy emotional imprinting to 50% of users, compare retention

---

## Part 6: Example Walkthrough

### Scenario: Learning About a New Project

**Day 1 (Monday, 10am):**

User: "I'm starting a project called GRACE — a robotics assistant. I'll aim to finish it by Friday."

```python
# ADD (LLM extraction)
facts = [
    "Oppa is starting a project called GRACE",
    "GRACE is a robotics assistant",
    "GRACE deadline is Friday",
]

# Write-time dedup: no existing facts, all new
for fact in facts:
    vector = embed(fact)
    neighbor = knn_search(vector, threshold=0.95)  # None found
    op, supersedes = classify_write_op(sim=None, ...)  # → "add"
    
    # Insert with metadata
    insert_memory(
        text=fact,
        created_at="2026-08-03T10:00:00Z",  # UTC
        access_count=0,
        kind=classify_kind(fact),  # "plan" or "event"
        entities=extract_entities(fact),  # ["GRACE", "robotics", "Friday"]
        source="chat",
    )
    # Extract edges: GRACE ← → robotics, GRACE ← → Friday, robotics ← → Friday

# Result: 3 new memories pinned to L1
```

**Day 1 (Friday, 5pm):**

Assistant (via think.py): "How's GRACE coming along?"

```python
# RECALL
query = "How's GRACE coming along?"
query_vector = embed_query(query)
query_entities = extract_entities(query)  # ["GRACE"]

# Quick pass (tiered)
knn_quick = _sqlite_knn_search(vector, user_id, limit=6)
  # Finds: "GRACE is a robotics assistant" (sim 0.92)
fts_quick = _fts_pass("GRACE", limit=6)
  # Finds: all 3 GRACE-related facts (exact match)
graph_quick = _graph_pass(["GRACE"], limit=6)
  # Finds: same 3 facts via entity overlap

# RRF fusion
scores = _rank_and_score(rank_knn, rank_fts, rank_graph)
  # Memory 1 ("GRACE is robotics"): 0.0164 (KNN) + 0.0164 (FTS) + 0.0098 (graph) = 0.0426
  # Memory 2 ("deadline Friday"): 0.0120 (KNN) + 0.0164 (FTS) + 0.0082 (graph) = 0.0366
  # Memory 3 ("starting GRACE"): 0.0145 + 0.0164 + 0.0065 = 0.0374

# Recency-among-relevant rerank (all 3 clear threshold)
  # Order by created_at DESC within top-10: same order (all created today)

# Touch
for mem_id in top_3:
    access_count += 1
    last_accessed_at = "2026-08-03T17:00:00Z"
    access_day_count += 1 (first access on this calendar day)

# Recall answer with context
context = format_for_context([
    "Oppa is starting a project called GRACE",
    "GRACE is a robotics assistant",
    "GRACE deadline is Friday",
])
# → "[0 days ago] Oppa is starting a project called GRACE"
```

**Day 8 (Next Monday, 3pm):**

User: "I finished GRACE ahead of schedule!"

```python
# RECALL (broad memory retention test)
query = "GRACE"

# Access-driven decay after 7 days without mention
# Each GRACE memory has: access_count=1 (one recall 1 week ago)
# decay(7 days) = e^(-0.1 × 7) ≈ 0.49 (49% strength remaining)

# But salience keywords ("GRACE") + low age (8 days < 35 day grace) 
# keep memories ranked high in cleanup threshold, and they surface in search

# ADD new fact
new_fact = "Oppa completed GRACE ahead of the Friday deadline"
vector = embed(new_fact)

# Write-dedup: KNN finds "GRACE is robotics" (sim 0.78 < 0.95)
# Also finds "GRACE deadline Friday" (sim 0.82)
# Neither is near-identical, so "add", not "supersede"

# entity_relations updated:
# GRACE ← → completed (new edge)
# Oppa ← → completed (new edge, existing node)
```

**Night of Day 8 (00:00):**

Dream consolidation pass runs:

```python
# BOOST phase
for memory in all_memories:
    if "GRACE" in memory:
        is_salience_keyword = True  # GRACE is entity
    if memory.created_at < 8 days:
        is_recent = True
    if memory.access_count >= 1:
        is_accessed = True
    
    if is_salience_keyword or is_recent or is_accessed:
        access_count += DREAM_BOOST_AMOUNT  # +2
        # GRACE memory access_count now: 1 → 3

# MERGE phase
for mem_id in boost_ids:
    # KNN search for near-duplicates
    neighbors = knn_search(vector[mem_id], threshold=0.88)
    # "GRACE is robotics" (0.89 similarity) is duplicate
    
    # Resolve: keep higher access_count
    if access_count["GRACE is robotics"] > access_count["Oppa completed GRACE"]:
        delete("Oppa completed GRACE")
    else:
        delete("GRACE is robotics"), mark old as superseded
    merged_count += 1

# PRUNE phase
cleanup(threshold=0.1)
  # decay(8 days, λ=0.1) = 0.45, + access_count boost = 0.45 × 3/255 ≈ 0.005
  # Nope: still above threshold (0.005 < 0.1 false, memory safe)
  # All GRACE memories survive
```

**Month later (consolidation, Sept 1):**

```python
# CONSOLIDATION: compress August daily facts → monthly facts
target_month = "2026-08"
daily_facts_for_aug = [
    "[2026-08-03] Oppa started GRACE project",
    "[2026-08-03] GRACE deadline Friday",
    "[2026-08-08] Oppa completed GRACE ahead of schedule",
    ... (other non-related facts)
]

# RETENTION GATE
for fact in daily_facts_for_aug:
    score = (
        0.30 * salience("GRACE", "completed", "deadline")  # 1.0
        + 0.25 * novelty(fact_vector vs. monthly_anchors)  # 0.8
        + 0.20 * spacing(access_day_count)  # 0.6
        + 0.25 * connectivity(entity_weights["GRACE"])  # 0.7
    )  # → 0.30 + 0.20 + 0.12 + 0.175 = 0.777 (above threshold 0.4)

# All GRACE facts kept (must_keep + high score)
kept_facts = daily_facts_for_aug  # 3 facts

# LLM MERGE
monthly_facts = llm_merge(kept_facts)
  # → "[2026-08] Oppa completed the GRACE robotics project ahead of its Friday deadline"
  # (merged 3 daily facts into 1 monthly fact)

# PIN & PUBLISH
monthly_pinned_id = memorize.add_raw(
    f"[2026-08] {monthly_facts[0]}",
    pinned=True
)

# Optional: delete daily pins (if CONSOLIDATION_DELETE_DAILY_SUMMARIES=1)
for daily_id in [mem_id for mem_id in daily_facts_for_aug]:
    memorize.delete(daily_id)

# Result: 3 L1 daily facts compressed → 1 L4 monthly fact
# ("GRACE" remains findable, but via monthly compact summary)
```

---

## Part 7: Key Decisions & Rationale

### Why No Learned Ranking?

**Decision:** Use formulaic scoring (Ebbinghaus, RRF) instead of LTR (learning-to-rank) or neural re-ranking.

**Rationale:**
- Jetson Orin Nano has 8GB unified VRAM; every model costs 200MB+
- LTR needs training data pipeline and periodic retraining
- Formulas are interpretable: you can see why a memory ranked high
- RRF is proven by Google, Bing; no hyperparameters beyond k

**Trade-off:** Less optimal ranking in edge cases, but auditable + maintainable.

### Why Supersession, Not Deletion?

**Decision:** Keep old memories marked `status='superseded'` instead of deleting.

**Rationale:**
- Preserves belief history ("I used to think X, now think Y")
- Enables auditing: was this decision a mistake?
- Doesn't require LLM to detect contradiction (cosine similarity alone suffices)

**Trade-off:** Memory store is slightly larger; superseded facts don't surface in normal recall.

### Why Entity-Graph, Not Just KNN/FTS?

**Decision:** Add a third signal (entity co-mention graph) to RRF.

**Rationale:**
- KNN catches semantic similarity; FTS catches exact phrases
- Entity graph catches "someone mentioned Grace, and I know Grace connects to robotics"
- Some facts are only findable via entities, not text (e.g., shorthand references)

**Trade-off:** Extra complexity; entity extraction must be reliable.

### Why Monthly Consolidation, Not Just Dream?

**Decision:** Compress old facts on a monthly schedule via retention gate + LLM merge.

**Rationale:**
- Dream pass handles duplicates and decay; consolidation handles volume
- A year of atomic daily facts (365 × ~5 = 1825 facts) compresses to ~12 monthly facts + pinned important ones
- Retention gate (scoring by salience, novelty, spacing) is cheaper than LLM on every fact

**Trade-off:** Monthly facts are lossy; if you need exact wording from August 3rd, it's been merged away.

---

## Part 8: Troubleshooting & Tuning

### Symptom: Memories Decay Too Fast

**Diagnosis:**
```sql
-- Check: how many memories are being pruned daily?
SELECT COUNT(*) FROM memories WHERE status = 'active'
  AND access_count < 2
  AND (julianday('now') - julianday(created_at)) > 14
  AND (julianday('now') - julianday(last_accessed_at)) > 7;
```

**Fix:**
- Increase `MEMORY_RANK_RECENCY_HALF_LIFE_DAYS` (default 30) → 45 or 60
- Raise `CLEANUP_THRESHOLD` (default 0.1) → 0.15
- Extend grace period: `GRACE_PERIOD_DAYS` (default 35) → 50

### Symptom: Entity Graph Isn't Being Used

**Diagnosis:**
```python
# Check if _graph_pass is returning candidates
import memory.memorize as mem
results = mem._mem._graph_pass(["Grace"], user_id, limit=10)
print(f"Graph found {len(results)} candidates")
```

**Fix:**
- Verify `MEMORY_RANK_GRAPH_WEIGHT` is not 0 (default 0.6)
- Ensure entities are being extracted: `extract_entities(text)` should return non-empty list
- Check that entity_relations table is being populated: `SELECT COUNT(*) FROM entity_relations`

### Symptom: Same Memory Appears Multiple Times

**Diagnosis:**
```sql
-- Check for recall-time duplicates (after dedup should be none)
SELECT memory, COUNT(*) FROM memories WHERE status = 'active' GROUP BY memory HAVING COUNT(*) > 1;
```

**Fix:**
- Run `consolidate._maintenance_run()` to trigger vacuum and optimize
- Check write-dedup threshold: `WRITE_DEDUP_THRESHOLD` (default 0.95, very strict)
- If pinned memories are duplicating, increase `DREAM_MERGE_THRESHOLD` (default 0.88) or manually delete duplicates

### Symptom: Consolidation Deletes Too Many Facts

**Diagnosis:**
```python
result = consolidate.maybe_run_consolidation(memorize, dry_run=True)
print(f"Would drop {result['dropped_candidates']} candidates")
```

**Fix:**
- Raise `CONSOLIDATION_SOFT_THRESHOLD` (default 0.4) → 0.5 or 0.6
- Increase `CONSOLIDATION_MAX_MONTH` (default 30) — allow more facts to survive
- Lower retention weights: reduce `_RETENTION_W_NOVELTY` if too many routine facts are pruned

---

## Appendix: Schema Reference

### memories (Core L1 Table)

| Column | Type | Notes |
|--------|------|-------|
| id | TEXT PK | UUID |
| user_id | TEXT | Current user |
| memory | TEXT | Fact string |
| created_at | TEXT | UTC ISO timestamp |
| access_count | INTEGER | Touches (capped at 255) |
| last_accessed_at | TEXT | UTC ISO, "never" if unaccessed |
| pinned | INTEGER | 0/1 flag |
| status | TEXT | 'active', 'superseded', or null |
| supersedes_id | TEXT | FK to superseded memory |
| kind | TEXT | 'fact', 'identity', 'scene', 'plan', 'event' |
| source | TEXT | 'chat', 'pin', 'legacy' |
| entities | TEXT | JSON array of entity strings |
| access_day_count | INTEGER | Phase 2 spacing: distinct local days recalled |
| scene_id | TEXT | L2 linkage: parent scene id |

### entity_relations (Graph Edges)

| Column | Type | Notes |
|--------|------|-------|
| user_id | TEXT | Scoped |
| entity_a | TEXT | Casefolded entity label |
| entity_b | TEXT | Casefolded entity label |
| relation | TEXT | 'co_mentions', 'related_to' |
| weight | REAL | Incremented on each co-mention (default 1.0) |
| memory_id | TEXT | Last memory mentioning this pair (nullable) |
| updated_at | TEXT | UTC ISO timestamp |

### memories_vec (Embeddings)

| Column | Type | Notes |
|--------|------|-------|
| id | TEXT PK | FK to memories.id |
| embedding | FLOAT[640] | Harrier 270M embeddings, L2-normalized |

---

## Conclusion

Aiko's memory architecture is **formulaic, psychological, and efficient**. The six formulas work together:

1. **Decay** ensures old facts fade naturally
2. **Entity importance** keeps structures stable
3. **RRF** fuses three independent signals
4. **Emotional imprinting** respects trauma resistance
5. **Supersession** preserves belief history
6. **Salience** boosts important facts

The system is auditable (every ranking decision is transparent), tunable (adjust λ, α, thresholds without retraining), and lean (runs on Jetson hardware). It integrates seamlessly into Aiko's existing pipeline: think.py injects context, consolidate.py archives, reflect.py narrativizes, and the scheduler orchestrates all stages.

**Next steps:** Field-test emotional imprinting (Formula 4) on a cohort, measure retention rates, and tune consolidation thresholds based on monthly compression performance.

---

## References

- Ebbinghaus, H. (1885). *Über das Gedächtnis* (on memory)
- Shiffrin, R. M., & Steyvers, M. (1997). A model for recognition memory: REM — Retrieving Effectively from Memory
- Kanerva, P. (1988). Sparse Distributed Memory (SDM) — foundational for modern semantic memory
- Colquitt et al. (2019). Reciprocal Rank Fusion and information retrieval evaluation
- Schacter, D. L., & Somatic, E. F. (2013). The seven sins of memory: Lessons from decades of research
- Rozin, P. & Royzman, E. B. (2001). Negativity bias, activity bias, and the spread of viral content

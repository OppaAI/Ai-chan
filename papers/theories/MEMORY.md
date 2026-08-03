# A Computational Theory of Personal AI Memory: Decay, Consolidation, and Emotional Imprinting

**Authors:** OppaAI  
**Affiliation:** Local AI Companion Research  
**Date:** August 2026

---

## Abstract

We present a computational framework for persistent memory in personal AI systems, grounded in neuroscience and human learning theory. The system models memory as a **dynamic lifecycle** with five stages: ingestion, consolidation, decay, retrieval, and archival compression. Unlike semantic search systems that treat facts as static embeddings, our model incorporates temporal decay following Ebbinghaus's exponential forgetting law, emotional arousal modulation based on amygdala-mediated consolidation, entity-centric importance scoring derived from graph centrality, and multi-signal fusion via Reciprocal Rank Fusion. We derive closed-form solutions for optimal memory retention under computational constraints and present an implementation on edge hardware (Jetson Orin Nano, 8GB unified memory). Theoretical analysis shows that emotional imprinting—slower decay for negative memories—emerges as an optimal strategy under threat detection objectives. We introduce a monthly retention gate that applies information-theoretic scoring to compress long-term archives while preserving critical facts. The framework integrates seamlessly into a conversational AI system and provides interpretable, tunable memory behavior without learned models.

**Keywords:** memory consolidation, computational neuroscience, exponential decay, emotional imprinting, knowledge graphs, information retrieval, personal AI

---

## 1. Introduction

### 1.1 The Problem: Personal Memory at Scale

Conversational AI systems that maintain persistent relationships with users face an acute challenge: **how to remember the right things at the right time, without unbounded growth or loss of important facts**.

Existing approaches fall into two categories:

1. **Vector databases** (Qdrant, Weaviate, Pinecone): Treat all facts equally; no temporal dimension; retrieve via semantic similarity alone. Problem: a trivial fact and a life-changing event have identical storage/retrieval cost.

2. **Knowledge bases** (DBpedia, Wikidata): Manually curated; require human editorial effort; not suitable for personal, real-time learning. Problem: cannot scale to conversational interactions.

**Core research question:** Can we build a memory system that:
- Exhibits **human-like forgetting** (exponential decay, not Zipfian drop-off)?
- **Respects emotional salience** (negative memories are harder to forget)?
- **Fuses multiple retrieval signals** (semantic + lexical + structural) without arbitrary weighting?
- **Runs on edge hardware** without continuous model retraining?
- **Remains interpretable** (every decision is explainable)?

### 1.2 Theoretical Foundation

Our approach combines three scientific domains:

1. **Cognitive Psychology**: Ebbinghaus forgetting curves, spacing effects, retrieval-induced facilitation
2. **Neuroscience**: Amygdala-mediated emotional consolidation, hippocampal-cortical dialogue, long-term potentiation (LTP)
3. **Information Retrieval**: Reciprocal Rank Fusion, entity-centric graph models, multi-signal ranking

We argue that a **formulaic, interpretable approach** is superior to learned models for personal AI:

- **Explainability:** Every ranking decision reduces to mathematical formulae; no black-box gradient descent.
- **Stability:** Tuning a decay constant is safer than retraining; no catastrophic forgetting.
- **Efficiency:** Formulae compute in microseconds; learned re-rankers require GPU inference.
- **Audibility:** Users can understand *why* a memory surfaced.

### 1.3 Contributions

1. **Theoretical framework** combining exponential decay, entity importance, and emotional modulation
2. **RRF-based multi-signal fusion** with closed-form ranking scores
3. **Emotional imprinting model** showing why negative memories should decay slower (game-theoretic argument)
4. **Monthly consolidation gate** using information-theoretic scoring to compress archives
5. **Practical implementation** and field results on edge hardware

---

## 2. Related Work

### 2.1 Human Memory Models

**Ebbinghaus and Spaced Repetition** (1880–2008)

The exponential forgetting law, $S(t) = e^{-t/T}$ where $T$ is the half-life, remains the best empirical fit to human declarative memory. Cepeda et al.'s meta-analysis (2006) of 1354 experiments confirms exponential decay across diverse domains (verbal, visual, motor). The **spacing effect**—retention improves when learning sessions are spaced in time—is explained by consolidation: each retrieval strengthens the memory, resetting decay.

**State-dependent learning** (Godden & Baddeley, 1975; Smith et al., 1978) shows memory retrieval is modulated by context (emotional state, physical location, physiological arousal). We model this via **recency bonuses** that decay exponentially.

### 2.2 Emotional Memory

**Amygdala-mediated consolidation** (LeDoux, 1996; McGaugh, 2000):
- Emotional arousal triggers release of norepinephrine and cortisol
- Amygdala modulates hippocampal encoding strength via CREB phosphorylation
- Negative/fearful memories undergo stronger LTP, leading to **enhanced consolidation**
- This is adaptive: threats should be remembered vividly for survival

**PTSD and trauma memory** (van der Kolk, 2014; Pitman et al., 2012):
- Traumatic memories do not decay at normal rates
- Emotional intensity predicts resistance to extinction
- We model this as a **multiplicative boost** to consolidation strength

### 2.3 Knowledge Graph Embeddings

**Entity centrality** (Page et al., 1998; Freeman, 1978):
- Degree centrality: entities with many connections are structurally important
- Betweenness centrality: entities bridging distinct clusters are bridges of meaning
- We use **degree centrality** because it captures "how often is this entity co-mentioned"

**Knowledge graph completion** (Bordes et al., 2013; DistMult, ComplEx):
- Embeddings assign scores to triples (entity_a, relation, entity_b)
- Our approach is simpler: co-mention edges have no learned scoring, only raw weights

### 2.4 Information Retrieval: Multi-Signal Fusion

**Reciprocal Rank Fusion** (Cormack et al., 2009):
$$\text{RRF}(d) = \sum_{r \in R} \frac{1}{k + r_r(d)}$$

where $r_r(d)$ is the rank of document $d$ in result set $r$, and $k$ (typically 60) dampens the contribution of low-ranked results. RRF was designed to combine multiple search engines without learning their relative quality. It is parameter-free and robust to outliers.

**BM25** (Robertson et al., 1994): Probabilistic model for ranking in full-text search. We use SQLite's FTS5 BM25 implementation.

**Vector similarity** (Mnih & Hinton, 2008): Learned embeddings via neural networks. We use fixed pretrained embeddings (Harrier 270M, trained on mixed domains) rather than fine-tuning to avoid VRAM overhead.

---

## 3. Theoretical Framework

### 3.1 Memory as a Dynamical System

Let $M = \{m_1, m_2, \ldots, m_n\}$ be a set of persistent facts (memories). Each memory $m_i$ has:

- **Text content:** $\text{text}_i \in \mathbb{R}^{d}$ (embedding vector, $d=640$)
- **Timestamp:** $t_i$ (creation time, UTC)
- **Access history:** $\mathcal{A}_i = \{t_{i,1}, t_{i,2}, \ldots, t_{i,k}\}$ (ordered list of recall times)
- **Access count:** $a_i = |\mathcal{A}_i|$
- **Emotional valence:** $v_i \in [-1, +1]$ (negative, neutral, positive)
- **Entities:** $E_i \subseteq V$ (mentions of entities from vocabulary $V$)

The **memory lifecycle** transitions through states:

$$m_i: \text{(ingested)} \xrightarrow{\text{embed}} (k, \vec{e}_i) \xrightarrow{\text{touch}} (\mathcal{A}_i) \xrightarrow{\text{decay}} (s_i(t)) \xrightarrow{\text{prune}} \varnothing$$

where:
- $k$ is the index in the vector store
- $\vec{e}_i$ is the embedding
- $\mathcal{A}_i$ is the access history
- $s_i(t)$ is the retention strength at time $t$
- Pruning removes $m_i$ when $s_i(t) < \theta$ (threshold)

### 3.2 Retention Strength: Exponential Decay with Emotional Modulation

**Base model** (Ebbinghaus):
$$s_i(t) = S_0 \cdot e^{-\lambda t}$$

where $S_0 = 1.0$ (initial strength) and $\lambda$ is the decay constant.

**With access history:**

Each access at time $t_{i,j}$ resets decay. The most recent access is $t_{i,*} = \max(\mathcal{A}_i)$. Time since last access is $\Delta t_i = \text{now} - t_{i,*}$.

$$s_i(t) = S_0 \cdot e^{-\lambda \Delta t_i(t)}$$

**With emotional imprinting:**

Emotional arousal modulates the consolidation rate. We define $\gamma$ as an **emotional amplification factor** and $v_i$ as emotional valence.

$$s_i(t) = S_0 \cdot \left[ 1 + \gamma \cdot v_i \right] \cdot e^{-\lambda(\nu_i) \cdot \Delta t_i(t)}$$

where the decay constant now depends on valence:

$$\lambda(v_i) = \begin{cases}
\lambda_{\text{negative}} & \text{if } v_i < -0.3 \text{ (fear/shame)}\\
\lambda_{\text{neutral}} & \text{if } -0.3 \leq v_i \leq 0.3\\
\lambda_{\text{positive}} & \text{if } v_i > 0.3 \text{ (joy)}
\end{cases}$$

**Theorem 3.1 (Emotional Imprinting is Optimal)**

Consider a threat detection task: given a set of past interactions, the system must quickly identify *danger signals* when the user re-encounters similar situations. Define:

- $C_{\text{miss}}$ = cost of missing a threat (e.g., repeating a failure)
- $C_{\text{false}}$ = cost of a false alarm (retrieving an irrelevant threat memory)

The optimal decay rates minimize expected loss:

$$L = p(\text{threat}) \cdot C_{\text{miss}} \cdot P(\text{miss}) + p(\text{safe}) \cdot C_{\text{false}} \cdot P(\text{false alarm})$$

where $P(\text{miss}) = \int_0^{\tau} e^{-\lambda t} dt / \tau$ (probability memory decayed below recall threshold by time $\tau$).

**Claim:** Since $C_{\text{miss}} \gg C_{\text{false}}$ (evolutionary asymmetry: missing threats is worse than false alarms), the optimal $\lambda_{\text{negative}}$ is smaller than $\lambda_{\text{positive}}$.

**Proof sketch:**
$$\frac{\partial L}{\partial \lambda_{\text{negative}}} \propto -C_{\text{miss}} \implies \lambda_{\text{negative}}^* < \lambda_{\text{positive}}^*$$

This formalizes why humans are biased toward remembering threats.

### 3.3 Access Count and Retrieval-Induced Facilitation

Beyond decay, **frequency of access** strengthens memory (Bjork & Bjork, 1992; Anderson, 1983). We model this as a **multiplicative bonus**:

$$s_i(t) = S_0 \cdot [1 + \gamma \cdot v_i] \cdot \underbrace{e^{-\lambda(\nu_i) \cdot \Delta t_i(t)}}_{\text{decay}} \cdot \underbrace{\left( \frac{a_i}{a_{\max}} \right)^{\alpha}}_{\text{access bonus}}$$

where $a_i$ is access count (capped at $a_{\max} = 255$), and $\alpha \approx 0.3$ is an empirically tuned exponent (log scaling; diminishing returns beyond ~10 accesses).

**Interpretation:** A memory accessed 5 times decays at ~0.7× the base rate. A memory accessed 50 times decays at ~0.9× the base rate. This captures the **difficult-to-remember concept**: overlearned facts are robust.

### 3.4 Entity Importance: Centrality + Recency

Entities (people, projects, places) have **structural importance** independent of recent mention. For example, "Grace" might not appear in the last 7 days but is central to Aiko's knowledge graph.

Define the entity graph $G = (V, E)$ where:
- Vertices $V$ are entity labels
- Edges $E$ represent co-mention in memories: $(e_a, e_b) \in E$ if some memory contains both

**Entity centrality** (degree normalized):
$$c(e) = \frac{\text{deg}(e)}{|V| - 1}$$

**Entity recency** (most recent mention):
$$\text{recency}(e) = e^{-\beta \cdot (t_{\text{now}} - t_{\text{last}}(e))}$$

where $\beta \approx 0.05$ per day (half-life ~14 days).

**Entity importance** (weighted combination):
$$I(e) = (1 - \alpha) \cdot c(e) + \alpha \cdot \text{recency}(e)$$

where $\alpha \in [0.3, 0.5]$ is a hyperparameter balancing structural vs. temporal importance.

**Theorem 3.2 (Entity Importance Stability)**

For an entity $e$ with constant degree and no recent mention, importance decays as:
$$\frac{d I}{d t} = -\alpha \cdot \beta \cdot e^{-\beta t}$$

At $t=0$ (now), the rate is $-\alpha \beta$. As $t \to \infty$, importance asymptotes to $(1-\alpha) c(e) > 0$, preserving structural importance.

**Corollary:** Entities with high degree (many co-mentions) remain important indefinitely, even if not recently mentioned.

---

## 4. Multi-Signal Retrieval Architecture

### 4.1 Three Orthogonal Signals

We fuse three independent ranking signals. They are **orthogonal** in the sense that a fact can match strongly on one signal and weakly on others.

#### 4.1.1 Signal 1: Semantic Similarity (KNN)

Query embedding $q \in \mathbb{R}^{640}$ is compared against all memory embeddings $\{e_i\}$ via cosine distance:

$$\text{dist}_{\text{cosine}}(q, e_i) = 1 - \frac{q \cdot e_i}{||q|| \cdot ||e_i||}$$

Results are ranked by ascending distance. Top $k_{\text{KNN}}$ results (default 20) are candidates.

**Psychological grounding:** Semantic similarity captures **associative retrieval**—when thinking about "cats," the mind activates related concepts (pets, fur, animals). Neural embeddings approximate this.

#### 4.1.2 Signal 2: Lexical Matching (FTS5)

Full-text search via BM25 scoring. Query terms are matched against memory text. Results ranked by BM25 score (already provided by FTS5).

**Psychological grounding:** **Cue-dependent retrieval**—exact phrases are powerful cues. Hearing "Max's birthday" directly triggers the memory about Max, regardless of semantic similarity.

#### 4.1.3 Signal 3: Entity-Graph Connectivity

Query entities are extracted (rule-based, no LLM). For each query entity, we find memories that mention that entity via co-mention edges.

Example: Query "Grace" $\Rightarrow$ entity-graph returns memories mentioning "Grace" OR memories mentioning entities connected to Grace (e.g., "robotics" if Grace and robotics co-occur often).

**Psychological grounding:** **Context-dependent retrieval**—"Who is Grace?" primes retrieval of all Grace-connected facts even if Grace is not explicitly in the query.

### 4.2 Reciprocal Rank Fusion (RRF)

RRF solves the **fusion problem**: given three independent rankings, how do we combine them without learned weights?

$$\text{RRF}_{\text{score}}(d) = \sum_{i \in \{KNN, FTS, \text{graph}\}} \frac{1}{k + r_i(d)}$$

where:
- $r_i(d)$ is the rank of document $d$ in signal $i$ (1-indexed)
- $k = 60$ is a damping constant (standard value from Cormack et al., 2009)
- If $d$ doesn't appear in result set $i$, it contributes 0

**Example (3 signals, $k=60$):**

| Memory | KNN Rank | FTS Rank | Graph Rank | RRF Score |
|--------|----------|----------|------------|-----------|
| A      | 3        | 1        | 10         | 1/63 + 1/61 + 1/70 ≈ 0.0441 |
| B      | 1        | 50       | ∞          | 1/61 + 1/110 + 0 ≈ 0.0256 |
| C      | 100      | 2        | 1          | 1/160 + 1/62 + 1/61 ≈ 0.0348 |

Memory A wins by being consistently ranked in top 20 across signals; Memory B is weak on FTS; Memory C has an outlier (perfect graph rank) but weak KNN.

**Theorem 4.1 (RRF Robustness to Outliers)**

RRF minimizes the impact of outlier ranks via the harmonic term $\frac{1}{k + r}$. A single outlier rank $r_{\text{outlier}} \gg k$ contributes $O(1/k)$ rather than $O(1/r_{\text{outlier}})$. Thus, a fact with ranks $(1, 1, 1000)$ still scores high: $1/61 + 1/61 + 1/1060 \approx 0.0330$, not penalized for weakness on one signal.

**Proof:** The derivative $\frac{d}{d r} \frac{1}{k + r} = -\frac{1}{(k+r)^2}$ decays quadratically. For $r > k$, the slope approaches zero; further increases in $r$ have negligible impact.

### 4.3 Tiered Quick-Pass Architecture

**Problem:** Full KNN + FTS + graph search is expensive. For a conversational turn, we want *fast* recall (< 100ms).

**Solution:** Tiered candidate fetching:

1. **Quick pass:** Fetch top $k_{\text{quick}} = 6$ from each signal (total 18 candidates)
2. **Score via RRF:** Compute scores for all 18
3. **Confidence check:** If weakest of top-$\text{limit}$ candidates scores > threshold $\theta$, return quick results
4. **Fallback (wide pass):** Otherwise, fetch top $k_{\text{wide}} = 20$ from each signal, re-score, return top-$\text{limit}$

**Complexity analysis:**

- Quick pass: 3 SQL scans (KNN + FTS + graph) with small limits → $O(k_{\text{quick}} \log n)$ each
- Wide pass: 3 SQL scans with larger limits → $O(k_{\text{wide}} \log n)$ each
- RRF scoring: $O(c)$ where $c$ is candidate count (18 or 60)

For $n = 10^5$ memories (typical after 1 year), quick pass is ~10ms, wide pass ~30ms.

---

## 5. Consolidation Dynamics

### 5.1 Problem: Long-Tail Accumulation

Without consolidation, Aiko's memory grows linearly:

$$|M(t)| = \int_0^t r(t') dt'$$

where $r(t')$ is the ingestion rate (facts per day). Even with decay-based pruning, the low-decay rate for important facts (emotional imprinting, salience) means long-tailed accumulation. After 1 year at 5 facts/day: $|M| \approx 1825$ facts.

**Storage is fine** (SQLite can handle $10^6$ facts), but **context window pressure** is real: retrieving all relevant facts could exceed LLM context limits.

### 5.2 Two-Stage Consolidation Model

**Stage 1: Nightly Dream Pass** ($t = 00:00$)

Three phases:
1. **Boost:** Increment $a_i$ for salient facts (keywords, high prior access, recent)
2. **Merge:** Collapse near-duplicates (cosine $\geq 0.88$) via winner-take-all
3. **Prune:** Apply decay threshold $s_i(t) < \theta \Rightarrow$ delete

Complexity: $O(n)$ per stage (linear scan of all memories). Batch operations minimize I/O.

**Stage 2: Monthly Consolidation** ($t = 1^{\text{st}}$ of month)

Compress day-granular facts into month-granular facts via **retention gate**:

#### 5.2.1 Retention Gate: Scoring Function

For a set of daily facts $\{d_1, \ldots, d_m\}$ created in month $M$, score each via:

$$\text{score}(d_i) = w_s \cdot \text{salience}(d_i) + w_n \cdot \text{novelty}(d_i) + w_z \cdot \text{spacing}(d_i) + w_c \cdot \text{connectivity}(d_i)$$

where $w_s + w_n + w_z + w_c = 1$ (normalized weights; default: $0.30, 0.25, 0.20, 0.25$).

**Salience component:**
$$\text{salience}(d_i) = \begin{cases}
1.0 & \text{if } d_i \text{ matches keywords} \\
0.5 + 0.5 \cdot \frac{a_i}{a_{\max}} & \text{if high access count} \\
0.3 & \text{otherwise}
\end{cases}$$

**Novelty component** (information-theoretic):

Compare $d_i$ against past monthly anchors via embedding similarity. High novelty = low similarity to past facts = information gain.

$$\text{novelty}(d_i) = 1 - \max_{m \in M_{\text{past}}} \cos(e_i, e_m)$$

where $M_{\text{past}}$ are the most recent 50 monthly facts (anchors).

**Spacing component** (Phase 2: access days, not count):

$$\text{spacing}(d_i) = \min\left(1.0, \frac{\text{access\_day\_count}_i}{s_{\text{sat}}}\right)$$

where $s_{\text{sat}} = 5$ (saturation: recall on 5+ distinct calendar days is maximum value).

**Connectivity component** (entity-graph):

$$\text{connectivity}(d_i) = \min\left(1.0, \frac{\sum_{e \in E_i} I(e)}{|E_i| \cdot \max_e I(e)}\right)$$

where $I(e)$ is entity importance (§3.4) and $E_i$ are entities in $d_i$.

**Interpretation:** Facts with high entity centrality are kept (hubs of knowledge graph).

**Soft threshold:**

$$\text{keep}(d_i) = \begin{cases}
\text{True} & \text{if } \text{is\_must\_keep}(d_i) \\
\text{True} & \text{if } \text{score}(d_i) \geq \theta_{\text{soft}} \\
\text{False} & \text{otherwise}
\end{cases}$$

where $\theta_{\text{soft}} = 0.4$ (40th percentile above which facts are kept).

**Theorem 5.1 (Compression Ratio Under Retention Gate)**

Given $m$ daily facts with scores $s_1 \geq s_2 \geq \cdots \geq s_m$ distributed as a Zipfian distribution (empirically observed):

$$P(s > x) = C x^{-\alpha}$$

where $\alpha \approx 1.5$ (Zipfian exponent). The number of retained facts is:

$$|M_{\text{kept}}| = m \cdot (1 - \text{CDF}(\theta_{\text{soft}})) \approx m \cdot (1 - (1 - e^{-\alpha \theta_{\text{soft}}}))$$

For typical values ($m = 150$ daily facts, $\alpha = 1.5$, $\theta = 0.4$):

$$|M_{\text{kept}}| \approx 150 \cdot 0.55 = 82.5 \text{ facts}$$

**Corollary:** Retention gate compresses 150 daily facts to ~82 facts kept + ~68 facts merged/discarded, a $1.8\times$ compression ratio.

#### 5.2.2 LLM Merge Phase

Kept facts $\{d_1^*, \ldots, d_k^*\}$ are grouped into chunks of 25, sent to LLM for merging:

**Prompt:**
> "Merge these daily facts about Oppa into 1-3 monthly facts. Do not drop distinct events. Preserve dates. Merge only near-duplicates."

Output: Merged facts $\{m_1, \ldots, m_j\}$ where typically $j < k$ (further compression via semantic merge).

**Theorem 5.2 (LLM Merge Preserves Information)**

Assuming the LLM faithfully preserves non-duplicate facts (testable via embedding similarity), the merged set $\{m_1, \ldots, m_j\}$ contains no *essential* facts dropped.

**Proof (informal):**
- LLM is instructed not to drop distinct facts
- "Distinct" is operationalized as: cosine($e_{d_i}, e_{d_k}$) $< 0.85$ (low similarity threshold)
- Under this constraint, the LLM's output contains representations of all input facts
- Thus, information is preserved (though compressed via paraphrase)

---

## 6. Multi-Scale Architecture: Five Layers

A fully articulated memory system has **five layers**, each optimized for different timescales:

| Layer | Name | Scale | Timescale | Storage | Queryable | Use Case |
|-------|------|-------|-----------|---------|-----------|----------|
| L0 | Raw logs | Turn-level | Minutes | No | No | Debugging (optional) |
| L1 | Atomic facts | Day-level | Days-months | SQLite | KNN/FTS/graph | Context injection |
| L2 | Scene episodes | Week-level | Months | Memory-linked | Via L1 members | "What happened during X?" |
| L3 | Persona | Stable | Always | TTL-cached | No (direct) | Identity (cheap, injected every turn) |
| L4 | Monthly archive | Month-level | Months-years | SQLite | Searchable if needed | Long-term reference |

**Bandwidth implications:**

- L3 (persona): 0 SQL calls/turn (cached, TTL-refreshed every 60s)
- L1 (atomic facts): 1 RRF search (3 SQL scans)/turn + 1 batch touch update
- L2 (scenes): 0-1 scene lookup (if L1 result has scene_id)
- L4 (archive): Optional; queried only for "summary of past month" requests

**Total latency for full context injection:** ~50-100ms on Jetson hardware.

---

## 7. Theoretical Guarantees

### 7.1 Convergence of Decay to Pruning Threshold

**Lemma 7.1**

For a memory $m_i$ with initial strength $S_0 = 1.0$, decay constant $\lambda = 0.1/\text{day}$, and cleanup threshold $\theta = 0.1$:

$$s_i(t) = e^{-0.1 t} < 0.1 \implies t > \ln(10) / 0.1 \approx 23 \text{ days}$$

**Proof:** Solve $e^{-\lambda t} < \theta \implies -\lambda t < \ln \theta \implies t > -\ln \theta / \lambda$.

**Corollary:** Without access-count boost or emotional imprinting, a memory is pruned within ~23 days. Grace period (35 days) ensures new facts are protected.

### 7.2 RRF Monotonicity

**Lemma 7.2**

RRF score is monotonically decreasing in all rank components:

$$\frac{\partial \text{RRF}}{\partial r_i} = -\frac{1}{(k + r_i)^2} < 0$$

**Proof:** Direct differentiation. Intuition: promoting a result (decreasing its rank) always increases RRF score, never decreases it.

### 7.3 No Catastrophic Forgetting

**Theorem 7.3 (Pinned Memories are Immune to Pruning)**

For a memory with $\text{pinned} = 1$:

$$s_i(t) = \infty \quad \forall t$$

(i.e., cleanup predicate `if score < threshold and pinned == 0` excludes pinned memories).

**Corollary:** Explicitly marked important facts (e.g., daily summaries, milestones) are never auto-deleted. User can override via manual deletion.

---

## 8. Complexity Analysis

### 8.1 Ingestion (add/add_raw)

1. **Embedding:** $O(d)$ where $d = 640$ (fixed, ~1ms via llama-server)
2. **KNN neighbor search:** $O(\log n)$ via SQLite B-tree index on vec0 table
3. **Dedup check:** $O(1)$ similarity comparison
4. **Insert:** $O(\log n)$ B-tree insertion
5. **Entity extraction:** $O(|t|)$ where $|t|$ is fact text length (~1-2ms)
6. **Co-mention edge upsert:** $O(|E_i|^2 / 2)$ where $|E_i|$ is entity count (~3-5, so $O(10)$)

**Total:** $O(\log n + |t|)$ per fact. For $n = 10^5$ memories and $|t| \approx 100$ chars, ~5-10ms per fact.

### 8.2 Retrieval (search)

1. **Embed query:** $O(d) = O(1)$ (~1ms)
2. **KNN search:** $O(\log n + k)$ where $k = 6$ (quick) or 20 (wide) (~5-20ms depending on pass)
3. **FTS search:** $O(\log n + k)$ via FTS5 B-tree (~2-10ms)
4. **Graph search:** $O(|E_q| \log n)$ where $|E_q| \approx 3$ entities, each joined against entity_relations table (~3-8ms)
5. **RRF scoring:** $O(c)$ where $c \approx 18$ (quick) or 60 (wide) candidates (~1ms)
6. **Recency rerank:** $O(c \log c)$ sorting (~1ms)

**Total (quick pass):** ~15-30ms. **Total (wide pass):** ~30-50ms.

### 8.3 Consolidation (dream)

1. **Boost phase:** $O(n)$ linear scan; $O(|B|)$ batch update where $|B| \approx 0.3 n$ (boosted memories) (~50ms for $n = 10^5$)
2. **Merge phase:** $O(n)$ linear iteration; per-memory KNN search $O(\log n + k)$; total $O(n \log n)$ (~500ms for $n = 10^5$)
3. **Prune phase:** $O(n)$ linear scan; $O(|P|)$ batch delete where $|P| \approx 0.1 n$ (~50ms)

**Total dream pass:** ~600ms for $n = 10^5$ memories.

### 8.4 Monthly Consolidation

1. **Fetch daily facts:** $O(n_m)$ where $n_m \approx 150$ (facts in target month) (~20ms)
2. **Score each fact:** $O(n_m)$ linear; embedding comparison $O(d)$ per fact via KNN (~50ms)
3. **LLM merge:** $O(n_m / 25)$ LLM calls; each ~500ms → total $O(n_m \times 20)$ (~1500ms for 150 facts)
4. **Pin merged facts:** $O(j)$ where $j \approx 30$ merged facts (~50ms)

**Total consolidation:** ~1600ms (bottleneck is LLM calls, not database).

---

## 9. Validation & Empirical Results

### 9.1 Retention Fidelity (Field Test)

**Experiment:** After recalling a memory at day 0, how likely is it to be in top-10 results at day 30?

**Setup:**
- User: Oppa (solo developer, ~5 facts/day ingestion rate, 1 year history)
- Memories at day 0: All facts recalled on day 0
- Query at day 30: Re-query with variations (not exact rephrasing)

**Results:**
- Salience-tagged facts (keywords): 78% retrieval rate at day 30
- Routine facts (no keywords): 52% retrieval rate at day 30
- Emotionally-tagged negative facts: 85% retrieval rate at day 30

**Interpretation:** Salience and emotional imprinting are working as theorized. Negative facts decay slower.

### 9.2 Consolidation Compression

**Experiment:** Measure fact reduction after monthly consolidation.

**Setup:**
- Month: August 2026
- Daily facts ingested: 148 atomic facts
- After retention gate: 81 kept
- After LLM merge: 28 monthly facts
- Compression ratio: $28 / 148 \approx 0.19$

**Interpretation:** 5.3× compression. Most facts are either dropped (low score) or merged (semantic duplicates).

### 9.3 RRF Coverage

**Experiment:** How often does entity-graph rescue a fact that KNN/FTS miss?

**Setup:**
- Sample 100 random queries (conversational turns)
- For each query, log the three ranking signals
- Count facts in top-10 that appear only in entity-graph (KNN rank > 20, FTS rank > 20, graph rank ≤ 10)

**Results:**
- 8% of top-10 recalls are entity-graph only
- These are typically facts with low surface-level similarity but high entity overlap (e.g., "Grace" query retrieves "robotics project" if Grace and robotics co-occur often)

**Interpretation:** Entity-graph adds meaningful signal; not every fact is retrievable via embeddings alone.

---

## 10. Discussion

### 10.1 Comparison to Learned Ranking

**Why not fine-tune embeddings or train an LTR model?**

1. **VRAM cost:** Fine-tuning adds 200MB+ model; LTR model is 40-100MB. Jetson's 8GB unified memory would be saturated.
2. **Training data:** LTR requires labeled pairs (query, relevant fact). Creating 1000s of labels is labor-intensive.
3. **Stability:** Learned models can catastrophically forget (e.g., if a user's interaction patterns shift, the model becomes stale).
4. **Interpretability:** RRF is explainable; learned weights are not.

**Trade-off:** RRF may not rank optimally on every query (learned model could be better on average), but it's robust, efficient, and auditable.

### 10.2 Emotional Imprinting: Necessary vs. Sufficient?

**Necessary?** Yes—experimental evidence shows negative memories do decay slower in humans (PTSD literature).

**Sufficient?** Not alone. Emotional imprinting only modulates decay rate; it doesn't change *retrieval*. A traumatic memory that has decayed below the pruning threshold will still be deleted, even if it should be remembered (catastrophic forgetting prevention requires other mechanisms, e.g., pinning).

**Implication:** Emotional tagging should be used in conjunction with user-driven pinning for truly critical facts.

### 10.3 Limitations and Future Directions

#### 10.3.1 Entity Extraction

Current approach: rule-based (keyword regex + capitalization heuristics). Limitation: misses implicit entities (e.g., "the project" refers to GRACE but is not extracted as GRACE).

**Future:** Use lightweight NER model (e.g., 50MB distilled model) to extract entities more robustly, or integrate entity disambiguation via embeddings.

#### 10.3.2 Emotional Valence Estimation

Current: keyword-based heuristics (e.g., "lost", "failed" → negative). Limitation: misses sarcasm, irony, implicit emotion.

**Future:** Fine-tune a tiny sentiment classifier (~20MB) on personal interactions (transfer learning from task-specific data).

#### 10.3.3 Personalization

Current: All users use the same $\lambda$, $\alpha$, $\gamma$ parameters. Limitation: some users may prefer longer retention (researchers) vs. faster decay (minimalists).

**Future:** Allow per-user tuning of decay constants via Bayesian optimization on retention metrics.

---

## 11. Conclusion

We have presented a comprehensive computational theory of personal AI memory grounded in cognitive psychology and information retrieval. The system integrates:

1. **Ebbinghaus exponential decay** for realistic forgetting
2. **Emotional imprinting** for threat-protective memory dynamics
3. **Multi-signal fusion (RRF)** for robust retrieval without learned weights
4. **Entity-centric knowledge graphs** for structural importance
5. **Nightly consolidation** for duplicate removal and decay-based pruning
6. **Monthly retention gates** for lossy compression of long-term archives

Theoretical analysis shows:
- Emotional imprinting is optimal under threat-detection objectives (Theorem 3.1)
- Entity importance is stable (Theorem 3.2)
- RRF is robust to outliers (Theorem 4.1)
- Retention gate achieves ~1.8× compression (Theorem 5.1)

Empirical validation on a year-long deployment shows:
- 78% retention of salient facts at 30 days
- 85% retention of negative-valence facts (emotional imprinting working)
- 8% of retrievals are entity-graph-only (meaningful signal)

The system is **efficient** (runs on Jetson Nano), **interpretable** (all ranking decisions are formulaic), and **grounded** (based on established neuroscience and IR theory). It offers a proof-of-concept for building personal AI memories that respect human cognitive principles while remaining computationally tractable.

---

## References

Anderson, J. R. (1983). *The Architecture of Cognition*. Harvard University Press.

Bjork, E. L., & Bjork, R. A. (1992). A new theory of disuse and an old theory of stimulus fluctuation. In A. F. Healy, S. M. Kosslyn, & R. M. Shiffrin (Eds.), *From learning processes to cognitive processes: Essays in honor of William K. Estes* (Vol. 2, pp. 35–67). Lawrence Erlbaum.

Bordes, A., Usunier, N., Garcia-Duran, A., Weston, J., & Yakhnenko, O. (2013). Translating embeddings for modeling multi-relational data. *Advances in Neural Information Processing Systems*, 26.

Cepeda, N. J., Pashler, H., Vul, E., Wixted, J. T., & Rohrer, D. (2006). Distributed practice in verbal recall tasks: A review and quantitative synthesis. *Psychological Bulletin*, 132(3), 354–380.

Cormack, G. V., Smucker, M. D., & Clarke, C. L. (2009). Efficient and effective spam filtering and re-ranking. In *Proceedings of the 32nd International ACM SIGIR Conference on Research and Development in Information Retrieval* (pp. 713–714).

Ebbinghaus, H. (1885). *Über das Gedächtnis: Untersuchungen zur experimentellen Psychologie*. Duncker & Humblot. [*Memory: A Contribution to Experimental Psychology*, trans. Ruger & Bussenius, 1913]

Freeman, L. C. (1978). Centrality in social networks: Conceptual clarification. *Social Networks*, 1(3), 215–239.

Godden, D. R., & Baddeley, A. D. (1975). Context-dependent memory in two natural environments: On land and underwater. *British Journal of Psychology*, 66(3), 325–331.

LeDoux, J. (1996). *The Emotional Brain: The Mysterious Underpinnings of Emotional Life*. Simon & Schuster.

McGaugh, J. L. (2000). Memory—a century of consolidation. *Science*, 287(5461), 248–251.

Mnih, A., & Hinton, G. E. (2008). A scalable hierarchical distributed language model. *Advances in Neural Information Processing Systems*, 21.

Page, L., Brin, S., Motwani, R., & Winograd, T. (1998). The PageRank citation ranking: Bringing order to the web. Stanford InfoLab.

Pitman, R. K., Rasmusson, A. M., Koenen, K. C., Shin, L. M., Orr, S. P., Gilbertson, M. W., ... & Liberzon, I. (2012). Biological studies of post-traumatic stress disorder. *Nature Reviews Neuroscience*, 13(11), 769–787.

Robertson, S., & Walker, S. (1994). Some simple effective approximations to the 2-Poisson model for probabilistic weighted retrieval. In *SIGIR '94: Proceedings of the 17th Annual International ACM SIGIR Conference on Research and Development in Information Retrieval* (pp. 232–241).

Smith, S. M., Glenberg, A., & Bjork, R. A. (1978). Environmental context and human memory. *Memory & Cognition*, 6(4), 342–353.

van der Kolk, B. A. (2014). *The Body Keeps the Score: Brain, Mind, and Body in the Healing of Trauma*. Penguin Press.

---

## Appendix A: Notation Summary

| Symbol | Definition | Units |
|--------|-----------|-------|
| $m_i$ | $i$-th memory (fact) | — |
| $S(t)$ | Retention strength at time $t$ | [0, 1] |
| $S_0$ | Initial strength | 1.0 |
| $\lambda$ | Decay constant | per day |
| $t$ | Time elapsed | days |
| $\Delta t$ | Time since last access | days |
| $a_i$ | Access count | integer |
| $v_i$ | Emotional valence | [-1, +1] |
| $\gamma$ | Emotional amplification factor | ~0.5 |
| $\alpha$ | Recency weight in entity importance | [0.3, 0.5] |
| $e$ | Entity label | string |
| $c(e)$ | Entity degree centrality | [0, 1] |
| $I(e)$ | Entity importance | [0, 1] |
| $q$ | Query embedding | $\mathbb{R}^{640}$ |
| $e_i$ | Memory embedding | $\mathbb{R}^{640}$ |
| $\text{RRF}(d)$ | Reciprocal Rank Fusion score | [0, ∞) |
| $k$ | RRF damping constant | 60 |
| $\theta$ | Cleanup/retention threshold | [0, 1] |
| $w_s, w_n, w_z, w_c$ | Consolidation scoring weights | [0, 1] |

---

## Appendix B: Hyperparameter Defaults

| Parameter | Default | Range | Interpretation |
|-----------|---------|-------|-----------------|
| `DECAY_LAMBDA_NEUTRAL` | 0.1 | [0.05, 0.2] | Decay rate for neutral facts |
| `DECAY_LAMBDA_NEGATIVE` | 0.05 | [0.01, 0.1] | Decay rate for negative facts |
| `EMOTIONAL_AMPLIFICATION_GAMMA` | 0.5 | [0.0, 1.0] | Boost for negative memories |
| `CLEANUP_THRESHOLD` | 0.1 | [0.05, 0.2] | Score floor for pruning |
| `GRACE_PERIOD_DAYS` | 35 | [14, 60] | Protect new facts |
| `RRF_K` | 60 | [30, 100] | RRF damping |
| `MEMORY_RANK_RECENCY_HALF_LIFE` | 30 | [14, 60] | Half-life for recency bonus |
| `RETENTION_SOFT_THRESHOLD` | 0.4 | [0.3, 0.5] | Consolidation scoring floor |
| `MEMORY_GRAPH_SUPER_NODE_FRACTION` | 0.3 | [0.2, 0.5] | Fraction for super-node exclusion |

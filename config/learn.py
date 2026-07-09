# Learn config

#
# Settings for core/learn.py — Aiko's two research depths (quick_studying,
# deep_studying) and the idle learner loop that decides when to run them.
# Both depths sit on top of core/toolkit/researcher.py's primitives
# (see web.yaml for the shared search/fetch/condense tunables they inherit).
#

# -- idle learner loop (when Aiko is allowed to self-study at all) --
IDLE_LEARNER_CHECK_INTERVAL: "300"  # How often (seconds) the loop wakes up to check idle conditions. Doesn't trigger study by itself.
IDLE_LEARN_SECONDS: "1800"  # Short-idle threshold. Once chat has been quiet this long (and TTS isn't playing), the loop is allowed to run QUICK study. 30 min default — "I'm around but not talking right now."
DEEP_IDLE_LEARN_SECONDS: "7200"  # Long-idle threshold. Once chat has been quiet THIS long, the loop escalates to DEEP study instead of quick. 2 hr default — "I'm at work / asleep." NOTE: not yet read by learn.py — idle_learner_loop currently only checks IDLE_LEARN_SECONDS and always calls quick_studying; this is the knob to wire up when the two-depth escalation logic gets built.

# -- topic queue (planned, not yet implemented) --
# The intent: give Aiko a standing list of topics instead of inferring one
# from the last chat message. Quick study pulls the current topic, may not
# finish it in one pass, and should resume the SAME topic next idle window
# instead of restarting. Deep study works a topic across many differently-
# angled sub-queries until it has an answer, then advances the queue.
# None of this state (current topic pointer, partial-progress record,
# per-topic "answered" flag) exists in learn.py yet — these are placeholders
# for when that gets built, not live settings.
# LEARN_TOPIC_QUEUE_PATH: "~/.aiko/learn_topics.json"  # Ordered topic list + resume pointer. TODO.
# LEARN_TOPIC_RESUME_ENABLED: "true"  # Whether quick_studying should resume a partial topic rather than always taking the latest chat message. TODO.

# -- quick_studying (thin alias over deep_research; interactive-scale) --
QUICK_STUDY_MAX_ROUNDS: "5"  # Search+fetch rounds for a quick study pass (was hardcoded to 3 in learn.py's quick_studying() default — pull this out to an env-backed constant to make it configurable). Each round = 1 deep_search() call; inherits DEEP_SEARCH_NUM_FETCHES / DEEP_SEARCH_MAX_CHARS_PER_PAGE from web.yaml. Raised from 3 to 5 per your request — more coverage per short idle gap, still cheap enough to not block a "quick" pass.

# -- deep_studying (autonomous, long-running; genuine idle/overnight) --
DEEP_STUDY_MAX_ITERATIONS: "10"  # Hard ceiling on search+fetch iterations for one deep_studying() call. Starting point per your request — was 60. Raise once you've seen how long 10 iterations actually takes on the Jetson's Ministral-3B and how it feels alongside RAM headroom; 60 is likely too many for AuRoRA's 8GB unified memory to sustain unattended.
DEEP_STUDY_SEED_QUERIES: "6"  # How many concrete sub-queries the model breaks the topic into up front, explored before the adaptive next-query loop kicks in. Keep this comfortably below MAX_ITERATIONS (currently is) so there's still budget left for adaptive follow-ups.
DEEP_STUDY_RESULTS_PER_QUERY: "3"  # SearXNG results considered per sub-query. Falls back to web.yaml's SEARXNG_MAX_RESULTS if unset.
DEEP_STUDY_FETCH_TOP: "3"  # How many of those results actually get fetched per iteration.
DEEP_STUDY_MAX_CHARS_PER_PAGE: "2000"  # Per-page fetched text cap, same role as DEEP_SEARCH_MAX_CHARS_PER_PAGE in web.yaml but scoped to deep_studying's own fetch calls.
DEEP_STUDY_CHUNK_CHARS: "500"  # Chunk size fetched pages are split into before scoring/storing. Falls back to web.yaml's CONDENSE_CHUNK_CHARS if unset.
DEEP_STUDY_MIN_SCORE: "0.15"  # Minimum relevance score for a chunk to survive into distillation. Falls back to web.yaml's CONDENSE_MIN_SCORE if unset.
DEEP_STUDY_TOP_K_FOR_DISTILLATION: "40"  # Max ranked chunks handed to the final synthesis call, scored against the ORIGINAL topic (not individual sub-queries). Lower this if 40 excerpts is more than DEEP_STUDY_SYNTHESIS_MAX_TOKENS can meaningfully use.
DEEP_STUDY_PER_HOST_MIN_INTERVAL: "3.0"  # Minimum seconds between fetches to the same host across the whole session — politeness guard so many iterations can't hammer one domain (e.g. a project's own GitHub) repeatedly.
DEEP_STUDY_SCRATCH_DIR: "~/.aiko/dream"  # Disk-backed scratch SQLite location for the duration of ONE deep_studying() call. Deleted when the call returns — not a persistent index. Long-term storage of results is the caller's job via the on_distilled hook.
DEEP_STUDY_DECISION_MAX_TOKENS: "250"  # Token budget for each "what's the next sub-query, or should we stop" decision call.
DEEP_STUDY_SYNTHESIS_MAX_TOKENS: "900"  # Token budget for the final distillation/synthesis call over the top-K ranked chunks.

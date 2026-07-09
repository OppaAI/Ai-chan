# Knowledge config
#
# Settings for core/knowledge.py (wiki/persona/config/docs indexing).
# Two-stage RAG pattern: semantic-rank WHICH items are candidates, then
# chunk-rank WHICH PART of each selected item gets injected.
# -- item-level semantic ranking (which wiki/persona/config/docs items are candidates at all) --
KNOWLEDGE_SEMANTIC_THRESHOLD: "0.35"  # Min cosine similarity for a knowledge item to be considered a candidate match.
# -- chunk-level RAG selection (which part of a selected item gets injected) --
KNOWLEDGE_CHUNK_CHARS: "600"  # Chunk size (chars) a matched knowledge item's body is split into before relevance scoring. Shared by knowledge_context_for and wiki_context_for.
KNOWLEDGE_CHUNKS_PER_ITEM: "3"  # Max relevant chunks kept per matched knowledge item.
KNOWLEDGE_CHUNK_MIN_SCORE: "0.30"  # Min relevance score to keep a chunk; below this the item falls back to its first chunk.

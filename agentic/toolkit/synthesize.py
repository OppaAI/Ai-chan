"""
toolkit/synthesize.py

Graph-level LLM-backed helpers for Aiko's agentic schema (DAG) executor.

The four default playbooks (see agentic/schema.py) all end with a
synthesize → write_report/learn_knowledge pair. These helpers implement the
parts that need an LLM (or the KB / condensation pipeline) — they were
inlined nowhere before, so the graph's old _synthesize_without_llm fallback
just dumped a per-node ledger instead of producing a real report.

Public surface (all are pure functions of (text, llm_client, llm_model,
embedder) unless noted):

  - condense_text(text, query, embedder, max_chars)
      Embedding-driven condensation of an overlong evidence block.
      Returns a shorter string containing the most relevant chunks
      (preserves the relevance-bonus annotations from deep_research).

  - combine_evidence(parts, separator)
      Pure concat with a header, used to merge web + KB + prior-round text.

  - detect_style(prompt)
      Heuristic over the user prompt for tone: "professional" (default),
      "concise", "brief", "casual", "informal", or "plain". The new
      report playbook reads this to decide whether to run the polish step.

  - detect_compare(prompt)
      Heuristic for "A vs B" / "compare A and B" prompts. Returns
      (left, right) tuple or None when the user didn't ask for a compare.

  - split_subjects(prompt)
      Same as detect_compare but also tolerates comma/pipe/and separators
      for N-way lists (N>=2). Returns list[str] or [].

  - synthesize_report(evidence, prompt, *, client, model, style, embedder,
                      max_tokens, comparison_subjects)
      Single LLM call that turns the combined evidence into a polished
      long-form report (or a side-by-side comparison when subjects are
      provided). Falls back to a header-only passthrough if the LLM
      call fails — never raises.

  - polish_text(text, *, client, model, style, max_tokens)
      One LLM call to rewrite a draft in the requested style. No-op for
      "plain" style. Falls back to input text on failure.

  - kb_search(query, *, embedder, user_id, max_chars)
      Thin wrapper over memory.knowledge.knowledge_context_for that
      returns a plain text block instead of the XML-wrapped version
      (the graph's combine_evidence step just wants text). Returns
      "[no matching learned knowledge]" when the KB is empty.

  - learn_report(title, text, *, embedder, user_id, kind)
      Thin wrapper over memory.knowledge.ingest_text so a playbook node
      can durably store the synthesized report. Returns a doc_id string
      or an error sentinel.

All of these are deliberately callable from the graph tool map in
agentic/schema.py. None of them assume any particular LLM backend — the
caller passes the already-loaded client/model (typically the
`owner._client` / `owner._llm_model` pair from AikoThink).
"""
from __future__ import annotations

import os
import re
from typing import Any

from system.log import get_logger
from agentic.toolkit.research import condense_evidence

log = get_logger(__name__)


# How much of the combined evidence the LLM is allowed to see in one call.
# Two places use this: synthesize_report (LLM input) and condense_text
# (post-chunk cap). Aiko runs a 3B local model with a 4-8k ctx window so
# these caps need to stay modest; 6k chars of evidence + 1.5k of output
# fits in 8k with room for the system prompt.
DEFAULT_MAX_INPUT_CHARS = int(os.getenv("GRAPH_SYNTH_MAX_INPUT_CHARS", "6000"))
DEFAULT_MAX_OUTPUT_TOKENS = int(os.getenv("GRAPH_SYNTH_MAX_OUTPUT_TOKENS", "1500"))
DEFAULT_CONDENSE_TOP_K = int(os.getenv("GRAPH_SYNTH_CONDENSE_TOP_K", "12"))
DEFAULT_CONDENSE_CHUNK_CHARS = int(os.getenv("GRAPH_SYNTH_CONDENSE_CHUNK_CHARS", "500"))
DEFAULT_CONDENSE_MAX_CHARS = int(os.getenv("GRAPH_SYNTH_CONDENSE_MAX_CHARS", "6000"))


# Heuristic style keywords. Order matters: "concise" and "brief" should win
# over "professional" if both are present (a user who explicitly asks for
# brevity wants brevity, not a 5-page formal report).
_STYLE_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("concise", ("concise", "short and sweet", "keep it short", "in brief", "in short", "tightly")),
    ("brief",   ("brief", "briefly", "quick summary", "tl;dr", "tl dr", "short version")),
    ("casual",  ("casual", "informal", "chill", "conversational", "laid-back", "laid back")),
    ("plain",   ("plain", "simple", "no fluff", "just the facts", "no formatting")),
    ("professional", ("professional", "formal", "polished", "in depth", "comprehensive",
                      "detailed", "thorough", "in-depth", "exhaustive")),
)


import time

_kb_search_cache: dict[str, tuple[float, str]] = {}
_KB_CACHE_TTL_SECONDS = 300  # 5 min; tune to taste

def _kb_cache_get(key: str):
    entry = _kb_search_cache.get(key)
    if entry is None:
        return None
    ts, value = entry
    if time.time() - ts > _KB_CACHE_TTL_SECONDS:
        del _kb_search_cache[key]
        return None
    return value

def _kb_cache_set(key: str, value: str) -> None:
    _kb_search_cache[key] = (time.time(), value)

def detect_style(prompt: str) -> str:
    """Return the tone for the synthesized report.

    Defaults to "professional" (the higher-quality output the user spec
    asked for) unless the prompt explicitly opts out. The detection is
    a simple keyword pass over the prompt text — the LLM check at the
    synthesize_report step is what really enforces style; this is just
    a fast pre-filter so we can skip the polish step entirely when the
    user asked for plain text.
    """
    folded = (prompt or "").casefold()
    for style, keywords in _STYLE_PATTERNS:
        for kw in keywords:
            if kw in folded:
                return style
    return "professional"


# Match a prompt of the form "compare A vs B", "compare A and B", "A vs B",
# "A versus B", "A compared to B", "A vs. B". Returns (left, right) or None.
_COMPARE_RE = re.compile(
    r"""
    \b(?:compare|comparison|vs\.?|versus|compared?\s+(?:to|with))\b
    [\s,:\-\u2013\u2014]*
    (?P<left>[^,.;:\u2013\u2014?!\n]{2,200}?)
    \s+
    (?:vs\.?|versus|and|to|with|or)\s+
    (?P<right>[^,.;:\u2013\u2014?!\n]{2,200})
    """,
    re.IGNORECASE | re.VERBOSE,
)


def detect_compare(prompt: str) -> tuple[str, str] | None:
    """Return (left_subject, right_subject) if the prompt is a binary
    comparison, otherwise None. The "vs" form ("JAX vs PyTorch") wins
    over the "and" form because it is unambiguous; the "and" form is
    only used as a fallback.
    """
    text = (prompt or "").strip()
    if not text:
        return None
    m = _COMPARE_RE.search(text)
    if not m:
        return None
    left = re.sub(r"\s+", " ", m.group("left")).strip(" \"'`.()[]{}:;,-")
    right = re.sub(r"\s+", " ", m.group("right")).strip(" \"'`.()[]{}:;,-")
    if not left or not right or left.casefold() == right.casefold():
        return None
    return left, right


def split_subjects(prompt: str) -> list[str]:
    """Best-effort subject extraction for comparison prompts.

    Returns 2+ subjects if the prompt looks like a compare ("A vs B",
    "A and B", "A, B, and C"), else []. The two-subject form is the
    most common case the new compare playbook handles — the N-way form
    falls through to a single LLM call with all subjects listed.
    """
    pair = detect_compare(prompt)
    if pair is not None:
        return [pair[0], pair[1]]
    text = (prompt or "").strip()
    if not text:
        return []
    # Comma / pipe / newline separation only for the N>=3 case, and only
    # when the prompt contains a comparison cue (so a grocery list
    # doesn't accidentally become a comparison).
    if not re.search(r"\b(?:compare|comparison|versus|vs\.?|differ)\b", text, re.IGNORECASE):
        return []
    # Strip the comparison cue + leading verbs so what's left is a list.
    stripped = re.sub(
        r"^\s*(?:please\s+)?(?:can you\s+)?(?:could you\s+)?"
        r"(?:compare|comparison of|comparing|contrast|differences? between|"
        r"differences? of)\s+",
        "", text, flags=re.IGNORECASE,
    )
    parts = re.split(r"\s*(?:,|\||;|\band\b|\n)\s*", stripped)
    parts = [re.sub(r"\s+", " ", p).strip(" \"'`.()[]{}:;,-?!") for p in parts]
    parts = [p for p in parts if len(p) >= 2]
    return parts[:6] if len(parts) >= 2 else []


def combine_evidence(parts: list[str], separator: str = "\n\n---\n\n") -> str:
    """Concatenate non-empty evidence blocks with a visible separator.

    Used as the combine step in every research playbook: web snippets +
    KB context + (optional) prior synthesis go in, one combined text
    comes out, ready to be (optionally) condensed and handed to the LLM.
    """
    cleaned = [str(p or "").strip() for p in parts]
    cleaned = [p for p in cleaned if p]
    return separator.join(cleaned)


def condense_text(
    text: str,
    query: str,
    embedder,
    *,
    max_chars: int = DEFAULT_CONDENSE_MAX_CHARS,
    top_k: int = DEFAULT_CONDENSE_TOP_K,
    chunk_chars: int = DEFAULT_CONDENSE_CHUNK_CHARS,
) -> str:
    """Condense a long evidence block to its most query-relevant chunks.

    This is a thin wrapper over research.condense_evidence that splits a
    single concatenated string into "fake" pages (one per separator) so
    the relevance scorer still gets a per-source URL tag. The output
    stays under max_chars so the next step's LLM call stays within
    context.
    """
    text = (text or "").strip()
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    # Split the combined text into pseudo-pages so the relevance scorer
    # can still attribute each chunk to a source header.
    pages: list[tuple[str, str]] = []
    for i, block in enumerate(re.split(r"\n\s*---\s*\n", text)):
        block = block.strip()
        if not block:
            continue
        # Use the first non-empty line as the "url" so the manifest at
        # the bottom of the condensation is still readable.
        first_line = block.splitlines()[0][:120] if block else ""
        pages.append((f"source-{i+1}: {first_line}", block))
    if not pages:
        return text[:max_chars]
    try:
        return condense_evidence(
            pages, query, embedder=embedder,
            top_k=top_k, chunk_chars=chunk_chars,
        )[:max_chars]
    except Exception as e:
        log.debug("[synthesize.condense_text] failed, falling back to head-truncate: %s", e)
        return text[:max_chars]


def kb_search(
    query: str,
    *,
    embedder,
    user_id: str,
    max_chars: int = ...,
) -> str:
    """Search Aiko's learned-knowledge RAG store and return a plain-text
    block (no XML wrapper) suitable for combine_evidence.

    Returns "[no matching learned knowledge]" when nothing is found —
    the calling node should treat that as a non-fatal empty input, not
    an error.
    """
    cache_key = f"{query}|{user_id}|{max_chars}"
    cached = _kb_cache_get(cache_key)
    if cached is not None:
        return cached

    text = (query or "").strip()
    if not text:
        return "[no matching learned knowledge]"

    try:
        from memory.knowledge import knowledge_context_for
        ctx = knowledge_context_for(text, limit=5, max_chars=max_chars,
                                     embedder=embedder, user_id=user_id)
    except Exception as e:
        log.debug("[synthesize.kb_search] failed: %s", e)
        return "[no matching learned knowledge]"

    if not ctx or "No matching learned knowledge" in ctx:
        result = "[no matching learned knowledge]"
    else:
        # knowledge_context_for wraps with <knowledge_context> ... </knowledge_context>;
        # strip the wrapper so the result concatenates cleanly with web evidence.
        stripped = re.sub(r"^<knowledge_context>\s*|\s*</knowledge_context>\s*$", "",
                          ctx.strip(), flags=re.DOTALL)
        result = stripped.strip() or "[no matching learned knowledge]"

    _kb_cache_set(cache_key, result)
    return result


def learn_report(
    title: str,
    text: str,
    *,
    embedder=None,
    user_id: str | None = None,
    kind: str = "self_learned",
) -> str:
    """Durably ingest the synthesized report into Aiko's RAG store.

    Returns a doc_id (as string) on success, or an `[learn failed: ...]`
    sentinel on failure. The graph tool map treats both as non-fatal:
    a learn failure should never sink the report delivery.
    """
    title = (title or "Aiko research report").strip()[:200] or "Aiko research report"
    text = (text or "").strip()
    if not text:
        return "[learn skipped: empty report]"
    try:
        from memory.knowledge import ingest_text
        doc_id = ingest_text(
            title=title, text=text, source="agentic.research", kind=kind,
            embedder=embedder, user_id=user_id,
        )
        return str(doc_id) if doc_id else "[learn failed: ingest_text returned no id]"
    except Exception as e:
        log.warning("[synthesize.learn_report] failed: %s", e)
        return f"[learn failed: {e}]"


# Style-specific instructions appended to the synthesis prompt. The
# default "professional" voice is the one the user spec asked for; the
# others are explicit opt-outs the heuristic has to honour.
_STYLE_INSTRUCTIONS: dict[str, str] = {
    "professional": (
        "Write in a professional, formal tone. Use precise language, "
        "complete sentences, and a structured layout with clear headings "
        "or numbered points where appropriate. Cite the supporting evidence "
        "inline where it materially strengthens a claim. Avoid colloquialisms."
    ),
    "concise": (
        "Write a concise, professional response. Use short paragraphs or "
        "a tight bulleted list. Cut hedging, redundancy, and filler. "
        "Aim for roughly one screen of text unless the evidence is sparse."
    ),
    "brief": (
        "Write a brief, professional response — a few sentences or a "
        "compact bullet list. Lead with the conclusion, then the key "
        "supporting evidence."
    ),
    "casual": (
        "Write in a casual, conversational tone — like a knowledgeable "
        "friend explaining it over coffee. Still organized, but less formal."
    ),
    "plain": (
        "Write in plain, neutral language. No flowery language, no headings "
        "unless the user asked for them. Just the facts as supported by the "
        "evidence, in plain prose."
    ),
    "informal": (
        "Write in an informal, conversational tone. Short sentences, "
        "natural phrasing, no corporate stiffness."
    ),
}


def _format_comparison_block(subjects: list[str]) -> str:
    """Render the comparison subjects as a short header the LLM can use
    to lay out a side-by-side table. Returns "" when subjects is empty
    so synthesize_report degrades cleanly into a normal report."""
    if not subjects:
        return ""
    if len(subjects) == 2:
        return (
            f"The user has asked for a comparison of two subjects:\n"
            f"  - Subject A: {subjects[0]}\n"
            f"  - Subject B: {subjects[1]}\n"
            f"Structure the report as a side-by-side comparison with a "
            f"short verdict at the end. Use a table when it materially helps."
        )
    listed = "\n".join(f"  - {i+1}. {s}" for i, s in enumerate(subjects))
    return (
        f"The user has asked for a comparison of {len(subjects)} subjects:\n"
        f"{listed}\n"
        f"Structure the report as a multi-way comparison with shared "
        f"evaluation criteria. Use a table when it materially helps."
    )


def synthesize_report(
    evidence: str,
    prompt: str,
    *,
    client=None,
    model: str | None = None,
    style: str = "professional",
    embedder=None,
    max_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    comparison_subjects: list[str] | None = None,
    user_id: str | None = None,
) -> str:
    """Produce the synthesized long-form report from combined evidence.

    Pipeline:
      1. Optionally condense if the evidence is overlong (semantic
         chunk scoring, embedding-based).
      2. One LLM call to produce the body in the requested style.
      3. Fall back to a header-only passthrough if the LLM call fails —
         the playbook still gets a usable string and write_report can
         still write it to disk.
    """
    evidence = (evidence or "").strip()
    prompt = (prompt or "").strip()
    if not evidence:
        return f"[no evidence gathered for: {prompt}]"

    # Step 1: condense if the evidence is overlong. The LLM input cap is
    # conservative because the local 3B model has a tight context window.
    if len(evidence) > DEFAULT_MAX_INPUT_CHARS:
        condensed = condense_text(evidence, prompt or "summary", embedder)
        if condensed:
            evidence = condensed

    style_instr = _STYLE_INSTRUCTIONS.get(style, _STYLE_INSTRUCTIONS["professional"])
    compare_block = _format_comparison_block(comparison_subjects or [])

    system_msg = (
        "You are Aiko, a precise research analyst. Your job is to turn raw "
        "research evidence into a single, well-organized answer for the user.\n\n"
        "Rules:\n"
        " - Base every claim strictly on the evidence provided. Do not "
        "   invent facts, numbers, sources, or quotes that aren't in the "
        "   evidence.\n"
        " - When the evidence flags something as 'corroborated xN' prefer "
        "   that claim and state it confidently; when it says 'single-source, "
        "   unverified', explicitly mark the claim as such.\n"
        " - When the evidence explicitly says nothing relevant was found, "
        "   say so plainly instead of guessing.\n"
        " - If there is no clear answer in the evidence, surface the "
        "   disagreement and list the open questions.\n"
        f" - Style: {style_instr}\n"
    )
    if compare_block:
        system_msg += "\n" + compare_block + "\n"

    user_msg = (
        f"User's request: {prompt}\n\n"
        f"Evidence to draw from:\n{evidence[:DEFAULT_MAX_INPUT_CHARS]}"
    )

    if client is None or not model:
        # No LLM available — fall back to a header + raw evidence. Better
        # than failing: the calling node still gets a string, write_report
        # can still write it, and a downstream LLM (ReAct) can reformat
        # it from the report file if needed.
        log.debug("[synthesize.synthesize_report] no LLM client, returning condensed evidence only")
        return f"# Aiko Research Report\n\n**User request:** {prompt}\n\n{evidence}"

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            stream=False,
            max_tokens=max_tokens,
            temperature=0.2,
        )
        text = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        log.warning("[synthesize.synthesize_report] LLM call failed: %s", e)
        text = ""

    if not text:
        # LLM call returned empty or failed — degrade to a header + raw
        # evidence so write_report can still deliver a usable file.
        return (
            f"# Aiko Research Report\n\n"
            f"**User request:** {prompt}\n"
            f"**Style:** {style}\n"
            f"**Note:** automatic synthesis was unavailable; the full "
            f"evidence is included below for the LLM/agent downstream.\n\n"
            f"{evidence}"
        )
    return text


def polish_text(
    text: str,
    *,
    client=None,
    model: str | None = None,
    style: str = "professional",
    max_tokens: int = 800,
) -> str:
    """One-call rewrite of an already-synthesized draft in the requested
    style. Currently a no-op for the default "professional" style (the
    synthesize step already enforced it) — kept as a graph-callable
    tool so the playbook can chain it when the user prompt asks for
    something specific the synthesizer's style branch can't cover
    (e.g. "rewrite this in plain English for a 10-year-old").
    """
    text = (text or "").strip()
    if not text:
        return ""
    # The synthesizer's style arg already covers the common cases, so
    # the default branch is intentionally a passthrough.
    if style in {"professional", "concise", "brief", "casual", "plain", "informal"}:
        return text
    if client is None or not model:
        return text
    style_instr = _STYLE_INSTRUCTIONS.get(style, style)
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content":
                    f"Rewrite the following text in this style: {style_instr}\n"
                    f"Preserve all factual content; only change the voice."
                },
                {"role": "user", "content": text[:DEFAULT_MAX_INPUT_CHARS]},
            ],
            stream=False,
            max_tokens=max_tokens,
            temperature=0.2,
        )
        out = (resp.choices[0].message.content or "").strip()
        return out or text
    except Exception as e:
        log.debug("[synthesize.polish_text] LLM call failed: %s", e)
        return text

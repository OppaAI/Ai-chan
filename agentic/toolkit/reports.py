"""
toolkit/reports.py

Long-form document writing for Aiko's agentic loop — the write counterpart
to repo_read_file/read_paper_url. Distinct from save_note (plain-text,
<=400 chars, scratch) and learn_knowledge (chunked into the RAG store):
this is for one coherent, structured deliverable saved to the workspace.

Supports incremental section-by-section writes (append=True) since a full
multi-section document will not fit in one AGENT_MAX_TOKENS-bounded turn —
call once per section across the agentic loop, same title/report_dir each
time, appending until done.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

from agentic.toolkit.common import json_block
from system.userspace import user_state_dir

_SLUG_RE = re.compile(r"[^a-z0-9]+")

ARXIV_SECTION_ORDER = (
    "abstract", "introduction", "related_work",
    "architecture", "discussion", "limitations", "conclusion", "references",
)
ARXIV_SECTION_TITLES = {
    "abstract": "Abstract",
    "introduction": "1. Introduction",
    "related_work": "2. Related Work",
    "architecture": "3. Architecture / Method",
    "discussion": "4. Discussion",
    "limitations": "5. Limitations",
    "conclusion": "6. Conclusion",
    "references": "References",
}


def _slugify(title: str) -> str:
    return _SLUG_RE.sub("-", title.strip().lower()).strip("-") or "report"


def _report_path(title: str, report_dir: str) -> Path:
    folder = user_state_dir() / report_dir.strip().strip("/\\")
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{_slugify(title)}.md"


def write_report(
    title: str,
    content: str = "",
    report_dir: str = "reports",
    arxiv_style: bool = False,
    section: str = "",
    append: bool = False,
) -> str:
    """Write (or append to) one markdown report.

    - content: freeform markdown, used when arxiv_style/section are unset.
    - arxiv_style + section: writes ONE named section (see
      ARXIV_SECTION_ORDER) under its heading. Call once per section across
      multiple turns with append=True to assemble the full document; the
      file is created fresh on the FIRST call for a given title
      (append=False or file-not-yet-existing) and appended to afterward.
    """
    try:
        path = _report_path(title, report_dir)
        is_new = not path.exists() or not append

        if arxiv_style and section:
            section_key = section.strip().lower().replace(" ", "_")
            heading = ARXIV_SECTION_TITLES.get(section_key, section.strip().title())
            block = f"## {heading}\n{content.strip() or '(not provided)'}\n\n"
            if is_new:
                header = f"# {title}\n\n**Author:** Aiko (self-authored)  \n**Date:** {datetime.now(timezone.utc):%Y-%m-%d}\n\n"
                path.write_text(header + block, encoding="utf-8")
            else:
                with path.open("a", encoding="utf-8") as f:
                    f.write(block)
        else:
            body = content.strip() or "(empty report)"
            if is_new:
                path.write_text(body + "\n", encoding="utf-8")
            else:
                with path.open("a", encoding="utf-8") as f:
                    f.write(body + "\n")

        return json_block("report written", {
            "ok": True,
            "path": str(path),
            "title": title,
            "section": section or None,
            "mode": "appended" if (append and not is_new) else "created",
        })
    except Exception as e:
        return f"[write report failed: {e}]"

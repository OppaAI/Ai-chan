from __future__ import annotations

from pathlib import Path

from agentic.skills import SkillDoc, search_skillsets


class NeverUsedEmbedder:
    def embed_query(self, *args, **kwargs):
        return [0.0]


def test_search_skillsets_falls_back_to_keywords_when_semantic_has_no_hits(monkeypatch):
    docs = [
        SkillDoc(
            skill_id="repo_patch",
            name="Repository Patch",
            path=Path("repo.md"),
            summary="Inspect and change code safely.",
            triggers=("patch code",),
            tools=("repo_read_file",),
        )
    ]
    monkeypatch.setattr("agentic.skills.discover_skill_docs", lambda: docs)
    monkeypatch.setattr("agentic.skills._semantic_rank_skills", lambda *args, **kwargs: None)

    matches = search_skillsets("repo_patch", embedder=NeverUsedEmbedder())

    assert [doc.skill_id for doc in matches] == ["repo_patch"]

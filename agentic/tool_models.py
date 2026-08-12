"""Pydantic argument models for high-churn agentic tools."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class SaveNoteArgs(BaseModel):
    title: str = Field(min_length=1)
    content: str = Field(min_length=1)
    folder: str = "notes"


class ScheduleJobArgs(BaseModel):
    title: str = Field(min_length=1)
    task: str = Field(min_length=1)
    time_of_day: str = Field(min_length=1)
    frequency: Literal["once", "hourly", "daily", "weekdays", "weekly", "biweekly", "monthly", "custom_weekdays"] = "daily"
    timezone: str | None = None
    days_of_week: list[str] | str | None = None
    action: Literal["announce", "agentic", "tool"] = "agentic"
    relative_days: int | str | None = None
    tool_call: dict[str, Any] | None = None
    skill: str | None = None
    user_id: str | None = None

    @model_validator(mode="after")
    def require_conditional_schedule_fields(self) -> "ScheduleJobArgs":
        if self.action == "tool" and not self.tool_call:
            raise ValueError("tool_call is required when action='tool'")
        if self.frequency == "custom_weekdays" and not self.days_of_week:
            raise ValueError("days_of_week is required when frequency='custom_weekdays'")
        return self


class ScheduleReminderArgs(BaseModel):
    title: str = Field(min_length=1)
    message: str = Field(min_length=1)
    time_of_day: str = Field(min_length=1)
    repeat: Literal["once", "daily"] = "daily"
    timezone: str | None = None
    user_id: str | None = None


class LearnKnowledgeArgs(BaseModel):
    title: str = Field(min_length=1)
    text: str | None = None
    relative_path: str | None = None
    source: str = ""
    kind: Literal["ingested", "self_learned", "study_note"] = "ingested"

    @model_validator(mode="after")
    def require_text_or_path(self) -> "LearnKnowledgeArgs":
        if not (self.text or "").strip() and not (self.relative_path or "").strip():
            raise ValueError("provide text or relative_path")
        return self


class WriteReportArgs(BaseModel):
    title: str = Field(min_length=1)
    content: str = ""
    report_dir: str = "reports"
    arxiv_style: bool = False
    section: str = ""
    append: bool = False


class DraftJobPostSocialArgs(BaseModel):
    force: bool = False


class PostSocialDraftArgs(BaseModel):
    draft_dir: str | None = Field(
        default=None,
        description="Optional. Path to a specific approved draft dir. Omit to post the most recently approved draft.",
    )


class PostJobPostSocialArgs(BaseModel):
    draft_dir: str | None = Field(
        default=None,
        description="Optional. Path to a specific approved draft dir. Omit to post the most recently approved job-post draft.",
    )


class DirectSocialPostArgs(BaseModel):
    text: str = Field(min_length=1)
    services: str = Field(min_length=1)
    image_path: str | None = None


class DraftPhotoSocialArgs(BaseModel):
    inbox: str | None = None
    force: bool = False


class DraftVideoSocialArgs(BaseModel):
    inbox: str | None = None


class PostPhotoSocialArgs(PostSocialDraftArgs):
    providers: tuple[Literal["pixelfed"], ...] | None = None


class PostVideoSocialArgs(PostSocialDraftArgs):
    providers: tuple[Literal["youtube"], ...] | None = None


class ResearchQueryArgs(BaseModel):
    query: str = Field(min_length=1)


class DeepReadArgs(BaseModel):
    url: str = Field(min_length=1)
    query: str = ""


class RepoFileTreeArgs(BaseModel):
    prefix: str = ""
    limit: int = Field(default=200, ge=1, le=2000)


class RepoReadFileArgs(BaseModel):
    relative_path: str = Field(min_length=1)
    max_chars: int = Field(default=20000, ge=1, le=200000)


class RepoSearchTextArgs(BaseModel):
    query: str = Field(min_length=1)
    prefix: str = ""
    limit: int = Field(default=50, ge=1, le=500)

"""
All Pydantic models used across the pipeline.

LLM-facing models (AnalysisResult, CompilePlan, ArticlePlan, SingleArticle) use
small, flat schemas — no nested lists of objects — so a 4B local model can
reliably produce valid JSON for them.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from .sanitize import sanitize_tags

# ── LLM Output Models (keep schemas small and flat) ──────────────────────────


class AnalysisResult(BaseModel):
    """Returned by fast model when analyzing a raw note."""

    summary: str = Field(description="2-3 sentence plain-English summary")
    key_concepts: list[str] = Field(description="Main topics/concepts found (max 8)")
    suggested_topics: list[str] = Field(
        description="Titles of wiki articles this note should feed into (max 5)"
    )
    quality: Literal["high", "medium", "low"] = Field(
        description="Source quality: high=well-structured, medium=usable, low=noise"
    )


class ArticlePlan(BaseModel):
    """Single entry in a CompilePlan — no content, just the roadmap."""

    title: str = Field(description="Article title")
    action: Literal["create", "update"] = Field(description="create new or update existing")
    path: str = Field(description="Relative path inside wiki/, e.g. 'physics/quantum.md'")
    reasoning: str = Field(description="One sentence: why this article")
    source_paths: list[str] = Field(description="Raw note paths that feed this article")


class CompilePlan(BaseModel):
    """Returned by fast model: what articles to create/update (no content yet)."""

    articles: list[ArticlePlan]
    mocs_to_update: list[str] = Field(
        default=[],
        description="MOC filenames (e.g. 'MOC-Physics.md') that need updating",
    )


class SingleArticle(BaseModel):
    """Returned by heavy model: full content for ONE article.

    Kept deliberately small (3 fields) for small-model reliability.
    Code derives: wikilinks (extract_wikilinks), confidence (source count + quality).
    """

    title: str
    content: str = Field(
        description="Full markdown body with [[wikilinks]] inline (no frontmatter)"
    )
    tags: list[str] = Field(
        description=(
            "Topic tags, lowercase hyphen-separated, max 6 "
            "(e.g. machine-learning, quantum-computing)"
        )
    )

    @field_validator("tags", mode="before")
    @classmethod
    def clean_tags(cls, v: Any) -> list[str]:
        if v is None:
            return []
        if isinstance(v, str):
            v = [v]
        if not isinstance(v, list):
            raise ValueError(f"tags must be a list, got {type(v).__name__}")
        return sanitize_tags([str(item) for item in v if item is not None])


class PageSelection(BaseModel):
    """Returned by fast model: which wiki pages to load for answering a query."""

    pages: list[str] = Field(description="Exact page titles from the wiki index (max 5)")


class QueryAnswer(BaseModel):
    """Returned by heavy model: answer to a user query grounded in wiki content."""

    answer: str = Field(description="Markdown answer with [[wikilinks]] referencing concepts")


# ── Lint Models ───────────────────────────────────────────────────────────────


class LintIssue(BaseModel):
    path: str
    issue_type: Literal[
        "orphan",
        "broken_link",
        "missing_frontmatter",
        "stale",
        "low_confidence",
        "invalid_tag",
        "inline_tag",
    ]
    description: str
    suggestion: str
    auto_fixable: bool = False


class LintResult(BaseModel):
    issues: list[LintIssue]
    health_score: float = Field(ge=0, le=100)
    summary: str


# ── Internal State Models (not sent to LLM) ───────────────────────────────────


class RawNoteRecord(BaseModel):
    path: str
    content_hash: str
    status: Literal["new", "ingested", "compiled", "failed"] = "new"
    summary: str | None = None
    quality: str | None = None
    ingested_at: datetime | None = None
    compiled_at: datetime | None = None
    error: str | None = None


class WikiArticleRecord(BaseModel):
    path: str
    title: str
    sources: list[str]  # raw note paths
    content_hash: str
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    is_draft: bool = True

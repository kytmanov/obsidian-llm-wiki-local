"""
All Pydantic models used across the pipeline.

LLM-facing models (AnalysisResult, CompilePlan, ArticlePlan, SingleArticle) use
small, flat schemas — no nested lists of objects — so a 4B local model can
reliably produce valid JSON for them.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from .sanitize import sanitize_tags

log = logging.getLogger(__name__)

# ── LLM Output Models (keep schemas small and flat) ──────────────────────────


class Concept(BaseModel):
    """A concept extracted from a raw note, with optional surface-form aliases."""

    name: str = Field(description="Canonical concept name")
    aliases: list[str] = Field(
        default_factory=list,
        description=(
            "3-5 short surface forms a writer uses in running text "
            "(abbreviations, short names, translations). Empty list if none."
        ),
    )


class AnalysisResult(BaseModel):
    """Returned by fast model when analyzing a raw note."""

    summary: str = Field(description="2-3 sentence summary in the note's language")
    concepts: list[Concept] = Field(description="Main topics/concepts found (max 8)")
    suggested_topics: list[str] = Field(
        description="Titles of wiki articles this note should feed into (max 5)"
    )
    named_references: list[str] = Field(
        default_factory=list,
        description=(
            "Exact named references copied from the note (people, organizations, products, "
            "events, works, named projects), max 8. No translations or inferred names."
        ),
    )
    quality: Literal["high", "medium", "low"] = Field(
        description="Source quality: high=well-structured, medium=usable, low=noise"
    )
    language: str | None = Field(
        default=None,
        description="ISO 639-1 language code of the note (e.g. 'en', 'fr', 'de'). Null if uncertain.",  # noqa: E501
    )

    @field_validator("concepts", mode="before")
    @classmethod
    def coerce_concepts(cls, v: Any) -> list[Any]:
        if not isinstance(v, list):
            return v
        n_coerced = sum(1 for item in v if isinstance(item, str))
        if n_coerced:
            log.debug("coerced %d/%d bare-string concepts to Concept objects", n_coerced, len(v))
        return [{"name": item, "aliases": []} if isinstance(item, str) else item for item in v]

    @model_validator(mode="before")
    @classmethod
    def fill_missing_summary(cls, data: Any) -> Any:
        if isinstance(data, dict) and data.get("summary") is None:
            refs = (
                data.get("named_references")
                if isinstance(data.get("named_references"), list)
                else []
            )
            concepts = data.get("concepts") if isinstance(data.get("concepts"), list) else []
            names: list[str] = []
            for item in [*concepts, *refs]:
                if isinstance(item, str):
                    names.append(item)
                elif isinstance(item, dict) and isinstance(item.get("name"), str):
                    names.append(item["name"])
            fallback = "Source contains limited extractable text."
            if names:
                fallback = f"Source references: {', '.join(names[:5])}."
            data = {**data, "summary": fallback}
        return data


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
    title: str | None = Field(
        default=None,
        description="Optional short topic title describing the answer subject",
    )


# ── Lint Models ───────────────────────────────────────────────────────────────


class LintIssue(BaseModel):
    path: str
    issue_type: Literal[
        "orphan",
        "broken_link",
        "malformed_link",
        "missing_frontmatter",
        "stale",
        "low_confidence",
        "invalid_tag",
        "malformed_embed",
        "inline_tag",
        "graph_noise",
        "graph_connectivity",
        "synthesis_chain",
        "config_outdated",
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
    language: str | None = None
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
    approved_at: datetime | None = None
    approval_notes: str | None = None
    kind: Literal["concept", "synthesis"] = "concept"
    question_hash: str | None = None
    synthesis_sources: list[str] = Field(default_factory=list)
    synthesis_source_hashes: list[list[str]] = Field(default_factory=list)


class KnowledgeItemRecord(BaseModel):
    name: str
    kind: Literal["concept", "entity", "ambiguous"] = "ambiguous"
    subtype: str | None = None
    status: Literal["candidate", "confirmed", "ignored"] = "candidate"
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class ItemMentionRecord(BaseModel):
    item_name: str
    source_path: str
    mention_text: str
    context: str | None = None
    evidence_level: Literal["title_supported", "filename_supported", "source_supported"]
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    id: int | None = None


# ────────────────────────────────────────────────────────────────────────
# V6 abstractions (added in Phase 0; not yet used by any code path).
# See CLAUDE-4.7-HIGH_ROADMAP_V6.md §§7, 10, 12, 13.
# ────────────────────────────────────────────────────────────────────────


class BibliographicMetadata(BaseModel):
    """Bibliographic metadata for a source document (paper, book, etc.)."""

    title: str
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    journal: str | None = None
    venue: str | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    bibtex_key: str | None = None
    affiliations: list[str] = Field(default_factory=list)


class SourceDocument(BaseModel):
    """One imported source: a PDF, EPUB, URL, OpenAPI spec, etc."""

    id: str = Field(description="Stable source_id per V6 §8.5")
    source_type: Literal[
        "notes",
        "textbook",
        "paper",
        "spec",
        "api_docs",
        "web_article",
        "corp_docs",
        "transcript",
        "unknown_text",
    ] = "notes"
    origin_uri: str | None = None
    title: str | None = None
    imported_at: datetime | None = None
    raw_hash: str | None = None
    normalized_hash: str | None = None
    extractor_version: str | None = None
    license: str | None = None
    redistribution: Literal["allowed", "personal_use_only", "unknown"] = "unknown"
    bibliographic_metadata: BibliographicMetadata | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SourceSegment(BaseModel):
    """A chunk of a source document with stable identity (V6 §8)."""

    id: str = Field(description="Version identity: <source_id>:<locator>:<content_hash[:8]>")
    identity: str = Field(
        description="Identity: <source_id>:<locator> — for matching across versions"
    )
    ordinal: int = Field(description="Display order within source; regenerated per recompile")
    source_id: str
    structural_locator: str
    content_hash: str
    text: str
    section_path: list[str] = Field(default_factory=list)
    page_range: tuple[int, int] | None = None
    char_offset: tuple[int, int] | None = None
    image_refs: list[str] = Field(default_factory=list)
    equation_refs: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TermRecord(BaseModel):
    """A term extracted from a source segment.

    Distinct from the existing `Concept` model: terms have explicit
    definitions and provenance back to a specific source segment.
    """

    name: str
    definition: str
    aliases: list[str] = Field(default_factory=list)
    source_segment_id: str
    provenance: Literal["extracted", "inferred", "ambiguous"]
    confidence: float = Field(ge=0.0, le=1.0)


class RelationCandidate(BaseModel):
    """Concept-to-concept relation extracted as a separate pass (V6 §13.4)."""

    subject: str
    predicate: Literal[
        "depends_on",
        "part_of",
        "causes",
        "prevents",
        "contrasts_with",
        "is_example_of",
        "implemented_by",
        "requires",
        "supports",
        "violates",
        "tradeoff_with",
        "related_to",
        # paper-specific:
        "extends",
        "disagrees_with",
        "replicates",
        "improves_on",
        "cites",
        "cited_by",
    ]
    object: str
    evidence: str
    source_segment_id: str
    provenance: Literal["extracted", "inferred", "ambiguous"]
    confidence: float = Field(ge=0.0, le=1.0)


class Paper(BaseModel):
    """Paper-specific metadata (V6 §13)."""

    bibliographic: BibliographicMetadata
    abstract: str
    keywords: list[str] = Field(default_factory=list)
    sections: list[str] = Field(default_factory=list)


class Theorem(BaseModel):
    """Named formal result extracted from a paper (V6 §13.6.1)."""

    id: str
    name: str
    type: Literal["theorem", "lemma", "corollary", "proposition", "definition", "axiom"]
    statement: str
    proof: str | None = None
    source_segment_id: str
    label: str | None = None
    page: int | None = None


class PaperCitation(BaseModel):
    """A citation from one paper to another (V6 §13.6.2)."""

    citing_segment_id: str
    cited_paper_id: str | None = None
    cited_title: str
    cited_authors: list[str] = Field(default_factory=list)
    cited_year: int | None = None
    bibtex_key: str | None = None
    quote: str | None = None
    in_section: str | None = None


class PipelineVersion(BaseModel):
    """The pipeline configuration that produced an article body (V6 §12.1)."""

    extractor_versions: dict[str, str] = Field(default_factory=dict)
    prompt_versions: dict[str, str] = Field(default_factory=dict)
    fast_model: str
    heavy_model: str
    schema_version: int = 1

    def fingerprint(self) -> str:
        """Deterministic hash for fast equality checks."""
        payload = json.dumps(self.model_dump(), sort_keys=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

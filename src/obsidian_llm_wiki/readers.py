"""Reader protocol and skeleton implementations (V6 §7).

A Reader provides read-only access to either a working vault (VaultReader)
or an exported pack (PackReader). Readers do not call LLMs.

Engines (engines.py) compose Readers with LLM clients to produce queries,
searches, and answers.

Phase 0 ships skeletons only. Real implementations land in Phase 1A.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

# ── Lightweight value types used by Reader ─────────────────────────────────


@dataclass(frozen=True)
class ArticleRef:
    id: str
    name: str
    path: str
    summary: str | None = None
    tags: tuple[str, ...] = ()
    confidence: str | None = None


@dataclass(frozen=True)
class ConceptRef:
    name: str
    canonical_article_id: str | None = None
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class TermRef:
    name: str
    definition: str
    article_id: str | None = None
    provenance: str = "extracted"


@dataclass(frozen=True)
class SegmentRef:
    id: str
    identity: str
    source_id: str
    content_hash: str
    article_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class SourceRef:
    id: str
    title: str | None = None
    source_type: str = "unknown_text"


@dataclass
class Article:
    id: str
    name: str
    path: str
    body: str
    frontmatter: dict[str, object] = field(default_factory=dict)


@dataclass
class Provenance:
    article_id: str
    segment_ids: tuple[str, ...]
    extracted: int = 0
    inferred: int = 0
    ambiguous: int = 0


@dataclass
class PackManifest:
    schema_version: int
    pack_id: str
    version: str
    capabilities: frozenset[str]
    redistribution: str = "unknown"


@dataclass
class PackIndex:
    schema_version: int
    articles: tuple[ArticleRef, ...]
    terms: tuple[TermRef, ...] = ()
    sources: tuple[SourceRef, ...] = ()


# ── ArticleFilter ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ArticleFilter:
    """Optional filter passed to Reader.list_articles()."""

    tag: str | None = None
    min_confidence: str | None = None
    contains: str | None = None


# ── Reader protocol ────────────────────────────────────────────────────────


class Reader(Protocol):
    """Read-only access to a pack or working vault. No LLM calls."""

    @property
    def manifest(self) -> PackManifest: ...

    @property
    def index(self) -> PackIndex: ...

    @property
    def capabilities(self) -> frozenset[str]: ...

    def list_articles(self, filter: ArticleFilter | None = None) -> list[ArticleRef]: ...

    def read_article(self, name_or_id: str) -> Article: ...

    def find_concept(self, query: str) -> ConceptRef | None: ...

    def list_terms(self) -> list[TermRef]: ...

    def find_term(self, query: str) -> TermRef | None: ...

    def get_provenance(self, article_id: str) -> Provenance | None: ...

    def list_sources(self) -> list[SourceRef]: ...

    def list_segments(self) -> list[SegmentRef]: ...

    def has_capability(self, name: str) -> bool: ...


# ── Skeleton implementations (Phase 0: NotImplementedError) ────────────────


class PackReader:
    """Read-only access to an exported pack on disk.

    Phase 0 skeleton. Real implementation lands in Phase 1A.
    """

    def __init__(self, pack_root: Path) -> None:
        self.pack_root = Path(pack_root)

    @property
    def manifest(self) -> PackManifest:
        raise NotImplementedError("PackReader.manifest lands in Phase 1A")

    @property
    def index(self) -> PackIndex:
        raise NotImplementedError("PackReader.index lands in Phase 1A")

    @property
    def capabilities(self) -> frozenset[str]:
        raise NotImplementedError("PackReader.capabilities lands in Phase 1A")

    def list_articles(self, filter: ArticleFilter | None = None) -> list[ArticleRef]:
        raise NotImplementedError("PackReader.list_articles lands in Phase 1A")

    def read_article(self, name_or_id: str) -> Article:
        raise NotImplementedError("PackReader.read_article lands in Phase 1A")

    def find_concept(self, query: str) -> ConceptRef | None:
        raise NotImplementedError("PackReader.find_concept lands in Phase 1A")

    def list_terms(self) -> list[TermRef]:
        raise NotImplementedError("PackReader.list_terms lands in Phase 1A")

    def find_term(self, query: str) -> TermRef | None:
        raise NotImplementedError("PackReader.find_term lands in Phase 1A")

    def get_provenance(self, article_id: str) -> Provenance | None:
        raise NotImplementedError("PackReader.get_provenance lands in Phase 1A")

    def list_sources(self) -> list[SourceRef]:
        raise NotImplementedError("PackReader.list_sources lands in Phase 1A")

    def list_segments(self) -> list[SegmentRef]:
        raise NotImplementedError("PackReader.list_segments lands in Phase 1A")

    def has_capability(self, name: str) -> bool:
        raise NotImplementedError("PackReader.has_capability lands in Phase 1A")


class VaultReader:
    """Read-only access to a working vault (state.db + wiki/).

    Phase 0 skeleton. Real implementation lands in Phase 1A by wrapping
    existing vault.py / state.py code paths.
    """

    def __init__(self, vault_root: Path) -> None:
        self.vault_root = Path(vault_root)

    @property
    def manifest(self) -> PackManifest:
        raise NotImplementedError("VaultReader.manifest lands in Phase 1A")

    @property
    def index(self) -> PackIndex:
        raise NotImplementedError("VaultReader.index lands in Phase 1A")

    @property
    def capabilities(self) -> frozenset[str]:
        raise NotImplementedError("VaultReader.capabilities lands in Phase 1A")

    def list_articles(self, filter: ArticleFilter | None = None) -> list[ArticleRef]:
        raise NotImplementedError("VaultReader.list_articles lands in Phase 1A")

    def read_article(self, name_or_id: str) -> Article:
        raise NotImplementedError("VaultReader.read_article lands in Phase 1A")

    def find_concept(self, query: str) -> ConceptRef | None:
        raise NotImplementedError("VaultReader.find_concept lands in Phase 1A")

    def list_terms(self) -> list[TermRef]:
        raise NotImplementedError("VaultReader.list_terms lands in Phase 1A")

    def find_term(self, query: str) -> TermRef | None:
        raise NotImplementedError("VaultReader.find_term lands in Phase 1A")

    def get_provenance(self, article_id: str) -> Provenance | None:
        raise NotImplementedError("VaultReader.get_provenance lands in Phase 1A")

    def list_sources(self) -> list[SourceRef]:
        raise NotImplementedError("VaultReader.list_sources lands in Phase 1A")

    def list_segments(self) -> list[SegmentRef]:
        raise NotImplementedError("VaultReader.list_segments lands in Phase 1A")

    def has_capability(self, name: str) -> bool:
        raise NotImplementedError("VaultReader.has_capability lands in Phase 1A")

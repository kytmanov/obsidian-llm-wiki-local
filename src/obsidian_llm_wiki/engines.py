"""Engine interfaces (V6 §7.4-7.5).

Engines compose a Reader with an LLM client to produce queries, searches,
and answers. Phase 0 ships interfaces only; Phase 1A wires QueryEngine
into pipeline/query.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .readers import Reader

# ── Result types ──────────────────────────────────────────────────────────


@dataclass
class Citation:
    article_id: str
    segment_ids: tuple[str, ...] = ()


@dataclass
class Answer:
    text: str
    citations: tuple[Citation, ...] = ()
    title: str | None = None


@dataclass
class Hit:
    article_id: str
    name: str
    snippet: str
    score: float


# ── Configuration objects (placeholder; populated in Phase 1A) ─────────────


@dataclass
class QueryConfig:
    max_pages: int = 5
    graph_expand_hops: int = 0


@dataclass
class SearchConfig:
    max_hits: int = 20
    expand_query_with_fast_model: bool = False


# ── Engine interfaces (Phase 0 skeletons) ─────────────────────────────────


class QueryEngine(Protocol):
    """Answers natural-language questions using a Reader + LLM client.

    Implementation lands in Phase 1A by wrapping pipeline/query.py.
    """

    reader: Reader

    def query(self, question: str) -> Answer: ...


class SearchEngine(Protocol):
    """Lexical/structural search across a Reader. May call fast model
    for query expansion.

    Implementation lands in Phase 1C.
    """

    reader: Reader

    def search(self, query: str) -> list[Hit]: ...


class MultiPackQueryEngine(Protocol):
    """Aggregates QueryEngines over multiple Readers with priority
    resolution.

    Implementation lands in Phase 1C.
    """

    engines: list[tuple[QueryEngine, int]]

    def query(self, question: str) -> Answer: ...


# ── Skeleton concrete classes (raise NotImplementedError) ─────────────────


class _BaseQueryEngine:
    """Phase 0 skeleton."""

    def __init__(self, reader: Reader) -> None:
        self.reader = reader

    def query(self, question: str) -> Answer:
        raise NotImplementedError("QueryEngine.query lands in Phase 1A")


class _BaseSearchEngine:
    """Phase 0 skeleton."""

    def __init__(self, reader: Reader) -> None:
        self.reader = reader

    def search(self, query: str) -> list[Hit]:
        raise NotImplementedError("SearchEngine.search lands in Phase 1C")


class _BaseMultiPackQueryEngine:
    """Phase 0 skeleton."""

    def __init__(self, engines: list[tuple[QueryEngine, int]] | None = None) -> None:
        self.engines = engines or []

    def query(self, question: str) -> Answer:
        raise NotImplementedError("MultiPackQueryEngine.query lands in Phase 1C")

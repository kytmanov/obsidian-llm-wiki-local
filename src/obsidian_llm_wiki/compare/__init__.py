"""
olw compare — multi-LLM comparison for the obsidian-llm-wiki pipeline.

Public API is intentionally tiny; the CLI command in cli.py is the
user-facing entry point.
"""

from __future__ import annotations

from .corpus import Corpus, CorpusMode, Note, Query, detect_mode, load_corpus, notes_set_hash
from .models import (
    CompareReport,
    ContestantResult,
    ContestantSpec,
    DimScore,
)

__all__ = [
    "CompareReport",
    "ContestantResult",
    "ContestantSpec",
    "Corpus",
    "CorpusMode",
    "DimScore",
    "Note",
    "Query",
    "detect_mode",
    "load_corpus",
    "notes_set_hash",
]

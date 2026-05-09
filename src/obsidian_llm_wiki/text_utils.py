"""
Text processing utilities.

Provides concept normalization functions used across multiple modules.
"""

from __future__ import annotations

import hashlib
import re


def normalize_concept(name: str, level: int = 3) -> str:
    """
    Concept normalization pipeline.

    Levels control normalization strength:
    - 1: Case folding
    - 2: + Punctuation stripping
    - 3: + Character normalization (for display/storage)
    - 4: + Whitespace folding (for comparison/fingerprint only)
    """
    text = name.strip()

    # Level 1: Case folding
    text = text.lower()

    if level >= 2:
        # Level 2: Punctuation stripping
        text = re.sub(r"[_\-/:]+", " ", text)

    if level >= 3:
        # Level 3: Character normalization
        text = re.sub(r"\(.*?\)", "", text)
        text = re.sub(r"[^\w\s]+", "", text)

    if level >= 4:
        # Level 4: Whitespace folding (fingerprint)
        text = re.sub(r"\s+", "", text)

    return text.strip()


def canonicalize_concept(name: str) -> str:
    """
    Canonicalize concept name for display/storage (level 3).

    Preserves readability while removing formatting variations.
    """
    return normalize_concept(name, level=3)


def fingerprint_concept(name: str) -> str:
    """
    Generate concept fingerprint for dedup comparison (level 4).

    Maximizes match rate by removing all whitespace and special characters.
    """
    return normalize_concept(name, level=4)


def hash_concepts(concepts: list[str]) -> str:
    """
    Generate a hash for a group of concepts.

    Used to uniquely identify dedup groups for skip tracking.
    Sorts concepts first to ensure consistent hashing regardless of order.

    Args:
        concepts: List of concept names.

    Returns:
        SHA256 hash of the sorted, joined concept names.
    """
    sorted_concepts = sorted(c.lower().strip() for c in concepts)
    key = "|".join(sorted_concepts)
    return hashlib.sha256(key.encode("utf-8")).hexdigest()

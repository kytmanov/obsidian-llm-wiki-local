"""
Tag sanitization utilities.

Leaf module — no project imports, safe to import from models.py and vault.py.
"""

from __future__ import annotations

import re

# Valid Obsidian tag: [a-zA-Z0-9][a-zA-Z0-9_/-]*
# We enforce lowercase by convention (consistent with hardcoded tags: source, meta, index, query).
_INVALID_CHARS = re.compile(r"[^a-zA-Z0-9_/\-]")
_LEADING_NON_ALNUM = re.compile(r"^[^a-zA-Z0-9]+")


def sanitize_tag(raw: str) -> str:
    """Convert arbitrary string to a valid Obsidian tag (lowercase convention).

    Steps:
      1. Strip whitespace
      2. Replace spaces with hyphens
      3. Remove chars not in [a-zA-Z0-9_/-]
      4. Strip leading non-alphanumeric chars
      5. Lowercase
      6. Return "" if nothing remains
    """
    tag = raw.strip()
    tag = tag.replace(" ", "-")
    tag = _INVALID_CHARS.sub("", tag)
    tag = _LEADING_NON_ALNUM.sub("", tag)
    tag = tag.lower()
    return tag


def sanitize_tags(raw_tags: list[str]) -> list[str]:
    """Map sanitize_tag over list, filter empties, deduplicate (preserving order)."""
    seen: set[str] = set()
    result: list[str] = []
    for raw in raw_tags:
        tag = sanitize_tag(raw)
        if tag and tag not in seen:
            seen.add(tag)
            result.append(tag)
    return result

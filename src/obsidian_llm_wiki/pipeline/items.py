"""Deterministic knowledge item extraction and audit helpers.

This intentionally avoids LLM calls. The goal is to preserve high-signal
non-concept references for later audit without promoting them to articles.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from ..models import ItemMentionRecord, KnowledgeItemRecord
from ..state import StateDB

_CONNECTOR_WORDS = {
    "a",
    "an",
    "and",
    "at",
    "for",
    "in",
    "of",
    "on",
    "the",
    "to",
    "with",
}
_PERSON_PAIR_RE = re.compile(r"\b([A-Z][a-z]{2,})\s+([A-Z][a-z]{2,})\b")
_CAMEL_ORG_RE = re.compile(r"\b[A-Z][A-Za-z]*[a-z][A-Z][A-Za-z]+\b")
_PRODUCT_MODEL_RE = re.compile(r"\b[A-Z][A-Za-z]+\s+[A-Z]{1,4}\d{1,4}\b")
_QUOTED_TITLE_RE = re.compile(
    r'"(.{4,80}?)"'
    r"|“(.{4,80}?)”"
    r"|„(.{4,80}?)[“”]"
    r"|«(.{4,80}?)»"
    r"|‹(.{4,80}?)›"
    r"|「(.{4,80}?)」"
    r"|『(.{4,80}?)』"
    r"|《(.{4,80}?)》"
)
_WORD_RE = re.compile(r"[^\W\d_]+")
_QUOTED_SEGMENT_SEPARATOR_RE = re.compile(r"\s*(?:[:|/]|-{1,2}|[–—])\s*")


@dataclass(frozen=True)
class ExtractedItem:
    name: str
    subtype: str
    mention_text: str
    evidence_level: str
    confidence: float
    context: str


def _clean_item_name(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip(" -_:;,.\t\n")).strip()


def _is_noisy_item(name: str) -> bool:
    lowered = name.casefold()
    if len(name) < 3:
        return True
    if "unknown_filename" in lowered or lowered.startswith("unknown"):
        return True
    if Path(name).suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".pdf"}:
        return True
    if lowered in _CONNECTOR_WORDS:
        return True
    return False


def _dedupe_items(items: list[ExtractedItem]) -> list[ExtractedItem]:
    seen: set[str] = set()
    result: list[ExtractedItem] = []
    for item in items:
        key = item.name.casefold()
        if key in seen or _is_noisy_item(item.name):
            continue
        seen.add(key)
        result.append(item)
    return result


def _quoted_match_text(match: re.Match[str]) -> str:
    for group in match.groups():
        if group is not None:
            return group
    return ""


def _has_quoted_item_substance(name: str) -> bool:
    compact = re.sub(r"\s+", "", name)
    if len(compact) < 4:
        return False

    words = _WORD_RE.findall(name)
    if not words:
        return False
    if len(words) >= 2:
        return True
    return sum(1 for char in name if char.isalnum()) >= 4


def _is_prominent_quoted_item(name: str, title: str, match: re.Match[str]) -> bool:
    """Keep only structurally prominent quoted candidates, without language word lists."""
    if not _has_quoted_item_substance(name):
        return False

    quote = match.group(0).strip()
    for segment in _QUOTED_SEGMENT_SEPARATOR_RE.split(title.strip()):
        normalized_segment = segment.strip().strip("()[]{}")
        if normalized_segment == quote:
            return True
    return False


def extract_title_items(title: str, source_path: str) -> list[ExtractedItem]:
    """Extract high-signal entity/ambiguous candidates from a title or filename."""
    title = _clean_item_name(title)
    if not title:
        return []

    items: list[ExtractedItem] = []

    for match in _PRODUCT_MODEL_RE.finditer(title):
        name = _clean_item_name(match.group(0))
        items.append(
            ExtractedItem(
                name=name,
                subtype="product",
                mention_text=name,
                evidence_level="title_supported",
                confidence=0.75,
                context=title,
            )
        )

    for match in _CAMEL_ORG_RE.finditer(title):
        name = _clean_item_name(match.group(0))
        items.append(
            ExtractedItem(
                name=name,
                subtype="event_or_org",
                mention_text=name,
                evidence_level="title_supported",
                confidence=0.65,
                context=title,
            )
        )

    for match in _QUOTED_TITLE_RE.finditer(title):
        name = _clean_item_name(_quoted_match_text(match))
        if not _is_prominent_quoted_item(name, title, match):
            continue
        items.append(
            ExtractedItem(
                name=name,
                subtype="quoted_title",
                mention_text=name,
                evidence_level="title_supported",
                confidence=0.55,
                context=title,
            )
        )

    for match in _PERSON_PAIR_RE.finditer(title):
        name = _clean_item_name(match.group(0))
        items.append(
            ExtractedItem(
                name=name,
                subtype="person",
                mention_text=name,
                evidence_level="title_supported",
                confidence=0.65,
                context=title,
            )
        )

    return _dedupe_items(items)


def store_extracted_items(db: StateDB, source_path: str, items: list[ExtractedItem]) -> None:
    for item in items:
        db.upsert_item(
            KnowledgeItemRecord(
                name=item.name,
                kind="ambiguous",
                subtype=item.subtype,
                status="candidate",
                confidence=item.confidence,
            )
        )
        db.add_item_mention(
            ItemMentionRecord(
                item_name=item.name,
                source_path=source_path,
                mention_text=item.mention_text,
                context=item.context,
                evidence_level=item.evidence_level,
                confidence=item.confidence,
            )
        )

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
_QUOTED_TITLE_RE = re.compile(r"[“\"«](.{6,80}?)[”\"»]")


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
        name = _clean_item_name(match.group(1))
        items.append(
            ExtractedItem(
                name=name,
                subtype="work",
                mention_text=name,
                evidence_level="title_supported",
                confidence=0.65,
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

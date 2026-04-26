"""Knowledge item extraction and audit helpers.

Named references are proposed by the ingest LLM and accepted only when exact
source evidence exists. Quoted-title extraction remains deterministic because it
uses structure rather than language-specific word lists.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from ..models import ItemMentionRecord, KnowledgeItemRecord
from ..state import StateDB

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
_URL_RE = re.compile(r"^(?:https?://|www\.)", re.I)
_MEDIA_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".pdf", ".svg", ".webp"}
_MIN_SOURCE_SUPPORTED_CHARS = 5


@dataclass(frozen=True)
class ExtractedItem:
    name: str
    subtype: str
    mention_text: str
    evidence_level: str
    confidence: float
    context: str


def _clean_item_name(name: str) -> str:
    name = unicodedata.normalize("NFKC", name)
    return re.sub(r"\s+", " ", name.strip(" -_:;,.\t\n")).strip()


def _evidence_text(*parts: str) -> str:
    return unicodedata.normalize("NFKC", " ".join(part for part in parts if part))


def _is_noisy_item(name: str) -> bool:
    lowered = name.casefold()
    if len(name) < 3:
        return True
    if "unknown_filename" in lowered:
        return True
    if _URL_RE.match(name):
        return True
    if Path(name).suffix.lower() in _MEDIA_SUFFIXES:
        return True
    if not any(char.isalnum() for char in name):
        return True
    if len(name) > 120:
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


def _has_exact_evidence(name: str, title: str, body: str, source_path: str) -> bool:
    needle = unicodedata.normalize("NFKC", name).strip()
    haystack = _evidence_text(title, body, Path(source_path).stem)
    if not needle or not haystack:
        return False
    if needle in haystack:
        return True
    return needle.casefold() in haystack.casefold()


def _matches_concept(name: str, concept_names: list[str]) -> bool:
    key = name.casefold().strip()
    return any(key == concept.casefold().strip() for concept in concept_names)


def _has_case_distinction(name: str) -> bool:
    return name.lower() != name.upper()


def _is_substantive_source_reference(name: str) -> bool:
    """Filter typo-like source references without relying on language word lists."""
    compact = re.sub(r"\s+", "", name)
    if len(compact) < _MIN_SOURCE_SUPPORTED_CHARS:
        return False
    if len(_WORD_RE.findall(name)) >= 2:
        return True
    if _has_case_distinction(name) and name != name.lower():
        return True
    if not _has_case_distinction(name):
        return True
    return any(char.isdigit() for char in name) or len(compact) >= 8


def extract_quoted_title_items(title: str, source_path: str) -> list[ExtractedItem]:
    """Extract structurally prominent quoted titles from a title or filename."""
    title = _clean_item_name(title)
    if not title:
        return []

    items: list[ExtractedItem] = []
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

    return _dedupe_items(items)


def extract_named_reference_items(
    references: list[str],
    title: str,
    body: str,
    source_path: str,
    concept_names: list[str],
) -> list[ExtractedItem]:
    """Accept LLM-proposed named references only when exact source evidence exists."""
    title = _clean_item_name(title)
    items: list[ExtractedItem] = []
    for raw_name in references:
        name = _clean_item_name(raw_name)
        if _is_noisy_item(name):
            continue
        if _matches_concept(name, concept_names):
            continue
        if not _has_exact_evidence(name, title, body, source_path):
            continue
        title_supported = _has_exact_evidence(name, title, "", source_path)
        if not title_supported and not _is_substantive_source_reference(name):
            continue
        items.append(
            ExtractedItem(
                name=name,
                subtype="named_reference",
                mention_text=name,
                evidence_level="title_supported" if title_supported else "source_supported",
                confidence=0.50,
                context=title or source_path,
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

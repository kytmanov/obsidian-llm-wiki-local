"""
Draft review helpers — testable logic separated from CLI concerns.

Used by `olw review` to list drafts, display diffs, and surface rejection history.
"""

from __future__ import annotations

import difflib
import logging
from dataclasses import dataclass
from pathlib import Path

from ..config import Config
from ..state import StateDB
from ..vault import parse_note, sanitize_filename

log = logging.getLogger(__name__)


@dataclass
class DraftSummary:
    path: Path
    title: str
    confidence: float
    source_count: int
    rejection_count: int
    has_published_version: bool
    has_annotations: bool


def list_drafts(config: Config, db: StateDB) -> list[DraftSummary]:
    """Return summaries of all drafts in wiki/.drafts/, sorted by rejection_count desc."""
    if not config.drafts_dir.exists():
        return []

    summaries = []
    for draft_path in sorted(config.drafts_dir.rglob("*.md")):
        try:
            meta, body = parse_note(draft_path)
        except Exception as e:
            log.warning("Could not read draft %s: %s", draft_path, e)
            continue

        title = meta.get("title", draft_path.stem)
        confidence = float(meta.get("confidence", 0.0))
        sources = meta.get("sources", [])
        source_count = len(sources) if isinstance(sources, list) else 0
        rejection_count = db.rejection_count(title)

        # Check for published version
        safe_name = sanitize_filename(title)
        wiki_path = config.wiki_dir / f"{safe_name}.md"
        has_published_version = wiki_path.exists()

        # Check for olw-auto annotations in body
        has_annotations = "<!-- olw-auto:" in body

        summaries.append(
            DraftSummary(
                path=draft_path,
                title=title,
                confidence=confidence,
                source_count=source_count,
                rejection_count=rejection_count,
                has_published_version=has_published_version,
                has_annotations=has_annotations,
            )
        )

    # Sort: most rejections first (need attention), then by title
    summaries.sort(key=lambda s: (-s.rejection_count, s.title.lower()))
    return summaries


def load_draft_content(draft_path: Path) -> tuple[dict, str]:
    """Return (frontmatter_dict, body) for a draft. Raises on failure."""
    return parse_note(draft_path)


def compute_diff(draft_path: Path, wiki_path: Path) -> str | None:
    """
    Compute unified diff between a draft and its published version.
    Returns None if the published version doesn't exist.
    """
    if not wiki_path.exists():
        return None
    try:
        _, draft_body = parse_note(draft_path)
        _, wiki_body = parse_note(wiki_path)
    except Exception as e:
        log.warning("Could not compute diff: %s", e)
        return None

    diff_lines = list(
        difflib.unified_diff(
            wiki_body.splitlines(keepends=True),
            draft_body.splitlines(keepends=True),
            fromfile="published",
            tofile="draft",
            lineterm="",
        )
    )
    if not diff_lines:
        return "(no differences)"
    return "".join(diff_lines)


def compute_rejection_diff(draft_path: Path, db: StateDB, concept: str) -> str | None:
    """
    Compute unified diff between current draft body and the last rejected body.
    Returns None if no rejection body is stored (pre-v0.2 rejection or no rejections).
    """
    rejections = db.get_rejections(concept, limit=1)
    if not rejections or not rejections[0].get("body"):
        return None

    try:
        _, current_body = parse_note(draft_path)
    except Exception as e:
        log.warning("Could not read current draft for diff: %s", e)
        return None

    rejected_body = rejections[0]["body"]
    diff_lines = list(
        difflib.unified_diff(
            rejected_body.splitlines(keepends=True),
            current_body.splitlines(keepends=True),
            fromfile="rejected",
            tofile="current",
            lineterm="",
        )
    )
    if not diff_lines:
        return "(no differences from rejected version)"
    return "".join(diff_lines)

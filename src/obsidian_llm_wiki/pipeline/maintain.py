"""
Wiki maintenance — self-initiated health operations.

Used by `olw maintain` to:
  - Create stub drafts for broken wikilinks
  - Suggest orphan link fixes
  - Suggest concept merges for near-duplicates
  - Report source quality distribution
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path

import frontmatter as fm_lib

from ..config import Config
from ..models import LintIssue
from ..state import StateDB
from ..vault import atomic_write, parse_note, sanitize_filename

log = logging.getLogger(__name__)

_STUB_BODY = """\
> [!info] This is a stub article — referenced by other pages but no source material yet.

Add raw notes about this topic to `raw/` and run `olw compile` to generate a full article.
"""

_CONCEPT_MERGE_THRESHOLD = 0.7


def create_stubs(
    config: Config,
    db: StateDB,
    broken_link_issues: list[LintIssue] | None = None,
    max_stubs: int = 5,
) -> list[Path]:
    """
    Create stub drafts for broken wikilinks.

    Finds [[Target]] references that have no matching article, creates placeholder
    drafts and registers them in the stubs table so compile can pick them up.

    Pass broken_link_issues to avoid re-running lint. If None, lint runs internally.
    """
    if broken_link_issues is None:
        from .lint import run_lint

        result = run_lint(config, db)
        broken_link_issues = [i for i in result.issues if i.issue_type == "broken_link"]

    # Extract target concept names from broken link descriptions
    # LintIssue description format: "[[Target]] not found" or similar
    created: list[Path] = []
    seen: set[str] = set()

    for issue in broken_link_issues:
        if len(created) >= max_stubs:
            log.info("Stub cap (%d) reached — stopping", max_stubs)
            break

        # Extract target from description (e.g. "[[Quantum Computing]] not found")
        target = _extract_link_target(issue.description)
        if not target:
            # Fall back to path stem
            target = Path(issue.path).stem

        if target in seen:
            continue
        seen.add(target)

        # Skip if already has a stub, draft, or published article
        if db.has_stub(target):
            continue
        safe_name = sanitize_filename(target)
        draft_path = config.drafts_dir / f"{safe_name}.md"
        wiki_path = config.wiki_dir / f"{safe_name}.md"
        if draft_path.exists() or wiki_path.exists():
            continue

        # Register in stubs table
        db.add_stub(target, source="auto")

        # Write placeholder draft
        config.drafts_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "title": target,
            "status": "stub",
            "tags": ["stub"],
            "sources": [],
            "confidence": 0.0,
            "created": datetime.now().strftime("%Y-%m-%d"),
            "updated": datetime.now().strftime("%Y-%m-%d"),
        }
        post = fm_lib.Post(_STUB_BODY, **meta)
        atomic_write(draft_path, fm_lib.dumps(post))
        created.append(draft_path)
        log.info("Stub created: %s", draft_path.name)

    return created


def _extract_link_target(description: str) -> str | None:
    """Extract [[Target]] from a lint issue description string."""
    match = re.search(r"\[\[([^\]]+)\]\]", description)
    return match.group(1) if match else None


def suggest_orphan_links(config: Config, db: StateDB) -> list[tuple[str, list[str]]]:
    """
    For each orphan article, find other articles that mention its title unlinked.

    Returns list of (orphan_title, [paths_that_mention_it]).
    """
    from .lint import run_lint

    result = run_lint(config, db)
    orphan_issues = [i for i in result.issues if i.issue_type == "orphan"]
    if not orphan_issues:
        return []

    # Load all published article bodies
    wiki_pages: dict[str, str] = {}
    if config.wiki_dir.exists():
        for p in config.wiki_dir.rglob("*.md"):
            if ".drafts" in p.parts:
                continue
            try:
                meta, body = parse_note(p)
                wiki_pages[str(p.relative_to(config.vault))] = body
            except Exception:
                pass

    suggestions = []
    for issue in orphan_issues:
        orphan_path = config.vault / issue.path
        try:
            meta, _ = parse_note(orphan_path)
            orphan_title = meta.get("title", orphan_path.stem)
        except Exception:
            orphan_title = orphan_path.stem

        # Find pages that mention the orphan title in plain text (not as wikilink)
        mentions = []
        title_pattern = re.compile(
            r"(?<!\[\[)\b" + re.escape(orphan_title) + r"\b(?!\]\])",
            re.IGNORECASE,
        )
        for page_path, body in wiki_pages.items():
            if page_path == issue.path:
                continue
            if title_pattern.search(body):
                mentions.append(page_path)

        if mentions:
            suggestions.append((orphan_title, mentions))

    return suggestions


def suggest_concept_merges(config: Config, db: StateDB) -> list[tuple[str, str, float]]:
    """
    Find near-duplicate concept pairs using Jaccard similarity on title tokens.

    Returns list of (concept_a, concept_b, similarity_score) for pairs > threshold.
    No LLM required — purely token-overlap based.
    """
    concepts = db.list_all_concept_names()
    if len(concepts) < 2:
        return []

    def tokenize(name: str) -> frozenset[str]:
        # Lowercase, split on spaces/hyphens/underscores, filter short tokens
        tokens = re.split(r"[\s\-_]+", name.lower())
        return frozenset(t for t in tokens if len(t) > 1)

    tokenized = [(c, tokenize(c)) for c in concepts]
    suggestions = []

    for i, (a, tokens_a) in enumerate(tokenized):
        for b, tokens_b in tokenized[i + 1 :]:
            if not tokens_a or not tokens_b:
                continue
            intersection = len(tokens_a & tokens_b)
            union = len(tokens_a | tokens_b)
            if union == 0:
                continue
            score = intersection / union
            if score >= _CONCEPT_MERGE_THRESHOLD:
                suggestions.append((a, b, round(score, 2)))

    suggestions.sort(key=lambda x: -x[2])
    return suggestions

from __future__ import annotations

from datetime import datetime, timedelta

from obsidian_llm_wiki.config import Config
from obsidian_llm_wiki.indexer import generate_index
from obsidian_llm_wiki.models import WikiArticleRecord
from obsidian_llm_wiki.state import StateDB
from obsidian_llm_wiki.vault import write_note


def test_generate_index_caps_and_orders_synthesis_entries(tmp_path):
    (tmp_path / "raw").mkdir()
    (tmp_path / "wiki").mkdir()
    (tmp_path / "wiki" / ".drafts").mkdir()
    (tmp_path / "wiki" / "sources").mkdir()
    (tmp_path / ".olw").mkdir()
    config = Config(vault=tmp_path)
    db = StateDB(config.state_db_path)

    base = datetime(2026, 5, 2, 12, 0, 0)
    expected_titles: list[str] = []
    for i in range(27):
        title = f"Topic {i:02d}"
        created_at = base - timedelta(minutes=i // 2)
        path = config.synthesis_dir / f"{title}.md"
        write_note(
            path,
            {"title": title, "tags": ["synthesis"], "kind": "synthesis", "status": "published"},
            "Body.",
        )
        db.upsert_article(
            WikiArticleRecord(
                path=str(path.relative_to(config.vault)),
                title=title,
                sources=[],
                content_hash=f"hash-{i}",
                created_at=created_at,
                updated_at=created_at,
                is_draft=False,
                kind="synthesis",
                question_hash=f"qh-{i}",
            )
        )

    ordered = sorted(
        db.list_articles(),
        key=lambda article: (-article.created_at.timestamp(), article.title.casefold()),
    )
    expected_titles = [article.title for article in ordered[:25] if article.kind == "synthesis"]

    index_text = generate_index(config, db).read_text(encoding="utf-8")

    assert "## Synthesis" in index_text
    for title in expected_titles:
        assert title in index_text
    assert "Topic 26" not in index_text or "Topic 25" not in index_text
    assert "_(2 more synthesis pages not shown)_" in index_text

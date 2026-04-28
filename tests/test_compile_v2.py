"""Tests for v0.2 compile.py additions: annotations, rejection feedback, selective compile."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from obsidian_llm_wiki.config import Config
from obsidian_llm_wiki.models import RawNoteRecord, WikiArticleRecord
from obsidian_llm_wiki.ollama_client import OllamaClient
from obsidian_llm_wiki.pipeline.compile import (
    _apply_draft_media_mode,
    _build_olw_annotations,
    _build_source_refs,
    _gather_sources,
    _inject_body_sections,
    _remove_dangling_open_brackets,
    _repair_bare_bracket_links,
    _repair_literal_newlines,
    _repair_malformed_embeds,
    _repair_malformed_wikilinks,
    _rewrite_citation_markers,
    _strip_empty_wikilinks,
    _strip_olw_annotations,
    _strip_self_wikilinks,
    _strip_unknown_wikilinks,
    _write_concept_prompt,
    approve_drafts,
    compile_concepts,
    reject_draft,
)
from obsidian_llm_wiki.state import StateDB


@pytest.fixture
def vault(tmp_path: Path) -> Path:
    (tmp_path / "raw").mkdir()
    (tmp_path / "wiki").mkdir()
    (tmp_path / "wiki" / ".drafts").mkdir()
    (tmp_path / ".olw").mkdir()
    return tmp_path


@pytest.fixture
def config(vault: Path) -> Config:
    return Config(vault=vault)


@pytest.fixture
def db(config: Config) -> StateDB:
    return StateDB(config.state_db_path)


def make_mock_client(response: str = "{}") -> OllamaClient:
    client = MagicMock(spec=OllamaClient)
    client.generate.return_value = response
    return client


# ── Annotations ───────────────────────────────────────────────────────────────


def test_build_annotations_low_confidence(db):
    annotations = _build_olw_annotations(0.3, ["raw/a.md"], db)
    assert any("low-confidence" in a for a in annotations)


def test_build_annotations_single_source(db):
    annotations = _build_olw_annotations(0.8, ["raw/a.md"], db)
    assert any("single-source" in a for a in annotations)


def test_build_no_annotations_above_threshold(db):
    rec_a = RawNoteRecord(path="raw/a.md", content_hash="h1", status="ingested", quality="high")
    rec_b = RawNoteRecord(path="raw/b.md", content_hash="h2", status="ingested", quality="high")
    db.upsert_raw(rec_a)
    db.upsert_raw(rec_b)
    annotations = _build_olw_annotations(0.75, ["raw/a.md", "raw/b.md"], db)
    # High confidence, multiple sources, high quality → no annotations
    assert annotations == []


def test_build_annotations_all_low_quality(db):
    rec_a = RawNoteRecord(path="raw/a.md", content_hash="h1", status="ingested", quality="low")
    rec_b = RawNoteRecord(path="raw/b.md", content_hash="h2", status="ingested", quality="low")
    db.upsert_raw(rec_a)
    db.upsert_raw(rec_b)
    annotations = _build_olw_annotations(0.6, ["raw/a.md", "raw/b.md"], db)
    assert any("low-quality" in a for a in annotations)


def test_strip_annotations_removes_olw_auto(db):
    body = "<!-- olw-auto: low-confidence -->\n\n## Overview\n\nContent."
    stripped = _strip_olw_annotations(body)
    assert "olw-auto" not in stripped
    assert "## Overview" in stripped
    assert "Content." in stripped


def test_strip_annotations_leaves_user_html_comments(db):
    body = "<!-- user comment -->\n\n<!-- olw-auto: single-source -->\n\nContent."
    stripped = _strip_olw_annotations(body)
    assert "<!-- user comment -->" in stripped
    assert "olw-auto" not in stripped


def test_strip_annotations_no_annotations_unchanged(db):
    body = "## Title\n\nNo annotations here."
    assert _strip_olw_annotations(body) == body


def test_inject_body_sections_always_includes_sources_heading(config):
    body = _inject_body_sections("## Overview\n\nContent.", [], config)
    assert "## Sources" in body


def test_build_source_refs_stable_order_and_metadata(vault):
    (vault / "raw" / "a.md").write_text("---\ntitle: Alpha Source\n---\nBody A.")
    (vault / "raw" / "b.md").write_text("Body B.")

    refs = _build_source_refs(["raw/a.md", "raw/b.md"], vault)

    assert [r.id for r in refs] == ["S1", "S2"]
    assert refs[0].raw_path == "raw/a.md"
    assert refs[0].title == "Alpha Source"
    assert refs[0].wiki_target == "sources/Alpha Source"
    assert refs[1].title == "B"


def test_gather_sources_with_refs_labels_source_ids(vault):
    (vault / "raw" / "a.md").write_text("---\ntitle: Alpha Source\n---\nBody A.")
    refs = _build_source_refs(["raw/a.md"], vault)

    text, resolved = _gather_sources(["raw/a.md"], vault, source_refs=refs)

    assert "## Source [S1]: Alpha Source (raw/a.md)" in text
    assert resolved == ["raw/a.md"]


def test_rewrite_citation_markers_single_and_multi(vault):
    (vault / "raw" / "a.md").write_text("---\ntitle: Alpha Source\n---\nA")
    (vault / "raw" / "b.md").write_text("---\ntitle: Beta Source\n---\nB")
    refs = _build_source_refs(["raw/a.md", "raw/b.md"], vault)

    body = _rewrite_citation_markers("Claim [S1]. Joint [S1, S2].", refs)

    assert "([[sources/Alpha Source|S1]])" in body
    assert "([[sources/Alpha Source|S1]], [[sources/Beta Source|S2]])" in body


def test_rewrite_citation_markers_legend_only_keeps_plain_ids(vault):
    (vault / "raw" / "a.md").write_text("---\ntitle: Alpha Source\n---\nA")
    (vault / "raw" / "b.md").write_text("---\ntitle: Beta Source\n---\nB")
    refs = _build_source_refs(["raw/a.md", "raw/b.md"], vault)

    body = _rewrite_citation_markers("Claim [S1]. Joint [S1, S2].", refs, link_inline=False)

    assert body == "Claim [S1](#Sources). Joint [S1,S2](#Sources)."
    assert "sources/" not in body


def test_rewrite_citation_markers_ignores_unknown_ids(vault):
    (vault / "raw" / "a.md").write_text("---\ntitle: Alpha Source\n---\nA")
    refs = _build_source_refs(["raw/a.md"], vault)

    body = _rewrite_citation_markers("Known [S1]. Unknown [S99]. Mixed [S1,S99].", refs)

    assert "Known ([[sources/Alpha Source|S1]])." in body
    assert "Unknown ." in body
    assert "Mixed ([[sources/Alpha Source|S1]])." in body


def test_rewrite_citation_markers_skips_code_embeds_and_existing_links(vault):
    (vault / "raw" / "a.md").write_text("---\ntitle: Alpha Source\n---\nA")
    refs = _build_source_refs(["raw/a.md"], vault)

    body = _rewrite_citation_markers(
        "`code [S1]`\n\n```\nblock [S1]\n```\n![[image [S1].png]] [[Existing [S1]]] prose [S1]",
        refs,
    )

    assert "`code [S1]`" in body
    assert "block [S1]" in body
    assert "![[image [S1].png]]" in body
    assert "[[Existing [S1]]]" in body
    assert "prose ([[sources/Alpha Source|S1]])" in body


def test_rewrite_citation_markers_restores_masks_after_length_changes(vault):
    (vault / "raw" / "a.md").write_text("---\ntitle: Alpha Source\n---\nA")
    refs = _build_source_refs(["raw/a.md"], vault)

    body = _rewrite_citation_markers("prose [S1] then `code [S1]` and [[Link [S1]]].", refs)

    assert body == ("prose ([[sources/Alpha Source|S1]]) then `code [S1]` and [[Link [S1]]].")


def test_repair_bare_bracket_links_converts_llm_slips():
    body = _repair_bare_bracket_links("See [API] and [Product Backlog].")

    assert body == "See [[API]] and [[Product Backlog]]."


def test_repair_bare_bracket_links_skips_markdown_citations_and_masks():
    body = _repair_bare_bracket_links(
        "Claim [S1]. Link [text](https://example.com). `code [API]` [[Already]] prose [API]"
    )

    assert body == (
        "Claim [S1]. Link [text](https://example.com). `code [API]` [[Already]] prose [[API]]"
    )


def test_repair_literal_newlines_converts_escaped_markdown():
    body = _repair_literal_newlines("## A\\n\\nBody\\n- item")

    assert body == "## A\n\nBody\n- item"


def test_strip_unknown_wikilinks_unwraps_invented_links():
    body = _strip_unknown_wikilinks(
        "Known [[API Testing]]. Unknown [[Название статьи]]. Aliased [[Missing|display]].",
        ["API Testing"],
    )

    assert body == "Known [[API Testing]]. Unknown Название статьи. Aliased display."


def test_strip_unknown_wikilinks_keeps_source_links():
    body = _strip_unknown_wikilinks(
        "Claim ([[sources/Api Testing Example|S1]]) and [[sources/Foo]].",
        [],
    )

    assert body == "Claim ([[sources/Api Testing Example|S1]]) and [[sources/Foo]]."


def test_strip_unknown_wikilinks_keeps_embeds():
    body = _strip_unknown_wikilinks("Diagram ![[file.pdf]] and unknown [[Ghost]].", [])

    assert body == "Diagram ![[file.pdf]] and unknown Ghost."


def test_strip_self_wikilinks_unwraps_self_links():
    body = _strip_self_wikilinks("[[Scrum]] and [[Scrum|this page]] link to [[Kanban]].", "Scrum")

    assert body == "Scrum and this page link to [[Kanban]]."


def test_strip_empty_wikilinks_removes_empty_targets():
    body = _strip_empty_wikilinks("Bad [[]]. Display [[|shown]]. Keep [[Kanban]].")

    assert body == "Bad . Display shown. Keep [[Kanban]]."


def test_repair_malformed_embeds_converts_bare_media_embed():
    body = _repair_malformed_embeds("Diagram !./_resources/file.pdf and ![[already.pdf]].")

    assert body == "Diagram ![[./_resources/file.pdf]] and ![[already.pdf]]."


def test_repair_malformed_embeds_converts_escaped_markdown_media_refs():
    body = _repair_malformed_embeds(
        "Images !\\[./_resources/a.jpeg\\]!\\[./_resources/b.jpeg\\!\\[./_resources/c.jpeg\\]"
    )

    assert "![[./_resources/a.jpeg]]" in body
    assert "![[./_resources/b.jpeg]]" in body
    assert "![[./_resources/c.jpeg]]" in body
    assert "!\\[" not in body


def test_remove_dangling_open_brackets():
    body = _remove_dangling_open_brackets("Check these aspects [\n\nNext paragraph.")

    assert body == "Check these aspects\n\nNext paragraph."


def test_repair_malformed_wikilinks_strips_quote_and_citation_debris():
    body = _repair_malformed_wikilinks(
        "See [[Independent publishing culture”]] and [[Documentary notes”, S2]].",
        ["Independent publishing culture", "Documentary notes"],
    )

    assert body == "See [[Independent publishing culture]] and [[Documentary notes]]."


def test_apply_draft_media_mode_reference_replaces_embeds():
    body = _apply_draft_media_mode("Diagram ![[./_resources/file.pdf]].", "reference")

    assert body == "Diagram Media reference: ./_resources/file.pdf."


def test_apply_draft_media_mode_embed_keeps_embeds():
    body = _apply_draft_media_mode("Diagram ![[./_resources/file.pdf]].", "embed")

    assert body == "Diagram ![[./_resources/file.pdf]]."


def test_apply_draft_media_mode_omit_removes_embeds():
    body = _apply_draft_media_mode("Diagram ![[./_resources/file.pdf]].", "omit")

    assert body == "Diagram ."


def test_inject_body_sections_uses_id_legend_when_enabled(config):
    (config.vault / "raw" / "a.md").write_text("---\ntitle: Alpha Source\n---\nA")
    config.pipeline.inline_source_citations = True

    body = _inject_body_sections(
        "See [[Concept]] and [[sources/Alpha Source|S1]].", ["raw/a.md"], config
    )

    assert "- [S1] [[sources/Alpha Source|Alpha Source]]" in body
    assert "- [[Concept]]" in body
    assert "- [[sources/Alpha Source]]" not in body


def test_inject_body_sections_skips_self_link_in_see_also(config):
    body = _inject_body_sections("See [[Scrum]] and [[Kanban]].", [], config, article_title="Scrum")

    assert "- [[Kanban]]" in body
    assert "- [[Scrum]]" not in body


def test_write_concept_prompt_citation_instructions_flagged():
    prompt = _write_concept_prompt("Topic", "source", [], inline_source_citations=True)
    assert "Inline source citations" in prompt
    assert "[S1,S2]" in prompt


def test_write_concept_prompt_no_citation_instructions_by_default():
    prompt = _write_concept_prompt("Topic", "source", [])
    assert "Inline source citations" not in prompt


def test_annotations_stripped_on_approve(config, db, tmp_path):
    """Annotations must be absent in published articles."""
    import frontmatter as fm_lib

    from obsidian_llm_wiki.vault import atomic_write, parse_note

    annotated_body = "<!-- olw-auto: low-confidence (0.25) -->\n\n## Overview\n\nContent."
    meta = {
        "title": "Test Article",
        "status": "draft",
        "tags": [],
        "sources": [],
        "confidence": 0.25,
        "created": "2024-01-01",
        "updated": "2024-01-01",
    }
    draft_path = config.drafts_dir / "Test_Article.md"
    atomic_write(draft_path, fm_lib.dumps(fm_lib.Post(annotated_body, **meta)))
    db.upsert_article(
        WikiArticleRecord(
            path=str(draft_path.relative_to(config.vault)),
            title="Test Article",
            sources=[],
            content_hash="h",
            is_draft=True,
        )
    )

    published = approve_drafts(config, db, [draft_path])
    assert len(published) == 1
    _, pub_body = parse_note(published[0])
    assert "olw-auto" not in pub_body
    assert "## Overview" in pub_body


# ── Rejection feedback in prompt ──────────────────────────────────────────────


def test_write_concept_prompt_no_rejection_history():
    prompt = _write_concept_prompt("Quantum Computing", "source text", [])
    assert "PREVIOUS REJECTIONS" not in prompt


def test_write_concept_prompt_with_rejection_history():
    history = ["Too vague", "Wrong format"]
    prompt = _write_concept_prompt(
        "Quantum Computing", "source text", [], rejection_history=history
    )
    assert "PREVIOUS REJECTIONS" in prompt
    assert "Too vague" in prompt
    assert "Wrong format" in prompt


def test_write_concept_prompt_deduplicates_feedback():
    history = ["Same feedback", "Same feedback", "Different feedback"]
    prompt = _write_concept_prompt("Topic", "source", [], rejection_history=history)
    # Should appear only once
    assert prompt.count("Same feedback") == 1
    assert "Different feedback" in prompt


# ── reject_draft stores feedback and body ────────────────────────────────────


def test_reject_draft_stores_feedback(config, db, tmp_path):
    import frontmatter as fm_lib

    from obsidian_llm_wiki.vault import atomic_write

    draft_path = config.drafts_dir / "Test.md"
    meta = {"title": "Test Topic", "status": "draft", "tags": [], "sources": [], "confidence": 0.5}
    atomic_write(draft_path, fm_lib.dumps(fm_lib.Post("Draft body.", **meta)))
    db.upsert_article(
        WikiArticleRecord(
            path=str(draft_path.relative_to(config.vault)),
            title="Test Topic",
            sources=[],
            content_hash="h",
            is_draft=True,
        )
    )

    reject_draft(draft_path, config, db, feedback="Needs more detail")

    assert not draft_path.exists()
    rejections = db.get_rejections("Test Topic")
    assert len(rejections) == 1
    assert rejections[0]["feedback"] == "Needs more detail"
    assert rejections[0]["body"] == "Draft body."


def test_reject_draft_no_feedback_no_rejection(config, db):
    import frontmatter as fm_lib

    from obsidian_llm_wiki.vault import atomic_write

    draft_path = config.drafts_dir / "Test.md"
    meta = {"title": "Test", "status": "draft", "tags": [], "sources": [], "confidence": 0.5}
    atomic_write(draft_path, fm_lib.dumps(fm_lib.Post("body", **meta)))
    db.upsert_article(
        WikiArticleRecord(
            path=str(draft_path.relative_to(config.vault)),
            title="Test",
            sources=[],
            content_hash="h",
            is_draft=True,
        )
    )

    reject_draft(draft_path, config, db, feedback="")
    assert db.rejection_count("Test") == 0


def test_reject_draft_blocks_after_cap(config, db):
    """5th rejection auto-blocks the concept."""
    import frontmatter as fm_lib

    from obsidian_llm_wiki.vault import atomic_write

    for i in range(4):
        db.add_rejection("Blocked Topic", f"feedback {i}", body="body")

    # 5th rejection via reject_draft
    draft_path = config.drafts_dir / "Blocked.md"
    meta = {
        "title": "Blocked Topic",
        "status": "draft",
        "tags": [],
        "sources": [],
        "confidence": 0.5,
    }  # noqa: E501
    atomic_write(draft_path, fm_lib.dumps(fm_lib.Post("body", **meta)))
    db.upsert_article(
        WikiArticleRecord(
            path=str(draft_path.relative_to(config.vault)),
            title="Blocked Topic",
            sources=[],
            content_hash="h",
            is_draft=True,
        )
    )

    reject_draft(draft_path, config, db, feedback="Still wrong")
    assert db.is_concept_blocked("Blocked Topic")


# ── Selective compile (concepts= param) ───────────────────────────────────────


def test_compile_concepts_selective(config, db, tmp_path):
    """compile_concepts(concepts=['Alpha']) should only compile Alpha, not Beta."""
    import json

    # Set up two concepts needing compile
    db.upsert_raw(RawNoteRecord(path="raw/a.md", content_hash="h1", status="ingested"))
    db.upsert_raw(RawNoteRecord(path="raw/b.md", content_hash="h2", status="ingested"))
    db.upsert_concepts("raw/a.md", ["Alpha"])
    db.upsert_concepts("raw/b.md", ["Beta"])

    # Write source files
    (config.vault / "raw" / "a.md").write_text("---\ntitle: A\n---\nContent about Alpha.")
    (config.vault / "raw" / "b.md").write_text("---\ntitle: B\n---\nContent about Beta.")

    mock_response = json.dumps({"title": "Alpha", "content": "Alpha content.", "tags": []})
    client = make_mock_client(mock_response)

    drafts, failed, _ = compile_concepts(config, client, db, concepts=["Alpha"])
    assert len(drafts) == 1
    assert "Alpha" in drafts[0].name or "alpha" in drafts[0].name.lower()
    # Beta should still be needing compile
    needing = db.concepts_needing_compile()
    assert "Beta" in needing


def test_compile_concepts_none_compiles_all(config, db):
    """concepts=None falls back to all needing compile."""
    import json

    db.upsert_raw(RawNoteRecord(path="raw/a.md", content_hash="h1", status="ingested"))
    db.upsert_concepts("raw/a.md", ["Topic"])
    (config.vault / "raw" / "a.md").write_text("---\ntitle: A\n---\nContent.")

    mock_response = json.dumps({"title": "Topic", "content": "Content.", "tags": []})
    client = make_mock_client(mock_response)

    drafts, failed, _ = compile_concepts(config, client, db, concepts=None)
    assert len(drafts) == 1


def test_compile_concepts_preserves_canonical_title(config, db):
    import json

    db.upsert_raw(RawNoteRecord(path="raw/a.md", content_hash="h1", status="ingested"))
    db.upsert_concepts("raw/a.md", ["Product Backlog"])
    db.upsert_aliases("Product Backlog", ["Продуктовый бэклог"])
    (config.vault / "raw" / "a.md").write_text("---\ntitle: A\n---\nContent.")

    mock_response = json.dumps({"title": "Продуктовый бэклог", "content": "Content.", "tags": []})
    client = make_mock_client(mock_response)

    drafts, failed, _ = compile_concepts(config, client, db, concepts=["Product Backlog"])

    assert failed == []
    assert len(drafts) == 1
    assert drafts[0].name == "Product Backlog.md"
    assert db.get_article("wiki/.drafts/Product Backlog.md").title == "Product Backlog"


def test_compile_concepts_legend_only_citations_do_not_link_inline(config, db):
    import json

    config.pipeline.inline_source_citations = True
    config.pipeline.source_citation_style = "legend-only"
    db.upsert_raw(RawNoteRecord(path="raw/a.md", content_hash="h1", status="ingested"))
    db.upsert_concepts("raw/a.md", ["Alpha"])
    (config.vault / "raw" / "a.md").write_text("---\ntitle: Alpha Source\n---\nContent.")

    mock_response = json.dumps({"title": "Alpha", "content": "Claim [S1].", "tags": []})
    client = make_mock_client(mock_response)

    drafts, failed, _ = compile_concepts(config, client, db, concepts=["Alpha"])

    assert failed == []
    body = drafts[0].read_text()
    assert "Claim [S1](#Sources)." in body
    assert "Claim ([[sources/Alpha Source|S1]])." not in body
    assert "- [S1] [[sources/Alpha Source|Alpha Source]]" in body


def test_compile_concepts_draft_media_reference_mode(config, db):
    import json

    config.pipeline.draft_media = "reference"
    db.upsert_raw(RawNoteRecord(path="raw/a.md", content_hash="h1", status="ingested"))
    db.upsert_concepts("raw/a.md", ["Alpha"])
    (config.vault / "raw" / "a.md").write_text("---\ntitle: Alpha Source\n---\nContent.")

    mock_response = json.dumps(
        {"title": "Alpha", "content": "Diagram ![[./_resources/file.pdf]].", "tags": []}
    )
    client = make_mock_client(mock_response)

    drafts, failed, _ = compile_concepts(config, client, db, concepts=["Alpha"])

    assert failed == []
    body = drafts[0].read_text()
    assert "![[./_resources/file.pdf]]" not in body
    assert "Media reference: ./_resources/file.pdf" in body


def test_compile_concepts_links_to_same_batch_concepts(config, db):
    import json

    db.upsert_raw(RawNoteRecord(path="raw/a.md", content_hash="h1", status="ingested"))
    db.upsert_raw(RawNoteRecord(path="raw/b.md", content_hash="h2", status="ingested"))
    db.upsert_concepts("raw/a.md", ["Alpha"])
    db.upsert_concepts("raw/b.md", ["Beta"])
    (config.vault / "raw" / "a.md").write_text("---\ntitle: A\n---\nAlpha mentions Beta.")
    (config.vault / "raw" / "b.md").write_text("---\ntitle: B\n---\nBeta.")

    client = make_mock_client()
    client.generate.side_effect = [
        json.dumps({"title": "Alpha", "content": "Alpha relates to Beta.", "tags": []}),
        json.dumps({"title": "Beta", "content": "Beta details.", "tags": []}),
    ]

    drafts, failed, _ = compile_concepts(config, client, db)

    assert failed == []
    alpha = next(path for path in drafts if path.name == "Alpha.md")
    assert "[[Beta]]" in alpha.read_text()


def test_compile_concepts_skips_pending_draft(config, db):
    import json

    import frontmatter as fm_lib

    from obsidian_llm_wiki.vault import atomic_write

    db.upsert_raw(RawNoteRecord(path="raw/a.md", content_hash="h1", status="ingested"))
    db.upsert_concepts("raw/a.md", ["Alpha"])
    (config.vault / "raw" / "a.md").write_text("---\ntitle: A\n---\nContent.")
    draft_path = config.drafts_dir / "Alpha.md"
    atomic_write(
        draft_path,
        fm_lib.dumps(
            fm_lib.Post("User-reviewed draft body.", title="Alpha", tags=[], status="draft")
        ),
    )

    mock_response = json.dumps({"title": "Alpha", "content": "New content.", "tags": []})
    client = make_mock_client(mock_response)

    drafts, failed, _ = compile_concepts(config, client, db, concepts=["Alpha"])

    assert drafts == []
    assert failed == []
    assert client.generate.call_count == 0
    assert "User-reviewed draft body." in draft_path.read_text()
    state = db.get_compile_state("Alpha", "raw/a.md")
    assert state is not None
    assert state["status"] == "deferred_draft"


def test_compile_concepts_marks_sources_after_each_success(config, db):
    import json

    db.upsert_raw(RawNoteRecord(path="raw/a.md", content_hash="h1", status="ingested"))
    db.upsert_raw(RawNoteRecord(path="raw/b.md", content_hash="h2", status="ingested"))
    db.upsert_concepts("raw/a.md", ["Alpha"])
    db.upsert_concepts("raw/b.md", ["Beta"])
    (config.vault / "raw" / "a.md").write_text("---\ntitle: A\n---\nContent.")
    (config.vault / "raw" / "b.md").write_text("---\ntitle: B\n---\nContent.")

    client = make_mock_client()
    client.generate.side_effect = [
        json.dumps({"title": "Alpha", "content": "Alpha content.", "tags": []}),
        "not valid json",
        "not valid json",
        "not valid json",
    ]

    drafts, failed, _ = compile_concepts(config, client, db)

    assert len(drafts) == 1
    assert failed == ["Beta"]
    assert db.get_raw("raw/a.md").status == "compiled"
    assert db.get_raw("raw/b.md").status == "ingested"


def test_compile_concepts_mixed_source_failure_keeps_only_failed_concept_queued(config, db):
    import json

    db.upsert_raw(RawNoteRecord(path="raw/a.md", content_hash="h1", status="ingested"))
    db.upsert_raw(RawNoteRecord(path="raw/b.md", content_hash="h2", status="ingested"))
    db.upsert_concepts("raw/a.md", ["Alpha", "Beta"])
    db.upsert_concepts("raw/b.md", ["Beta"])
    (config.vault / "raw" / "a.md").write_text("---\ntitle: A\n---\nContent A.")
    (config.vault / "raw" / "b.md").write_text("---\ntitle: B\n---\nContent B.")

    client = make_mock_client()
    client.generate.side_effect = [
        json.dumps({"title": "Alpha", "content": "Alpha content.", "tags": []}),
        "not valid json",
        "not valid json",
        "not valid json",
    ]

    drafts, failed, _ = compile_concepts(config, client, db, concepts=["Alpha", "Beta"])

    assert len(drafts) == 1
    assert failed == ["Beta"]
    assert db.get_compile_state("Alpha", "raw/a.md")["status"] == "compiled"
    assert db.get_compile_state("Beta", "raw/a.md")["status"] == "failed"
    assert db.get_compile_state("Beta", "raw/b.md")["status"] == "failed"
    assert db.get_raw("raw/a.md").status == "ingested"
    assert db.get_raw("raw/b.md").status == "ingested"
    assert db.concepts_needing_compile() == ["Beta"]


def test_compile_concepts_force_clears_manual_edit_defer(config, db):
    import json

    db.upsert_raw(RawNoteRecord(path="raw/a.md", content_hash="h1", status="ingested"))
    db.upsert_concepts("raw/a.md", ["Alpha"])
    (config.vault / "raw" / "a.md").write_text("---\ntitle: A\n---\nContent.")
    wiki_path = config.wiki_dir / "Alpha.md"
    wiki_path.write_text("---\ntitle: Alpha\n---\nEdited.")
    db.upsert_article(
        WikiArticleRecord(
            path=str(wiki_path.relative_to(config.vault)),
            title="Alpha",
            sources=["raw/a.md"],
            content_hash="old-hash",
            is_draft=False,
        )
    )

    client = make_mock_client(json.dumps({"title": "Alpha", "content": "New.", "tags": []}))

    drafts, failed, _ = compile_concepts(config, client, db)
    assert drafts == []
    assert failed == []
    assert db.get_compile_state("Alpha", "raw/a.md")["status"] == "deferred_manual_edit"

    drafts, failed, _ = compile_concepts(config, client, db, force=True, concepts=["Alpha"])
    assert len(drafts) == 1
    assert failed == []
    assert db.get_compile_state("Alpha", "raw/a.md")["status"] == "compiled"


def test_approve_and_reject_update_compile_state(config, db):
    import frontmatter as fm_lib

    from obsidian_llm_wiki.vault import atomic_write

    db.upsert_raw(RawNoteRecord(path="raw/a.md", content_hash="h1", status="ingested"))
    db.upsert_concepts("raw/a.md", ["Alpha"])
    draft_path = config.drafts_dir / "Alpha.md"
    meta = {"title": "Alpha", "status": "draft", "tags": [], "sources": ["raw/a.md"]}
    atomic_write(draft_path, fm_lib.dumps(fm_lib.Post("Body.", **meta)))
    db.upsert_article(
        WikiArticleRecord(
            path=str(draft_path.relative_to(config.vault)),
            title="Alpha",
            sources=["raw/a.md"],
            content_hash="h",
            is_draft=True,
        )
    )
    db.mark_concept_compile_state("Alpha", ["raw/a.md"], "deferred_draft")

    approve_drafts(config, db, [draft_path])
    assert db.get_compile_state("Alpha", "raw/a.md")["status"] == "compiled"

    draft_path = config.drafts_dir / "Alpha.md"
    atomic_write(draft_path, fm_lib.dumps(fm_lib.Post("Body.", **meta)))
    db.upsert_article(
        WikiArticleRecord(
            path=str(draft_path.relative_to(config.vault)),
            title="Alpha",
            sources=["raw/a.md"],
            content_hash="h2",
            is_draft=True,
        )
    )
    reject_draft(draft_path, config, db, feedback="Nope")
    assert db.get_compile_state("Alpha", "raw/a.md")["status"] == "pending"


# ── Approval records approved_at ─────────────────────────────────────────────


def test_approve_drafts_sets_approved_at(config, db):
    import frontmatter as fm_lib

    from obsidian_llm_wiki.vault import atomic_write

    draft_path = config.drafts_dir / "Article.md"
    meta = {"title": "Article", "status": "draft", "tags": [], "sources": [], "confidence": 0.7}
    atomic_write(draft_path, fm_lib.dumps(fm_lib.Post("## Body\n\nContent.", **meta)))
    db.upsert_article(
        WikiArticleRecord(
            path=str(draft_path.relative_to(config.vault)),
            title="Article",
            sources=[],
            content_hash="h",
            is_draft=True,
        )
    )

    published = approve_drafts(config, db, [draft_path])
    assert len(published) == 1
    art = db.get_article(str(published[0].relative_to(config.vault)))
    assert art is not None
    assert art.approved_at is not None


def test_approve_drafts_skips_paths_outside_drafts(config, db):
    outside = config.vault / "elsewhere.md"
    outside.write_text("---\ntitle: Elsewhere\nstatus: draft\n---\nBody.")

    published = approve_drafts(config, db, [outside])

    assert published == []
    assert outside.exists()


def test_approve_drafts_records_untracked_on_disk_draft(config, db):
    import frontmatter as fm_lib

    from obsidian_llm_wiki.vault import atomic_write

    draft_path = config.drafts_dir / "Untracked.md"
    meta = {"title": "Untracked", "status": "draft", "tags": [], "sources": ["raw/a.md"]}
    atomic_write(draft_path, fm_lib.dumps(fm_lib.Post("Body.", **meta)))

    published = approve_drafts(config, db, [draft_path])

    assert len(published) == 1
    art = db.get_article("wiki/Untracked.md")
    assert art is not None
    assert art.title == "Untracked"
    assert art.is_draft is False


# ── _gather_sources ───────────────────────────────────────────────────────────


def test_gather_sources_missing_file_skipped(vault):
    """Source path that doesn't exist → skipped, no crash."""
    text, resolved = _gather_sources(["raw/nonexistent.md"], vault)
    assert text == ""
    assert resolved == []


def test_gather_sources_unreadable_file_skipped(vault):
    """Source file that can't be parsed → skipped, no crash."""
    bad = vault / "raw" / "bad.md"
    bad.write_bytes(b"\xff\xfe")  # invalid UTF-8 triggers parse error
    text, resolved = _gather_sources(["raw/bad.md"], vault)
    # May or may not resolve depending on parse_note tolerance — just no exception
    assert isinstance(text, str)
    assert isinstance(resolved, list)


def test_gather_sources_combines_multiple(vault):
    (vault / "raw" / "a.md").write_text("---\ntitle: A\n---\nContent A.")
    (vault / "raw" / "b.md").write_text("---\ntitle: B\n---\nContent B.")
    text, resolved = _gather_sources(["raw/a.md", "raw/b.md"], vault)
    assert "Content A." in text
    assert "Content B." in text
    assert len(resolved) == 2


def test_gather_sources_bare_filename_resolved(vault):
    """Model sometimes returns bare filename without raw/ prefix."""
    (vault / "raw" / "note.md").write_text("---\ntitle: Note\n---\nBody.")
    text, resolved = _gather_sources(["note.md"], vault)
    assert "Body." in text
    assert len(resolved) == 1


# ── Stub compile path ─────────────────────────────────────────────────────────


def test_compile_concepts_stub_produces_draft(config, db):
    """Stub concept → fast model called → draft created → stub removed from DB."""
    import json

    db.add_stub("Orphan Topic")
    mock_response = json.dumps({"title": "Orphan Topic", "content": "Stub content.", "tags": []})
    client = make_mock_client(mock_response)

    drafts, failed, _ = compile_concepts(config, client, db)
    assert len(drafts) == 1
    assert not db.has_stub("Orphan Topic")
    assert "Orphan" in drafts[0].name or "orphan" in drafts[0].name.lower()


def test_compile_concepts_stub_failure_adds_to_failed(config, db):
    """StructuredOutputError during stub compile → concept added to failed."""

    db.add_stub("Bad Stub")
    client = make_mock_client("not valid json at all {{{")
    # structured_output will exhaust retries and raise StructuredOutputError
    # Mock client to always return garbage
    client.generate.return_value = "garbage"

    drafts, failed, _ = compile_concepts(config, client, db)
    assert "Bad Stub" in failed


def test_compile_concepts_stub_llm_bad_request_adds_to_failed(config, db):
    """LLMBadRequestError during stub compile → concept in failed, no crash."""
    from obsidian_llm_wiki.openai_compat_client import LLMBadRequestError

    db.add_stub("Context Overflow Stub")
    client = make_mock_client()
    client.generate.side_effect = LLMBadRequestError("HTTP 400: context too long")

    drafts, failed, _ = compile_concepts(config, client, db)
    assert "Context Overflow Stub" in failed
    assert drafts == []


def test_compile_concepts_article_llm_bad_request_adds_to_failed(config, db):
    """LLMBadRequestError during article compile → concept in failed, no crash."""
    from obsidian_llm_wiki.openai_compat_client import LLMBadRequestError

    db.upsert_raw(RawNoteRecord(path="raw/src.md", content_hash="h1", status="ingested"))
    db.upsert_concepts("raw/src.md", ["Heavy Concept"])
    (config.vault / "raw" / "src.md").write_text(
        "---\ntitle: Source\n---\nContent about Heavy Concept."
    )

    client = make_mock_client()
    client.generate.side_effect = LLMBadRequestError("HTTP 400: n_keep > ctx")

    drafts, failed, _ = compile_concepts(config, client, db)
    assert "Heavy Concept" in failed
    assert drafts == []

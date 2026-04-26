from __future__ import annotations

from click.testing import CliRunner

from obsidian_llm_wiki.pipeline.items import (
    extract_named_reference_items,
    extract_quoted_title_items,
    store_extracted_items,
)
from obsidian_llm_wiki.state import StateDB


def test_items_audit_cli_shows_candidates(vault, config, db):
    from obsidian_llm_wiki.cli import cli

    items = extract_named_reference_items(
        ["Example Reference"],
        "Notes about Example Reference",
        "The note mentions Example Reference explicitly.",
        "raw/talk.md",
        [],
    )
    store_extracted_items(db, "raw/talk.md", items)

    result = CliRunner().invoke(cli, ["items", "audit", "--vault", str(vault)])

    assert result.exit_code == 0
    assert "Example Reference" in result.output


def test_items_show_cli_shows_mentions(vault, config, db):
    from obsidian_llm_wiki.cli import cli

    items = extract_named_reference_items(
        ["Example Reference"],
        "Notes about Example Reference",
        "The note mentions Example Reference explicitly.",
        "raw/talk.md",
        [],
    )
    store_extracted_items(db, "raw/talk.md", items)

    result = CliRunner().invoke(cli, ["items", "show", "--vault", str(vault), "Example Reference"])

    assert result.exit_code == 0
    assert "title_supported" in result.output
    assert "raw/talk.md" in result.output


def test_extract_named_reference_items_keeps_evidenced_non_latin_reference():
    items = extract_named_reference_items(
        ["設計の思想"],
        "会議メモ",
        "このメモでは設計の思想について説明する。",
        "raw/design.md",
        [],
    )

    assert any(item.name == "設計の思想" and item.subtype == "named_reference" for item in items)
    assert items[0].evidence_level == "source_supported"


def test_extract_named_reference_items_rejects_unevidenced_reference():
    items = extract_named_reference_items(
        ["Missing Reference"],
        "Notes",
        "The body does not contain the proposed item.",
        "raw/notes.md",
        [],
    )

    assert items == []


def test_extract_named_reference_items_rejects_matching_concept():
    items = extract_named_reference_items(
        ["Example Concept"],
        "Example Concept notes",
        "Example Concept is the main topic.",
        "raw/concept.md",
        ["Example Concept"],
    )

    assert items == []


def test_extract_named_reference_items_rejects_noise():
    items = extract_named_reference_items(
        ["unknown_filename.pdf", "https://example.com", "image.png", "!!!"],
        "unknown_filename.pdf",
        "https://example.com image.png !!!",
        "raw/unknown_filename.pdf.md",
        [],
    )

    assert items == []


def test_extract_named_reference_items_rejects_short_source_only_typos():
    items = extract_named_reference_items(
        ["гниги"],
        "Lecture Notes",
        "The OCR text contains гниги as a typo-like token.",
        "raw/lecture.md",
        [],
    )

    assert items == []


def test_extract_quoted_title_items_ignores_unknown_filename():
    items = extract_quoted_title_items("unknown_filename.pdf", "raw/unknown_filename.pdf.md")

    assert items == []


def test_extract_quoted_title_items_ignores_lowercase_quoted_fragments():
    items = extract_quoted_title_items(
        "The article says that the phrase «draft notes» was misquoted",
        "raw/quoted-fragment.md",
    )

    assert not any(item.name == "draft notes" for item in items)


def test_extract_quoted_title_items_keeps_separator_delimited_quoted_titles():
    items = extract_quoted_title_items(
        "Notes - «thinking in systems»",
        "raw/book.md",
    )

    assert any(
        item.name == "thinking in systems" and item.subtype == "quoted_title" for item in items
    )


def test_extract_quoted_title_items_keeps_whole_quoted_title():
    items = extract_quoted_title_items("«thinking in systems»", "raw/book.md")

    assert any(item.name == "thinking in systems" and item.confidence == 0.55 for item in items)


def test_extract_quoted_title_items_keeps_non_latin_quoted_titles():
    items = extract_quoted_title_items("Notes - 「設計の思想」", "raw/design.md")

    assert any(item.name == "設計の思想" and item.subtype == "quoted_title" for item in items)


def test_extract_quoted_title_items_rejects_context_only_quoted_phrases():
    items = extract_quoted_title_items(
        "Review of the book «thinking in systems»",
        "raw/book.md",
    )

    assert not any(item.name == "thinking in systems" for item in items)


def test_store_extracted_items_records_item_and_mention(tmp_path):
    db = StateDB(tmp_path / ".olw" / "state.db")
    items = extract_named_reference_items(
        ["Example Reference"],
        "Notes about Example Reference",
        "The note mentions Example Reference explicitly.",
        "raw/talk.md",
        [],
    )

    store_extracted_items(db, "raw/talk.md", items)

    item = db.get_item("Example Reference")
    assert item is not None
    assert item.kind == "ambiguous"
    assert item.subtype == "named_reference"
    mentions = db.get_item_mentions("Example Reference")
    assert len(mentions) == 1
    assert mentions[0].evidence_level == "title_supported"

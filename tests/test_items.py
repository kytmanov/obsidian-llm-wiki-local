from __future__ import annotations

from click.testing import CliRunner

from obsidian_llm_wiki.pipeline.items import extract_title_items, store_extracted_items
from obsidian_llm_wiki.state import StateDB


def test_items_audit_cli_shows_candidates(vault, config, db):
    from obsidian_llm_wiki.cli import cli

    items = extract_title_items(
        "A new look at an old country Mark Power at TEDxKrakow",
        "raw/talk.md",
    )
    store_extracted_items(db, "raw/talk.md", items)

    result = CliRunner().invoke(cli, ["items", "audit", "--vault", str(vault)])

    assert result.exit_code == 0
    assert "Mark Power" in result.output
    assert "TEDxKrakow" in result.output


def test_items_show_cli_shows_mentions(vault, config, db):
    from obsidian_llm_wiki.cli import cli

    items = extract_title_items(
        "A new look at an old country Mark Power at TEDxKrakow",
        "raw/talk.md",
    )
    store_extracted_items(db, "raw/talk.md", items)

    result = CliRunner().invoke(cli, ["items", "show", "--vault", str(vault), "Mark Power"])

    assert result.exit_code == 0
    assert "title_supported" in result.output
    assert "raw/talk.md" in result.output


def test_extract_title_items_person_and_event():
    items = extract_title_items(
        "A new look at an old country Mark Power at TEDxKrakow",
        "raw/A new look at an old country Mark Power at TEDxKrakow.md",
    )

    by_name = {item.name: item for item in items}
    assert by_name["Mark Power"].subtype == "person"
    assert by_name["TEDxKrakow"].subtype == "event_or_org"


def test_extract_title_items_product_model():
    items = extract_title_items("Mazda CX90 Fuse diagram", "raw/Mazda CX90 Fuse diagram.md")

    assert any(item.name == "Mazda CX90" and item.subtype == "product" for item in items)


def test_extract_title_items_ignores_unknown_filename():
    items = extract_title_items("unknown_filename.pdf", "raw/unknown_filename.pdf.md")

    assert items == []


def test_store_extracted_items_records_item_and_mention(tmp_path):
    db = StateDB(tmp_path / ".olw" / "state.db")
    items = extract_title_items(
        "A new look at an old country Mark Power at TEDxKrakow",
        "raw/talk.md",
    )

    store_extracted_items(db, "raw/talk.md", items)

    mark = db.get_item("Mark Power")
    assert mark is not None
    assert mark.kind == "ambiguous"
    assert mark.subtype == "person"
    mentions = db.get_item_mentions("Mark Power")
    assert len(mentions) == 1
    assert mentions[0].evidence_level == "title_supported"

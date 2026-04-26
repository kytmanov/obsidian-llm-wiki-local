from __future__ import annotations

from click.testing import CliRunner

from obsidian_llm_wiki.pipeline.items import extract_title_items, store_extracted_items
from obsidian_llm_wiki.state import StateDB


def test_items_audit_cli_shows_candidates(vault, config, db):
    from obsidian_llm_wiki.cli import cli

    items = extract_title_items(
        "Field report by Ada Lovelace at LocalTechConf",
        "raw/talk.md",
    )
    store_extracted_items(db, "raw/talk.md", items)

    result = CliRunner().invoke(cli, ["items", "audit", "--vault", str(vault)])

    assert result.exit_code == 0
    assert "Ada Lovelace" in result.output
    assert "LocalTechConf" in result.output


def test_items_show_cli_shows_mentions(vault, config, db):
    from obsidian_llm_wiki.cli import cli

    items = extract_title_items(
        "Field report by Ada Lovelace at LocalTechConf",
        "raw/talk.md",
    )
    store_extracted_items(db, "raw/talk.md", items)

    result = CliRunner().invoke(cli, ["items", "show", "--vault", str(vault), "Ada Lovelace"])

    assert result.exit_code == 0
    assert "title_supported" in result.output
    assert "raw/talk.md" in result.output


def test_extract_title_items_person_and_event():
    items = extract_title_items(
        "Field report by Ada Lovelace at LocalTechConf",
        "raw/Field report by Ada Lovelace at LocalTechConf.md",
    )

    by_name = {item.name: item for item in items}
    assert by_name["Ada Lovelace"].subtype == "person"
    assert by_name["LocalTechConf"].subtype == "event_or_org"


def test_extract_title_items_product_model():
    items = extract_title_items("ExampleCam X200 wiring notes", "raw/ExampleCam X200.md")

    assert any(item.name == "ExampleCam X200" and item.subtype == "product" for item in items)


def test_extract_title_items_ignores_unknown_filename():
    items = extract_title_items("unknown_filename.pdf", "raw/unknown_filename.pdf.md")

    assert items == []


def test_extract_title_items_ignores_lowercase_quoted_fragments():
    items = extract_title_items(
        "The article says that the phrase «draft notes» was misquoted",
        "raw/quoted-fragment.md",
    )

    assert not any(item.name == "draft notes" for item in items)


def test_extract_title_items_keeps_separator_delimited_quoted_titles():
    items = extract_title_items(
        "Notes - «thinking in systems»",
        "raw/book.md",
    )

    assert any(
        item.name == "thinking in systems" and item.subtype == "quoted_title" for item in items
    )


def test_extract_title_items_keeps_whole_quoted_title():
    items = extract_title_items("«thinking in systems»", "raw/book.md")

    assert any(item.name == "thinking in systems" and item.confidence == 0.55 for item in items)


def test_extract_title_items_keeps_non_latin_quoted_titles():
    items = extract_title_items("Notes - 「設計の思想」", "raw/design.md")

    assert any(item.name == "設計の思想" and item.subtype == "quoted_title" for item in items)


def test_extract_title_items_rejects_context_only_quoted_phrases():
    items = extract_title_items(
        "Review of the book «thinking in systems»",
        "raw/book.md",
    )

    assert not any(item.name == "thinking in systems" for item in items)


def test_store_extracted_items_records_item_and_mention(tmp_path):
    db = StateDB(tmp_path / ".olw" / "state.db")
    items = extract_title_items(
        "Field report by Ada Lovelace at LocalTechConf",
        "raw/talk.md",
    )

    store_extracted_items(db, "raw/talk.md", items)

    person = db.get_item("Ada Lovelace")
    assert person is not None
    assert person.kind == "ambiguous"
    assert person.subtype == "person"
    mentions = db.get_item_mentions("Ada Lovelace")
    assert len(mentions) == 1
    assert mentions[0].evidence_level == "title_supported"

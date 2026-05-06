"""Tests for the Reader protocol and skeleton implementations."""

from __future__ import annotations

from pathlib import Path

import pytest

from obsidian_llm_wiki.readers import (
    ArticleFilter,
    ArticleRef,
    PackIndex,
    PackManifest,
    PackReader,
    Reader,
    VaultReader,
)


def test_pack_reader_init(tmp_path: Path) -> None:
    reader = PackReader(tmp_path)
    assert reader.pack_root == tmp_path


def test_vault_reader_init(tmp_path: Path) -> None:
    reader = VaultReader(tmp_path)
    assert reader.vault_root == tmp_path


def test_pack_reader_methods_raise_not_implemented(tmp_path: Path) -> None:
    reader = PackReader(tmp_path)
    methods = [
        lambda: reader.manifest,
        lambda: reader.index,
        lambda: reader.capabilities,
        lambda: reader.list_articles(),
        lambda: reader.read_article("x"),
        lambda: reader.find_concept("x"),
        lambda: reader.list_terms(),
        lambda: reader.find_term("x"),
        lambda: reader.get_provenance("x"),
        lambda: reader.list_sources(),
        lambda: reader.list_segments(),
        lambda: reader.has_capability("x"),
    ]
    for call in methods:
        with pytest.raises(NotImplementedError):
            call()


def test_vault_reader_methods_raise_not_implemented(tmp_path: Path) -> None:
    reader = VaultReader(tmp_path)
    methods = [
        lambda: reader.manifest,
        lambda: reader.index,
        lambda: reader.capabilities,
        lambda: reader.list_articles(),
        lambda: reader.read_article("x"),
        lambda: reader.find_concept("x"),
        lambda: reader.list_terms(),
        lambda: reader.find_term("x"),
        lambda: reader.get_provenance("x"),
        lambda: reader.list_sources(),
        lambda: reader.list_segments(),
        lambda: reader.has_capability("x"),
    ]
    for call in methods:
        with pytest.raises(NotImplementedError):
            call()


def test_article_ref_is_immutable() -> None:
    ref = ArticleRef(id="01HXX", name="Vector Clocks", path="articles/Vector-Clocks.md")
    with pytest.raises((AttributeError, TypeError)):
        ref.name = "Different"  # type: ignore[misc]


def test_article_filter_optional_fields() -> None:
    article_filter = ArticleFilter()
    assert article_filter.tag is None
    assert article_filter.min_confidence is None
    article_filter = ArticleFilter(tag="distributed-systems", min_confidence="high")
    assert article_filter.tag == "distributed-systems"


def test_pack_manifest_construction() -> None:
    manifest = PackManifest(
        schema_version=1,
        pack_id="ostep",
        version="1.0.0",
        capabilities=frozenset({"articles", "concepts"}),
    )
    assert "articles" in manifest.capabilities
    assert manifest.redistribution == "unknown"


def test_pack_index_construction() -> None:
    index = PackIndex(schema_version=1, articles=())
    assert index.articles == ()


def test_protocol_recognized_structurally() -> None:
    """Reader is a Protocol; PackReader and VaultReader satisfy it structurally."""

    def takes_reader(_reader: Reader) -> None:
        return None

    takes_reader(PackReader(Path(".")))
    takes_reader(VaultReader(Path(".")))
    assert hasattr(PackReader, "list_articles")
    assert hasattr(VaultReader, "list_articles")

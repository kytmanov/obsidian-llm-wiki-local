"""Tests for engine interface skeletons."""

from __future__ import annotations

from pathlib import Path

import pytest

from obsidian_llm_wiki.engines import (
    Answer,
    Citation,
    Hit,
    QueryConfig,
    SearchConfig,
    _BaseMultiPackQueryEngine,
    _BaseQueryEngine,
    _BaseSearchEngine,
)
from obsidian_llm_wiki.readers import VaultReader


def test_query_config_defaults() -> None:
    cfg = QueryConfig()
    assert cfg.max_pages == 5
    assert cfg.graph_expand_hops == 0


def test_search_config_defaults() -> None:
    cfg = SearchConfig()
    assert cfg.max_hits == 20
    assert cfg.expand_query_with_fast_model is False


def test_query_engine_skeleton_raises(tmp_path: Path) -> None:
    reader = VaultReader(tmp_path)
    engine = _BaseQueryEngine(reader)
    with pytest.raises(NotImplementedError):
        engine.query("anything")


def test_search_engine_skeleton_raises(tmp_path: Path) -> None:
    reader = VaultReader(tmp_path)
    engine = _BaseSearchEngine(reader)
    with pytest.raises(NotImplementedError):
        engine.search("anything")


def test_multi_pack_query_engine_skeleton_raises() -> None:
    engine = _BaseMultiPackQueryEngine()
    with pytest.raises(NotImplementedError):
        engine.query("anything")


def test_answer_construction() -> None:
    answer = Answer(text="hello", citations=(Citation(article_id="01HXX"),))
    assert answer.text == "hello"
    assert answer.citations[0].article_id == "01HXX"


def test_hit_construction() -> None:
    hit = Hit(article_id="01HXX", name="Vector Clocks", snippet="...", score=0.85)
    assert hit.score == 0.85

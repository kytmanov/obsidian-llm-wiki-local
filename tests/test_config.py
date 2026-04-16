"""Tests for config.py — PipelineConfig and default_wiki_toml."""

from __future__ import annotations

import tomllib

from obsidian_llm_wiki.config import PipelineConfig, default_wiki_toml


def test_pipeline_config_language_default_none():
    cfg = PipelineConfig()
    assert cfg.language is None


def test_pipeline_config_language_from_dict():
    cfg = PipelineConfig(**{"language": "fr"})
    assert cfg.language == "fr"


def test_pipeline_config_language_from_toml(tmp_path):
    toml_content = default_wiki_toml()
    # default_wiki_toml has language commented out — parse should give None
    data = tomllib.loads(toml_content)
    pipeline_data = data.get("pipeline", {})
    cfg = PipelineConfig(**pipeline_data)
    assert cfg.language is None


def test_default_wiki_toml_contains_language_comment():
    toml = default_wiki_toml()
    assert "language" in toml
    assert "ISO 639-1" in toml


def test_pipeline_config_accepts_explicit_language():
    cfg = PipelineConfig(language="de")
    assert cfg.language == "de"

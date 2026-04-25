"""Tests for config.py — PipelineConfig and default_wiki_toml."""

from __future__ import annotations

import tomllib
from pathlib import Path

from obsidian_llm_wiki.config import Config, PipelineConfig, default_wiki_toml


def test_pipeline_config_language_default_none():
    cfg = PipelineConfig()
    assert cfg.language is None


def test_pipeline_config_inline_source_citations_default_false():
    cfg = PipelineConfig()
    assert cfg.inline_source_citations is False


def test_pipeline_config_graph_defaults():
    cfg = PipelineConfig()
    assert cfg.source_citation_style == "legend-only"
    assert cfg.draft_media == "reference"
    assert cfg.graph_quality_checks is True


def test_pipeline_config_inline_source_citations_from_dict():
    cfg = PipelineConfig(**{"inline_source_citations": True})
    assert cfg.inline_source_citations is True


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


def test_default_wiki_toml_contains_inline_source_citations_comment():
    toml = default_wiki_toml()
    assert "inline_source_citations" in toml
    assert "Experimental" in toml


def test_default_wiki_toml_contains_graph_quality_defaults():
    toml = default_wiki_toml()
    assert "source_citation_style" in toml
    assert "draft_media" in toml
    assert "graph_quality_checks = true" in toml


def test_default_wiki_toml_can_enable_inline_source_citations():
    toml = default_wiki_toml(inline_source_citations=True)
    data = tomllib.loads(toml)
    assert data["pipeline"]["inline_source_citations"] is True


def test_from_vault_loads_inline_source_citations(tmp_path):
    _write_wiki_toml(
        tmp_path,
        """
[models]
fast = "gemma4:e4b"
heavy = "gemma4:e4b"

[pipeline]
inline_source_citations = true
""",
    )
    cfg = Config.from_vault(tmp_path)
    assert cfg.pipeline.inline_source_citations is True


def test_from_vault_loads_graph_quality_config(tmp_path):
    _write_wiki_toml(
        tmp_path,
        """
[models]
fast = "gemma4:e4b"
heavy = "gemma4:e4b"

[pipeline]
source_citation_style = "inline-wikilink"
draft_media = "embed"
graph_quality_checks = false
""",
    )
    cfg = Config.from_vault(tmp_path)
    assert cfg.pipeline.source_citation_style == "inline-wikilink"
    assert cfg.pipeline.draft_media == "embed"
    assert cfg.pipeline.graph_quality_checks is False


def test_pipeline_config_accepts_explicit_language():
    cfg = PipelineConfig(language="de")
    assert cfg.language == "de"


# ── from_vault deep-merge ──────────────────────────────────────────────────────


def _write_wiki_toml(path: Path, body: str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "wiki.toml").write_text(body)


def test_from_vault_partial_models_override_preserves_siblings(tmp_path):
    """Override `models.fast` alone must NOT wipe out `models.heavy`."""
    _write_wiki_toml(
        tmp_path,
        """
[models]
fast = "orig-fast"
heavy = "orig-heavy"

[ollama]
url = "http://localhost:11434"
""",
    )
    cfg = Config.from_vault(tmp_path, models={"fast": "override-fast"})
    assert cfg.models.fast == "override-fast"
    assert cfg.models.heavy == "orig-heavy"


def test_from_vault_provider_override_adds_new_section(tmp_path):
    """Passing provider={'name':'groq','url':...} should inject provider section."""
    _write_wiki_toml(
        tmp_path,
        """
[models]
fast = "gemma4:e4b"
heavy = "gemma4:e4b"

[ollama]
url = "http://localhost:11434"
""",
    )
    cfg = Config.from_vault(
        tmp_path,
        provider={"name": "groq", "url": "https://api.groq.com/openai/v1"},
    )
    assert cfg.provider is not None
    assert cfg.provider.name == "groq"
    assert cfg.provider.url == "https://api.groq.com/openai/v1"
    eff = cfg.effective_provider
    assert eff.name == "groq"


def test_from_vault_provider_partial_override_preserves_file_keys(tmp_path):
    """Partial provider override merges with file's [provider] block."""
    _write_wiki_toml(
        tmp_path,
        """
[models]
fast = "gemma4:e4b"
heavy = "gemma4:e4b"

[provider]
name = "groq"
url = "https://api.groq.com/openai/v1"
timeout = 300
fast_ctx = 8192
heavy_ctx = 32768
""",
    )
    cfg = Config.from_vault(tmp_path, provider={"url": "https://alt.example/v1"})
    assert cfg.provider.name == "groq"  # preserved from file
    assert cfg.provider.url == "https://alt.example/v1"  # overridden
    assert cfg.provider.fast_ctx == 8192  # preserved


def test_from_vault_none_override_ignored(tmp_path):
    _write_wiki_toml(
        tmp_path,
        """
[models]
fast = "orig"
heavy = "orig"
""",
    )
    cfg = Config.from_vault(tmp_path, models=None)
    assert cfg.models.fast == "orig"

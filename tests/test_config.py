"""Tests for config — TOML loading, defaults, and derived paths."""

from __future__ import annotations

from pathlib import Path

from obsidian_llm_wiki.config import Config, ModelsConfig, OllamaConfig, PipelineConfig, RagConfig

# ── Default values ────────────────────────────────────────────────────────────


def test_default_models():
    m = ModelsConfig()
    assert m.fast == "gemma4:e4b"
    assert m.heavy == "qwen2.5:14b"
    assert m.embed == "nomic-embed-text"


def test_default_ollama():
    o = OllamaConfig()
    assert o.url == "http://localhost:11434"
    assert o.timeout == 600.0
    assert o.fast_ctx == 8192
    assert o.heavy_ctx == 16384


def test_default_pipeline():
    p = PipelineConfig()
    assert p.auto_approve is False
    assert p.auto_commit is True
    assert p.watch_debounce == 3.0
    assert p.max_concepts_per_source == 8


def test_default_rag():
    r = RagConfig()
    assert r.chunk_size == 512
    assert r.chunk_overlap == 50
    assert r.similarity_threshold == 0.7


# ── Config construction ──────────────────────────────────────────────────────


def test_config_with_vault_path(tmp_path: Path):
    c = Config(vault=tmp_path)
    assert c.vault == tmp_path.resolve()


def test_config_resolves_relative_vault():
    c = Config(vault=Path("."))
    assert c.vault.is_absolute()


# ── Derived paths ─────────────────────────────────────────────────────────────


def test_derived_paths(tmp_path: Path):
    c = Config(vault=tmp_path)
    assert c.raw_dir == tmp_path / "raw"
    assert c.wiki_dir == tmp_path / "wiki"
    assert c.drafts_dir == tmp_path / "wiki" / ".drafts"
    assert c.olw_dir == tmp_path / ".olw"
    assert c.state_db_path == tmp_path / ".olw" / "state.db"
    assert c.chroma_dir == tmp_path / ".olw" / "chroma"
    assert c.sources_dir == tmp_path / "wiki" / "sources"
    assert c.queries_dir == tmp_path / "wiki" / "queries"
    assert c.schema_path == tmp_path / "vault-schema.md"


# ── from_vault ────────────────────────────────────────────────────────────────


def test_from_vault_no_toml(tmp_path: Path):
    """Config.from_vault works even without wiki.toml — uses defaults."""
    c = Config.from_vault(tmp_path)
    assert c.vault == tmp_path.resolve()
    assert c.models.fast == "gemma4:e4b"


def test_from_vault_with_toml(tmp_path: Path):
    toml_content = """\
[models]
fast = "llama3.2:3b"
heavy = "llama3.1:8b"

[ollama]
url = "http://192.168.1.100:11434"
timeout = 120

[pipeline]
auto_approve = true
auto_commit = false
"""
    (tmp_path / "wiki.toml").write_text(toml_content)
    c = Config.from_vault(tmp_path)
    assert c.models.fast == "llama3.2:3b"
    assert c.models.heavy == "llama3.1:8b"
    assert c.ollama.url == "http://192.168.1.100:11434"
    assert c.ollama.timeout == 120
    assert c.pipeline.auto_approve is True
    assert c.pipeline.auto_commit is False


def test_from_vault_with_overrides(tmp_path: Path):
    toml_content = """\
[models]
fast = "llama3.2:3b"
"""
    (tmp_path / "wiki.toml").write_text(toml_content)
    c = Config.from_vault(tmp_path, models={"fast": "override:1b", "heavy": "override:7b"})
    assert c.models.fast == "override:1b"
    assert c.models.heavy == "override:7b"


def test_from_vault_partial_toml(tmp_path: Path):
    """Only some sections specified — rest use defaults."""
    toml_content = """\
[models]
fast = "tiny:1b"
"""
    (tmp_path / "wiki.toml").write_text(toml_content)
    c = Config.from_vault(tmp_path)
    assert c.models.fast == "tiny:1b"
    assert c.models.heavy == "qwen2.5:14b"  # default
    assert c.ollama.url == "http://localhost:11434"  # default
    assert c.pipeline.auto_approve is False  # default


def test_from_vault_with_rag_config(tmp_path: Path):
    toml_content = """\
[rag]
chunk_size = 1024
chunk_overlap = 100
similarity_threshold = 0.5
"""
    (tmp_path / "wiki.toml").write_text(toml_content)
    c = Config.from_vault(tmp_path)
    assert c.rag.chunk_size == 1024
    assert c.rag.chunk_overlap == 100
    assert c.rag.similarity_threshold == 0.5

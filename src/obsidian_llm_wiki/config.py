from __future__ import annotations

import tomllib
from pathlib import Path

from pydantic import BaseModel, field_validator


def _toml_quote(value: str) -> str:
    """Return a safely quoted TOML basic string, escaping backslashes, quotes, and control chars."""
    escaped = (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )
    return f'"{escaped}"'


def default_wiki_toml(
    fast_model: str = "gemma4:e4b",
    heavy_model: str = "qwen2.5:14b",
    ollama_url: str = "http://localhost:11434",
) -> str:
    """Generate wiki.toml content, optionally pre-filled from global config."""
    return (
        f"[models]\n"
        f"fast = {_toml_quote(fast_model)}\n"
        f"heavy = {_toml_quote(heavy_model)}\n"
        f"# Optional: set heavy = fast to use a single model for everything\n\n"
        f"[ollama]\n"
        f"url = {_toml_quote(ollama_url)}\n"
        f"timeout = 600\n"
        f"fast_ctx = 16384                  # context window for fast model (tokens)\n"
        f"heavy_ctx = 32768                 # context window for heavy model (tokens)\n\n"
        f"[pipeline]\n"
        f"auto_approve = false\n"
        f"auto_commit = true\n"
        f"auto_maintain = false\n"
        f"watch_debounce = 3.0\n"
        f"max_concepts_per_source = 8\n"
        f"ingest_parallel = false   # true = parallel chunks (needs OLLAMA_NUM_PARALLEL>=4)\n"
    )


class ModelsConfig(BaseModel):
    fast: str = "gemma4:e4b"
    heavy: str = "qwen2.5:14b"
    embed: str = "nomic-embed-text"  # used only when RAG optional dependency is installed


class OllamaConfig(BaseModel):
    url: str = "http://localhost:11434"
    timeout: float = 600.0  # seconds; 14B models over network need >5min
    fast_ctx: int = 16384
    heavy_ctx: int = 32768


class PipelineConfig(BaseModel):
    auto_approve: bool = False
    auto_commit: bool = True
    watch_debounce: float = 3.0
    max_concepts_per_source: int = 8
    auto_maintain: bool = False
    ingest_parallel: bool = False  # parallel chunk analysis (needs OLLAMA_NUM_PARALLEL≥4)


class RagConfig(BaseModel):
    chunk_size: int = 512
    chunk_overlap: int = 50
    similarity_threshold: float = 0.7


class Config(BaseModel):
    vault: Path
    models: ModelsConfig = ModelsConfig()
    ollama: OllamaConfig = OllamaConfig()
    pipeline: PipelineConfig = PipelineConfig()
    rag: RagConfig = RagConfig()

    @field_validator("vault", mode="before")
    @classmethod
    def resolve_vault(cls, v: str | Path) -> Path:
        return Path(v).expanduser().resolve()

    @property
    def raw_dir(self) -> Path:
        return self.vault / "raw"

    @property
    def wiki_dir(self) -> Path:
        return self.vault / "wiki"

    @property
    def drafts_dir(self) -> Path:
        return self.vault / "wiki" / ".drafts"

    @property
    def olw_dir(self) -> Path:
        return self.vault / ".olw"

    @property
    def state_db_path(self) -> Path:
        return self.olw_dir / "state.db"

    @property
    def chroma_dir(self) -> Path:
        return self.olw_dir / "chroma"

    @property
    def sources_dir(self) -> Path:
        return self.vault / "wiki" / "sources"

    @property
    def queries_dir(self) -> Path:
        return self.vault / "wiki" / "queries"

    @property
    def schema_path(self) -> Path:
        return self.vault / "vault-schema.md"

    @classmethod
    def from_vault(cls, vault_path: Path, **overrides) -> Config:
        vault = Path(vault_path).expanduser().resolve()
        config_file = vault / "wiki.toml"
        file_config: dict = {}
        if config_file.exists():
            with open(config_file, "rb") as f:
                file_config = tomllib.load(f)
        # Merge overrides into nested dicts
        for key, val in overrides.items():
            if val is not None:
                file_config[key] = val
        return cls(vault=vault, **file_config)

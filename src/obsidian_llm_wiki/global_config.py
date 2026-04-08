"""
Global user-level config for olw, stored at:
  - Mac/Linux: ~/.config/olw/config.toml  (or $XDG_CONFIG_HOME/olw/config.toml)
  - Windows:   %%APPDATA%%\\olw\\config.toml

This is separate from the per-vault wiki.toml. It stores user preferences
(default vault, Ollama URL, model choices) so commands work without --vault.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path

from pydantic import BaseModel, ConfigDict


class GlobalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    vault: str | None = None
    ollama_url: str | None = None
    fast_model: str | None = None
    heavy_model: str | None = None


def _global_config_path() -> Path:
    if os.name == "nt":
        appdata = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
        return Path(appdata) / "olw" / "config.toml"
    xdg = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(xdg) / "olw" / "config.toml"


def load_global_config() -> GlobalConfig | None:
    """Load global config. Returns None if missing or malformed — never raises."""
    path = _global_config_path()
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)
        return GlobalConfig(**data)
    except Exception:
        return None


def save_global_config(cfg: GlobalConfig) -> None:
    """Write global config to disk. Creates parent directory if needed."""
    path = _global_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    if cfg.vault is not None:
        lines.append(f"vault = {_toml_str(cfg.vault)}")
    if cfg.ollama_url is not None:
        lines.append(f"ollama_url = {_toml_str(cfg.ollama_url)}")
    if cfg.fast_model is not None:
        lines.append(f"fast_model = {_toml_str(cfg.fast_model)}")
    if cfg.heavy_model is not None:
        lines.append(f"heavy_model = {_toml_str(cfg.heavy_model)}")
    path.write_text("\n".join(lines) + "\n" if lines else "", encoding="utf-8")


def _toml_str(value: str) -> str:
    """Minimal safe TOML string quoting — escapes backslashes, double quotes, and control chars."""
    escaped = (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )
    return f'"{escaped}"'

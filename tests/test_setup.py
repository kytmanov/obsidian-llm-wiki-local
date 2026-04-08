"""
Tests for global_config module and olw setup command.
All tests are offline — no Ollama instance required.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from obsidian_llm_wiki.cli import cli
from obsidian_llm_wiki.global_config import (
    GlobalConfig,
    _global_config_path,
    _toml_str,
    load_global_config,
    save_global_config,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def cfg_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect global config to a temp dir."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    # On Windows, patch APPDATA too
    monkeypatch.setenv("APPDATA", str(tmp_path))
    return tmp_path


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


# ── _toml_str ─────────────────────────────────────────────────────────────────


def test_toml_str_simple():
    assert _toml_str("hello") == '"hello"'


def test_toml_str_escapes_backslashes():
    assert _toml_str("C:\\Users\\alex") == '"C:\\\\Users\\\\alex"'


def test_toml_str_escapes_quotes():
    assert _toml_str('say "hi"') == '"say \\"hi\\""'


def test_toml_str_combined():
    result = _toml_str('C:\\path\\to\\my "wiki"')
    assert result == '"C:\\\\path\\\\to\\\\my \\"wiki\\""'


def test_toml_str_escapes_control_chars():
    assert _toml_str("line1\nline2") == '"line1\\nline2"'
    assert _toml_str("col1\tcol2") == '"col1\\tcol2"'
    assert _toml_str("cr\rhere") == '"cr\\rhere"'


def test_toml_str_control_chars_produce_valid_toml(cfg_dir: Path):
    """Paths with control chars must still round-trip through save/load."""
    cfg = GlobalConfig(fast_model="model\twith\ttabs")
    save_global_config(cfg)
    loaded = load_global_config()
    assert loaded is not None
    assert loaded.fast_model == "model\twith\ttabs"


# ── save / load round-trip ────────────────────────────────────────────────────


def test_save_load_full_config(cfg_dir: Path):
    cfg = GlobalConfig(
        vault="/tmp/my-wiki",
        ollama_url="http://localhost:11434",
        fast_model="gemma4:e4b",
        heavy_model="qwen2.5:14b",
    )
    save_global_config(cfg)
    loaded = load_global_config()
    assert loaded == cfg


def test_save_creates_parent_dirs(cfg_dir: Path):
    cfg = GlobalConfig(fast_model="gemma4:e4b")
    save_global_config(cfg)
    path = _global_config_path()
    assert path.exists()
    assert path.parent.is_dir()


def test_saved_file_is_valid_toml(cfg_dir: Path):
    cfg = GlobalConfig(
        vault="/tmp/wiki",
        fast_model="gemma4:e4b",
        heavy_model="qwen2.5:14b",
        ollama_url="http://localhost:11434",
    )
    save_global_config(cfg)
    path = _global_config_path()
    with open(path, "rb") as f:
        data = tomllib.load(f)
    assert data["fast_model"] == "gemma4:e4b"
    assert data["heavy_model"] == "qwen2.5:14b"


def test_partial_config_no_null_keys(cfg_dir: Path):
    """None fields must not appear in the written TOML file."""
    cfg = GlobalConfig(fast_model="gemma4:e4b")
    save_global_config(cfg)
    path = _global_config_path()
    raw = path.read_text()
    assert "vault" not in raw
    assert "ollama_url" not in raw
    assert "heavy_model" not in raw
    assert "fast_model" in raw


def test_empty_config_writes_empty_file(cfg_dir: Path):
    save_global_config(GlobalConfig())
    path = _global_config_path()
    assert path.read_text() == ""


# ── load error handling ───────────────────────────────────────────────────────


def test_load_missing_file_returns_none(cfg_dir: Path):
    result = load_global_config()
    assert result is None


def test_load_malformed_toml_returns_none(cfg_dir: Path):
    path = _global_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("this is not [ valid toml !!!!", encoding="utf-8")
    result = load_global_config()
    assert result is None


def test_load_unknown_fields_returns_none(cfg_dir: Path):
    """Extra keys not in GlobalConfig schema should cause load to return None."""
    path = _global_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('unknown_key = "value"\n', encoding="utf-8")
    result = load_global_config()
    assert result is None


# ── _load_config fallback ─────────────────────────────────────────────────────


def test_load_config_uses_global_vault(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cfg_dir: Path
):
    """_load_config(None) should fall back to global config vault."""
    vault = tmp_path / "wiki"
    vault.mkdir()
    (vault / "wiki.toml").write_text(
        '[models]\nfast = "gemma4:e4b"\nheavy = "qwen2.5:14b"\n[ollama]\nurl = "http://localhost:11434"\n'
    )

    save_global_config(GlobalConfig(vault=str(vault)))

    # Import here to use real _load_config
    from obsidian_llm_wiki.cli import _load_config

    config = _load_config(None)
    assert config.vault == vault


def test_load_config_no_vault_exits(monkeypatch: pytest.MonkeyPatch, cfg_dir: Path):
    """_load_config(None) with no global config should sys.exit(1)."""
    from obsidian_llm_wiki.cli import _load_config

    monkeypatch.delenv("OLW_VAULT", raising=False)
    with pytest.raises(SystemExit) as exc:
        _load_config(None)
    assert exc.value.code == 1


# ── olw setup --non-interactive ───────────────────────────────────────────────


def test_setup_non_interactive_no_config(runner: CliRunner, cfg_dir: Path):
    result = runner.invoke(cli, ["setup", "--non-interactive"])
    assert result.exit_code == 0
    assert "No global config" in result.output


def test_setup_non_interactive_with_config(runner: CliRunner, cfg_dir: Path):
    save_global_config(
        GlobalConfig(
            fast_model="gemma4:e4b",
            heavy_model="qwen2.5:14b",
            ollama_url="http://192.168.1.10:11434",
        )
    )
    result = runner.invoke(cli, ["setup", "--non-interactive"])
    assert result.exit_code == 0
    assert "gemma4:e4b" in result.output
    assert "qwen2.5:14b" in result.output
    assert "192.168.1.10" in result.output


# ── olw setup --reset ─────────────────────────────────────────────────────────


def test_setup_reset_clears_config(runner: CliRunner, cfg_dir: Path):
    save_global_config(GlobalConfig(fast_model="old-model"))

    # Provide all wizard inputs via stdin so it runs non-interactively in tests
    # Inputs: URL (default), fast model (default), heavy model (default), vault (skip)
    result = runner.invoke(
        cli,
        ["setup", "--reset"],
        input="\n\n\n\n",
        catch_exceptions=False,
    )
    assert result.exit_code == 0
    loaded = load_global_config()
    # After reset + wizard with defaults, old-model should be gone
    assert loaded is not None
    assert loaded.fast_model != "old-model"


# ── olw setup wizard (stdin input) ───────────────────────────────────────────


def test_setup_wizard_saves_config(runner: CliRunner, cfg_dir: Path):
    """Wizard with all-default inputs should create a valid config."""
    with patch("obsidian_llm_wiki.ollama_client.OllamaClient") as MockClient:
        instance = MagicMock()
        instance.healthcheck.return_value = False
        instance.list_models_detailed.return_value = []
        MockClient.return_value = instance

        result = runner.invoke(
            cli,
            ["setup"],
            input="\ngemma4:e4b\nqwen2.5:14b\n\n",  # URL default, fast, heavy, no vault
            catch_exceptions=False,
        )

    assert result.exit_code == 0
    cfg = load_global_config()
    assert cfg is not None
    assert cfg.fast_model == "gemma4:e4b"
    assert cfg.heavy_model == "qwen2.5:14b"


def test_setup_wizard_model_number_selection(runner: CliRunner, cfg_dir: Path):
    """Selecting model by number from the list should resolve to model name."""
    with patch("obsidian_llm_wiki.ollama_client.OllamaClient") as MockClient:
        instance = MagicMock()
        instance.healthcheck.return_value = True
        instance.list_models_detailed.return_value = [
            {"name": "gemma4:e4b", "size_gb": "4.3 GB"},
            {"name": "qwen2.5:14b", "size_gb": "8.7 GB"},
        ]
        MockClient.return_value = instance

        result = runner.invoke(
            cli,
            ["setup"],
            input="\n1\n2\n\n",  # URL default, pick #1 fast, pick #2 heavy, no vault
            catch_exceptions=False,
        )

    assert result.exit_code == 0
    cfg = load_global_config()
    assert cfg is not None
    assert cfg.fast_model == "gemma4:e4b"
    assert cfg.heavy_model == "qwen2.5:14b"


def test_setup_wizard_whitespace_input_uses_default(runner: CliRunner, cfg_dir: Path):
    """Spaces-only model input should fall back to default, not save a blank model name."""
    with patch("obsidian_llm_wiki.ollama_client.OllamaClient") as MockClient:
        instance = MagicMock()
        instance.healthcheck.return_value = False
        instance.list_models_detailed.return_value = []
        MockClient.return_value = instance

        # Send spaces for fast and heavy model prompts (no table, free-text path)
        result = runner.invoke(
            cli,
            ["setup"],
            input="\n   \n   \n\n",  # URL default, spaces for fast, spaces for heavy, no vault
            catch_exceptions=False,
        )

    assert result.exit_code == 0
    cfg = load_global_config()
    assert cfg is not None
    # Should have fallen back to defaults, not saved empty/whitespace strings
    assert cfg.fast_model and cfg.fast_model.strip() != ""
    assert cfg.heavy_model and cfg.heavy_model.strip() != ""


def test_setup_wizard_with_vault(runner: CliRunner, cfg_dir: Path, tmp_path: Path):
    """Setting a vault path in wizard should save it to global config."""
    vault = tmp_path / "my-wiki"

    with patch("obsidian_llm_wiki.ollama_client.OllamaClient") as MockClient:
        instance = MagicMock()
        instance.healthcheck.return_value = False
        instance.list_models_detailed.return_value = []
        MockClient.return_value = instance

        result = runner.invoke(
            cli,
            ["setup"],
            input=f"\ngemma4:e4b\nqwen2.5:14b\n{vault}\n",
            catch_exceptions=False,
        )

    assert result.exit_code == 0
    cfg = load_global_config()
    assert cfg is not None
    assert cfg.vault == str(vault.resolve())


# ── olw init uses global config models ───────────────────────────────────────


def test_init_uses_global_config_models(runner: CliRunner, cfg_dir: Path, tmp_path: Path):
    """olw init should pre-fill wiki.toml from global config models."""
    save_global_config(
        GlobalConfig(
            fast_model="llama3.2:3b",
            heavy_model="llama3.1:8b",
            ollama_url="http://192.168.1.5:11434",
        )
    )
    vault = tmp_path / "test-vault"
    result = runner.invoke(cli, ["init", str(vault)])
    assert result.exit_code == 0

    toml_path = vault / "wiki.toml"
    assert toml_path.exists()
    content = toml_path.read_text()
    assert "llama3.2:3b" in content
    assert "llama3.1:8b" in content
    assert "192.168.1.5" in content


def test_init_defaults_without_global_config(runner: CliRunner, cfg_dir: Path, tmp_path: Path):
    """olw init without global config should use built-in defaults."""
    vault = tmp_path / "test-vault"
    result = runner.invoke(cli, ["init", str(vault)])
    assert result.exit_code == 0

    content = (vault / "wiki.toml").read_text()
    assert "gemma4:e4b" in content
    assert "qwen2.5:14b" in content


def test_init_syncs_models_into_existing_wiki_toml(
    runner: CliRunner, cfg_dir: Path, tmp_path: Path
):
    """olw init on an existing vault should patch models from global config."""
    vault = tmp_path / "existing-vault"
    vault.mkdir()
    # Simulate old wiki.toml with stale heavy model
    old_toml = (
        '[models]\nfast = "gemma4:e4b"\nheavy = "qwen2.5:14b"\n\n'
        '[ollama]\nurl = "http://localhost:11434"\ntimeout = 600\n\n'
        "[pipeline]\nauto_approve = false\nauto_commit = true\n"
    )
    (vault / "wiki.toml").write_text(old_toml)

    save_global_config(
        GlobalConfig(
            fast_model="gemma4:e4b",
            heavy_model="gemma4:e4b",
            ollama_url="http://localhost:11434",
        )
    )
    result = runner.invoke(cli, ["init", str(vault)])
    assert result.exit_code == 0

    content = (vault / "wiki.toml").read_text()
    # heavy should now be gemma4:e4b (patched from global config)
    assert 'heavy = "gemma4:e4b"' in content
    # pipeline settings must be preserved
    assert "auto_approve = false" in content

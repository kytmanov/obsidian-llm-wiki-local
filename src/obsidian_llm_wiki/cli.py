"""
obsidian-llm-wiki CLI (olw)

Commands:
  init     — create vault structure (or adopt existing)
  ingest   — analyze raw notes
  compile  — synthesize notes into wiki articles (writes to .drafts/)
  approve  — publish drafts to wiki/
  reject   — discard a draft
  status   — show vault health and pending drafts
  undo     — revert last N [olw] git commits
  query    — RAG-powered Q&A (Phase 2)
  lint     — check wiki health (Phase 2)
  watch    — file watcher daemon (Phase 3)
"""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.prompt import Prompt
from rich.table import Table

console = Console()
err_console = Console(stderr=True, style="bold red")

_EXPERIMENTAL_CITATIONS_COPY = (
    "Inline source citations link generated claims back to source pages. "
    "Experimental: small models may omit citations or add noisy markers. "
    "Default: off. Turn off later with `olw config inline-source-citations off --vault <path>`."
)


def _format_optional_bool(value: bool | None) -> str:
    if value is None:
        return "[dim]not set[/dim]"
    return "on" if value else "off"


class InlineSourceCitationsConfigError(Exception):
    """Raised when inline citation config cannot be read safely."""


def _read_inline_source_citations_setting(toml_path: Path, *, strict: bool = False) -> bool | None:

    if not toml_path.exists():
        return None
    try:
        with open(toml_path, "rb") as f:
            data = tomllib.load(f)
    except tomllib.TOMLDecodeError as exc:
        if strict:
            raise InlineSourceCitationsConfigError(f"Invalid TOML in {toml_path}: {exc}") from exc
        return None
    except OSError as exc:
        if strict:
            raise InlineSourceCitationsConfigError(f"Could not read {toml_path}: {exc}") from exc
        return None
    pipeline = data.get("pipeline", {})
    value = pipeline.get("inline_source_citations") if isinstance(pipeline, dict) else None
    if value is not None and not isinstance(value, bool):
        if strict:
            raise InlineSourceCitationsConfigError(
                "Invalid pipeline.inline_source_citations in "
                f"{toml_path}: expected boolean true/false, got {type(value).__name__}"
            )
        return None
    return value if isinstance(value, bool) else None


def _set_inline_source_citations(toml_path: Path, enabled: bool) -> None:
    """Patch one pipeline key while preserving unrelated wiki.toml content."""
    from .vault import atomic_write

    if not toml_path.exists():
        raise FileNotFoundError(toml_path)

    text = toml_path.read_text(encoding="utf-8")
    line = f"inline_source_citations = {'true' if enabled else 'false'}"
    section_match = re.search(r"(?m)^\[pipeline\]\s*$", text)

    if section_match is None:
        separator = "" if text.endswith("\n") or not text else "\n"
        atomic_write(toml_path, f"{text}{separator}\n[pipeline]\n{line}\n")
        return

    section_start = section_match.end()
    next_section = re.search(r"(?m)^\[[^\]]+\]\s*$", text[section_start:])
    section_end = section_start + next_section.start() if next_section else len(text)
    section = text[section_start:section_end]
    key_re = re.compile(r"(?m)^(\s*)#?\s*inline_source_citations\s*=.*$")

    if key_re.search(section):
        new_section = key_re.sub(rf"\1{line}", section, count=1)
    else:
        insertion = ("" if section.endswith("\n") or not section else "\n") + line + "\n"
        new_section = section + insertion

    atomic_write(toml_path, text[:section_start] + new_section + text[section_end:])


# ── Context helpers ───────────────────────────────────────────────────────────


def _resolve_vault_path(vault_str: str | None) -> Path:
    from .global_config import load_global_config

    if vault_str is None:
        gcfg = load_global_config()
        vault_str = gcfg.vault if gcfg and gcfg.vault else None

    if not vault_str:
        cwd = Path.cwd()
        for parent in [cwd, *cwd.parents]:
            if (parent / "wiki.toml").exists():
                vault_str = str(parent)
                break

    if not vault_str:
        click.echo(
            "Error: no vault specified. Use --vault, set OLW_VAULT, run `olw setup`, "
            "or cd into a vault directory.",
            err=True,
        )
        sys.exit(1)
    vault_path = Path(vault_str).expanduser().resolve()
    if not vault_path.exists():
        click.echo(
            f"Error: vault path does not exist: {vault_path}\n"
            f"Run `olw init {vault_path}` to create it, or re-run `olw setup` "
            f"to update the default vault.",
            err=True,
        )
        sys.exit(1)
    if not vault_path.is_dir():
        click.echo(
            f"Error: vault path is not a directory: {vault_path}\n"
            "A vault is a directory containing wiki.toml. "
            "Point --vault / OLW_VAULT at the parent directory instead.",
            err=True,
        )
        sys.exit(1)
    return vault_path


def _load_config(vault_str: str | None, **kwargs):
    from .config import Config

    return Config.from_vault(_resolve_vault_path(vault_str), **kwargs)


def _load_db(config):
    from .state import StateDB

    return StateDB(config.state_db_path)


def _model_override_options(f):
    """Shared decorator adding --fast-model/--heavy-model/--provider/--provider-url."""
    f = click.option(
        "--fast-model",
        "fast_model",
        default=None,
        help="Override fast model for this invocation",
    )(f)
    f = click.option(
        "--heavy-model",
        "heavy_model",
        default=None,
        help="Override heavy model for this invocation",
    )(f)
    f = click.option(
        "--provider",
        "provider_name",
        default=None,
        help="Override provider name (ollama, groq, openai, azure, ...)",
    )(f)
    f = click.option(
        "--provider-url",
        "provider_url",
        default=None,
        help="Override provider base URL (e.g. https://api.groq.com/openai/v1)",
    )(f)
    return f


def _model_override_kwargs(
    fast_model: str | None,
    heavy_model: str | None,
    provider_name: str | None,
    provider_url: str | None,
) -> dict:
    """Pack CLI model-override flags into kwargs for Config.from_vault."""
    kwargs: dict = {}
    models: dict = {}
    if fast_model:
        models["fast"] = fast_model
    if heavy_model:
        models["heavy"] = heavy_model
    if models:
        kwargs["models"] = models
    provider: dict = {}
    if provider_name:
        provider["name"] = provider_name
    if provider_url:
        provider["url"] = provider_url
    if provider:
        kwargs["provider"] = provider
    return kwargs


def _resolve_draft_arg(config, raw_path: str | Path) -> Path:
    """Resolve a CLI draft argument relative to wiki/.drafts/ when appropriate."""
    path = Path(raw_path).expanduser()
    candidates: list[Path]
    if path.is_absolute():
        candidates = [path]
    else:
        candidates = [config.drafts_dir / path, config.vault / path, path]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def _load_deps(config):
    from .client_factory import LLMError, build_client

    client = build_client(config)
    try:
        client.require_healthy()
    except LLMError as e:
        err_console.print(str(e))
        sys.exit(1)
    db = _load_db(config)
    return client, db


# ── CLI root ──────────────────────────────────────────────────────────────────


@click.group()
@click.version_option(package_name="obsidian-llm-wiki")
def cli():
    """obsidian-llm-wiki (olw) — 100% local Obsidian → wiki pipeline.

    Run `olw setup` for interactive configuration.
    """
    import logging

    from rich.logging import RichHandler

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(console=console, show_path=False, show_time=False)],
    )
    # Silence noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


# ── init ──────────────────────────────────────────────────────────────────────


@cli.command()
@click.argument("vault_path", type=click.Path())
@click.option("--existing", is_flag=True, help="Adopt an existing Obsidian vault")
@click.option("--non-interactive", is_flag=True)
def init(vault_path: str, existing: bool, non_interactive: bool):
    """Create vault structure and initialise olw."""
    from .git_ops import git_init

    vault = Path(vault_path).expanduser().resolve()
    vault.mkdir(parents=True, exist_ok=True)

    if existing:
        _init_existing(vault, non_interactive)
    else:
        _init_fresh(vault)

    # Write or sync wiki.toml from global config
    toml_path = vault / "wiki.toml"
    from .config import default_wiki_toml
    from .global_config import load_global_config

    gcfg = load_global_config()
    provider_name = gcfg.provider_name if gcfg and gcfg.provider_name else "ollama"
    # Only fall back to Ollama-specific model names when using Ollama; cloud providers
    # must have been configured explicitly via `olw setup`.
    _ollama = provider_name == "ollama"
    fast = gcfg.fast_model if gcfg and gcfg.fast_model else ("gemma4:e4b" if _ollama else "")
    heavy = gcfg.heavy_model if gcfg and gcfg.heavy_model else ("qwen2.5:14b" if _ollama else "")
    provider_url = gcfg.provider_url if gcfg and gcfg.provider_url else None
    ollama_url = gcfg.ollama_url if gcfg and gcfg.ollama_url else "http://localhost:11434"
    effective_url = provider_url or ollama_url
    azure_api_version = gcfg.azure_api_version if gcfg and gcfg.azure_api_version else None

    if not toml_path.exists():
        from .providers import get_provider

        prov_info = get_provider(provider_name)
        timeout = prov_info.default_timeout if prov_info else 600.0
        toml_path.write_text(
            default_wiki_toml(
                fast,
                heavy,
                ollama_url=ollama_url,
                provider_name=provider_name,
                provider_url=effective_url if provider_name != "ollama" else None,
                provider_timeout=timeout,
                azure_api_version=azure_api_version,
                inline_source_citations=(
                    bool(gcfg.experimental_inline_source_citations) if gcfg else False
                ),
            )
        )
    else:
        # Existing vault: patch model/URL fields from global config so that
        # olw setup changes are reflected without overwriting pipeline settings.
        _sync_wiki_toml_models(
            toml_path,
            fast,
            heavy,
            effective_url,
            provider_name=provider_name if provider_name != "ollama" else None,
        )

    # Init git
    git_init(vault)

    # Create .gitignore
    gi = vault / ".gitignore"
    if not gi.exists():
        gi.write_text(
            ".DS_Store\n.olw/chroma/\n.olw/state.db\n.olw/compare/\n.obsidian/workspace.json\n*.log\n"
        )

    console.print(f"[green]Vault initialised:[/green] {vault}")
    console.print("Next steps:")
    console.print("  1. Drop .md notes into [bold]raw/[/bold]")
    console.print("  2. Run [bold]olw run[/bold]  (ingest + compile + lint in one step)")
    console.print("  3. Review drafts: [bold]olw review[/bold]")


def _sync_wiki_toml_models(
    toml_path: Path,
    fast: str,
    heavy: str,
    ollama_url: str,
    provider_name: str | None = None,
) -> None:
    """Patch fast/heavy model, URL, and optionally provider name in an existing wiki.toml.

    Preserves all other settings (pipeline, rag, etc.) so user customisations
    are not lost. Only updates fields that come from global config.

    URL is only updated within the [ollama] or [provider] section, never globally,
    so switching providers cannot overwrite unrelated url= fields.
    """
    import re

    text = toml_path.read_text(encoding="utf-8")
    original = text

    def _replace_value(t: str, key: str, value: str) -> str:
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return re.sub(
            rf'^({re.escape(key)}\s*=\s*)".+"',
            rf'\g<1>"{escaped}"',
            t,
            flags=re.MULTILINE,
        )

    def _replace_in_section(t: str, section: str, key: str, value: str) -> str:
        """Replace key=value only within the named TOML section."""
        escaped_val = value.replace("\\", "\\\\").replace('"', '\\"')
        # Match the section header then capture everything until the next section or EOF
        pattern = rf'(\[{re.escape(section)}\][^\[]*?)^({re.escape(key)}\s*=\s*)".+"'
        replacement = rf'\1\2"{escaped_val}"'
        return re.sub(pattern, replacement, t, flags=re.MULTILINE | re.DOTALL)

    text = _replace_value(text, "fast", fast)
    text = _replace_value(text, "heavy", heavy)
    # Update URL only in [ollama] or [provider] sections to avoid clobbering other urls
    for section in ("ollama", "provider"):
        text = _replace_in_section(text, section, "url", ollama_url)
    if provider_name is not None:
        if "[provider]" not in text:
            console.print(
                f"  [yellow]Warning:[/yellow] wiki.toml has no [provider] section — "
                f"provider '{provider_name}' not applied. "
                f"Delete wiki.toml and re-run [bold]olw init[/bold] to regenerate it."
            )
        else:
            text = _replace_in_section(text, "provider", "name", provider_name)

    if text != original:
        toml_path.write_text(text, encoding="utf-8")
        console.print(f"[dim]wiki.toml updated: fast={fast}, heavy={heavy}, url={ollama_url}[/dim]")


def _init_fresh(vault: Path) -> None:
    for d in ["raw", "wiki", "wiki/.drafts", "wiki/sources", ".olw", ".olw/chroma"]:
        (vault / d).mkdir(parents=True, exist_ok=True)
    _write_vault_schema(vault)
    _write_index(vault)
    console.print("[dim]Created fresh vault structure[/dim]")


def _init_existing(vault: Path, non_interactive: bool) -> None:
    note_count = sum(1 for _ in vault.rglob("*.md"))
    console.print(f"Found [bold]{note_count}[/bold] existing .md files in {vault}")

    for d in ["raw", "wiki", "wiki/.drafts", "wiki/sources", ".olw", ".olw/chroma"]:
        (vault / d).mkdir(parents=True, exist_ok=True)

    if not non_interactive and note_count > 0:
        if click.confirm(f"Treat existing notes as raw source material? ({note_count} files)"):
            console.print("[dim]Existing notes will be ingested as raw material.[/dim]")
            console.print("[dim]Run [bold]olw ingest --all[/bold] to process them.[/dim]")

    _write_vault_schema(vault)
    _write_index(vault)
    _cleanup_legacy_index(vault)


def _cleanup_legacy_index(vault: Path) -> None:
    """Remove wiki/INDEX.md if it's the bootstrap stub and is distinct from wiki/index.md."""
    old = vault / "wiki" / "INDEX.md"
    new = vault / "wiki" / "index.md"
    if not old.exists():
        return
    # On case-insensitive FS old and new are the same file — don't delete
    if new.exists():
        try:
            if old.samefile(new):
                return
        except OSError:
            return
    try:
        content = old.read_text(encoding="utf-8")
        if content == _INDEX_STUB:
            old.unlink()
    except Exception:
        pass


def _write_vault_schema(vault: Path) -> None:
    schema_path = vault / "vault-schema.md"
    if not schema_path.exists():
        schema_path.write_text(
            "# Vault Schema\n\n"
            "## Folder Structure\n"
            "- `raw/` — input notes (immutable, never edited by olw)\n"
            "- `wiki/` — AI-synthesised articles (managed by olw)\n"
            "- `wiki/.drafts/` — pending human review\n\n"
            "## Note Format\n"
            "Every wiki note has YAML frontmatter with: title, tags, sources, "
            "confidence, status, created, updated.\n\n"
            "## Links\n"
            "Use `[[Article Title]]` wikilinks between notes.\n"
        )


_INDEX_STUB = (
    "---\ntitle: Index\ntags: [index]\nstatus: published\n---\n\n"
    "# Wiki Index\n\n_Updated automatically by olw._\n"
)


def _write_index(vault: Path) -> None:
    index = vault / "wiki" / "index.md"
    if not index.exists():
        index.parent.mkdir(parents=True, exist_ok=True)
        index.write_text(_INDEX_STUB)


# ── setup ─────────────────────────────────────────────────────────────────────


def _pick_model(
    console: Console,
    client,
    step_label: str,
    description: str,
    default_fallback: str,
    connected: bool,
) -> str:
    """Interactive model selector — shows table if models available, else free-text."""
    console.print()
    console.print(f"  [bold]{step_label}[/bold]  {description}")

    models: list[dict] = []
    if connected:
        models = client.list_models_detailed()

    if models:
        table = Table(show_header=True, box=None, padding=(0, 2))
        table.add_column("#", style="dim", width=3)
        table.add_column("Model")
        table.add_column("Size", style="dim")
        for i, m in enumerate(models, 1):
            table.add_row(str(i), m["name"], m["size_gb"])
        console.print(table)
        console.print()
        raw = Prompt.ask("    Select (number or name)", default="1", console=console).strip()
        if not raw:
            return default_fallback
        if raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(models):
                return models[idx]["name"]
            console.print(f"    [yellow]Invalid number, using {default_fallback}[/yellow]")
            return default_fallback
        return raw
    else:
        if connected:
            console.print(
                "    [yellow]No models found.[/yellow] "
                "Pull one first: [bold]ollama pull gemma4:e4b[/bold]"
            )
        console.print("    (e.g. gemma4:e4b, llama3.2:3b, qwen2.5:14b)")
        raw = Prompt.ask("    Model name", default=default_fallback, console=console).strip()
        return raw if raw else default_fallback


@cli.command()
@click.option("--non-interactive", is_flag=True, help="Print current config and exit")
@click.option("--reset", is_flag=True, help="Clear saved config and re-run wizard")
@click.option(
    "--provider",
    "provider_preset",
    default=None,
    help="Skip provider selection (e.g. groq, lm_studio)",
)
def setup(non_interactive: bool, reset: bool, provider_preset: str | None):
    """Interactive wizard: configure provider, models, and default vault."""
    from .global_config import GlobalConfig, load_global_config, save_global_config
    from .providers import PROVIDER_REGISTRY, get_provider, list_all_providers

    # ── non-interactive: show current config ──────────────────────────────────
    if non_interactive:
        gcfg = load_global_config()
        if not gcfg:
            console.print(
                "[dim]No global config found. Run [bold]olw setup[/bold] to configure.[/dim]"
            )
            return
        table = Table(title="Global config", show_header=False, box=None, padding=(0, 2))
        table.add_column("Key", style="bold")
        table.add_column("Value")
        prov_display = gcfg.provider_name or (gcfg.ollama_url and "ollama") or "[dim]not set[/dim]"
        table.add_row("Provider", prov_display)
        table.add_row("URL", gcfg.provider_url or gcfg.ollama_url or "[dim]not set[/dim]")
        table.add_row("API key", "***" if gcfg.api_key else "[dim]not set[/dim]")
        table.add_row("Fast model", gcfg.fast_model or "[dim]not set[/dim]")
        table.add_row("Heavy model", gcfg.heavy_model or "[dim]not set[/dim]")
        table.add_row("Default vault", gcfg.vault or "[dim]not set[/dim]")
        table.add_row(
            "Inline source citations for new vaults",
            _format_optional_bool(gcfg.experimental_inline_source_citations),
        )
        console.print(table)
        return

    # ── reset: wipe config before wizard ─────────────────────────────────────
    if reset:
        save_global_config(GlobalConfig())
        console.print("[dim]Config cleared.[/dim]")

    try:
        # ── Header ───────────────────────────────────────────────────────────
        console.print()
        from importlib.metadata import version as _pkg_version

        try:
            _ver = _pkg_version("obsidian-llm-wiki")
        except Exception:
            _ver = "unknown"
        console.print(
            Panel(
                f"[bold]obsidian-llm-wiki[/bold] v{_ver}  ·  setup",
                expand=False,
                border_style="blue",
                padding=(0, 4),
            )
        )
        console.print()

        all_providers = list_all_providers()

        # ── Step 1 — Provider selection ───────────────────────────────────────
        if provider_preset:
            chosen_prov = get_provider(provider_preset)
            if chosen_prov is None:
                console.print(
                    f"    [yellow]Unknown provider '{provider_preset}', using Ollama.[/yellow]"
                )
                chosen_prov = PROVIDER_REGISTRY["ollama"]
            chosen_name = chosen_prov.name
        else:
            console.print("  [bold]Step 1[/bold]  Provider\n")

            # Build numbered list
            local_provs = [p for p in all_providers if p.is_local]
            cloud_provs = [p for p in all_providers if not p.is_local and p.name != "custom"]

            console.print("    [bold]Local[/bold] (no API key needed):")
            idx_map: dict[int, str] = {}
            counter = 1
            for p in local_provs:
                marker = "  [default]" if p.name == "ollama" else ""
                console.print(f"      {counter:2}. {p.display_name:<14} {p.default_url}{marker}")
                idx_map[counter] = p.name
                counter += 1

            console.print()
            console.print("    [bold]Cloud[/bold] (API key required):")
            for p in cloud_provs:
                url_hint = p.default_url if p.default_url else "(enter URL manually)"
                console.print(f"      {counter:2}. {p.display_name:<14} {url_hint}")
                idx_map[counter] = p.name
                counter += 1

            console.print()
            console.print(f"      {counter:2}. Custom         (enter URL manually)")
            idx_map[counter] = "custom"

            console.print()
            raw = Prompt.ask(
                "    Select provider (number or name)", default="1", console=console
            ).strip()

            if raw.isdigit():
                num = int(raw)
                chosen_name = idx_map.get(num, "ollama")
            elif raw in PROVIDER_REGISTRY:
                chosen_name = raw
            else:
                console.print(f"    [yellow]Unknown '{raw}', defaulting to Ollama.[/yellow]")
                chosen_name = "ollama"

            chosen_prov = PROVIDER_REGISTRY[chosen_name]

        # ── Step 2 — URL ──────────────────────────────────────────────────────
        console.print()
        console.print("  [bold]Step 2[/bold]  URL")
        default_url = chosen_prov.default_url or ""
        if chosen_name == "azure":
            console.print(
                "    Azure format: https://{resource}.openai.azure.com/openai/deployments/{model}"
            )
        provider_url = Prompt.ask("    Base URL", default=default_url, console=console).strip()
        if not provider_url:
            provider_url = default_url
        if not provider_url and chosen_name in ("custom", "azure"):
            console.print(
                "    [red]URL is required for this provider. "
                "Run [bold]olw setup[/bold] again and enter a valid URL.[/red]"
            )
            sys.exit(1)

        # ── Step 3 — API key (cloud + custom providers) ───────────────────────
        import os

        needs_key_prompt = chosen_prov.requires_auth or chosen_name == "custom"
        api_key: str | None = None
        if needs_key_prompt:
            console.print()
            console.print("  [bold]Step 3[/bold]  API key")
            if chosen_prov.env_var:
                env_hint = f"  [dim](or set {chosen_prov.env_var} env var)[/dim]"
            elif chosen_name == "custom":
                env_hint = "  [dim](optional — press Enter to skip)[/dim]"
            else:
                env_hint = ""
            console.print(f"    API key{env_hint}")
            raw_key = Prompt.ask("    Key", default="", password=True, console=console).strip()
            api_key = raw_key if raw_key else None

        # ── Build a temp client to probe for model list ───────────────────────
        if chosen_name == "ollama":
            from .ollama_client import OllamaClient

            temp_client = OllamaClient(base_url=provider_url, timeout=5)
        else:
            from .openai_compat_client import OpenAICompatClient

            resolved_key = api_key
            if not resolved_key and chosen_prov.env_var:
                resolved_key = os.environ.get(chosen_prov.env_var)
            if not resolved_key:
                resolved_key = os.environ.get("OLW_API_KEY")
            temp_client = OpenAICompatClient(
                base_url=provider_url,
                provider_name=chosen_name,
                api_key=resolved_key,
                timeout=5,
                supports_json_mode=chosen_prov.supports_json_mode,
                supports_embeddings=chosen_prov.supports_embeddings,
                azure=chosen_prov.azure,
            )
        connected = temp_client.healthcheck()
        if connected:
            console.print("    [green]✓ connected[/green]")
        else:
            console.print(
                f"    [yellow]Warning:[/yellow] Cannot reach {provider_url} — continuing anyway."
            )

        # ── Default model names per provider ──────────────────────────────────
        # For non-Ollama providers, leave defaults empty — model names are
        # provider-specific and must be entered by the user.
        default_fast = "gemma4:e4b" if chosen_name == "ollama" else ""
        default_heavy = "qwen2.5:14b" if chosen_name == "ollama" else ""
        if chosen_name != "ollama" and not connected:
            console.print(
                "    [dim]Tip: enter the model name exactly as the provider lists it "
                "(e.g. llama-3.1-70b-versatile for Groq).[/dim]"
            )

        step_offset = 1 if needs_key_prompt else 0

        # ── Step 4 — Fast model ───────────────────────────────────────────────
        fast_model = _pick_model(
            console=console,
            client=temp_client,
            step_label=f"Step {3 + step_offset}",
            description="Fast model  [dim](analysis & routing · 3–8B recommended)[/dim]",
            default_fallback=default_fast,
            connected=connected,
        )

        # ── Step 5 — Heavy model ──────────────────────────────────────────────
        heavy_model = _pick_model(
            console=console,
            client=temp_client,
            step_label=f"Step {4 + step_offset}",
            description="Heavy model  [dim](article writing · 7–14B recommended)[/dim]",
            default_fallback=default_heavy,
            connected=connected,
        )

        temp_client.close()

        # ── Final step — Default vault ────────────────────────────────────────
        console.print()
        step_label = f"Step {5 + step_offset}"
        console.print(
            f"  [bold]{step_label}[/bold]  Default vault path  [dim](press Enter to skip)[/dim]"
        )
        vault_input = Prompt.ask("    Vault path", default="", console=console)
        vault_path: str | None = None
        if vault_input.strip():
            vault_path = str(Path(vault_input).expanduser().resolve())

        # ── Experimental features ─────────────────────────────────────────────
        console.print()
        step_label = f"Step {6 + step_offset}"
        console.print(f"  [bold]{step_label}[/bold]  Experimental features (optional)")
        console.print(f"    {_EXPERIMENTAL_CITATIONS_COPY}")
        raw_citations = (
            Prompt.ask(
                "    Enable inline source citations for new vaults?",
                choices=["y", "n"],
                default="n",
                show_choices=False,
                console=console,
            )
            .strip()
            .lower()
        )
        experimental_inline_source_citations = raw_citations == "y"

        applied_to_existing_vault = False
        current_vault_setting: bool | None = None
        current_toml_exists = False
        if vault_path:
            current_toml = Path(vault_path) / "wiki.toml"
            current_toml_exists = current_toml.exists()
            current_vault_setting = _read_inline_source_citations_setting(current_toml)
            if current_toml_exists:
                apply_now = (
                    Prompt.ask(
                        f"    Apply this setting to {current_toml} now?",
                        choices=["y", "n"],
                        default="n",
                        show_choices=False,
                        console=console,
                    )
                    .strip()
                    .lower()
                )
                if apply_now == "y":
                    _set_inline_source_citations(current_toml, experimental_inline_source_citations)
                    current_vault_setting = experimental_inline_source_citations
                    applied_to_existing_vault = True

        # ── Save ──────────────────────────────────────────────────────────────
        # Preserve existing azure_api_version so re-running setup doesn't reset it.
        existing_cfg = load_global_config()
        if chosen_name == "azure":
            azure_api_ver = (
                existing_cfg.azure_api_version
                if existing_cfg and existing_cfg.azure_api_version
                else "2024-02-15-preview"
            )
        else:
            azure_api_ver = None

        # Keep ollama_url for backward compat when Ollama is selected
        cfg = GlobalConfig(
            vault=vault_path,
            ollama_url=provider_url if chosen_name == "ollama" else None,
            fast_model=fast_model if fast_model else None,
            heavy_model=heavy_model if heavy_model else None,
            provider_name=chosen_name,
            provider_url=provider_url,
            api_key=api_key,
            azure_api_version=azure_api_ver,
            experimental_inline_source_citations=experimental_inline_source_citations,
        )
        save_global_config(cfg)

        # ── Summary panel ─────────────────────────────────────────────────────
        init_target = vault_path or "~/my-wiki"
        summary_lines = [
            "[green]✓[/green]  Setup complete\n",
            f"  Provider:     [bold]{chosen_prov.display_name}[/bold]",
            f"  URL:          {provider_url}",
        ]
        if api_key:
            summary_lines.append("  API key:      ***")
        if fast_model:
            summary_lines.append(f"  Fast model:   [bold]{fast_model}[/bold]")
        if heavy_model:
            summary_lines.append(f"  Heavy model:  [bold]{heavy_model}[/bold]")
        if vault_path:
            summary_lines.append(f"  Vault:        {vault_path}")
        summary_lines.append(
            "  Inline source citations: "
            f"{'on' if experimental_inline_source_citations else 'off'} for new vaults"
        )
        if vault_path:
            if current_toml_exists:
                current_display = (
                    _format_optional_bool(current_vault_setting)
                    if current_vault_setting is not None
                    else "[dim]not set (default: off)[/dim]"
                )
                suffix = " [dim](updated)[/dim]" if applied_to_existing_vault else ""
            else:
                current_display = (
                    f"[dim]not initialized yet; will be "
                    f"{'on' if experimental_inline_source_citations else 'off'} after init[/dim]"
                )
                suffix = ""
            summary_lines.append(f"  Current vault: {current_display}{suffix}")
        summary_lines += [
            "",
            "  Next steps:",
            f"    [bold]olw init {init_target}[/bold]",
            "    [bold]olw run[/bold]  (or: olw ingest --all && olw compile)",
        ]
        console.print()
        console.print(
            Panel("\n".join(summary_lines), border_style="green", expand=False, padding=(0, 2))
        )

    except (EOFError, KeyboardInterrupt):
        console.print("\n[yellow]Setup interrupted.[/yellow]")
        sys.exit(1)


# ── config ───────────────────────────────────────────────────────────────────


@cli.group(name="config")
def config_cmd():
    """Inspect or update vault-local configuration."""


@config_cmd.command(name="inline-source-citations")
@click.argument("action", type=click.Choice(["on", "off", "status"]))
@click.option("--vault", "vault_str", envvar="OLW_VAULT", default=None)
def config_inline_source_citations(action: str, vault_str: str | None):
    """Enable, disable, or inspect inline source citations for one vault."""
    vault_path = _resolve_vault_path(vault_str)
    toml_path = vault_path / "wiki.toml"
    if not toml_path.exists():
        click.echo(
            f"Error: {toml_path} not found. Run `olw init {vault_path}` first.",
            err=True,
        )
        sys.exit(1)

    if action == "status":
        try:
            setting = _read_inline_source_citations_setting(toml_path, strict=True)
        except InlineSourceCitationsConfigError as exc:
            click.echo(f"Error: {exc}", err=True)
            sys.exit(1)
        if setting is None:
            status = "not set (default: disabled)"
        else:
            status = "enabled" if setting else "disabled"
        console.print(f"inline_source_citations: {status} in {toml_path}")
        return

    enabled = action == "on"
    _set_inline_source_citations(toml_path, enabled)
    console.print(f"inline_source_citations = {'true' if enabled else 'false'} in {toml_path}")
    if enabled:
        console.print(
            "[dim]Turn off later with `olw config inline-source-citations off --vault "
            f"{vault_path}`.[/dim]"
        )


# ── ingest ────────────────────────────────────────────────────────────────────


@cli.command()
@click.option("--vault", "vault_str", envvar="OLW_VAULT", default=None)
@click.option("--all", "ingest_all", is_flag=True, help="Ingest all files in raw/")
@click.option("--force", is_flag=True, help="Re-ingest already-processed notes")
@click.argument("paths", nargs=-1, type=click.Path(exists=True))
@_model_override_options
def ingest(
    vault_str,
    ingest_all,
    force,
    paths,
    fast_model,
    heavy_model,
    provider_name,
    provider_url,
):
    """Analyze raw notes: extract concepts, quality, suggested topics."""

    overrides = _model_override_kwargs(fast_model, heavy_model, provider_name, provider_url)
    config = _load_config(vault_str, **overrides)
    client, db = _load_deps(config)

    if ingest_all:
        target_paths = [
            p
            for p in config.raw_dir.rglob("*.md")
            if "processed" not in p.parts and not p.name.startswith(".")
        ]
    elif paths:
        target_paths = [Path(p).resolve() for p in paths]
    else:
        click.echo("Specify --all or provide file paths.", err=True)
        sys.exit(1)

    if not target_paths:
        console.print("[yellow]No notes found in raw/[/yellow]")
        return

    skipped = ingested = failed = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Ingesting...", total=len(target_paths))

        for path in target_paths:
            progress.update(task, description=f"[dim]{path.name}[/dim]")
            from .pipeline.ingest import ingest_note as _ingest_note

            result = _ingest_note(
                path=path,
                config=config,
                client=client,
                db=db,
                force=force,
            )
            if result is None:
                # Distinguish skip vs failure by checking DB status
                rel = str(path.relative_to(config.vault))
                rec = db.get_raw(rel)
                if rec and rec.status == "failed":
                    failed += 1
                else:
                    skipped += 1
            else:
                ingested += 1
            progress.advance(task)

    console.print(
        f"[green]Done.[/green] Ingested: {ingested}  Skipped: {skipped}  Failed: {failed}"
    )

    # Update index and log
    from .indexer import append_log, generate_index

    generate_index(config, db)
    if ingested:
        append_log(config, f"ingest | {ingested} notes ingested")

    if ingested and config.pipeline.auto_commit:
        from .git_ops import git_commit

        git_commit(
            config.vault,
            f"ingest: {ingested} notes",
            paths=["raw/", "wiki/sources/", "wiki/index.md", "wiki/log.md", "vault-schema.md"],
        )
        console.print("[dim]Git commit created.[/dim]")


# ── compile ───────────────────────────────────────────────────────────────────


@cli.command()
@click.option("--vault", "vault_str", envvar="OLW_VAULT", default=None)
@click.option("--dry-run", is_flag=True, help="Show plan, write nothing")
@click.option("--auto-approve", is_flag=True, help="Publish immediately (skip draft review)")
@click.option("--force", is_flag=True, help="Recompile even manually-edited articles")
@click.option("--legacy", is_flag=True, help="Use legacy LLM-planning compile (CompilePlan)")
@click.option(
    "--concept",
    "concepts",
    multiple=True,
    help="Compile specific concept(s), even if not currently pending",
)
@click.option(
    "--retry-failed",
    "retry_failed",
    is_flag=True,
    help="Re-ingest failed raw notes and retry failed concept compiles",
)
@_model_override_options
def compile(
    vault_str,
    dry_run,
    auto_approve,
    force,
    legacy,
    concepts,
    retry_failed,
    fast_model,
    heavy_model,
    provider_name,
    provider_url,
):
    """Synthesize ingested notes into wiki article drafts."""
    from .git_ops import git_commit
    from .pipeline.compile import approve_drafts, compile_concepts, compile_notes

    overrides = _model_override_kwargs(fast_model, heavy_model, provider_name, provider_url)
    config = _load_config(vault_str, **overrides)
    client, db = _load_deps(config)

    explicit_concepts: list[str] | None = None
    if concepts:
        known_concepts = {name.casefold(): name for name in db.list_all_concept_names()}
        known_stubs = {name.casefold(): name for name in db.get_stubs()}
        resolved = []
        unresolved = []
        seen = set()
        for concept in concepts:
            canonical = db.resolve_alias(concept) or concept
            canonical_lookup = known_concepts.get(canonical.casefold()) or known_stubs.get(
                canonical.casefold()
            )
            if canonical_lookup is None:
                unresolved.append(concept)
                continue
            if canonical_lookup.casefold() not in seen:
                seen.add(canonical_lookup.casefold())
                resolved.append(canonical_lookup)
        for concept in unresolved:
            console.print(f"[yellow]Unknown concept, skipping:[/yellow] {concept}")
        if not resolved:
            console.print("[red]No valid concepts to compile.[/red]")
            sys.exit(1)
        explicit_concepts = resolved

    # Re-ingest previously failed notes before compiling
    if retry_failed:
        failed_recs = db.list_raw(status="failed")
        if not failed_recs:
            console.print("[dim]No failed notes to retry.[/dim]")
        else:
            console.print(f"[yellow]Retrying {len(failed_recs)} failed note(s)...[/yellow]")
            from .pipeline.ingest import ingest_note as _ingest_note

            retried = 0
            for rec in failed_recs:
                p = config.vault / rec.path
                if not p.exists():
                    console.print(f"  [red]Not found, skipping:[/red] {rec.path}")
                    continue
                db.mark_raw_status(rec.path, "new")
                result = _ingest_note(path=p, config=config, client=client, db=db, force=True)
                if result is not None:
                    retried += 1
            console.print(f"[green]Re-ingested {retried}/{len(failed_recs)} note(s).[/green]")

        failed_concepts = db.list_failed_concepts()
        if failed_concepts:
            console.print(f"[yellow]Retrying {len(failed_concepts)} failed concept(s)...[/yellow]")
            if explicit_concepts is None:
                explicit_concepts = failed_concepts
            else:
                for concept in failed_concepts:
                    if concept.casefold() not in {name.casefold() for name in explicit_concepts}:
                        explicit_concepts.append(concept)
        else:
            console.print("[dim]No failed concepts to retry.[/dim]")

    if dry_run:
        console.print("[dim]Dry run — no files will be written.[/dim]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        if legacy:
            task = progress.add_task("Planning compilation (legacy)...", total=None)
            draft_paths, failed = compile_notes(
                config=config,
                client=client,
                db=db,
                dry_run=dry_run,
            )
        else:
            task = progress.add_task("Compiling concepts...", total=1)

            def _on_progress(idx: int, total: int, name: str) -> None:
                progress.update(
                    task,
                    total=total,
                    completed=idx - 1,
                    description=f"[dim]{name}[/dim]",
                )

            draft_paths, failed, _ = compile_concepts(
                config=config,
                client=client,
                db=db,
                force=force,
                dry_run=dry_run,
                on_progress=_on_progress,
                concepts=explicit_concepts,
            )
            progress.update(task, completed=progress.tasks[task].total or 1)

    if dry_run:
        return

    if draft_paths:
        console.print(f"\n[green]{len(draft_paths)} draft(s) written:[/green]")
        for p in draft_paths:
            console.print(f"  {p.relative_to(config.vault)}")

    if failed:
        console.print(f"[yellow]{len(failed)} article(s) failed:[/yellow] {', '.join(failed)}")

    # Update index and log
    from .indexer import append_log, generate_index

    generate_index(config, db)
    if draft_paths:
        append_log(config, f"compile | {len(draft_paths)} drafts written")

    if auto_approve and draft_paths:
        published = approve_drafts(config, db, draft_paths)
        generate_index(config, db)
        append_log(config, f"approve | {len(published)} articles published")
        if config.pipeline.auto_commit:
            git_commit(
                config.vault, f"compile: {len(published)} articles", paths=["wiki/", ".olw/"]
            )
        console.print(f"[green]Published {len(published)} articles.[/green]")
    elif draft_paths:
        console.print("\nReview drafts in [bold]wiki/.drafts/[/bold], then run:")
        console.print("  [bold]olw approve --all[/bold]")


# ── approve ───────────────────────────────────────────────────────────────────


@cli.command()
@click.option("--vault", "vault_str", envvar="OLW_VAULT", default=None)
@click.option("--all", "approve_all", is_flag=True)
@click.argument("files", nargs=-1, type=click.Path())
def approve(vault_str, approve_all, files):
    """Publish draft(s) from wiki/.drafts/ to wiki/."""
    from .git_ops import git_commit
    from .pipeline.compile import approve_drafts

    config = _load_config(vault_str)
    db = _load_db(config)

    if approve_all:
        paths = None  # approve_drafts handles all
    elif files:
        paths = [_resolve_draft_arg(config, f) for f in files]
    else:
        click.echo("Specify --all or file paths.", err=True)
        sys.exit(1)

    published = approve_drafts(config, db, paths)
    if not published:
        console.print("[yellow]No drafts to approve.[/yellow]")
        return

    console.print(f"[green]Published {len(published)} article(s).[/green]")

    # Update index and log
    from .indexer import append_log, generate_index

    generate_index(config, db)
    append_log(config, f"approve | {len(published)} articles published")

    if config.pipeline.auto_commit:
        git_commit(
            config.vault, f"approve: {len(published)} articles published", paths=["wiki/", ".olw/"]
        )
        console.print("[dim]Git commit created.[/dim]")


# ── reject ────────────────────────────────────────────────────────────────────


@cli.command()
@click.option("--vault", "vault_str", envvar="OLW_VAULT", default=None)
@click.option("--all", "reject_all", is_flag=True, help="Reject all drafts in wiki/.drafts/")
@click.option("--feedback", default="", help="Reason for rejection")
@click.argument("files", nargs=-1, type=click.Path())
def reject(vault_str, reject_all, feedback, files):
    """Discard draft article(s) and store rejection feedback for future recompiles."""
    from .pipeline.compile import reject_draft

    config = _load_config(vault_str)
    db = _load_db(config)

    if reject_all:
        draft_paths = list(config.drafts_dir.rglob("*.md")) if config.drafts_dir.exists() else []
        if not draft_paths:
            console.print("[yellow]No drafts to reject.[/yellow]")
            return
        if not feedback:
            feedback = click.prompt("Reason for rejecting all drafts?", default="")
    elif files:
        draft_paths = [_resolve_draft_arg(config, f) for f in files]
        for p in draft_paths:
            if not p.exists():
                click.echo(f"File not found: {p}", err=True)
                sys.exit(1)
        if not feedback:
            feedback = click.prompt("Reason for rejection?", default="")
    else:
        click.echo("Specify --all or provide file paths.", err=True)
        sys.exit(1)

    from .vault import parse_note as _parse

    for draft_path in draft_paths:
        title = draft_path.stem
        try:
            meta, _ = _parse(draft_path)
            title = meta.get("title", draft_path.stem)
        except Exception:
            pass

        reject_draft(draft_path, config, db, feedback=feedback)
        console.print(f"[yellow]Draft rejected:[/yellow] {draft_path.name}")

        if feedback:
            count = db.rejection_count(title)
            if db.is_concept_blocked(title):
                console.print(
                    f"[red]⚠ '{title}' blocked after {count} rejections. "
                    f'Use [bold]olw unblock "{title}"[/bold] to re-enable.[/red]'
                )
            else:
                console.print(
                    f"[dim]Feedback saved. Next compile of '{title}' will address it. "
                    f"({count}/{db._REJECTION_CAP} rejections)[/dim]"
                )

    if len(draft_paths) > 1:
        console.print(f"[green]Rejected {len(draft_paths)} draft(s).[/green]")


# ── status ────────────────────────────────────────────────────────────────────


@cli.command()
@click.option("--vault", "vault_str", envvar="OLW_VAULT", default=None)
@click.option("--failed", "show_failed", is_flag=True, help="List failed notes with error messages")
def status(vault_str, show_failed):
    """Show vault health, pending drafts, and pipeline stats."""
    from .models import WikiArticleRecord

    config = _load_config(vault_str)
    db = _load_db(config)

    stats = db.stats(config.vault)
    raw = stats.get("raw", {})

    table = Table(title="Vault Status", show_header=True)
    table.add_column("Category")
    table.add_column("Count", justify="right")

    table.add_row("Raw: new", str(raw.get("new", 0)))
    table.add_row("Raw: ingested", str(raw.get("ingested", 0)))
    table.add_row("Raw: compiled", str(raw.get("compiled", 0)))
    table.add_row("Raw: failed", str(raw.get("failed", 0)))
    table.add_row("Drafts pending", str(stats["drafts"]))
    table.add_row("Published articles", str(stats["published"]))

    console.print(table)

    # List pending drafts
    drafts = db.list_articles(drafts_only=True)
    known_draft_paths = {article.path for article in drafts}
    if config.drafts_dir.exists():
        from .vault import list_draft_articles

        for title, path, sources in list_draft_articles(config.drafts_dir):
            rel_path = str(path.relative_to(config.vault))
            if rel_path in known_draft_paths:
                continue
            drafts.append(
                WikiArticleRecord(
                    path=rel_path,
                    title=title,
                    sources=sources,
                    content_hash="",
                    is_draft=True,
                )
            )
    if drafts:
        console.print(f"\n[bold]{len(drafts)} draft(s) pending review:[/bold]")
        for article in drafts:
            sources_str = ", ".join(Path(s).name for s in article.sources)
            console.print(f"  [dim]{article.path}[/dim]  (from: {sources_str})")
        console.print("\nRun [bold]olw approve --all[/bold] to publish.")

    # List failed notes if requested (or if there are any)
    if show_failed or raw.get("failed", 0):
        failed_recs = db.list_raw(status="failed")
        if failed_recs:
            console.print(f"\n[red][bold]{len(failed_recs)} failed note(s):[/bold][/red]")
            for rec in failed_recs:
                err = rec.error or "unknown error"
                console.print(f"  [dim]{rec.path}[/dim]")
                console.print(f"    [red]{err}[/red]")
            console.print("\nRun [bold]olw compile --retry-failed[/bold] to re-attempt.")

    # Show blocked concepts
    blocked = db.list_blocked_concepts()
    if blocked:
        console.print(f"\n[red][bold]{len(blocked)} blocked concept(s):[/bold][/red]")
        for concept in blocked:
            count = db.rejection_count(concept)
            console.print(f"  {concept} [dim]({count} rejections)[/dim]")
        console.print('[dim]Run [bold]olw unblock "Concept"[/bold] to re-enable.[/dim]')

    # Show pipeline lock status
    from .pipeline.lock import has_invalid_lock_file, lock_holder_pid

    pid = lock_holder_pid(config.vault)
    if pid is not None:
        import os

        try:
            os.kill(pid, 0)
            console.print(f"\n[yellow]⚠ Pipeline lock held by PID {pid}[/yellow]")
        except (ProcessLookupError, PermissionError):
            console.print(f"\n[dim]Lock file present (PID {pid}) but process not running[/dim]")
    elif has_invalid_lock_file(config.vault):
        console.print("\n[dim]Lock file present but invalid; no live process holds it[/dim]")


# ── undo ─────────────────────────────────────────────────────────────────────


@cli.command()
@click.option("--vault", "vault_str", envvar="OLW_VAULT", default=None)
@click.option("--steps", default=1, show_default=True)
def undo(vault_str, steps):
    """Revert last N [olw] auto-commits (uses git revert — safe)."""
    from .git_ops import git_undo

    config = _load_config(vault_str)
    reverted = git_undo(config.vault, steps=steps)
    if reverted:
        console.print(f"[green]Reverted {len(reverted)} commit(s):[/green]")
        for msg in reverted:
            console.print(f"  {msg}")
    else:
        console.print("[yellow]No [olw] commits found to revert.[/yellow]")


# ── clean ─────────────────────────────────────────────────────────────────────


@cli.command()
@click.option("--vault", "vault_str", envvar="OLW_VAULT", default=None)
@click.option("--yes", is_flag=True, help="Skip confirmation prompt")
def clean(vault_str, yes):
    """Clear state DB, wiki/, and drafts — raw/ notes are kept.

    Use this to start fresh without deleting your source material.
    """
    import shutil

    config = _load_config(vault_str)

    targets = [
        ("state DB", config.state_db_path),
        ("wiki/", config.wiki_dir),
    ]

    console.print("[bold yellow]This will delete:[/bold yellow]")
    for label, path in targets:
        if path.exists():
            console.print(f"  {label}: {path}")
    console.print("Raw notes in [bold]raw/[/bold] are NOT touched.")

    if not yes:
        click.confirm("Proceed?", abort=True)

    for label, path in targets:
        if path.exists():
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            console.print(f"  [dim]Deleted {label}[/dim]")

    # Re-create empty wiki/ structure
    config.wiki_dir.mkdir(parents=True, exist_ok=True)
    config.drafts_dir.mkdir(parents=True, exist_ok=True)
    config.sources_dir.mkdir(parents=True, exist_ok=True)

    console.print("[green]Clean complete.[/green] Run [bold]olw ingest --all[/bold] to restart.")


# ── doctor ───────────────────────────────────────────────────────────────────


@cli.command()
@click.option("--vault", "vault_str", envvar="OLW_VAULT", default=None)
def doctor(vault_str):
    """Check LLM provider connection, model availability, and vault health."""
    from .client_factory import LLMError, build_client

    config = _load_config(vault_str)
    db = _load_db(config)
    ok = True
    prov = config.effective_provider

    console.print("[bold]olw doctor[/bold]\n")

    # ── Vault structure ──────────────────────────────────────────────────────
    console.print("[bold]Vault structure[/bold]")
    toml_path = config.vault / "wiki.toml"
    if not toml_path.exists():
        console.print(
            f"  [red]✗[/red] wiki.toml missing — vault not initialised.\n"
            f"    Run: [bold]olw init {config.vault}[/bold]"
        )
        console.print("\n[red][bold]Vault not initialised. Remaining checks skipped.[/bold][/red]")
        sys.exit(1)

    checks = {
        "raw/": config.raw_dir,
        "wiki/": config.wiki_dir,
        "wiki/.drafts/": config.drafts_dir,
        "wiki/sources/": config.sources_dir,
        ".olw/": config.olw_dir,
        "wiki.toml": toml_path,
    }
    for name, path in checks.items():
        if path.exists():
            console.print(f"  [green]✓[/green] {name}")
        else:
            console.print(f"  [yellow]![/yellow] {name} missing (run [bold]olw init[/bold])")

    # ── Provider connection ───────────────────────────────────────────────────
    console.print(f"\n[bold]{prov.name}[/bold]")
    client = build_client(config)
    try:
        client.require_healthy()
        console.print(f"  [green]✓[/green] Reachable at {prov.url}")
    except LLMError as e:
        console.print(f"  [red]✗[/red] {e}")
        ok = False

    # ── Model availability ────────────────────────────────────────────────────
    console.print("\n[bold]Models[/bold]")
    try:
        available_models = client.list_models()
    except Exception:
        available_models = []

    for label, model_name in [("fast", config.models.fast), ("heavy", config.models.heavy)]:
        if any(model_name in a for a in available_models):
            console.print(f"  [green]✓[/green] {label}: {model_name}")
        else:
            pull_hint = (
                f"run: [bold]ollama pull {model_name}[/bold]"
                if prov.name == "ollama"
                else "check provider model list"
            )
            console.print(f"  [yellow]![/yellow] {label}: {model_name} not found — {pull_hint}")
            ok = False

    # ── Vault stats ───────────────────────────────────────────────────────────
    console.print("\n[bold]Vault stats[/bold]")
    stats = db.stats(config.vault)
    raw = stats.get("raw", {})
    console.print(f"  Raw notes:         {sum(raw.values())}")
    console.print(f"  Ingested:          {raw.get('ingested', 0) + raw.get('compiled', 0)}")
    console.print(f"  Drafts pending:    {stats['drafts']}")
    console.print(f"  Published:         {stats['published']}")

    draft_graph_filter = [
        "-path:raw",
        "-path:wiki/sources",
        "-path:_resources",
        "-file:Welcome",
    ]
    published_graph_filter = [
        "-path:raw",
        "-path:wiki/sources",
        "-path:wiki/.drafts",
        "-path:_resources",
        "-file:Welcome",
    ]
    graph_notes: list[str] = []
    if (config.vault / "Welcome.md").exists():
        graph_notes.append("Welcome.md is present and can create starter graph noise")
    if config.raw_dir.exists() and any(config.raw_dir.rglob("*.md")):
        graph_notes.append("raw/ notes are visible unless filtered")
    if config.sources_dir.exists() and any(config.sources_dir.glob("*.md")):
        graph_notes.append("wiki/sources/ pages can dominate graph when citations are enabled")
    if config.drafts_dir.exists() and any(config.drafts_dir.rglob("*.md")):
        graph_notes.append("wiki/.drafts/ pages are review artifacts, not published wiki")

    console.print("\n[bold]Graph view[/bold]")
    if graph_notes:
        for note in graph_notes:
            console.print(f"  [yellow]![/yellow] {note}")
    else:
        console.print("  [green]✓[/green] No obvious graph-noise layers detected")
    console.print("  Draft review graph filter:")
    console.print(f"  [dim]{' '.join(draft_graph_filter)}[/dim]")
    console.print("  Published-only graph filter:")
    console.print(f"  [dim]{' '.join(published_graph_filter)}[/dim]")

    console.print()
    if ok:
        console.print("[green][bold]All checks passed.[/bold][/green]")
    else:
        console.print("[yellow][bold]Some checks need attention (see above).[/bold][/yellow]")


# ── query ─────────────────────────────────────────────────────────────────────


@cli.command()
@click.option("--vault", "vault_str", envvar="OLW_VAULT", default=None)
@click.option("--save", is_flag=True, help="Save answer to wiki/queries/")
@click.argument("question")
def query(vault_str, question, save):
    """Answer a question using your wiki as context (no embeddings needed)."""
    from rich.markdown import Markdown

    from .pipeline.query import run_query

    config = _load_config(vault_str)
    client, db = _load_deps(config)

    with console.status("[bold]Searching wiki index…"):
        answer, pages = run_query(config, client, db, question, save=save)

    if pages:
        console.print(f"[dim]Sources: {', '.join(pages)}[/dim]")
    console.print()
    console.print(Markdown(answer))
    if save:
        console.print("\n[green]Answer saved to wiki/queries/[/green]")


# ── lint ──────────────────────────────────────────────────────────────────────


@cli.command()
@click.option("--vault", "vault_str", envvar="OLW_VAULT", default=None)
@click.option("--fix", is_flag=True, help="Auto-fix simple issues (missing frontmatter fields)")
def lint(vault_str, fix):
    """Check wiki health: orphans, broken links, missing frontmatter, low confidence."""
    from .pipeline.lint import run_lint

    config = _load_config(vault_str)
    db = _load_db(config)

    result = run_lint(config, db, fix=fix)

    # Score bar
    score = result.health_score
    colour = "green" if score >= 80 else "yellow" if score >= 50 else "red"
    console.print(f"\n[bold {colour}]Health: {score}/100[/bold {colour}]  {result.summary}")

    if result.issues:
        console.print()
        _TYPE_ICON = {
            "orphan": "○",
            "broken_link": "⛓",
            "missing_frontmatter": "⚙",
            "stale": "✎",
            "low_confidence": "↓",
            "config_outdated": "⚠",
        }
        from rich.markup import escape

        for iss in result.issues:
            icon = _TYPE_ICON.get(iss.issue_type, "!")
            fix_tag = " [dim][auto-fixable][/dim]" if iss.auto_fixable else ""
            console.print(f"  {icon} [bold]{iss.issue_type}[/bold]{fix_tag}  {escape(iss.path)}")
            console.print(f"     {escape(iss.description)}")
            console.print(f"     [dim]→ {escape(iss.suggestion)}[/dim]")
        console.print()

    if fix:
        fixed = sum(1 for i in result.issues if i.auto_fixable)
        if fixed:
            console.print(f"[green]Auto-fixed {fixed} issue(s).[/green]")


# ── watch ─────────────────────────────────────────────────────────────────────


@cli.command()
@click.option("--vault", "vault_str", envvar="OLW_VAULT", default=None)
@click.option(
    "--auto-approve", is_flag=True, help="Publish drafts immediately without manual review"
)
def watch(vault_str, auto_approve):
    """Watch raw/ for new/changed notes → auto-ingest + compile."""
    from .pipeline.lock import pipeline_lock
    from .pipeline.orchestrator import PipelineOrchestrator
    from .watcher import watch as _watch

    config = _load_config(vault_str)
    client, db = _load_deps(config)
    orchestrator = PipelineOrchestrator(config, client, db)

    debounce = config.pipeline.watch_debounce
    console.print(f"[bold]Watching[/bold] {config.raw_dir}  (debounce={debounce:.0f}s)")
    console.print("[dim]Ctrl+C to stop.[/dim]\n")

    def _on_event(paths: list[str]) -> None:
        md_paths = [p for p in paths if p.endswith(".md")]
        if not md_paths:
            return

        console.rule(f"[dim]{len(md_paths)} file(s) changed[/dim]")

        with pipeline_lock(config.vault) as acquired:
            if not acquired:
                console.print("[yellow]⚠ compile skipped — pipeline already running[/yellow]")
                return
            try:
                report = orchestrator.run(
                    paths=md_paths,
                    auto_approve=auto_approve or config.pipeline.auto_approve,
                    fix=config.pipeline.auto_maintain,
                )
            except Exception as exc:
                console.print(f"[red]Pipeline error:[/red] {exc}")
                return

        if report.ingested:
            console.print(f"  [green]✓[/green] ingested {report.ingested} note(s)")
        if report.compiled:
            console.print(f"  [green]✓[/green] {report.compiled} draft(s) compiled")
        if report.failed:
            failed_str = ", ".join(report.failed_names)
            console.print(
                f"  [yellow]![/yellow] {len(report.failed)} concept(s) failed: {failed_str}"
            )
        if report.published:
            console.print(f"  [green]✓[/green] {report.published} article(s) published")
        elif report.compiled:
            console.print("  [dim]Run [bold]olw approve --all[/bold] to publish drafts.[/dim]")
        if report.stubs_created:
            console.print(f"  [dim]Created {report.stubs_created} stub(s) for broken links[/dim]")

    _watch(config=config, client=client, db=db, on_event=_on_event)


# ── run ───────────────────────────────────────────────────────────────────────


@cli.command()
@click.option("--vault", "vault_str", envvar="OLW_VAULT", default=None)
@click.option("--auto-approve", is_flag=True, help="Publish drafts immediately")
@click.option("--fix", is_flag=True, help="Create stubs for broken wikilinks")
@click.option("--max-rounds", default=2, show_default=True, help="Max compile rounds")
@click.option("--dry-run", is_flag=True, help="Report what would happen, make no changes")
@_model_override_options
def run(
    vault_str,
    auto_approve,
    fix,
    max_rounds,
    dry_run,
    fast_model,
    heavy_model,
    provider_name,
    provider_url,
):
    """Run full pipeline: ingest → compile → lint → [approve]."""
    from .pipeline.lock import pipeline_lock
    from .pipeline.orchestrator import PipelineOrchestrator

    overrides = _model_override_kwargs(fast_model, heavy_model, provider_name, provider_url)
    config = _load_config(vault_str, **overrides)
    client, db = _load_deps(config)

    if dry_run:
        console.print("[dim]Dry run — no changes will be made.[/dim]\n")

    with pipeline_lock(config.vault) as acquired:
        if not acquired:
            err_console.print("Pipeline already running — lock held. Check `olw status`.")
            sys.exit(1)
        orchestrator = PipelineOrchestrator(config, client, db)
        report = orchestrator.run(
            auto_approve=auto_approve,
            fix=fix,
            max_rounds=max_rounds,
            dry_run=dry_run,
        )

    table = Table(title="Pipeline Report", show_header=True)
    table.add_column("Step")
    table.add_column("Count", justify="right")
    table.add_column("Time", justify="right")

    table.add_row("Ingested", str(report.ingested), f"{report.timings.get('ingest', 0):.1f}s")
    table.add_row(
        "Compiled",
        str(report.compiled),
        f"{report.timings.get('compile_r1', 0) + report.timings.get('compile_r2', 0):.1f}s",
    )
    table.add_row("Published", str(report.published), "")
    table.add_row("Lint issues", str(report.lint_issues), "")
    table.add_row("Stubs created", str(report.stubs_created), "")
    if report.rounds > 1:
        table.add_row("Compile rounds", str(report.rounds), "")
    console.print(table)

    if report.failed:
        console.print(f"\n[yellow]{len(report.failed)} concept(s) failed:[/yellow]")
        for f in report.failed:
            console.print(f"  [dim]{f.concept}[/dim] ({f.reason.value})")


# ── review ────────────────────────────────────────────────────────────────────


@cli.command()
@click.option("--vault", "vault_str", envvar="OLW_VAULT", default=None)
def review(vault_str):
    """Interactive draft review: approve, reject, edit, or diff drafts."""

    from .pipeline.compile import approve_drafts, reject_draft
    from .pipeline.review import (
        compute_diff,
        compute_rejection_diff,
        list_drafts,
        load_draft_content,
    )

    config = _load_config(vault_str)
    db = _load_db(config)

    while True:
        summaries = list_drafts(config, db)
        if not summaries:
            console.print("[dim]No drafts pending review.[/dim]")
            return

        # Build menu table
        table = Table(title="Drafts Pending Review", show_header=True, show_lines=False)
        table.add_column("#", justify="right", style="dim")
        table.add_column("Title")
        table.add_column("Conf", justify="right")
        table.add_column("Sources", justify="right")
        table.add_column("Rejections", justify="right")
        table.add_column("Flags", justify="left")

        for i, s in enumerate(summaries, 1):
            flags = ""
            if s.has_annotations:
                flags += "⚠ annotations  "
            if s.rejection_count > 0:
                flags += f"{'🔴' if s.rejection_count >= 3 else '🟡'} rejected"
            conf_color = (
                "green" if s.confidence >= 0.6 else "yellow" if s.confidence >= 0.4 else "red"
            )
            table.add_row(
                str(i),
                s.title,
                f"[{conf_color}]{s.confidence:.2f}[/{conf_color}]",
                str(s.source_count),
                str(s.rejection_count),
                flags.strip(),
            )

        console.print(table)
        console.print("\n[dim]  Type: number=open draft, a=approve all, x=reject all, q=quit[/dim]")
        choice = click.prompt("\nChoice", prompt_suffix=" > ").strip().lower()

        if choice == "q":
            return
        elif choice == "a":
            all_paths = [s.path for s in summaries]
            published = approve_drafts(config, db, all_paths)
            console.print(f"[green]Published {len(published)} article(s).[/green]")
            from .indexer import append_log, generate_index

            generate_index(config, db)
            append_log(config, f"review | approved {len(published)} articles")
            return
        elif choice == "x":
            reason = click.prompt("Reason for rejecting all", default="")
            for s in summaries:
                reject_draft(s.path, config, db, feedback=reason)
            console.print(f"[yellow]Rejected {len(summaries)} draft(s).[/yellow]")
            return
        elif choice.isdigit():
            idx = int(choice) - 1
            if idx < 0 or idx >= len(summaries):
                console.print("[red]Invalid selection.[/red]")
                continue
            _review_single(
                summaries[idx],
                config,
                db,
                approve_drafts,
                reject_draft,
                compute_diff,
                compute_rejection_diff,
                load_draft_content,
            )
        else:
            console.print("[red]Unknown command.[/red]")


def _review_single(
    summary,
    config,
    db,
    approve_drafts,
    reject_draft,
    compute_diff,
    compute_rejection_diff,
    load_draft_content,
):
    """Handle single-draft review loop."""
    from rich.panel import Panel

    from .vault import sanitize_filename

    while True:
        if not summary.path.exists():
            console.print("[yellow]Draft no longer exists.[/yellow]")
            return

        try:
            meta, body = load_draft_content(summary.path)
        except Exception as e:
            console.print(f"[red]Could not read draft: {e}[/red]")
            return

        # Show rejection history
        rejections = db.get_rejections(summary.title, limit=3)
        if rejections:
            console.print(
                Panel(
                    "\n".join(f"• {r['feedback']}" for r in rejections),
                    title=f"[red]Previous rejections ({len(rejections)})[/red]",
                    border_style="red",
                )
            )

        # Show metadata
        console.print(
            f"[bold]{summary.title}[/bold]  "
            f"conf={meta.get('confidence', 0):.2f}  "
            f"sources={summary.source_count}  "
            f"rejections={summary.rejection_count}"
        )

        # Show body
        console.print(Panel(body[:3000] + ("…" if len(body) > 3000 else ""), title="Draft"))

        console.print(
            "\n[dim]Type: a=approve, r=reject, e=edit, "
            "d=diff vs published, v=rejection diff, s=skip[/dim]"
        )
        raw_action = click.prompt("\nAction", prompt_suffix=" > ").strip()
        action = raw_action.lower()

        if action == "s":
            return
        elif action == "a":
            if not summary.path.exists():
                console.print("[yellow]Draft disappeared.[/yellow]")
                return
            published = approve_drafts(config, db, [summary.path])
            console.print(f"[green]Published:[/green] {published[0].name if published else '?'}")
            from .indexer import append_log, generate_index

            generate_index(config, db)
            append_log(config, f"review | approved {summary.title}")
            return
        elif action == "r":
            reason = click.prompt("Reason?", default="")
            if not summary.path.exists():
                console.print("[yellow]Draft disappeared.[/yellow]")
                return
            reject_draft(summary.path, config, db, feedback=reason)
            console.print("[yellow]Rejected.[/yellow]")
            if reason:
                count = db.rejection_count(summary.title)
                if db.is_concept_blocked(summary.title):
                    console.print(f"[red]⚠ '{summary.title}' is now blocked.[/red]")
                else:
                    console.print(f"[dim]({count}/{db._REJECTION_CAP} rejections)[/dim]")
            return
        elif action == "e":
            import os
            import subprocess

            editor = os.environ.get("VISUAL") or os.environ.get("EDITOR") or "vi"
            subprocess.call([editor, str(summary.path)])
        elif action == "d":
            safe_name = sanitize_filename(summary.title)
            wiki_path = config.wiki_dir / f"{safe_name}.md"
            diff = compute_diff(summary.path, wiki_path)
            if diff is None:
                console.print("[dim]No published version — this is a new article.[/dim]")
            else:
                console.print(diff)
        elif action == "v":
            diff = compute_rejection_diff(summary.path, db, summary.title)
            if diff is None:
                console.print("[dim]No rejected body stored for this concept.[/dim]")
            else:
                console.print(diff)
        else:
            console.print("[red]Unknown action.[/red]")


# ── maintain ──────────────────────────────────────────────────────────────────


@cli.command()
@click.option("--vault", "vault_str", envvar="OLW_VAULT", default=None)
@click.option(
    "--fix", is_flag=True, help="Auto-fix missing frontmatter, invalid tags, create stubs"
)  # noqa: E501
@click.option("--stubs-only", is_flag=True, help="Only create stub articles")
@click.option("--dry-run", is_flag=True, help="Report issues without making changes")
def maintain(vault_str, fix, stubs_only, dry_run):
    """Wiki maintenance: lint, stub creation, orphan suggestions, concept merge hints."""
    from .pipeline.lint import run_lint
    from .pipeline.lock import pipeline_lock
    from .pipeline.maintain import (
        create_stubs,
        fix_broken_links,
        normalize_published_alias_links,
        suggest_concept_merges,
        suggest_orphan_links,
    )

    config = _load_config(vault_str)
    db = _load_db(config)

    if dry_run:
        console.print("[dim]Dry run — no changes will be made.[/dim]\n")

    with pipeline_lock(config.vault) as acquired:
        if not acquired:
            err_console.print("Pipeline already running — lock held.")
            sys.exit(1)

        # Quality warning
        quality = db.quality_stats()
        total_sources = sum(quality.values())
        if total_sources > 0:
            low_pct = round(100 * quality["low"] / total_sources)
            if low_pct > 60:
                console.print(
                    f"[yellow]⚠ {low_pct}% of sources are low quality — "
                    f"articles will have low confidence.[/yellow]"
                )

        # Blocked concepts
        blocked = db.list_blocked_concepts()
        if blocked:
            console.print(f"\n[red]{len(blocked)} blocked concept(s):[/red]")
            for concept in blocked:
                count = db.rejection_count(concept)
                console.print(f"  {concept} ({count} rejections)")
            console.print('[dim]Use [bold]olw unblock "Concept"[/bold] to re-enable.[/dim]')

        if stubs_only:
            if not dry_run:
                created = create_stubs(config, db)
                console.print(f"[green]Created {len(created)} stub(s).[/green]")
            else:
                result = run_lint(config, db)
                broken = [i for i in result.issues if i.issue_type == "broken_link"]
                console.print(f"[dim]Would create up to {min(len(broken), 5)} stub(s).[/dim]")
            return

        # Full lint
        result = run_lint(config, db, fix=fix and not dry_run)
        score = result.health_score
        colour = "green" if score >= 80 else "yellow" if score >= 50 else "red"
        console.print(f"\n[bold {colour}]Health: {score}/100[/bold {colour}]  {result.summary}")

        if result.issues:
            console.print()
            for iss in result.issues:
                fix_tag = " [dim][auto-fixable][/dim]" if iss.auto_fixable else ""
                console.print(f"  [bold]{iss.issue_type}[/bold]{fix_tag}  {iss.path}")
                console.print(f"    {iss.description}")

        # Alias link normalization in published articles (fix [[Alias]] → [[Canonical|Alias]])
        # Runs independently of broken-link detection: lint resolves aliases so they never
        # appear as broken, but published articles may still have raw alias-form links.
        if fix and not stubs_only:
            alias_normalized = normalize_published_alias_links(config, db, dry_run=dry_run)
            if alias_normalized:
                console.print(
                    f"\n[green]Normalized alias links in {alias_normalized} article(s).[/green]"
                )

        # Broken link repair + stub creation
        broken = [i for i in result.issues if i.issue_type == "broken_link"]
        if broken:
            if fix and not stubs_only:
                repair = fix_broken_links(config, db, broken, dry_run=dry_run)
                if repair.repaired:
                    console.print(f"\n[green]Repaired {repair.repaired} broken link(s).[/green]")
                remaining = repair.still_broken
                if remaining and not dry_run:
                    created = create_stubs(config, db, broken_link_issues=remaining)
                    if created:
                        console.print(f"[green]Created {len(created)} stub(s).[/green]")
                elif remaining:
                    console.print(
                        f"[dim]{len(remaining)} link(s) unresolvable"
                        f" — stubs would be created.[/dim]"
                    )
            elif fix:
                created = create_stubs(config, db, broken_link_issues=broken)
                if created:
                    console.print(f"\n[green]Created {len(created)} stub(s).[/green]")
            else:
                console.print(
                    f"\n[dim]{len(broken)} broken link(s) — "
                    f"run [bold]olw maintain --fix[/bold] to repair or create stubs.[/dim]"
                )

        # Orphan suggestions
        orphan_suggestions = suggest_orphan_links(config, db)
        if orphan_suggestions:
            console.print(f"\n[bold]Orphan link suggestions ({len(orphan_suggestions)}):[/bold]")
            for title, mentioners in orphan_suggestions[:5]:
                console.print(f"  {title} — mentioned in:")
                for m in mentioners[:3]:
                    console.print(f"    [dim]{m}[/dim]")

        # Concept merge suggestions
        merges = suggest_concept_merges(config, db)
        if merges:
            console.print(f"\n[bold]Possible concept duplicates ({len(merges)}):[/bold]")
            for a, b, score in merges[:5]:
                console.print(f"  '{a}' ≈ '{b}'  [dim](similarity={score:.0%})[/dim]")


# ── unblock ───────────────────────────────────────────────────────────────────


@cli.command()
@click.option("--vault", "vault_str", envvar="OLW_VAULT", default=None)
@click.argument("concept")
def unblock(vault_str, concept):
    """Re-enable a concept that was blocked after too many rejections."""
    config = _load_config(vault_str)
    db = _load_db(config)

    if not db.is_concept_blocked(concept):
        console.print(f"[yellow]'{concept}' is not blocked.[/yellow]")
        return

    db.unblock_concept(concept)
    count = db.rejection_count(concept)
    console.print(f"[green]'{concept}' unblocked.[/green]")
    console.print(
        f"[dim]{count} rejection(s) remain on record. Next compile will include this concept.[/dim]"
    )


# ── items ─────────────────────────────────────────────────────────────────────


@cli.group()
def items():
    """Audit preserved non-concept knowledge item candidates."""


@items.command("audit")
@click.option("--vault", "vault_str", envvar="OLW_VAULT", default=None)
@click.option("--limit", default=30, show_default=True, help="Maximum items to show")
def items_audit(vault_str, limit):
    """Show ambiguous/entity candidates preserved during ingest."""
    config = _load_config(vault_str)
    db = _load_db(config)
    candidates = [item for item in db.list_items(status="candidate") if item.kind != "concept"]
    if not candidates:
        console.print("[green]No candidate knowledge items found.[/green]")
        return

    console.print(f"[bold]{len(candidates)} candidate knowledge item(s)[/bold]\n")
    for item in candidates[:limit]:
        mentions = db.get_item_mentions(item.name)
        console.print(
            f"[yellow]{item.name}[/yellow]  "
            f"kind={item.kind} subtype={item.subtype or 'unknown'} "
            f"confidence={item.confidence:.2f} mentions={len(mentions)}"
        )
        for mention in mentions[:3]:
            console.print(
                f"  - {mention.evidence_level}: {mention.source_path} ({mention.mention_text})"
            )
        console.print("  suggested: classify / ignore / keep candidate\n")


@items.command("show")
@click.option("--vault", "vault_str", envvar="OLW_VAULT", default=None)
@click.argument("name")
def items_show(vault_str, name):
    """Show one preserved knowledge item and its mentions."""
    config = _load_config(vault_str)
    db = _load_db(config)
    item = db.get_item(name)
    if item is None:
        console.print(f"[red]Item not found:[/red] {name}")
        raise SystemExit(1)
    console.print(f"[bold]{item.name}[/bold]")
    console.print(f"  kind: {item.kind}")
    console.print(f"  subtype: {item.subtype or 'unknown'}")
    console.print(f"  status: {item.status}")
    console.print(f"  confidence: {item.confidence:.2f}")
    mentions = db.get_item_mentions(item.name)
    console.print(f"\n[bold]Mentions ({len(mentions)})[/bold]")
    for mention in mentions:
        console.print(f"- {mention.evidence_level}: {mention.source_path}")
        if mention.context:
            console.print(f"  {mention.context}")


# ── compare ───────────────────────────────────────────────────────────────────


def _is_cloud_provider(provider_name: str | None) -> bool:
    from .providers import get_provider

    pname = provider_name or "ollama"
    info = get_provider(pname)
    if info is None:
        return pname != "ollama"
    return info is not None and not info.is_local


def _validate_compare_out_dir(out: Path, config) -> Path:
    out = out.expanduser().resolve()
    raw_dir = config.raw_dir.resolve()
    wiki_dir = config.wiki_dir.resolve()
    olw_dir = config.olw_dir.resolve()
    compare_root = (config.olw_dir / "compare").resolve()

    def _is_within(path: Path, root: Path) -> bool:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            return False

    if _is_within(out, raw_dir) or _is_within(out, wiki_dir):
        raise click.BadParameter("--out must not be inside raw/ or wiki/")
    if _is_within(out, olw_dir) and not _is_within(out, compare_root):
        raise click.BadParameter("--out under .olw/ is only allowed inside .olw/compare/")
    return out


def _validate_compare_inputs(config, queries_path: str | None) -> None:
    from .compare.runner import _collect_raw_notes, _validate_queries_path

    try:
        _collect_raw_notes(config.raw_dir)
    except ValueError as e:
        raise click.BadParameter(str(e)) from e
    if queries_path:
        try:
            _validate_queries_path(Path(queries_path))
        except ValueError as e:
            raise click.BadParameter(str(e)) from e


def _validate_compare_sample_n(_ctx, _param, value: int | None) -> int | None:
    if value is None or value >= 1:
        return value
    raise click.BadParameter("must be at least 1")


@cli.command(name="compare")
@click.option("--vault", "vault_str", envvar="OLW_VAULT", default=None)
@_model_override_options
@click.option(
    "--queries",
    "queries_path",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Optional compare queries.toml.",
)
@click.option(
    "--out",
    "out_dir",
    type=click.Path(file_okay=False),
    default=None,
    show_default=True,
    help="Output directory (default: .olw/compare in the active vault).",
)
@click.option("--keep-artifacts", is_flag=True, help="Do not delete ephemeral vaults.")
@click.option(
    "--allow-cloud-upload",
    is_flag=True,
    help="Required when the challenger uses a cloud provider.",
)
@click.option(
    "--format",
    "report_format",
    type=click.Choice(["md", "json", "both"]),
    default="both",
    show_default=True,
    help="Report output format.",
)
@click.option(
    "--sample-n",
    "sample_n",
    type=int,
    default=None,
    callback=_validate_compare_sample_n,
    help="Limit compare to first N raw notes (useful for a quick spot-check on large vaults).",
)
def compare(
    vault_str,
    fast_model,
    heavy_model,
    provider_name,
    provider_url,
    queries_path,
    out_dir,
    keep_artifacts,
    allow_cloud_upload,
    report_format,
    sample_n,
):
    """Preview whether switching LLM config would improve your vault."""
    from .compare.runner import run_compare

    config = _load_config(vault_str)
    challenger_kwargs = _model_override_kwargs(fast_model, heavy_model, provider_name, provider_url)
    if not challenger_kwargs:
        err_console.print("Provide at least one challenger override, e.g. --heavy-model.")
        sys.exit(1)
    challenger_config = _load_config(vault_str, **challenger_kwargs)

    current_summary = (
        config.models.fast,
        config.models.heavy,
        config.effective_provider.name,
        config.effective_provider.url,
    )
    challenger_summary = (
        challenger_config.models.fast,
        challenger_config.models.heavy,
        challenger_config.effective_provider.name,
        challenger_config.effective_provider.url,
    )
    if challenger_summary == current_summary:
        err_console.print("Challenger config is identical to current config.")
        sys.exit(1)

    _validate_compare_inputs(config, queries_path)

    if _is_cloud_provider(challenger_config.effective_provider.name) and not allow_cloud_upload:
        err_console.print(
            "Cloud challenger requires --allow-cloud-upload "
            "(your raw notes will be sent to the provider)."
        )
        sys.exit(1)

    out = (
        _validate_compare_out_dir(Path(out_dir), config)
        if out_dir
        else (config.olw_dir / "compare").resolve()
    )
    out.mkdir(parents=True, exist_ok=True)

    sample_label = f"first {sample_n} notes" if sample_n is not None else "all notes"
    console.print(
        f"[bold]olw compare[/bold] — active vault preview\n"
        f"  vault={config.vault}\n"
        f"  current: fast={config.models.fast} heavy={config.models.heavy} "
        f"provider={config.effective_provider.name}\n"
        f"  challenger: fast={challenger_config.models.fast} "
        f"heavy={challenger_config.models.heavy} "
        f"provider={challenger_config.effective_provider.name}\n"
        f"  queries={'enabled' if queries_path else 'disabled'}\n"
        f"  scope={sample_label}\n"
        f"  Active vault will not be modified."
    )

    report = run_compare(
        current_config=config,
        challenger_config=challenger_config,
        out_dir=out,
        queries_path=Path(queries_path) if queries_path else None,
        keep_artifacts=keep_artifacts,
        sample_n=sample_n,
    )

    from .compare.report import (
        render_json,
        render_markdown,
        render_summary_json,
        render_switch_config_toml,
        resolve,
    )

    resolve(report)

    run_dir = out / report.run_id / "results"
    if report_format in ("md", "both"):
        (run_dir / "report.md").write_text(render_markdown(report))
    if report_format in ("json", "both"):
        (run_dir / "report.json").write_text(render_json(report))
    (run_dir / "summary.json").write_text(render_summary_json(report))

    from .compare.models import AdvisorVerdict

    console.print()
    console.print(f"[green]Run complete:[/green] {report.run_id}")
    console.print(f"Artifacts: {out / report.run_id}")
    console.print(f"[bold]Verdict:[/bold] {report.verdict.value}")
    for reason in report.reasons:
        console.print(f"  · {reason}")
    if report.verdict == AdvisorVerdict.SWITCH:
        console.print("\n[bold]Next step:[/bold] edit wiki.toml and set:")
        console.print(
            render_switch_config_toml(
                fast_model=challenger_config.models.fast,
                heavy_model=challenger_config.models.heavy,
                provider_name=challenger_config.effective_provider.name,
                provider_url=challenger_config.effective_provider.url,
            ),
            markup=False,
        )

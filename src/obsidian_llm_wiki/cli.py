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

import logging
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

console = Console()
err_console = Console(stderr=True, style="bold red")


# ── Context helpers ───────────────────────────────────────────────────────────


def _load_config(vault_str: str | None, **kwargs):
    from .config import Config

    if not vault_str:
        click.echo("Error: --vault required (or set OLW_VAULT env var)", err=True)
        sys.exit(1)
    return Config.from_vault(Path(vault_str), **kwargs)


def _load_db(config):
    from .state import StateDB

    return StateDB(config.state_db_path)


def _load_deps(config):
    from .ollama_client import OllamaClient, OllamaError

    client = OllamaClient(base_url=config.ollama.url, timeout=config.ollama.timeout)
    try:
        client.require_healthy()
    except OllamaError as e:
        err_console.print(str(e))
        sys.exit(1)
    db = _load_db(config)
    return client, db


# ── CLI root ──────────────────────────────────────────────────────────────────


@click.group()
@click.version_option(package_name="obsidian-llm-wiki")
@click.option("-v", "--verbose", is_flag=True, default=False, help="Enable DEBUG logging.")
@click.option(
    "-q",
    "--quiet",
    is_flag=True,
    default=False,
    help="WARNING-only logging; suppress progress bars. Overrides --verbose.",
)
@click.pass_context
def cli(ctx: click.Context, verbose: bool, quiet: bool):
    """obsidian-llm-wiki (olw) — 100% local Obsidian → wiki pipeline."""
    ctx.ensure_object(dict)
    ctx.obj["quiet"] = quiet

    if quiet:
        level = logging.WARNING
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO

    fmt = (
        "%(asctime)s %(name)s %(levelname)s %(message)s"
        if verbose
        else "%(levelname)s: %(message)s"
    )
    logging.basicConfig(level=level, format=fmt, force=True)


def _is_quiet() -> bool:
    """Return True when ``--quiet`` was passed to the CLI root."""
    ctx = click.get_current_context(silent=True)
    if ctx is None:
        return False
    obj = ctx.find_root().obj
    return bool(obj and obj.get("quiet"))


# ── init ──────────────────────────────────────────────────────────────────────


@cli.command()
@click.argument("vault_path", type=click.Path())
@click.option("--existing", is_flag=True, help="Adopt an existing Obsidian vault")
@click.option("--non-interactive", is_flag=True)
def init(vault_path: str, existing: bool, non_interactive: bool):
    """Create vault structure and initialise olw."""
    from .config import DEFAULT_WIKI_TOML
    from .git_ops import git_init

    vault = Path(vault_path).expanduser().resolve()
    vault.mkdir(parents=True, exist_ok=True)

    if existing:
        _init_existing(vault, non_interactive)
    else:
        _init_fresh(vault)

    # Write wiki.toml
    toml_path = vault / "wiki.toml"
    if not toml_path.exists():
        toml_path.write_text(DEFAULT_WIKI_TOML)

    # Init git
    git_init(vault)

    # Create .gitignore
    gi = vault / ".gitignore"
    if not gi.exists():
        gi.write_text(".DS_Store\n.olw/chroma/\n.olw/state.db\n.obsidian/workspace.json\n*.log\n")

    console.print(f"[green]Vault initialised:[/green] {vault}")
    console.print("Next steps:")
    console.print("  1. Drop .md notes into [bold]raw/[/bold]")
    console.print("  2. Run [bold]olw ingest --all[/bold]")
    console.print("  3. Run [bold]olw compile[/bold]")
    console.print("  4. Run [bold]olw approve --all[/bold]")


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


def _write_index(vault: Path) -> None:
    index = vault / "wiki" / "INDEX.md"
    if not index.exists():
        index.parent.mkdir(parents=True, exist_ok=True)
        index.write_text(
            "---\ntitle: Index\ntags: [index]\nstatus: published\n---\n\n"
            "# Wiki Index\n\n_Updated automatically by olw._\n"
        )


# ── ingest ────────────────────────────────────────────────────────────────────


@cli.command()
@click.option("--vault", "vault_str", envvar="OLW_VAULT", required=True)
@click.option("--all", "ingest_all", is_flag=True, help="Ingest all files in raw/")
@click.option("--force", is_flag=True, help="Re-ingest already-processed notes")
@click.argument("paths", nargs=-1, type=click.Path(exists=True))
def ingest(vault_str, ingest_all, force, paths):
    """Analyze raw notes: extract concepts, quality, suggested topics."""

    config = _load_config(vault_str)
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
        disable=_is_quiet(),
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
@click.option("--vault", "vault_str", envvar="OLW_VAULT", required=True)
@click.option("--dry-run", is_flag=True, help="Show plan, write nothing")
@click.option("--auto-approve", is_flag=True, help="Publish immediately (skip draft review)")
@click.option("--force", is_flag=True, help="Recompile even manually-edited articles")
@click.option("--legacy", is_flag=True, help="Use legacy LLM-planning compile (CompilePlan)")
@click.option(
    "--retry-failed",
    "retry_failed",
    is_flag=True,
    help="Re-ingest raw notes that previously failed, then compile",
)
def compile(vault_str, dry_run, auto_approve, force, legacy, retry_failed):
    """Synthesize ingested notes into wiki article drafts."""
    from .git_ops import git_commit
    from .pipeline.compile import approve_drafts, compile_concepts, compile_notes

    config = _load_config(vault_str)
    client, db = _load_deps(config)

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

    if dry_run:
        console.print("[dim]Dry run — no files will be written.[/dim]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
        disable=_is_quiet(),
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

            draft_paths, failed = compile_concepts(
                config=config,
                client=client,
                db=db,
                force=force,
                dry_run=dry_run,
                on_progress=_on_progress,
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
@click.option("--vault", "vault_str", envvar="OLW_VAULT", required=True)
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
        paths = [Path(f) for f in files]
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
@click.option("--vault", "vault_str", envvar="OLW_VAULT", required=True)
@click.option("--feedback", default="", help="Reason for rejection (logged)")
@click.argument("file", type=click.Path(exists=True))
def reject(vault_str, feedback, file):
    """Discard a draft article."""
    from .pipeline.compile import reject_draft

    config = _load_config(vault_str)
    db = _load_db(config)
    reject_draft(Path(file), config, db, feedback=feedback)
    console.print(f"[yellow]Draft rejected:[/yellow] {file}")


# ── status ────────────────────────────────────────────────────────────────────


@cli.command()
@click.option("--vault", "vault_str", envvar="OLW_VAULT", required=True)
@click.option("--failed", "show_failed", is_flag=True, help="List failed notes with error messages")
def status(vault_str, show_failed):
    """Show vault health, pending drafts, and pipeline stats."""
    config = _load_config(vault_str)
    db = _load_db(config)

    stats = db.stats()
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


# ── undo ─────────────────────────────────────────────────────────────────────


@cli.command()
@click.option("--vault", "vault_str", envvar="OLW_VAULT", required=True)
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
@click.option("--vault", "vault_str", envvar="OLW_VAULT", required=True)
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
@click.option("--vault", "vault_str", envvar="OLW_VAULT", required=True)
def doctor(vault_str):
    """Check Ollama connection, model availability, and vault health."""
    from .ollama_client import OllamaClient, OllamaError

    config = _load_config(vault_str)
    db = _load_db(config)
    ok = True

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

    # ── Ollama connection ────────────────────────────────────────────────────
    console.print("\n[bold]Ollama[/bold]")
    client = OllamaClient(base_url=config.ollama.url, timeout=10)
    try:
        client.require_healthy()
        console.print(f"  [green]✓[/green] Reachable at {config.ollama.url}")
    except OllamaError as e:
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
            console.print(
                f"  [yellow]![/yellow] {label}: {model_name} not found — "
                f"run: [bold]ollama pull {model_name}[/bold]"
            )
            ok = False

    # ── Vault stats ───────────────────────────────────────────────────────────
    console.print("\n[bold]Vault stats[/bold]")
    stats = db.stats()
    raw = stats.get("raw", {})
    console.print(f"  Raw notes:         {sum(raw.values())}")
    console.print(f"  Ingested:          {raw.get('ingested', 0) + raw.get('compiled', 0)}")
    console.print(f"  Drafts pending:    {stats['drafts']}")
    console.print(f"  Published:         {stats['published']}")

    console.print()
    if ok:
        console.print("[green][bold]All checks passed.[/bold][/green]")
    else:
        console.print("[yellow][bold]Some checks need attention (see above).[/bold][/yellow]")


# ── query ─────────────────────────────────────────────────────────────────────


@cli.command()
@click.option("--vault", "vault_str", envvar="OLW_VAULT", required=True)
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
@click.option("--vault", "vault_str", envvar="OLW_VAULT", required=True)
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
        }
        for iss in result.issues:
            icon = _TYPE_ICON.get(iss.issue_type, "!")
            fix_tag = " [dim][auto-fixable][/dim]" if iss.auto_fixable else ""
            console.print(f"  {icon} [bold]{iss.issue_type}[/bold]{fix_tag}  {iss.path}")
            console.print(f"     {iss.description}")
            console.print(f"     [dim]→ {iss.suggestion}[/dim]")
        console.print()

    if fix:
        fixed = sum(1 for i in result.issues if i.auto_fixable)
        if fixed:
            console.print(f"[green]Auto-fixed {fixed} issue(s).[/green]")


# ── watch ─────────────────────────────────────────────────────────────────────


@cli.command()
@click.option("--vault", "vault_str", envvar="OLW_VAULT", required=True)
@click.option(
    "--auto-approve", is_flag=True, help="Publish drafts immediately without manual review"
)
def watch(vault_str, auto_approve):
    """Watch raw/ for new/changed notes → auto-ingest + compile."""
    from .git_ops import git_commit
    from .indexer import append_log, generate_index
    from .pipeline.compile import approve_drafts, compile_concepts
    from .pipeline.ingest import ingest_note as _ingest_note
    from .watcher import watch as _watch

    config = _load_config(vault_str)
    client, db = _load_deps(config)

    debounce = config.pipeline.watch_debounce
    console.print(f"[bold]Watching[/bold] {config.raw_dir}  (debounce={debounce:.0f}s)")
    console.print("[dim]Ctrl+C to stop.[/dim]\n")

    def _on_event(paths: list[str]) -> None:
        md_paths = [p for p in paths if p.endswith(".md")]
        if not md_paths:
            return

        console.rule(f"[dim]{len(md_paths)} file(s) changed[/dim]")

        # Ingest each changed file
        ingested = 0
        for raw_path in md_paths:
            p = Path(raw_path)
            if not p.exists() or config.raw_dir not in p.parents:
                continue
            try:
                result = _ingest_note(path=p, config=config, client=client, db=db)
                if result is not None:
                    ingested += 1
                    console.print(f"  [green]✓[/green] ingested {p.name}")
                else:
                    console.print(f"  [dim]~ skipped {p.name} (duplicate/unchanged)[/dim]")
            except Exception as exc:
                console.print(f"  [red]✗[/red] ingest failed {p.name}: {exc}")

        if not ingested:
            return

        generate_index(config, db)
        append_log(config, f"watch | ingested {ingested} note(s)")

        # Compile
        try:
            draft_paths, failed = compile_concepts(config=config, client=client, db=db)
        except Exception as exc:
            console.print(f"[red]Compile error:[/red] {exc}")
            return

        if draft_paths:
            console.print(f"  [green]✓[/green] {len(draft_paths)} draft(s) compiled")

        if failed:
            failed_str = ", ".join(failed)
            console.print(f"  [yellow]![/yellow] {len(failed)} concept(s) failed: {failed_str}")

        if auto_approve and draft_paths:
            published = approve_drafts(config, db, draft_paths)
            generate_index(config, db)
            append_log(config, f"approve | {len(published)} articles published")
            if config.pipeline.auto_commit:
                git_commit(
                    config.vault, f"watch: {len(published)} articles", paths=["wiki/", ".olw/"]
                )
            console.print(f"  [green]✓[/green] {len(published)} article(s) published")
        elif draft_paths:
            console.print("  [dim]Run [bold]olw approve --all[/bold] to publish drafts.[/dim]")

    _watch(config=config, client=client, db=db, on_event=_on_event)

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install (editable, auto-detects uv/pip)
python install.py

# Dependencies
uv sync --group dev

# Tests (all offline, no Ollama required)
uv run pytest
uv run pytest tests/test_ingest.py::test_function_name -v   # single test
uv run pytest --cov=src/obsidian_llm_wiki                   # with coverage

# Lint & format
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
uv run ruff format src/ tests/        # auto-fix

# Smoke tests (user preference: run against LM Studio with gemma4:e4b)
# Requires LM Studio server at http://localhost:1234/v1 with gemma4:e4b loaded.
PROVIDER=lm_studio FAST_MODEL=gemma4:e4b HEAVY_MODEL=gemma4:e4b bash scripts/smoke_test.sh
PROVIDER=lm_studio FAST_MODEL=gemma4:e4b HEAVY_MODEL=gemma4:e4b bash scripts/compare_smoke.sh
# If LM Studio rejects the alias, use the exact loaded id it reports, e.g.
# PROVIDER=lm_studio FAST_MODEL=google/gemma-4-e4b HEAVY_MODEL=google/gemma-4-e4b bash scripts/smoke_test.sh
# The main smoke can take 15+ minutes with gemma4:e4b; use a long command timeout.
```

## Architecture

Three-stage local LLM pipeline turning Obsidian raw notes into a synthesized wiki via Ollama.

**Data flow:** `raw/*.md` → **ingest** (fast model → AnalysisResult) → **compile** (heavy model → SingleArticle drafts) → **approve** (publish to `wiki/`)

**Vault structure:** `raw/` (immutable user notes), `wiki/` (published articles), `wiki/.drafts/`, `wiki/sources/`, `wiki/queries/`, `.olw/state.db`

### Key modules (`src/obsidian_llm_wiki/`)

- `cli.py` — Click-based CLI entry point, all commands registered here
- `config.py` / `global_config.py` — Two-tier config: per-vault `wiki.toml` + user-level `~/.config/olw/config.toml`
- `models.py` — Pydantic v2 schemas for LLM I/O (AnalysisResult, SingleArticle, PageSelection, QueryAnswer) and internal state (RawNoteRecord, WikiArticleRecord)
- `state.py` — SQLite state DB tracking note lifecycle (new → ingested → compiled → published/failed), concepts, and articles
- `ollama_client.py` — Thin httpx wrapper around Ollama HTTP API (no langchain)
- `structured_output.py` — 3-tier JSON extraction fallback: native `format=json` → regex extraction → retry with error feedback
- `vault.py` — Frontmatter parsing (python-frontmatter), wikilink extraction, atomic writes
- `indexer.py` — Generates `wiki/index.md` and append-only operation log
- `git_ops.py` — Auto-commit with `[olw]` prefix, safe undo via `git revert`
- `watcher.py` — Debounced file watcher (watchdog + threading.Timer)

### Pipeline stages (`pipeline/`)

- `ingest.py` — Fast model analyzes raw notes, extracts concepts, creates source summary pages
- `compile.py` — Default: concept-driven (one article per concept, incremental). Legacy: two-step LLM planning (`--legacy`). Manual-edit protection via content_hash comparison
- `query.py` — Index-based page routing (no embeddings), fast model selects pages, heavy model answers
- `lint.py` — Static health checks (orphans, broken links, stale articles, missing frontmatter)

## Conventions

- **Two LLM tiers:** fast model (gemma4:e4b, 8K ctx) for analysis/routing, heavy model (qwen2.5:14b, 16K ctx) for writing. For manual/smoke testing, use gemma4:e4b for both fast and heavy
- **Pydantic models for LLM output:** Keep schemas small and flat (no nested lists of objects) for 4B model reliability. JSON schema is injected into system prompts
- **Atomic writes:** `vault.atomic_write()` uses temp file + rename for crash safety
- **Content hashing:** SHA256 on note body (excluding frontmatter) for dedup and manual-edit detection
- **Concept normalization:** Case-insensitive matching against existing canonical names during ingest
- **Config loading order:** `--vault` flag → `OLW_VAULT` env var → global config default vault → error
- **Git safety:** All auto-operations use `[olw]`-prefixed commits; undo uses `git revert` (never destructive)
- **Error handling:** LLM failures log + mark note as "failed" + continue (no crash). Config loading returns None on failure (fail open)
- **Testing:** All 200 tests mock OllamaClient via MagicMock, use tmp_path vaults and in-memory SQLite. Fixtures in `tests/fixtures/`. Integration tests marked with `@pytest.mark.integration`
- **Python 3.11+:** Uses `tomllib` (stdlib), `from __future__ import annotations`, full type hints
- **Ruff config:** rules E, F, I, UP; line-length 100

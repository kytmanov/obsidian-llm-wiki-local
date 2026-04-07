# Development Plan — Staged, Testable & Deployable

Each stage below is **independently testable** (offline unit tests pass) and
**independently deployable** (the CLI works for the features delivered so far).

Tests are cumulative — every stage keeps all previous tests green.

---

## Stage 1 — Core Foundation

**Goal:** Vault utilities, configuration, state DB, and Pydantic models.

**Modules delivered:**
- `vault.py` — parse/write notes, wikilinks, chunking, atomic writes
- `config.py` — TOML-based configuration, derived paths
- `state.py` — SQLite state tracking (raw notes, concepts, articles)
- `models.py` — Pydantic schemas for LLM I/O and internal records

**Testable:** `pytest tests/test_vault.py tests/test_state.py tests/test_config.py`

**Deployable:** `olw init` creates vault structure (no LLM required).

**Acceptance criteria:**
- All vault utilities round-trip correctly
- StateDB CRUD, deduplication, and migration work
- Config loads from TOML with defaults and overrides
- `olw init` creates correct directory structure

---

## Stage 2 — Ingest Pipeline

**Goal:** Analyze raw notes, extract concepts, create source summaries.

**Modules delivered (cumulative):**
- `ollama_client.py` — HTTP wrapper for Ollama API
- `structured_output.py` — three-tier JSON extraction from LLMs
- `pipeline/ingest.py` — note analysis, concept extraction, source pages
- `indexer.py` — index and log generation

**Testable:** `pytest tests/test_ingest.py tests/test_structured_output.py tests/test_ollama_client.py tests/test_indexer.py`

**Deployable:** `olw ingest --all` processes raw notes (requires Ollama).

**Acceptance criteria:**
- Ingest extracts concepts and writes source summary pages
- Duplicate detection works (content hash)
- Structured output fallback tiers function correctly
- OllamaClient methods handle errors gracefully
- Index generation produces valid wikilinks

---

## Stage 3 — Compile & Approve Pipeline

**Goal:** Generate wiki articles from extracted concepts, draft review workflow.

**Modules delivered (cumulative):**
- `pipeline/compile.py` — concept-driven compilation, draft management
- `git_ops.py` — auto-commit, safe undo via git revert

**Testable:** `pytest tests/test_compile.py tests/test_git_ops.py`

**Deployable:** `olw compile`, `olw approve`, `olw reject`, `olw undo` work end-to-end.

**Acceptance criteria:**
- Compile creates drafts with correct frontmatter
- Manual-edit protection skips edited articles
- Approve moves drafts to wiki, updates index
- Git commit/undo lifecycle works correctly
- Dry-run writes nothing

---

## Stage 4 — Lint & Query

**Goal:** Wiki health checks and RAG-powered Q&A.

**Modules delivered (cumulative):**
- `pipeline/lint.py` — orphan detection, broken links, health scoring
- `pipeline/query.py` — index-routed Q&A with save support

**Testable:** `pytest tests/test_lint.py tests/test_query.py`

**Deployable:** `olw lint`, `olw lint --fix`, `olw query`, `olw query --save` work.

**Acceptance criteria:**
- Lint detects orphans, broken links, missing frontmatter, stale articles
- Lint --fix adds missing fields
- Query routes through index and returns grounded answers
- Query --save persists to wiki/queries/

---

## Stage 5 — CLI & Watcher

**Goal:** Full CLI integration, file watcher daemon, end-to-end workflow.

**Modules delivered (cumulative):**
- `cli.py` — Click-based CLI with all commands
- `watcher.py` — filesystem watcher with debouncing

**Testable:** `pytest tests/test_cli.py tests/test_watcher.py`

**Deployable:** All `olw` commands work. `olw watch` auto-processes new notes.

**Acceptance criteria:**
- All CLI commands produce correct exit codes
- CLI error handling (missing vault, missing Ollama) works
- Watcher debounces rapid events and filters non-.md files
- `olw doctor`, `olw status`, `olw clean` work correctly

---

## Smoke Test Coverage

`scripts/smoke_test.sh` covers the full end-to-end workflow against a live
Ollama instance. It is structured by stage:

| Section              | Stage | What it tests                              |
|----------------------|-------|--------------------------------------------|
| Prerequisites        | —     | uv, Ollama reachable, models present       |
| Install + Init       | 1     | `olw init`, vault structure, wiki.toml     |
| Doctor               | 1     | `olw doctor` health check                  |
| Ingest               | 2     | `olw ingest --all`, source pages, index    |
| Compile              | 3     | `olw compile`, drafts created              |
| Approve + Undo       | 3     | `olw approve --all`, git commits, undo     |
| Incremental compile  | 3     | Add 3rd note, only new concepts compiled   |
| Manual edit protect  | 3     | Edited article skipped in recompile        |
| Duplicate detection  | 2     | Duplicate raw note skipped                 |
| Lint                 | 4     | `olw lint`, health score, `--fix`          |
| Query                | 4     | `olw query`, `--save`                      |
| Retry failed         | 3     | `olw compile --retry-failed`               |
| Status               | 5     | `olw status` output                        |
| Clean                | 5     | `olw clean` resets state                   |

---

## Test Coverage Targets

| Module               | Before | After  | Target | Strategy                              |
|----------------------|--------|--------|--------|---------------------------------------|
| `vault.py`           |   80%  |  80%   |  90%+  | Add edge-case tests                   |
| `config.py`          |   83%  | 100%   | 100%   | ✅ Complete                           |
| `state.py`           |   92%  |  93%   |  95%+  | Add migration, edge-case tests        |
| `models.py`          |  100%  | 100%   | 100%   | ✅ Complete                           |
| `ollama_client.py`   |   31%  |  89%   |  85%+  | ✅ Complete                           |
| `structured_output.py`|  92%  |  92%   |  95%+  | Already good, minor gaps              |
| `indexer.py`         |   69%  |  92%   |  90%+  | ✅ Complete                           |
| `git_ops.py`         |    0%  |  93%   |  85%+  | ✅ Complete                           |
| `pipeline/ingest.py` |   85%  |  85%   |  90%+  | Already good                          |
| `pipeline/compile.py`|   81%  |  81%   |  85%+  | Already good                          |
| `pipeline/lint.py`   |   86%  |  86%   |  90%+  | Already good                          |
| `pipeline/query.py`  |   92%  |  92%   |  95%+  | Already good                          |
| `cli.py`             |    0%  |  53%   |  50%+  | ✅ Complete                           |
| `watcher.py`         |   79%  |  79%   |  85%+  | Already good                          |
| **TOTAL**            | **58%**| **79%**|        | **+21% overall improvement**          |

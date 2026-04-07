# Development Plan — Staged, Testable & Deployable

Each stage below is **independently testable** (offline unit tests pass) and
**independently deployable** (the CLI works for the features delivered so far).

Tests are cumulative — every stage keeps all previous tests green.

---

## Architecture Note: CLI is Cross-Cutting

`cli.py` is the entry point for **all** stages — `olw init` (Stage 1), `olw
ingest` (Stage 2), `olw compile`/`olw approve` (Stage 3), etc. It is **not**
a Stage 5 deliverable. Each stage adds CLI commands as needed. `watcher.py` is
the only genuinely Stage 5 module.

The "Deployable" line in each stage refers to the CLI commands that work after
that stage's modules are complete.

---

## Stage 1 — Core Foundation (✅ Done)

**Goal:** Vault utilities, configuration, state DB, and Pydantic models.

**Modules delivered:**
- `vault.py` — parse/write notes, wikilinks, chunking, atomic writes
- `config.py` — TOML-based configuration, derived paths
- `state.py` — SQLite state tracking (raw notes, concepts, articles)
- `models.py` — Pydantic schemas for LLM I/O and internal records
- `cli.py` (partial) — `olw init`, `olw doctor`, `olw status`, `olw clean`

**Testable:** `pytest tests/test_vault.py tests/test_state.py tests/test_config.py tests/test_cli.py`

**Deployable:** `olw init` creates vault structure (no LLM required).

**Acceptance criteria:**
- All vault utilities round-trip correctly
- StateDB CRUD, deduplication, and migration work
- Config loads from TOML with defaults and overrides
- `olw init` creates correct directory structure

---

## Stage 2 — Ingest Pipeline (✅ Done)

**Goal:** Analyze raw notes, extract concepts, create source summaries.

**Modules delivered (cumulative):**
- `ollama_client.py` — HTTP wrapper for Ollama API
- `structured_output.py` — three-tier JSON extraction from LLMs
- `pipeline/ingest.py` — note analysis, concept extraction, source pages
- `indexer.py` — index and log generation
- `cli.py` (partial) — `olw ingest`

**Testable:** `pytest tests/test_ingest.py tests/test_structured_output.py tests/test_ollama_client.py tests/test_indexer.py`

**Deployable:** `olw ingest --all` processes raw notes (requires Ollama).

**Acceptance criteria:**
- Ingest extracts concepts and writes source summary pages
- Duplicate detection works (content hash)
- Structured output fallback tiers function correctly
- OllamaClient methods handle errors gracefully
- Index generation produces valid wikilinks

---

## Stage 3 — Compile & Approve Pipeline (✅ Done)

**Goal:** Generate wiki articles from extracted concepts, draft review workflow.

**Modules delivered (cumulative):**
- `pipeline/compile.py` — concept-driven compilation, draft management
- `git_ops.py` — auto-commit, safe undo via git revert
- `cli.py` (partial) — `olw compile`, `olw approve`, `olw reject`, `olw undo`

**Testable:** `pytest tests/test_compile.py tests/test_git_ops.py`

**Deployable:** `olw compile`, `olw approve`, `olw reject`, `olw undo` work end-to-end.

**Acceptance criteria:**
- Compile creates drafts with correct frontmatter
- Manual-edit protection skips edited articles
- Approve moves drafts to wiki, updates index
- Git commit/undo lifecycle works correctly
- Dry-run writes nothing

**Legacy mode:** `compile_notes` (the `--legacy` flag) is kept as a fallback.
Decision: deprecate with a log warning and remove in next major version.

---

## Stage 4 — Lint & Query (✅ Done)

**Goal:** Wiki health checks and RAG-powered Q&A.

**Modules delivered (cumulative):**
- `pipeline/lint.py` — orphan detection, broken links, health scoring
- `pipeline/query.py` — index-routed Q&A with save support
- `cli.py` (partial) — `olw lint`, `olw query`

**Testable:** `pytest tests/test_lint.py tests/test_query.py`

**Deployable:** `olw lint`, `olw lint --fix`, `olw query`, `olw query --save` work.

**Acceptance criteria:**
- Lint detects orphans, broken links, missing frontmatter, stale articles
- Lint --fix adds missing fields
- Query routes through index and returns grounded answers
- Query --save persists to wiki/queries/

---

## Stage 5 — Watcher (✅ Done)

**Goal:** File watcher daemon for auto-processing new notes.

**Modules delivered (cumulative):**
- `watcher.py` — filesystem watcher with debouncing
- `cli.py` (partial) — `olw watch`

**Testable:** `pytest tests/test_watcher.py`

**Deployable:** `olw watch` auto-processes new notes dropped into `raw/`.

**Acceptance criteria:**
- Watcher debounces rapid events and filters non-.md files
- `olw watch --auto-approve` skips draft review

---

## Stage 6 — Hardening (✅ Done)

**Goal:** Close coverage gaps, add error-path tests, enforce quality gates in CI.

### 6a. Close coverage gaps (✅ Done)

| Module                | Before  | After  | Target | Status |
|-----------------------|---------|--------|--------|--------|
| `vault.py`            |   80%   | 100%   |  90%+  | ✅     |
| `state.py`            |   93%   | 100%   |  95%+  | ✅     |
| `structured_output.py`|   92%   | 100%   |  95%+  | ✅     |
| `pipeline/ingest.py`  |   85%   | 100%   |  90%+  | ✅     |
| `pipeline/compile.py` |   81%   |  97%   |  85%+  | ✅     |
| `pipeline/lint.py`    |   86%   | 100%   |  90%+  | ✅     |
| `pipeline/query.py`   |   92%   | 100%   |  95%+  | ✅     |
| `watcher.py`          |   79%   |  99%   |  85%+  | ✅     |
| `cli.py`              |   53%   |  53%   |  65%+  | Deferred (CLI is cross-cutting; testing requires Click test runner) |

### 6b. Add error-path tests (✅ Done)

Created `tests/test_error_paths.py` (17 tests) covering:
- ✅ Ollama returns invalid JSON → StructuredOutputError after retries exhausted
- ✅ Ollama timeout during ingest → note marked `failed`, pipeline continues
- ✅ `raw/` contains empty files, binary files, non-UTF-8 files → skip with warning
- ✅ Disk full during atomic write → no partial files left behind
- ✅ Frontmatter with unexpected types → parse_note still works
- ✅ Concurrent StateDB access → safe (check_same_thread=False)
- ✅ Compile/query/lint with empty wiki → no crash, helpful output

### 6c. Add contract tests between stages (✅ Done)

Created `tests/test_contracts.py` (11 tests) covering:
- ✅ Ingest→Compile contract: concepts+raw_notes match what compile reads
- ✅ Compile→Approve contract: draft frontmatter → approve moves to wiki
- ✅ Approve→Lint contract: published article passes lint
- ✅ Approve→Query contract: published articles findable by query
- ✅ Full pipeline flow: ingest → compile → approve → lint → query
- ✅ State DB round-trip: all fields preserved through serialization
- ✅ WikiArticleRecord sources JSON serialization round-trip

### 6d. CI enforcement (✅ Done)

Updated `.github/workflows/ci.yml`:
- ✅ Added `--cov-fail-under=80` to prevent coverage regressions
- ✅ Added `--cov-report=xml` and artifact upload for PR review
- Deferred: nightly Ollama smoke test (requires containerized Ollama setup)

### 6e. Smoke test improvements (TODO)

Fix false-confidence issues in `scripts/smoke_test.sh`:
- Replace loose `grep -qi 'retry\|failed\|not found\|re-ingest'` patterns with
  exact expected strings — the current checks match error messages and success
  messages alike
- Add error-path smoke sections: feed an empty `.md` file, a binary file, a
  file with only frontmatter — verify graceful handling
- Capture and report wall-clock timings for ingest/compile/query sections as
  performance baselines

---

## Stage 7 — Observability & Debugging (✅ Done)

**Goal:** Make pipeline failures debuggable without reading source code.

### 7a. Structured logging (✅ Done)

- ✅ Added `--verbose` / `-v` flag to CLI root (sets `logging.DEBUG`)
- ✅ Added `--quiet` / `-q` flag (sets `logging.WARNING`)
- ✅ LLM prompt/response pairs logged at DEBUG level in `structured_output.py`
- ✅ Verbose mode includes timestamps in log format

### 7b. State DB versioning (✅ Done)

- ✅ Added `schema_version` table to `state.db` (single row: `version INTEGER`)
- ✅ `StateDB.__init__` checks version, runs only needed migrations, guards future versions
- ✅ Test: v0 DB (no summary/quality) → migrated correctly, data preserved
- ✅ Test: future version → `RuntimeError` with clear message
- ✅ Test: fresh DB gets current version; reopen preserves version

### 7c. Deprecate legacy compile mode (✅ Done)

- ✅ `compile_notes()` emits `DeprecationWarning` on every call
- ✅ Test verifies warning is emitted
- ✅ `README.md` has no `--legacy` references (none existed)

---

## Stage 8 — Scale & Portability (TODO)

**Goal:** Prepare for larger vaults and non-Linux users.

### 8a. Performance baselines

- Document expected timings: `olw ingest` < 30s per note (4B model),
  `olw compile` < 60s per concept (14B model), `olw query` < 10s
- Add `--timings` flag to CLI that prints wall-clock time for each operation
- Smoke test records timings and fails if >3x baseline (regression detection)

### 8b. Cross-platform testing

- Add macOS (`macos-latest`) to CI matrix — `watchdog` uses kqueue on macOS vs
  inotify on Linux, and path separator handling differs
- Document Windows limitations (no `olw watch` support without WSL; `shasum` in
  smoke test is bash-only) or add Windows CI with equivalent checks

### 8c. Scaling considerations

- Document the ~100-source-note practical ceiling (README already hints at this)
- Plan for future: optional `chromadb` or `sqlite-vec` backend behind a feature
  flag for vaults with 100+ notes (not needed now, but reserve the config key
  `[rag] backend = "index"` to avoid breaking changes later)
- `state.db` concurrent access: add WAL mode (`PRAGMA journal_mode=WAL`) to
  prevent locking issues when watcher and manual CLI commands run simultaneously

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
| Error paths          | 6     | Empty file, binary file → graceful skip    |
| Timings              | 8     | Wall-clock per section, fail if >3x baseline |

---

## Test Coverage Targets

| Module               | Before | Current | Target | Status                                |
|----------------------|--------|---------|--------|---------------------------------------|
| `vault.py`           |   80%  |  100%   |  90%+  | ✅ Done                              |
| `config.py`          |   83%  |  100%   | 100%   | ✅ Done                              |
| `state.py`           |   92%  |  100%   |  95%+  | ✅ Done                              |
| `models.py`          |  100%  |  100%   | 100%   | ✅ Done                              |
| `ollama_client.py`   |   31%  |   89%   |  85%+  | ✅ Done                              |
| `structured_output.py`|  92%  |  100%   |  95%+  | ✅ Done                              |
| `indexer.py`         |   69%  |   92%   |  90%+  | ✅ Done                              |
| `git_ops.py`         |    0%  |   93%   |  85%+  | ✅ Done                              |
| `pipeline/ingest.py` |   85%  |  100%   |  90%+  | ✅ Done                              |
| `pipeline/compile.py`|   81%  |   97%   |  85%+  | ✅ Done                              |
| `pipeline/lint.py`   |   86%  |  100%   |  90%+  | ✅ Done                              |
| `pipeline/query.py`  |   92%  |  100%   |  95%+  | ✅ Done                              |
| `cli.py`             |    0%  |   53%   |  65%+  | Deferred (cross-cutting Click tests)  |
| `watcher.py`         |   79%  |   99%   |  85%+  | ✅ Done                              |
| **TOTAL**            | **58%**|  **86%**| **85%+**| ✅ **Target reached**                |

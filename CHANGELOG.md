# Changelog

## [0.2.0] - 2026-04-11

### Highlights

**v0.2 turns the wiki into a self-improving system.** Drafts can be rejected with feedback, and the next compile automatically addresses it. A new review interface makes approving and rejecting drafts fast. Maintenance tooling keeps the wiki healthy over time.

### New Features

- **Rejection feedback loop** — Reject a draft with a reason (`olw review`), and the next compile injects that feedback into the prompt. Concepts rejected 5+ times are blocked and surfaced in `olw status`.

- **Draft review interface** (`olw review`) — Interactive numbered menu for reviewing drafts. Approve, reject with feedback, edit in `$EDITOR`, diff against published version, or view rejection diff vs previous attempt.

- **Pipeline orchestrator** (`olw run`) — Single command runs the full ingest → compile → lint → approve sequence. Handles selective recompile (only concepts linked to changed notes), transient failure retry, and optional auto-approve.

- **Self-maintenance** (`olw maintain`) — Detects broken wikilinks and auto-creates stub articles for them. Reports orphan articles, near-duplicate concept suggestions, and source quality warnings.

- **Inline draft annotations** — Low-confidence drafts get HTML comment annotations (`<!-- olw-auto: ... -->`) flagging single-source articles, low-quality sources, or uncertain content. Annotations are stripped automatically on approve.

- **Long-note chunked ingest** — Notes larger than the context window are split into chunks, analyzed in parallel (with `ingest_parallel = true`), and merged. Enables processing notes of any length without truncation.

- **Pipeline concurrency lock** — `olw watch` and manual `olw run` no longer race. Advisory file lock (`fcntl.flock`) prevents concurrent pipeline runs from corrupting state.

### Improvements

- Per-concept compile timings in `olw run` output (avg + top-3 slowest)
- `olw setup` shows installed version in header
- Structured output uses fill-in templates instead of raw JSON Schema — reduces model confusion and schema-echo failures
- Ollama requests now set `num_predict` to prevent mid-article JSON truncation
- Schema versioning added — future DB migrations are ordered and transactional

### Bug Fixes

- `httpx.ReadTimeout` during compile now caught and classified as a transient failure (concept is retried, pipeline continues)
- Re-approving a previously published article no longer raises a UNIQUE constraint violation
- `reject_draft` path resolution fixed on macOS

## [0.1.3] - 2026-04-08

- CI: add PyPI publish job to release workflow

## [0.1.2] - 2026-04-08

- Fix release script — bump version via PR branch, push tag after merge

## [0.1.1] - 2026-04-07

- Initial public release

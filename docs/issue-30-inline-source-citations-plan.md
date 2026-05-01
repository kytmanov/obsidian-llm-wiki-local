# Issue #30: Inline Source Citations Status And Follow-Up

## Status

Phase 1 of inline source citations is already implemented on this branch.

This document replaces the earlier implementation plan, which assumed the feature had not been built yet. That is no longer true. An agent should treat the core feature as shipped-and-tested, then focus only on follow-up gaps or refinements.

## What Exists Today

The current implementation already provides the main Phase 1 behavior behind a vault-local opt-in flag.

### Config And CLI

- `pipeline.inline_source_citations` exists and defaults to `false`: `src/obsidian_llm_wiki/config.py`
- `pipeline.source_citation_style` exists with:
  - `legend-only` (default)
  - `inline-wikilink`
- `default_wiki_toml()` renders the option
- `olw config inline-source-citations on|off|status` exists: `src/obsidian_llm_wiki/cli.py`
- Global setup can seed new vault defaults via `experimental_inline_source_citations`: `src/obsidian_llm_wiki/global_config.py`

### Compile Pipeline

- Source refs are built deterministically as `S1`, `S2`, ... via `SourceRef` and `_build_source_refs()`: `src/obsidian_llm_wiki/pipeline/compile.py`
- Gathered source material is labeled with source ids via `_gather_sources()` when citations are enabled
- Prompt instructions are gated by `inline_source_citations` in `_write_concept_prompt()`
- Draft writing rewrites citation markers via `_rewrite_citation_markers()`
- Two citation output modes exist:
  - `legend-only`: prose stays graph-quiet as `[S1](#Sources)`
  - `inline-wikilink`: prose becomes `([[sources/...|S1]])`
- `## Sources` includes a legend when inline citations are enabled
- `## See Also` explicitly excludes `sources/...` links

### Query And Lint

- Query resolves explicit `sources/...` targets in `_find_page()`: `src/obsidian_llm_wiki/pipeline/query.py`
- Lint accepts valid `[[sources/...]]` links and flags missing ones: `tests/test_lint.py`
- Lint repair normalizes plain `[S1]` citations to `[S1](#Sources)` and cleans linked legend labels

### Tests And Docs

- Config coverage exists in `tests/test_config.py` and `tests/test_setup.py`
- Compile coverage exists in `tests/test_compile_v2.py`
- Query coverage exists in `tests/test_query.py`
- Lint coverage exists in `tests/test_lint.py`
- User docs already mention setup and behavior: `README.md`
- Smoke script support already exists via `INLINE_SOURCE_CITATIONS=1`: `scripts/smoke_test.sh`

## Current Behavior

When `inline_source_citations = false`:

- compile behaves as before
- article provenance remains in `## Sources`
- no citation-specific prompt instructions are added

When `inline_source_citations = true` and `source_citation_style = "legend-only"`:

- the model is asked to emit `[S1]`-style markers
- valid markers are rewritten to `[S1](#Sources)`
- `## Sources` contains the source legend and the actual `[[sources/...]]` links
- inline citations do not create extra graph edges by default

When `inline_source_citations = true` and `source_citation_style = "inline-wikilink"`:

- valid markers are rewritten inline to `([[sources/...|S1]])`
- `## Sources` still contains the legend

## Why The Previous Plan Was Wrong

The older version of this document described several items as future work even though they already exist:

- opt-in config flag
- CLI toggle/status support
- deterministic source-id registry
- prompt gating
- citation marker rewriting
- `## Sources` legend formatting
- `## See Also` filtering for source links
- query resolution for `sources/...`
- smoke-script enablement

Following that older plan would cause duplicate implementation work and risk regressions in behavior that is already covered by tests.

## Remaining Work

The remaining work is no longer "implement Phase 1". It is follow-up validation and polish.

### 1. Validate Real-Model Reliability

Goal:
Confirm that real backends, especially smaller local models such as `gemma4:e4b`, produce useful citation markers often enough for the feature to be worth enabling.

Work:

- run smoke tests with `INLINE_SOURCE_CITATIONS=1`
- measure how often models emit valid `[S1]` markers
- inspect whether prompt instructions degrade prose quality, brevity, or wikilink quality
- compare `legend-only` versus `inline-wikilink` if needed

Suggested commands:

```bash
PROVIDER=lm_studio FAST_MODEL=gemma4:e4b HEAVY_MODEL=gemma4:e4b INLINE_SOURCE_CITATIONS=1 bash scripts/smoke_test.sh
PROVIDER=lm_studio FAST_MODEL=gemma4:e4b HEAVY_MODEL=gemma4:e4b INLINE_SOURCE_CITATIONS=1 bash scripts/compare_smoke.sh
```

Acceptance criteria:

- compile succeeds with citations enabled
- no citation-related broken links are introduced
- generated prose quality remains acceptable
- at least some drafts contain valid citation output under representative prompts

### 2. Decide Whether Default Style Should Stay `legend-only`

Status:
Current default is `legend-only`, which is intentionally graph-quiet and already documented in `README.md`.

Open question:
Whether `inline-wikilink` should remain an expert opt-in only, or whether any additional UX/docs changes are needed to make the tradeoff clearer.

This is a product decision, not missing core implementation.

### 3. Add Any Missing E2E Assertions Discovered During Smoke Runs

If smoke validation exposes gaps, add the smallest offline regression coverage needed. Possible examples:

- end-to-end assertions around approved articles containing repaired citations
- targeted tests for malformed mixed valid/invalid marker output from smaller models
- additional lint fix coverage if new malformed patterns appear in practice

Do not add speculative tests unless smoke runs reveal a concrete missing case.

## Non-Goals

These items are still out of scope for Phase 1:

- exact raw-note excerpt linking
- block-reference citations
- new nested LLM output schemas
- making article compilation fail when citations are absent or malformed

## Guidance For Future Agents

If continuing work on issue #30:

1. Start from the current implementation, not from scratch.
2. Read the existing tests before changing compile behavior.
3. Prefer extending current helpers over adding parallel citation systems.
4. Treat smoke validation and quality tuning as the main next step.
5. Preserve current default behavior when the feature flag is off.

## Verification Shortlist

Before merging follow-up changes related to inline citations, run:

```bash
uv run pytest tests/test_config.py tests/test_setup.py tests/test_compile_v2.py tests/test_lint.py tests/test_query.py
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
```

Run real-model smoke validation separately when the environment is available.

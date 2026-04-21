# Changelog

## [0.6.0] - 2026-04-21

### Highlights

**v0.6.0 adds `olw compare`, a safe way to preview a model or provider switch before changing your vault config.** Instead of benchmarking abstract prompt quality, `olw compare` rebuilds isolated previews from your current vault's `raw/` notes, compares the current setup against one challenger, and tells you whether to switch, keep the current config, or inspect the results manually.

### New Features

- **`olw compare`** — compare the active vault config against a single challenger model/provider configuration using the same source notes. The command produces a recommendation-first report and shows the exact `wiki.toml` config needed to reproduce the challenger setup.

### Changes

- **Safer preview runs** — compare never mutates the active vault outside `.olw/compare/`, rejects unsafe path/symlink inputs, and requires explicit acknowledgement before sending vault content to a cloud provider.
- **Better compare guidance** — reports now include clearer verdicts, config-change instructions, and artifact output designed for manual review instead of benchmark-style winner framing.
- **Docs and smoke coverage** — README and smoke tests now cover the new compare workflow, including LM Studio / OpenAI-compatible local backends.

## [0.5.1] - 2026-04-19

### Highlights

**v0.5.1 hardens the pipeline against a common quirk of small local models.** When the fast model returns `concepts` as a flat list of strings instead of `{name, aliases}` objects, ingest no longer fails — the payload is coerced transparently and the note proceeds. The root cause, an over-literal JSON template passed to the model, is also fixed so every `list[Model]` field benefits from the same resilience.

### Bug Fixes

- **String-concept tolerance (#32)** — `AnalysisResult` accepts a flat list of concept strings from the LLM and wraps them into `Concept` objects with empty aliases. Mixed payloads (some strings, some dicts) also validate. A `log.debug` line fires when coercion triggers, so the signal stays observable for prompt-tuning.
- **Structured-output template renders nested objects** — the fill-in JSON example passed to the fast model now expands `list[Concept]`, `list[ArticlePlan]`, and `list[LintIssue]` into their real shape, including enum discriminator values. Previously the model saw only the array's description string and sometimes produced wrong item types. `Optional[...]` fields still carry their outer description through `anyOf`.

### Changes

- **Cleaner error when the vault path is missing** — `olw <cmd>` now prints a short, actionable message pointing at `olw init` or `olw setup` instead of surfacing a raw config-loading traceback.
- **Smoke test**: new pre-LLM "Structured output resilience" block pins the validator + template + `request_structured` end-to-end against a stub client, so the class of failure above is caught in the installed artifact without requiring a running model.
- README: concept-block rejection / `olw unblock` section trimmed and clarified.

## [0.5.0] - 2026-04-18

### Highlights

**v0.5 makes the wiki resilient to the natural way humans refer to concepts.** When you write about "Program Counter" once and later mention "PC", `olw` now recognizes they're the same thing. Aliases are extracted at ingest, stored alongside articles, and used everywhere the pipeline resolves a concept name — so broken links to common abbreviations get repaired automatically, and queries using either form land on the right article.

### New Features

- **Concept aliases** — during ingest, the fast model extracts short aliases for each concept (e.g. "Program Counter (PC)" → alias `PC`). Aliases are stored in the state DB and written into each article's `aliases:` frontmatter, so Obsidian picks them up for its own link suggestions.

- **Alias-aware link repair** — `olw maintain --fix` now rewrites broken `[[Alias]]` wikilinks to `[[Canonical|Alias]]` when an unambiguous alias is known. Links inside code fences are left untouched. Still-unresolvable targets fall through to stub creation as before.

- **Alias-aware queries and health checks** — `olw query` resolves page routes through the alias map when the canonical name misses, and `olw lint` counts alias forms as valid links so they don't get flagged as broken.

- **`olw status --failed`** — filter the status report to only failed notes with their error messages. Useful when triaging after a bulk ingest.

### Changes

- The auto-generated wiki index is now `wiki/index.md` (lowercase). Vaults created before v0.5 with `wiki/INDEX.md` are handled transparently — no manual rename needed.
- Smoke test suite expanded: 105 end-to-end assertions now cover lint issue-type coverage, `maintain --fix` idempotency, stub frontmatter shape, legacy-compile bit-rot, and the `status --failed` filter.

### Breaking Changes

None. Existing vaults upgrade automatically on first `olw` run: the DB gains a `concept_aliases` table, aliases are backfilled from current concept titles (deterministic, no LLM calls), and any legacy `INDEX.md` is treated like the new lowercase form.

## [0.4.0] - 2026-04-16

### Highlights

**v0.4 makes `olw` smarter about where you are and what you're writing.** Notes are now analyzed in the language they were written in, and articles are compiled in the same language — no more English-only output in a multilingual vault. You can also reject an entire batch of drafts in one go. Two usability papercuts are fixed: the tool now finds your vault automatically when you're already inside it, and the review menu works correctly in all terminals.

### New Features

- **Multi-language support** — `olw ingest` detects the language of each note and stores it. `olw compile` then writes each article in the appropriate language: the detected language if all sources agree, or the vault-wide `language` setting from `wiki.toml` if configured. French notes produce French articles, Japanese notes produce Japanese articles, and so on.

- **`olw reject --all`** — Reject every pending draft in one step with a shared feedback message. Useful when a bad prompt version or wrong model produced a full batch of unusable drafts.

- **Vault auto-detection** — Commands no longer fail when `--vault` is not passed and no default was saved in `olw setup`. If you're inside a vault directory (or any subdirectory), `olw` finds it automatically by walking up to the nearest `wiki.toml`. The "press Enter to skip" flow in setup now actually makes sense.

### Bug Fixes

- **Review menu shortcuts were invisible in some terminals** — The `olw review` action menu (`[a]pprove`, `[r]eject`, etc.) used Rich markup that silently ate the first letter of each word in terminals with limited style support, showing `pprove eject dit` instead. Prompts now render as plain text (`a=approve, r=reject, e=edit, ...`) and work correctly everywhere.

## [0.3.0] - 2026-04-15

### Highlights

**v0.3 opens the pipeline to any LLM provider.** Ollama stays the default, but you can now point `olw` at Groq, Together AI, Mistral, LM Studio, vLLM, Azure OpenAI, or any other OpenAI-compatible endpoint — local or cloud. The setup wizard gained a provider selection step, API keys are stored securely in the global config (never in `wiki.toml`), and all pipeline internals use a shared protocol so adding new providers requires no pipeline changes.

### New Features

- **Multi-provider support** — `olw setup` now presents a numbered provider menu covering 9 local runtimes (LM Studio, vLLM, llama.cpp, LocalAI, TGI, SGLang, Llamafile, Lemonade, Ollama) and 11 cloud providers (Groq, Together AI, Fireworks AI, DeepInfra, OpenRouter, Mistral AI, DeepSeek, SiliconFlow, Perplexity, xAI / Grok, Azure OpenAI) plus a free-form Custom option. Select by number or name.

- **OpenAI-compatible client** (`openai_compat_client.py`) — thin `httpx` wrapper matching the `OllamaClient` interface. Handles bearer auth, Azure `api-key` header + `?api-version=` query param, JSON mode with auto-downgrade on HTTP 400, and ordered embedding responses.

- **`[provider]` section in `wiki.toml`** — non-Ollama vaults use a `[provider]` block with `name`, `url`, `timeout`, `fast_ctx`, `heavy_ctx` (and `azure_api_version` for Azure). The `[ollama]` section still works unchanged — existing vaults need no migration.

- **API key security** — keys are stored in `~/.config/olw/config.toml` (user-private global config) and resolved at runtime from the provider-specific env var (e.g. `GROQ_API_KEY`), the generic `OLW_API_KEY` env var, or the global config `api_key` field. They are never written to `wiki.toml`.

- **`LLMClientProtocol`** — structural interface (`typing.Protocol`) that both `OllamaClient` and `OpenAICompatClient` satisfy. All pipeline stages (`ingest`, `compile`, `query`, `orchestrator`, `structured_output`) now depend on the protocol, not the concrete Ollama class.

- **`build_client()` factory** — resolves the correct client from `config.effective_provider`. Pipelines and tests need no awareness of which provider is active.

### Improvements

- `olw doctor` is now provider-aware: shows the active provider name and URL, and gives provider-appropriate hints for missing models
- `olw setup` temp client correctly passes `azure=True` and `supports_embeddings` flags so the health probe uses the right auth header
- Empty URL for `custom` or `azure` provider now exits with a clear error instead of producing cryptic HTTP failures later
- `azure_api_version` flows from `olw setup` → global config → `wiki.toml` (Azure vaults) so each vault can override the API version independently
- `embed_batch` now catches `TimeoutException` and wraps it as `LLMError` (previously only `generate` did)
- `LLMClientProtocol` declares `__enter__`/`__exit__` so type checkers recognise both clients as context managers

### Breaking Changes

None. Ollama remains the default. Existing `wiki.toml` files with `[ollama]` sections work without modification.

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

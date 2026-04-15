# Changelog

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

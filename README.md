# obsidian-llm-wiki

**Turn your raw notes into a self-maintaining, interlinked wiki — powered by a local LLM.**

Drop a markdown file into a folder. An AI reads it, extracts concepts, and either creates new wiki articles or updates existing ones with the new knowledge. Over time your wiki compounds: every note you add makes the whole smarter and more connected.

**100% local. No cloud, no API keys, no telemetry.** Just [Ollama](https://ollama.com) running on your machine.

---

## The idea (Karpathy's LLM Wiki)

This project is a practical implementation of the pattern described by Andrej Karpathy in [**"The LLM Wiki"**](https://karpathy.ai/llmwiki) — a vision for a personal knowledge base where:

> *"The LLM doesn't just store what you tell it — it synthesizes, cross-references, and keeps everything current. You add raw material; it does the bookkeeping."*

The key insight: treat your notes as **source material**, not as the final artifact. The LLM compiles them into a structured wiki that grows smarter as you add more. Unlike a chatbot that forgets, the wiki **persists and compounds**.

```
You write raw notes  →  LLM extracts concepts  →  Wiki articles created/updated
     raw/                    (automatic)                    wiki/
  quantum.md          "Qubit", "Superposition"       Qubit.md  ←──┐
  ml-basics.md        "Neural Network", "SGD"    Superposition.md  │
  physics.md          "Qubit" (again!)           Neural Network.md │
                                                      ↑  linked via [[wikilinks]]
```

The wiki lives in Obsidian, so you get the graph view, backlinks, and Dataview queries for free.

---

## Features

- **Concept-driven, incremental compilation** — each concept gets its own article, updated only when its source notes change
- **Manual edit protection** — edited an article by hand? The compiler detects the change and skips it
- **Source traceability** — every article links back to the raw notes it was built from
- **Draft review workflow** — articles land in `.drafts/` for human approval before publishing
- **File watcher** — `olw watch` auto-processes anything dropped into `raw/`
- **Wiki health checks** — `olw lint` detects orphans, broken links, stale articles (no LLM needed)
- **Query your wiki** — `olw query "what is X?"` answers from your published articles
- **Git safety net** — every auto-action is committed; `olw undo` reverts safely
- **Offline test suite** — all 139 tests run without Ollama

---

## Quick start

### 1. Install

**From PyPI** (recommended for most users):

```bash
pip install obsidian-llm-wiki
# or with uv (faster):
uv tool install obsidian-llm-wiki
```

**From source** (clone and install with one command):

```bash
git clone https://github.com/kytmanov/obsidian-llm-wiki
cd obsidian-llm-wiki
python install.py
```

`install.py` detects `uv` or falls back to `pip`, verifies the install, and tells you to run the next step.

### 2. Install and start Ollama

```bash
# Install Ollama: https://ollama.com/download
ollama pull gemma4:e4b      # fast model — analysis and routing
ollama pull qwen2.5:14b     # heavy model — article writing (optional, 7B+ recommended)
```

> **Minimal setup:** pull only `gemma4:e4b` and set both `fast` and `heavy` to it in the wizard.

### 3. Run the setup wizard

```bash
olw setup
```

An interactive wizard configures your Ollama URL, fast and heavy models, and an optional default vault path. Takes ~30 seconds.

```
╭──────────────────────────────────────────────────╮
│      obsidian-llm-wiki  ·  first run setup       │
╰──────────────────────────────────────────────────╯

  Step 1/4  Ollama connection
    Trying http://localhost:11434 …  ✓ connected

  Step 2/4  Fast model (analysis · 3–8B recommended)
    #  Model           Size
    1  gemma4:e4b      9.6 GB
    2  phi4-mini       2.5 GB
    Select (number or name) [1]: _
  ...
```

Settings are saved to `~/.config/olw/config.toml` (Mac/Linux) or `%APPDATA%\olw\config.toml` (Windows).

### 4. Set up your vault

```bash
olw init ~/my-wiki
```

This creates the folder structure and a `wiki.toml` pre-filled with your setup wizard choices.

### 5. Add some notes

Drop any `.md` files into `~/my-wiki/raw/`. Web clips, book notes, meeting notes, anything.

```
~/my-wiki/raw/
  quantum-computing.md
  ml-fundamentals.md
  physics-lecture.md
```

### 6. Run the pipeline

```bash
# Analyze notes, extract concepts
olw ingest --all

# Generate wiki articles (lands in .drafts/)
olw compile

# Review drafts, then publish
olw approve --all
```

If you set a default vault in `olw setup`, the `--vault` flag is optional. Otherwise use `--vault ~/my-wiki` or `export OLW_VAULT=~/my-wiki`.

Open `~/my-wiki` as an Obsidian vault. The graph view shows your connected wiki.

### 7. Keep it running (optional)

```bash
olw watch
# Drop a file in raw/ → ingest + compile happen automatically
```

---

## How it works

The pipeline has three stages, each using the LLM for a different purpose:

```
raw/note.md
    │
    ▼ olw ingest
    Fast LLM (3B–8B)
    • Reads note
    • Extracts concept names
    • Writes quality score + summary to state.db
    • Creates wiki/sources/Note.md (source summary page)
    │
    ▼ olw compile
    Heavy LLM (7B–14B)
    • For each concept: gathers all source notes that mention it
    • Writes a wiki article with [[wikilinks]] to related concepts
    • Lands in wiki/.drafts/ for review
    │
    ▼ olw approve
    • Moves draft to wiki/
    • Updates wiki/index.md (navigation layer)
    • Git commits the change
```

**No vector databases, no embeddings.** `wiki/index.md` acts as the routing layer for `olw query`. This keeps the setup simple and works well up to ~100 source notes.

---

## Vault structure

```
my-wiki/
├── raw/                        ← YOUR NOTES (never modified by olw)
│   ├── quantum-computing.md
│   └── ml-fundamentals.md
├── wiki/
│   ├── Quantum Computing.md    ← concept articles (flat, one per concept)
│   ├── Machine Learning.md
│   ├── sources/                ← auto-generated source summaries
│   │   ├── Quantum Computing Fundamentals.md
│   │   └── ML Fundamentals.md
│   ├── queries/                ← saved Q&A answers (olw query --save)
│   ├── .drafts/                ← pending human review
│   ├── index.md                ← auto-generated navigation + routing layer
│   └── log.md                  ← append-only operation history
├── vault-schema.md             ← LLM context: conventions for this vault
├── wiki.toml                   ← configuration
└── .olw/
    └── state.db                ← SQLite: notes, concepts, articles, hashes
```

`raw/` is immutable — `olw` never writes to it. All metadata lives in `state.db`.

---

## Configuration

`wiki.toml` (created by `olw init`):

```toml
[models]
fast = "gemma4:e4b"        # extraction, analysis, query routing
heavy = "qwen2.5:14b"     # article generation, Q&A answers
# Single-model: set heavy = fast

[ollama]
url = "http://localhost:11434"   # supports LAN: http://192.168.1.x:11434
timeout = 600
fast_ctx = 8192                  # context window for fast model (tokens)
heavy_ctx = 16384                # context window for heavy model (tokens)

[pipeline]
auto_approve = false             # true = skip draft review
auto_commit = true               # git commit after each operation
max_concepts_per_source = 8      # limit concepts extracted per note
watch_debounce = 3.0             # seconds after last file event before processing
```

### Tuning context windows

`heavy_ctx` controls how much source material the heavy model reads when writing articles (`source budget = heavy_ctx / 2` chars) and how long the generated article can be. The default of `16384` is sized for 7–14B models. **If you use a model with a large context window (e.g. `gemma4:e4b` supports 128K), increase it.**

| RAM available | Recommended `heavy_ctx` | Source budget | Notes |
|---|---|---|---|
| 8 GB | `8192` | ~4K chars | Minimum; short articles |
| 16 GB | `16384` | ~8K chars | Default |
| 16 GB+ | `32768` | ~16K chars | Recommended for `gemma4:e4b` |
| 32 GB+ | `65536` | ~32K chars | Rich multi-source articles |

`fast_ctx` (used for ingest/routing) rarely needs changing — single notes fit comfortably in 8K.

After editing `wiki.toml`, no reinstall is needed. Run `olw compile --force` to regenerate articles with the new context budget.

---

## Commands

| Command | Description |
|---------|-------------|
| `olw setup` | Interactive setup wizard (first run) |
| `olw init PATH` | Create vault structure and git repo |
| `olw init PATH --existing` | Adopt an existing Obsidian vault |
| `olw doctor` | Check Ollama, models, vault structure |
| `olw ingest --all` | Analyze all raw notes |
| `olw ingest FILE` | Analyze one note |
| `olw compile` | Generate wiki articles → `.drafts/` |
| `olw compile --retry-failed` | Retry previously failed notes |
| `olw approve --all` | Publish all drafts |
| `olw approve FILE` | Publish one draft |
| `olw reject FILE` | Discard a draft |
| `olw status` | Show pipeline state and pending drafts |
| `olw status --failed` | List failed notes with error messages |
| `olw query "question"` | Answer from your wiki |
| `olw query "..." --save` | Answer and save to `wiki/queries/` |
| `olw lint` | Health check: orphans, broken links, stale articles |
| `olw lint --fix` | Auto-fix missing frontmatter fields |
| `olw watch` | File watcher — auto-pipeline on new notes |
| `olw watch --auto-approve` | Watch + auto-publish (no manual review) |
| `olw undo` | Revert last `[olw]` git commit |
| `olw clean` | Clear state DB + wiki/, keep raw/ notes |

All commands accept `--vault PATH` or the env var `OLW_VAULT`.

---

## Model recommendations

| Role | Recommended | Minimum |
|------|-------------|---------|
| Fast (analysis + routing) | `gemma4:e4b`, `llama3.2:3b` | any 3B with JSON format |
| Heavy (article writing) | `qwen2.5:14b`, `llama3.1:8b` | any 7B |
| Single model (everything) | `llama3.1:8b`, `mistral:7b` | any 7B |

Any [Ollama model](https://ollama.com/library) with JSON format support works. The tool degrades gracefully with smaller models — they produce shorter, simpler articles but the pipeline still functions.

---

## Obsidian tips

- **Graph view** — concept pages link to source pages and each other via `[[wikilinks]]`; the graph shows how your knowledge connects
- **Dataview** — query by `status: published`, `confidence: > 0.7`, `tags: [physics]`, etc.
- **Backlinks** — every concept page shows which source pages mention it
- **Web Clipper** — save web articles directly to `raw/` (see [docs/web-clipper-setup.md](docs/web-clipper-setup.md))

---

## Running the tests

All tests are offline — no Ollama required.

```bash
git clone https://github.com/kytmanov/obsidian-llm-wiki
cd obsidian-llm-wiki
uv sync --group dev
uv run pytest
```

For the full end-to-end smoke test (requires a running Ollama instance):

```bash
OLLAMA_URL=http://localhost:11434 bash scripts/smoke_test.sh
```

---

## FAQ

**Q: I ran `olw compile` but nothing appears in Obsidian.**

Drafts land in `wiki/.drafts/` — Obsidian hides dotfolders by default so they won't show in the graph yet. Run:

```bash
olw approve --all
```

Articles move to `wiki/` and become fully visible. Open `~/my-wiki` as an Obsidian vault (**File → Open vault**) if you haven't already.

---

**Q: Compile says "2 article(s) failed: Methodology, Sprints" — what do I do?**

Failed concepts are not recorded in the DB, so simply re-running compile retries them automatically:

```bash
olw compile
```

If the same concepts keep failing, the LLM is likely struggling with JSON output for those specific titles. Try:

```bash
# More room for the model to produce clean output
# Edit wiki.toml: heavy_ctx = 32768, then:
olw compile
```

See [Tuning context windows](#tuning-context-windows) for the `heavy_ctx` table.

---

**Q: I see `structured_output attempt N failed` messages during compile — is something broken?**

No. This is the built-in 3-tier retry system working as designed. The model occasionally echoes the JSON schema structure instead of flat output — the retry corrects it. Articles are still generated. These messages are debug-level noise; a real failure surfaces as `article(s) failed: ...` in the summary line.

---

**Q: `olw ingest --all && olw compile` gives "Missing option '--vault'".**

Run `olw setup` first to configure a default vault, or pass it explicitly:

```bash
export OLW_VAULT=~/my-wiki
olw ingest --all && olw compile
```

---

**Q: I changed models in `olw setup` but `olw compile` still uses the old model.**

Re-run `olw init` on your vault — it now syncs the model settings from your global config into `wiki.toml`:

```bash
olw init ~/my-wiki
```

---

## Why not just use a chatbot?

Chatbots forget. Every conversation starts fresh. This tool builds a **persistent artifact** — a wiki that grows with every note you add, that you can open in Obsidian, search, query, and edit by hand.

The LLM is a compiler, not a conversation partner. You give it raw material; it produces structured knowledge. The output is plain markdown files you own forever.

---

## License

MIT — see [LICENSE](LICENSE).

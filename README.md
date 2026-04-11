# obsidian-llm-wiki

**Turn your raw notes into a self-improving, interlinked wiki — powered by a local LLM.**

Drop a markdown file into a folder. The pipeline reads it, extracts concepts, and creates or updates wiki articles with the new knowledge. Reject a draft and explain why — the next compile addresses your feedback. Over time your wiki compounds: every note you add (and every draft you review) makes the whole smarter.

**100% local. No cloud, no API keys, no telemetry.** Just [Ollama](https://ollama.com) running on your machine.

---

## The idea (Karpathy's LLM Wiki)

This project is a practical implementation of the pattern described by Andrej Karpathy in [**"The LLM Wiki"**](https://karpathy.ai/llmwiki) — a vision for a personal knowledge base where:

> *"The LLM doesn't just store what you tell it — it synthesizes, cross-references, and keeps everything current. You add raw material; it does the bookkeeping."*

The key insight: treat your notes as **source material**, not as the final artifact. The LLM compiles them into a structured wiki that grows smarter as you add more. Unlike a chatbot that forgets, the wiki **persists and compounds**.

```
You write raw notes  →  LLM extracts concepts  →  Wiki articles created/updated
     raw/                    (automatic)                    wiki/
  quantum.md          "Qubit", "Superposition"       Qubit.md   ←──┐
  ml-basics.md        "Neural Network", "SGD"    Superposition.md  │
  physics.md          "Qubit" (again!)           Neural Network.md │
                                                      ↑  linked via [[wikilinks]]
```

The wiki lives in Obsidian, so you get the graph view, backlinks, and Dataview queries for free.

---

## Features

- **Concept-driven, incremental compilation** — each concept gets its own article, updated only when its source notes change
- **Rejection feedback loop** — reject a draft with a reason; the next compile injects your feedback into the prompt so the model addresses it
- **Draft annotations** — low-confidence or single-source drafts are flagged with `<!-- olw-auto: ... -->` comments (invisible in Obsidian, stripped on approval)
- **Rich review interface** — `olw review` lists drafts ranked by rejection count, shows diffs vs published and vs the last rejected version
- **Pipeline orchestrator** — `olw run` runs ingest → compile → lint → [approve] as one command with timing and failure classification
- **Selective recompile** — after a file save, only concepts linked to that source are recompiled (not the entire wiki)
- **Self-maintenance** — `olw maintain` creates stub articles for broken wikilinks, suggests orphan link fixes, and warns about low-quality source distribution
- **Manual edit protection** — edited an article by hand? The compiler detects the change and skips it
- **Source traceability** — every article links back to the raw notes it was built from
- **File watcher** — `olw watch` auto-processes anything dropped into `raw/`
- **Wiki health checks** — `olw lint` detects orphans, broken links, stale articles (no LLM needed)
- **Query your wiki** — `olw query "what is X?"` answers from your published articles
- **Git safety net** — every auto-action is committed; `olw undo` reverts safely
- **Offline test suite** — all 322 tests run without Ollama

---

## Quick start

### 1. Install

**From PyPI** (recommended):

```bash
pip install obsidian-llm-wiki
```

Or with `uv`:

```bash
uv tool install obsidian-llm-wiki
```

**From source** (latest development version):

```bash
git clone https://github.com/kytmanov/obsidian-llm-wiki-local
cd obsidian-llm-wiki-local
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

### 6. Run the full pipeline

```bash
# One command: ingest + compile + lint + optional auto-approve
olw run

# Or step by step:
olw ingest --all
olw compile
olw review        # interactive draft review
```

If you set a default vault in `olw setup`, the `--vault` flag is optional. Otherwise use `--vault ~/my-wiki` or `export OLW_VAULT=~/my-wiki`.

Open `~/my-wiki` as an Obsidian vault. The graph view shows your connected wiki.

### 7. Keep it running (optional)

```bash
olw watch
# Drop a file in raw/ → ingest + compile happen automatically (selective: only linked concepts)
```

---

## How it works

The pipeline has three stages, each using the LLM for a different purpose:

```
raw/note.md
    │
    ▼ olw ingest  (or olw run)
    Fast LLM (3B–8B)
    • Reads note
    • Extracts concept names
    • Writes quality score + summary to state.db
    • Creates wiki/sources/Note.md (source summary page)
    │
    ▼ olw compile
    Heavy LLM (7B–14B)
    • For each concept: gathers all source notes that mention it
    • Injects rejection feedback from previous reviews into the prompt
    • Writes a wiki article with [[wikilinks]] to related concepts
    • Adds quality annotations if confidence is low or sources are sparse
    • Lands in wiki/.drafts/ for review
    │
    ▼ olw review  (or olw approve)
    • Interactive numbered menu — approve / reject / diff / edit
    • Rejection feedback stored and injected into next compile
    • On approve: annotations stripped, article published to wiki/
    • Updates wiki/index.md (navigation layer)
    • Git commits the change
```

**No vector databases, no embeddings.** `wiki/index.md` acts as the routing layer for `olw query`. This keeps the setup simple and works well up to ~100 source notes.

---

## Rejection feedback loop

The core v0.2 feature. When you reject a draft:

```bash
olw review
# Select draft → [r]eject → "The overview section is too vague, needs concrete examples"
```

The feedback is stored in the state database. On the next compile of that concept, the prompt includes:

```
PREVIOUS REJECTIONS — address these issues:
- The overview section is too vague, needs concrete examples
```

After 5 rejections of the same concept without an approval, the concept is **auto-blocked** and excluded from future compiles until you explicitly re-enable it:

```bash
olw unblock "Quantum Computing"
```

---

## Draft annotations

Drafts with low confidence or sparse sources are annotated with HTML comments that are invisible in Obsidian's preview but visible in the editor:

```markdown
<!-- olw-auto: low-confidence (0.32) — verify before publishing -->
<!-- olw-auto: single-source — cross-reference recommended -->

## Overview
...
```

Annotations are stripped automatically when you approve a draft. `olw review` surfaces them as a warning in the draft list.

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
    ├── state.db                ← SQLite: notes, concepts, articles, rejections, stubs
    └── pipeline.lock           ← advisory lock (auto-released when the holding process exits)
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
fast_ctx = 16384                 # context window for fast model (tokens)
heavy_ctx = 32768                # context window for heavy model (tokens)

[pipeline]
auto_approve = false             # true = skip draft review
auto_commit = true               # git commit after each operation
auto_maintain = false            # true = run maintain checks after each compile
max_concepts_per_source = 8      # limit concepts extracted per note
watch_debounce = 3.0             # seconds after last file event before processing
ingest_parallel = false          # true = parallel chunk analysis (needs OLLAMA_NUM_PARALLEL>=4)
```

### Tuning context windows

`heavy_ctx` controls how much source material the heavy model reads when writing articles (`source budget = heavy_ctx / 2` chars) and how long the generated article can be. Defaults target **16 GB VRAM**. **If you use a model with a large context window (e.g. `gemma4:e4b` supports 128K), increase it.**

| VRAM | Recommended `heavy_ctx` | Source budget | Notes |
|---|---|---|---|
| 8 GB | `8192` | ~4K chars | Minimum; short articles |
| 16 GB | `32768` | ~16K chars | Default |
| 32 GB+ | `65536` | ~32K chars | Rich multi-source articles |

`fast_ctx` controls ingest analysis. Notes longer than `fast_ctx / 2` chars are automatically split into chunks and analyzed in sequence — all content is covered, no truncation.

| VRAM | Recommended `fast_ctx` | Notes per chunk |
|---|---|---|
| 8 GB | `8192` | ~4K chars |
| 16 GB | `16384` | ~8K chars — Default |
| 32 GB+ | `32768` | ~16K chars |

### Speeding up long-note ingest

For vaults with many long notes (>8K chars), enable parallel chunk analysis:

```toml
[pipeline]
ingest_parallel = true   # requires OLLAMA_NUM_PARALLEL>=4
```

Also set in your shell before starting Ollama:

```bash
OLLAMA_NUM_PARALLEL=4 ollama serve
```

This lets Ollama process multiple chunks simultaneously. On 16 GB VRAM with `gemma4:e4b` (9.6 GB), 4 parallel slots fit comfortably (~12.8 GB total). Wall time for a 25K-char note drops from ~39s to ~14s.

After editing `wiki.toml`, no reinstall is needed. Run `olw compile --force` to regenerate articles with the new context budget.

---

## Commands

| Command | Description |
|---------|-------------|
| `olw setup` | Interactive setup wizard (first run) |
| `olw init PATH` | Create vault structure and git repo |
| `olw init PATH --existing` | Adopt an existing Obsidian vault |
| `olw doctor` | Check Ollama, models, vault structure |
| `olw run` | Full pipeline: ingest → compile → lint → [approve] |
| `olw run --auto-approve` | Full pipeline, publish without review |
| `olw run --dry-run` | Report what would happen, make no changes |
| `olw ingest --all` | Analyze all raw notes |
| `olw ingest FILE` | Analyze one note |
| `olw compile` | Generate wiki articles → `.drafts/` |
| `olw compile --retry-failed` | Retry previously failed notes |
| `olw review` | Interactive draft review (approve / reject / diff) |
| `olw approve --all` | Publish all drafts without review |
| `olw approve FILE` | Publish one draft |
| `olw reject FILE` | Discard a draft (prompts for feedback) |
| `olw reject FILE --feedback "..."` | Discard with feedback for next compile |
| `olw unblock "Concept"` | Re-enable a concept blocked after 5 rejections |
| `olw maintain` | Health check + stubs + orphan and merge suggestions |
| `olw maintain --fix` | Auto-fix issues and create stub articles |
| `olw maintain --dry-run` | Report issues without making changes |
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
git clone https://github.com/kytmanov/obsidian-llm-wiki-local
cd obsidian-llm-wiki-local
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
olw review        # interactive review
# or
olw approve --all
```

Articles move to `wiki/` and become fully visible.

---

**Q: Compile says "2 article(s) failed: Methodology, Sprints" — what do I do?**

Failed concepts are retried automatically on the next run. Or force a retry:

```bash
olw compile --retry-failed
```

If the same concepts keep failing, the LLM is likely struggling with JSON output for those specific titles. Try increasing `heavy_ctx` in `wiki.toml` (see [Tuning context windows](#tuning-context-windows)).

---

**Q: I see `structured_output attempt N failed` messages during compile — is something broken?**

No. This is the built-in 3-tier retry system working as designed. The model occasionally echoes the JSON schema structure instead of flat output — the retry corrects it. Articles are still generated. A real failure surfaces as `article(s) failed: ...` in the summary line.

---

**Q: `olw ingest --all && olw compile` gives "Missing option '--vault'".**

Run `olw setup` first to configure a default vault, or pass it explicitly:

```bash
export OLW_VAULT=~/my-wiki
olw ingest --all && olw compile
```

---

**Q: I changed models in `olw setup` but `olw compile` still uses the old model.**

Re-run `olw init` on your vault — it syncs the model settings from your global config into `wiki.toml`:

```bash
olw init ~/my-wiki
```

---

**Q: A concept keeps getting rejected — how do I stop it from recompiling?**

After 5 rejections the concept is auto-blocked and excluded from future compiles. If it blocks earlier than you'd like:

```bash
olw unblock "Concept Name"   # re-enable
```

Or manually block it:

```bash
# Mark as blocked so compile skips it
olw status   # shows blocked concepts
```

---

## Why not just use a chatbot?

Chatbots forget. Every conversation starts fresh. This tool builds a **persistent artifact** — a wiki that grows with every note you add, that you can open in Obsidian, search, query, and edit by hand.

The LLM is a compiler, not a conversation partner. You give it raw material; it produces structured knowledge. The output is plain markdown files you own forever.

---

## License

MIT — see [LICENSE](LICENSE).

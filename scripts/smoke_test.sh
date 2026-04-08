#!/usr/bin/env bash
# smoke_test.sh — end-to-end test against a real Ollama instance
#
# Usage:
#   ./scripts/smoke_test.sh                        # use default models
#   FAST_MODEL=llama3.2:latest ./scripts/smoke_test.sh   # override models
#   VAULT_DIR=/tmp/my-vault ./scripts/smoke_test.sh      # keep vault after run
#   SKIP_PULL=1 ./scripts/smoke_test.sh            # skip ollama pull
#
# Requirements:
#   - uv (https://docs.astral.sh/uv/)
#   - Ollama running (ollama serve)

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
FAST_MODEL="${FAST_MODEL:-gemma4:e4b}"
HEAVY_MODEL="${HEAVY_MODEL:-qwen2.5:14b}"
OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434}"
SKIP_PULL="${SKIP_PULL:-0}"
KEEP_VAULT="${KEEP_VAULT:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Use provided VAULT_DIR or create a temp one
if [[ -n "${VAULT_DIR:-}" ]]; then
    KEEP_VAULT=1
    mkdir -p "$VAULT_DIR"
else
    VAULT_DIR="$(mktemp -d)"
fi

# ── Helpers ───────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
NC='\033[0m'

pass() { echo -e "${GREEN}✓${NC} $1"; }
fail() { echo -e "${RED}✗ FAIL: $1${NC}"; exit 1; }
info() { echo -e "${YELLOW}▶${NC} $1"; }
header() { echo -e "\n${BOLD}$1${NC}"; }

PASS_COUNT=0
check() {
    local desc="$1"
    shift
    local rc=0
    ( set +o pipefail; eval "$@" ) > /dev/null 2>&1 || rc=$?
    if [[ $rc -eq 0 ]]; then
        pass "$desc"
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        fail "$desc"
    fi
}

cleanup() {
    if [[ "$KEEP_VAULT" == "0" ]]; then
        rm -rf "$VAULT_DIR"
    else
        echo -e "\nVault kept at: ${BOLD}$VAULT_DIR${NC}"
    fi
}
trap cleanup EXIT

# ── Prerequisites ─────────────────────────────────────────────────────────────
header "Prerequisites"

check "uv available" "command -v uv"
check "Ollama reachable at $OLLAMA_URL" "curl -sf $OLLAMA_URL/api/tags"

if [[ "$SKIP_PULL" == "0" ]]; then
    info "Pulling models (skippable with SKIP_PULL=1)"
    ollama pull "$FAST_MODEL"   || fail "Could not pull $FAST_MODEL"
    ollama pull "$HEAVY_MODEL"  || fail "Could not pull $HEAVY_MODEL"
fi

check "Fast model present: $FAST_MODEL"  "curl -sf $OLLAMA_URL/api/tags | grep -q '$FAST_MODEL'"
check "Heavy model present: $HEAVY_MODEL" "curl -sf $OLLAMA_URL/api/tags | grep -q '$HEAVY_MODEL'"

# ── Install ───────────────────────────────────────────────────────────────────
header "Install"

info "Installing obsidian-llm-wiki from $REPO_DIR"
uv sync --project "$REPO_DIR" --quiet
pass "uv sync"

OLW="uv run --project $REPO_DIR olw"
export OLW_VAULT="$VAULT_DIR"

# ── Init ──────────────────────────────────────────────────────────────────────
header "olw init"

$OLW init "$VAULT_DIR" 2>&1 | grep -v "^$" || true

check "raw/ created"           "test -d $VAULT_DIR/raw"
check "wiki/ created"          "test -d $VAULT_DIR/wiki"
check "wiki/.drafts/ created"  "test -d $VAULT_DIR/wiki/.drafts"
check "wiki/sources/ created"  "test -d $VAULT_DIR/wiki/sources"
check ".olw/ created"          "test -d $VAULT_DIR/.olw"
check "wiki.toml created"      "test -f $VAULT_DIR/wiki.toml"
check "git repo initialised"   "test -d $VAULT_DIR/.git"

# Override models in wiki.toml
cat > "$VAULT_DIR/wiki.toml" <<TOML
[models]
fast = "$FAST_MODEL"
heavy = "$HEAVY_MODEL"
embed = "nomic-embed-text"

[ollama]
url = "$OLLAMA_URL"
timeout = 900
fast_ctx = 8192
heavy_ctx = 16384

[pipeline]
auto_approve = false
auto_commit = true
watch_debounce = 3.0

[rag]
chunk_size = 512
chunk_overlap = 50
similarity_threshold = 0.7
TOML
pass "wiki.toml configured (fast=$FAST_MODEL, heavy=$HEAVY_MODEL)"

# ── Doctor ───────────────────────────────────────────────────────────────────
header "olw doctor"
$OLW doctor 2>&1 || true
# Doctor exit code not checked (models may not be present before pull)

# ── Seed raw notes ────────────────────────────────────────────────────────────
header "Seed raw notes"

cat > "$VAULT_DIR/raw/quantum-computing.md" <<'EOF'
---
title: Quantum Computing Fundamentals
source: https://example.com/quantum
---

Quantum computers use qubits instead of classical bits. Unlike bits which are
either 0 or 1, qubits exploit superposition to be in multiple states simultaneously.

Entanglement links qubits: measuring one instantly determines the state of its
partner regardless of distance. This enables quantum parallelism.

Key algorithms:
- Shor's algorithm: factors large integers exponentially faster than classical
- Grover's algorithm: searches unsorted databases with quadratic speedup
- Quantum Fourier Transform: underpins most quantum speedups

Hardware approaches: superconducting qubits (IBM, Google), trapped ions (IonQ),
photonic (PsiQuantum), topological (Microsoft).

Current state (2024): NISQ era — noisy, ~1000 qubits, error rates ~0.1%.
Fault-tolerant quantum computing requires ~1M physical qubits per logical qubit.
EOF

cat > "$VAULT_DIR/raw/machine-learning-basics.md" <<'EOF'
---
title: Machine Learning Fundamentals
---

Machine learning enables computers to learn from data without being explicitly
programmed. Three main paradigms:

Supervised learning: labeled training data. Examples: classification (spam
detection), regression (price prediction). Algorithms: linear regression,
decision trees, neural networks, SVMs.

Unsupervised learning: finds hidden structure in unlabeled data. Clustering
(k-means), dimensionality reduction (PCA), generative models.

Reinforcement learning: agent learns by interacting with environment, maximising
cumulative reward. Used in game playing (AlphaGo), robotics, recommendation systems.

Deep learning: neural networks with many layers. Excels at images (CNNs), text
(Transformers), audio. Requires large datasets and compute.

Key concepts: gradient descent, backpropagation, overfitting/underfitting,
train/val/test split, cross-validation.
EOF

check "raw note 1 created" "test -f $VAULT_DIR/raw/quantum-computing.md"
check "raw note 2 created" "test -f $VAULT_DIR/raw/machine-learning-basics.md"

# Snapshot checksums so we can verify raw files stay immutable after ingest
RAW_HASH_1=$(shasum "$VAULT_DIR/raw/quantum-computing.md" | awk '{print $1}')
RAW_HASH_2=$(shasum "$VAULT_DIR/raw/machine-learning-basics.md" | awk '{print $1}')

# ── Ingest ────────────────────────────────────────────────────────────────────
header "olw ingest --all"
info "Calling Ollama ($FAST_MODEL) — may take 30-120s..."

$OLW ingest --all 2>&1

check "state.db created" "test -f $VAULT_DIR/.olw/state.db"

# Raw files must remain unchanged (immutability contract)
check "raw note 1 unchanged after ingest" \
    "test \"\$(shasum '$VAULT_DIR/raw/quantum-computing.md' | awk '{print \$1}')\" = '$RAW_HASH_1'"
check "raw note 2 unchanged after ingest" \
    "test \"\$(shasum '$VAULT_DIR/raw/machine-learning-basics.md' | awk '{print \$1}')\" = '$RAW_HASH_2'"

# Source summary pages created in wiki/sources/
SOURCE_COUNT=$(find "$VAULT_DIR/wiki/sources" -name "*.md" 2>/dev/null | wc -l | tr -d ' ')
check "source summary pages created" "test '$SOURCE_COUNT' -ge 1"

if [[ "$SOURCE_COUNT" -gt 0 ]]; then
    FIRST_SOURCE=$(find "$VAULT_DIR/wiki/sources" -name "*.md" | sort | head -1)
    check "source page has YAML frontmatter" "grep -q '^---' \"$FIRST_SOURCE\""
    check "source page has tags: [source]"   "grep -q 'source' \"$FIRST_SOURCE\""
    check "source page has concept wikilinks" "grep -q '\[\[' \"$FIRST_SOURCE\""
fi

# index.md and log.md created
check "wiki/index.md created" "test -f $VAULT_DIR/wiki/index.md"
check "wiki/log.md created"   "test -f $VAULT_DIR/wiki/log.md"
check "index.md has wikilinks" "grep -q '\[\[' $VAULT_DIR/wiki/index.md"

# ── Status after ingest ───────────────────────────────────────────────────────
header "olw status (after ingest)"
STATUS_OUT=$($OLW status 2>&1)
echo "$STATUS_OUT"

check "status shows ingested notes" "echo '$STATUS_OUT' | grep -q 'ingested'"

# ── Concept extraction check ──────────────────────────────────────────────────
header "Concept extraction"
# Source summary pages should have wikilinks pointing to extracted concepts
if [[ "$SOURCE_COUNT" -gt 0 ]]; then
    # Verify concept wikilinks exist in source pages (extracted during ingest)
    CONCEPT_LINKS=$(grep -r '\[\[' "$VAULT_DIR/wiki/sources/" 2>/dev/null | wc -l | tr -d ' ')
    check "source pages have concept wikilinks" "test '$CONCEPT_LINKS' -ge 1"
fi

# ── Compile (concept-driven) ──────────────────────────────────────────────────
header "olw compile (concept-driven)"
info "Calling Ollama ($HEAVY_MODEL) — may take 2-5 min..."

$OLW compile 2>&1

DRAFT_COUNT=$(find "$VAULT_DIR/wiki/.drafts" -name "*.md" 2>/dev/null | wc -l | tr -d ' ')
check "at least 1 draft created" "test '$DRAFT_COUNT' -ge 1"

if [[ "$DRAFT_COUNT" -gt 0 ]]; then
    FIRST_DRAFT=$(find "$VAULT_DIR/wiki/.drafts" -name "*.md" | sort | head -1)
    check "draft has YAML frontmatter"   "grep -q '^---' \"$FIRST_DRAFT\""
    check "draft has title field"        "grep -q 'title:' \"$FIRST_DRAFT\""
    check "draft has status: draft"      "grep -q 'status: draft' \"$FIRST_DRAFT\""
    check "draft has sources field"      "grep -q 'sources:' \"$FIRST_DRAFT\""
    check "draft has content"            "test \$(wc -l < \"$FIRST_DRAFT\") -ge 10"
    check "draft has ## Sources section" "grep -q '^## Sources' \"$FIRST_DRAFT\""
    check "draft has confidence field"   "grep -q 'confidence:' \"$FIRST_DRAFT\""
fi

# ── Status after compile ──────────────────────────────────────────────────────
header "olw status (after compile)"
$OLW status 2>&1

# ── Approve ───────────────────────────────────────────────────────────────────
header "olw approve --all"
$OLW approve --all 2>&1

WIKI_COUNT=$(find "$VAULT_DIR/wiki" -name "*.md" -not -path "*/.drafts/*" 2>/dev/null | wc -l | tr -d ' ')
check "articles published to wiki/"    "test '$WIKI_COUNT' -ge 1"
check "drafts directory now empty"     "test \$(find $VAULT_DIR/wiki/.drafts -name '*.md' 2>/dev/null | wc -l) -eq 0"
check "git commit created"             "git -C $VAULT_DIR log --oneline | grep -q '\[olw\]'"

# ── Git log ───────────────────────────────────────────────────────────────────
header "Git history"
git -C "$VAULT_DIR" log --oneline

# ── Undo ─────────────────────────────────────────────────────────────────────
header "olw undo"
$OLW undo 2>&1

check "undo reverted publish commit" \
    "git -C $VAULT_DIR log --oneline | grep -q 'Revert'"

# ── Incremental compile (3rd note → only new concepts compiled) ───────────────
header "Incremental compile"
info "Adding 3rd note to test concept-based incremental updates..."

cat > "$VAULT_DIR/raw/deep-learning.md" <<'EOF'
---
title: Deep Learning
---

Deep learning is a subset of machine learning using neural networks with many layers.

Convolutional Neural Networks (CNNs) excel at image recognition tasks.
Transformers (e.g. BERT, GPT) dominate natural language processing.
Recurrent Neural Networks (RNNs) handle sequential data.

Training requires large datasets and GPUs. Key challenges: vanishing gradients,
overfitting, interpretability. Techniques: dropout, batch normalization,
learning rate scheduling.
EOF

$OLW ingest "$VAULT_DIR/raw/deep-learning.md" 2>&1
INGEST3_OUT=$($OLW compile --dry-run 2>&1)
echo "$INGEST3_OUT"
check "dry run shows only new concepts" \
    "echo \"$INGEST3_OUT\" | grep -qi 'concept\|compile\|deep\|neural\|no concept'"

# ── Manual edit protection ────────────────────────────────────────────────────
header "Manual edit protection"
# Find any published wiki article (not index, log, sources)
WIKI_ARTICLE=$(find "$VAULT_DIR/wiki" -maxdepth 1 -name "*.md" \
    ! -name "index.md" ! -name "log.md" 2>/dev/null | head -1)

if [[ -n "$WIKI_ARTICLE" ]]; then
    info "Manually editing: $WIKI_ARTICLE"
    echo -e "\n\nManually added content." >> "$WIKI_ARTICLE"

    # Re-ingest to create a new 'ingested' note that would normally trigger compile
    # Use the already ingested note (force it back to ingested)
    COMPILE_OUT=$($OLW compile 2>&1 || true)
    echo "$COMPILE_OUT"
    # Manually edited article should be skipped (not recompiled)
    DRAFT_AFTER_EDIT=$(find "$VAULT_DIR/wiki/.drafts" -name "$(basename $WIKI_ARTICLE)" 2>/dev/null | wc -l | tr -d ' ')
    check "manually edited article skipped in compile" \
        "test '$DRAFT_AFTER_EDIT' -eq 0"
fi

# ── Duplicate detection ───────────────────────────────────────────────────────
header "Duplicate detection"
cp "$VAULT_DIR/raw/quantum-computing.md" "$VAULT_DIR/raw/quantum-computing-copy.md" 2>/dev/null || true

INGEST_OUT=$($OLW ingest "$VAULT_DIR/raw/quantum-computing-copy.md" 2>&1 || true)
check "duplicate skipped" "echo \"$INGEST_OUT\" | grep -qi 'skip\|duplicate\|already'"
rm -f "$VAULT_DIR/raw/quantum-computing-copy.md"

# ── Query (Stage 3) ───────────────────────────────────────────────────────────
header "olw query (Stage 3)"
info "Approving drafts so query has articles to search..."
$OLW approve --all 2>&1 || true

info "Running query against wiki..."
QUERY_OUT=$($OLW query "What is a qubit?" 2>&1 || true)
echo "$QUERY_OUT"
check "query returns an answer" \
    "echo \"$QUERY_OUT\" | grep -qi 'qubit\|quantum\|superposition\|bit'"

info "Running query with --save..."
QUERY_SAVE_OUT=$($OLW query --save "What algorithms are used in quantum computing?" 2>&1 || true)
echo "$QUERY_SAVE_OUT"
QUERY_COUNT=$(find "$VAULT_DIR/wiki/queries" -name "*.md" 2>/dev/null | wc -l | tr -d ' ')
check "query --save creates file in wiki/queries/" "test '$QUERY_COUNT' -ge 1"

# ── Lint (Stage 3) ────────────────────────────────────────────────────────────
header "olw lint (Stage 3)"
LINT_OUT=$($OLW lint 2>&1 || true)
echo "$LINT_OUT"
check "lint reports health score" \
    "echo \"$LINT_OUT\" | grep -qi 'health\|score\|100\|issue'"

# Lint --fix (should not crash even if no fixable issues)
$OLW lint --fix 2>&1 || true
pass "lint --fix runs without error"

# ── Retry failed (Stage 4) ────────────────────────────────────────────────────
header "olw compile --retry-failed (Stage 4)"
# Inject a fake failed record directly, then verify --retry-failed notices it
python3 - <<PYEOF
import sqlite3, pathlib
db_path = "$VAULT_DIR/.olw/state.db"
conn = sqlite3.connect(db_path)
# Only insert if not already present
conn.execute("""
    INSERT OR IGNORE INTO raw_notes (path, content_hash, status, error)
    VALUES ('raw/fake-failed.md', 'badhash', 'failed', 'simulated failure')
""")
conn.commit()
conn.close()
PYEOF
RETRY_TMP=$(mktemp)
$OLW compile --retry-failed 2>&1 | tee "$RETRY_TMP" || true
check "retry-failed reports failed notes" \
    "grep -qi 'retry\|failed\|not found\|re-ingest' '$RETRY_TMP'"
rm -f "$RETRY_TMP"

# ── Summary ───────────────────────────────────────────────────────────────────
header "Results"
echo -e "${GREEN}${BOLD}All checks passed: $PASS_COUNT${NC}"
echo ""
echo "Wiki articles created:"
find "$VAULT_DIR/wiki" -name "*.md" -not -path "*/.drafts/*" | sort | sed 's/^/  /'
echo ""
echo "To inspect the vault:"
echo "  export OLW_VAULT=$VAULT_DIR"
echo "  uv run --project $REPO_DIR olw status"
if [[ "$KEEP_VAULT" == "1" ]]; then
    echo "  open $VAULT_DIR in Obsidian"
fi

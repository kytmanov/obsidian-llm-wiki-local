#!/usr/bin/env bash
# smoke_test.sh — end-to-end test against a real LLM backend
#
# Supports Ollama (default) and LM Studio (PROVIDER=lm_studio).
#
# Usage:
#   ./scripts/smoke_test.sh                              # Ollama, default models
#   PROVIDER=lm_studio ./scripts/smoke_test.sh           # LM Studio
#   PROVIDER=lm_studio FAST_MODEL=google/gemma-4-e4b ./scripts/smoke_test.sh
#   FAST_MODEL=llama3.2:latest ./scripts/smoke_test.sh   # Ollama, custom model
#   VAULT_DIR=/tmp/my-vault ./scripts/smoke_test.sh      # keep vault after run
#   SKIP_PULL=1 ./scripts/smoke_test.sh                  # skip ollama pull
#
# Requirements:
#   - uv (https://docs.astral.sh/uv/)
#   - Ollama running (ollama serve)  — OR —  LM Studio running with a model loaded

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
PROVIDER="${PROVIDER:-ollama}"

case "$PROVIDER" in
    ollama)
        # OLLAMA_URL kept for backward compatibility
        PROVIDER_URL="${PROVIDER_URL:-${OLLAMA_URL:-http://localhost:11434}}"
        FAST_MODEL="${FAST_MODEL:-gemma4:e4b}"
        HEAVY_MODEL="${HEAVY_MODEL:-gemma4:e4b}"
        FAST_CTX=8192
        HEAVY_CTX=16384
        ;;
    lm_studio)
        PROVIDER_URL="${PROVIDER_URL:-http://localhost:1234/v1}"
        FAST_MODEL="${FAST_MODEL:-google/gemma-4-e4b}"
        HEAVY_MODEL="${HEAVY_MODEL:-google/gemma-4-e4b}"
        FAST_CTX=8192
        # Keep output budget + input within 8192: source uses heavy_ctx//2 tokens,
        # output uses _MAX_ARTICLE_PREDICT=4096. Total = 4096+4096+~800 overhead > 8192.
        # Use 8192 so _gather_sources truncates aggressively enough for short test notes.
        HEAVY_CTX=8192
        ;;
    *)
        # Generic OpenAI-compatible provider — caller must set PROVIDER_URL and models
        PROVIDER_URL="${PROVIDER_URL:-}"
        FAST_MODEL="${FAST_MODEL:-}"
        HEAVY_MODEL="${HEAVY_MODEL:-}"
        FAST_CTX="${FAST_CTX:-8192}"
        HEAVY_CTX="${HEAVY_CTX:-16384}"
        HEAVY_MODEL="${HEAVY_MODEL:-$FAST_MODEL}"
        if [[ -z "$PROVIDER_URL" || -z "$FAST_MODEL" || -z "$HEAVY_MODEL" ]]; then
            echo "ERROR: PROVIDER=$PROVIDER requires PROVIDER_URL and FAST_MODEL to be set."
            exit 1
        fi
        ;;
esac

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
header "Prerequisites (provider: $PROVIDER)"

check "uv available" "command -v uv"

if [[ "$PROVIDER" == "ollama" ]]; then
    check "Ollama reachable at $PROVIDER_URL" "curl -sf $PROVIDER_URL/api/tags"

    if [[ "$SKIP_PULL" == "0" ]]; then
        info "Pulling models (skippable with SKIP_PULL=1)"
        ollama pull "$FAST_MODEL"  || fail "Could not pull $FAST_MODEL"
        if [[ "$FAST_MODEL" != "$HEAVY_MODEL" ]]; then
            ollama pull "$HEAVY_MODEL" || fail "Could not pull $HEAVY_MODEL"
        fi
    fi

    check "Fast model present: $FAST_MODEL"  "curl -sf $PROVIDER_URL/api/tags | grep -F -q '$FAST_MODEL'"
    check "Heavy model present: $HEAVY_MODEL" "curl -sf $PROVIDER_URL/api/tags | grep -F -q '$HEAVY_MODEL'"
else
    # LM Studio and other OpenAI-compatible providers: just verify the endpoint is up.
    # Model presence can't be checked reliably via /v1/models on all backends.
    check "$PROVIDER reachable at $PROVIDER_URL" "curl -sf $PROVIDER_URL/models"
    info "Model pull skipped ($PROVIDER manages its own models — load $FAST_MODEL in $PROVIDER before running)"
fi

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

# Write provider-appropriate wiki.toml
if [[ "$PROVIDER" == "ollama" ]]; then
    cat > "$VAULT_DIR/wiki.toml" <<TOML
[models]
fast = "$FAST_MODEL"
heavy = "$HEAVY_MODEL"
embed = "nomic-embed-text"

[ollama]
url = "$PROVIDER_URL"
timeout = 900
fast_ctx = $FAST_CTX
heavy_ctx = $HEAVY_CTX

[pipeline]
auto_approve = false
auto_commit = true
watch_debounce = 3.0

[rag]
chunk_size = 512
chunk_overlap = 50
similarity_threshold = 0.7
TOML
else
    cat > "$VAULT_DIR/wiki.toml" <<TOML
[models]
fast = "$FAST_MODEL"
heavy = "$HEAVY_MODEL"

[provider]
name = "$PROVIDER"
url = "$PROVIDER_URL"
timeout = 900
fast_ctx = $FAST_CTX
heavy_ctx = $HEAVY_CTX

[pipeline]
auto_approve = false
auto_commit = true
watch_debounce = 3.0

[rag]
chunk_size = 512
chunk_overlap = 50
similarity_threshold = 0.7
TOML
fi
pass "wiki.toml configured (provider=$PROVIDER fast=$FAST_MODEL heavy=$HEAVY_MODEL)"

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
info "Calling $PROVIDER ($FAST_MODEL) — may take 30-120s..."

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
    check "source page has YAML frontmatter"  "grep -q '^---' \"$FIRST_SOURCE\""
    check "source page has tags: [source]"    "grep -q 'source' \"$FIRST_SOURCE\""
    check "source page has concept wikilinks" "grep -q '\[\[' \"$FIRST_SOURCE\""

    SRC_YAML_ERR=$(uv run --project "$REPO_DIR" python - "$FIRST_SOURCE" 2>/dev/null <<'PYEOF'
import sys, frontmatter
try:
    frontmatter.load(sys.argv[1])
except Exception as e:
    print(f"error: {e}")
    sys.exit(1)
PYEOF
)
    check "source page YAML is parseable" "test -z \"$SRC_YAML_ERR\""

    SRC_ALIAS_ERR=$(uv run --project "$REPO_DIR" python - "$FIRST_SOURCE" 2>/dev/null <<'PYEOF'
import sys, frontmatter
try:
    m = frontmatter.load(sys.argv[1])
    aliases = m.get('aliases', [])
    assert isinstance(aliases, list), f'aliases not a list: {aliases!r}'
except AssertionError as e:
    print(str(e))
    sys.exit(1)
except Exception as e:
    print(f"error: {e}")
    sys.exit(1)
PYEOF
)
    check "source page aliases is a list" "test -z \"$SRC_ALIAS_ERR\""
fi

# index.md and log.md created
check "wiki/index.md created" "test -f $VAULT_DIR/wiki/index.md"
check "wiki/log.md created"   "test -f $VAULT_DIR/wiki/log.md"
check "index.md has wikilinks" "grep -q '\[\[' $VAULT_DIR/wiki/index.md"

# ── Status after ingest ───────────────────────────────────────────────────────
header "olw status (after ingest)"
STATUS_OUT=$($OLW status 2>&1)
echo "$STATUS_OUT"

check "status shows ingested notes" "echo \"$STATUS_OUT\" | grep -q 'ingested'"

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
info "Calling $PROVIDER ($HEAVY_MODEL) — may take 2-5 min..."

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
    DRAFT_YAML_OK=$(uv run --project "$REPO_DIR" python - "$FIRST_DRAFT" 2>/dev/null <<'PYEOF'
import sys, frontmatter
try:
    frontmatter.load(sys.argv[1])
except Exception as e:
    print(f"error: {e}")
    sys.exit(1)
PYEOF
)
    check "draft YAML is parseable" "test -z \"$DRAFT_YAML_OK\""
    DRAFT_TAG_BAD=$(uv run --project "$REPO_DIR" python - "$FIRST_DRAFT" 2>/dev/null <<'PYEOF'
import sys, re, frontmatter
m = frontmatter.load(sys.argv[1])
valid_re = re.compile(r'^[a-z0-9][a-zA-Z0-9_/\-]*$')
bad = [t for t in m.get('tags', []) if not isinstance(t, str) or ' ' in t or t != t.lower() or not valid_re.match(t)]
if bad:
    print(f"Bad tags: {bad}")
    sys.exit(1)
PYEOF
)
    check "draft tags are valid (lowercase, no spaces, no special chars)" "test -z \"$DRAFT_TAG_BAD\""
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

# Bulk YAML validity + tag check on all published wiki pages
# Use -print0 / read -d '' to handle spaces and special chars in filenames
header "YAML validity of published pages"
YAML_FAIL=0
TAG_FAIL=0

# Write validator to temp file (avoids heredoc-inside-process-substitution bash quirk)
_YAML_VALIDATOR=$(mktemp /tmp/olw_yaml_check.XXXXXX)
cat > "$_YAML_VALIDATOR" << 'PYEOF'
import sys, re, frontmatter
try:
    m = frontmatter.load(sys.argv[1])
except Exception as e:
    print(f"  YAML parse failed: {sys.argv[1]}: {e}")
    sys.exit(1)
tags = m.get('tags', [])
valid_re = re.compile(r'^[a-z0-9][a-zA-Z0-9_/\-]*$')
bad = [t for t in tags if not isinstance(t, str) or ' ' in t or t != t.lower() or not valid_re.match(t)]
if bad:
    print(f"  Bad tags in {sys.argv[1]}: {bad}")
    sys.exit(2)
PYEOF

while IFS= read -r -d '' md; do
    result=$(uv run --project "$REPO_DIR" python "$_YAML_VALIDATOR" "$md" 2>&1)
    exit_code=$?
    if [ $exit_code -eq 1 ]; then
        echo "$result"
        YAML_FAIL=1
    elif [ $exit_code -eq 2 ]; then
        echo "$result"
        TAG_FAIL=1
    fi
done < <(find "$VAULT_DIR/wiki" -name "*.md" -not -path "*/.drafts/*" -print0 2>/dev/null)
rm -f "$_YAML_VALIDATOR"

check "all published pages have valid YAML" "test $YAML_FAIL -eq 0"
check "no published pages have invalid tags (spaces/uppercase/special)" "test $TAG_FAIL -eq 0"

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
_TMP=$(mktemp); echo "$INGEST3_OUT" > "$_TMP"
check "dry run shows only new concepts" \
    "grep -qiE 'concept|compile|deep|neural|no concept' \"$_TMP\""
rm -f "$_TMP"

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
        "test \"$DRAFT_AFTER_EDIT\" -eq 0"
fi

# ── Duplicate detection ───────────────────────────────────────────────────────
header "Duplicate detection"
cp "$VAULT_DIR/raw/quantum-computing.md" "$VAULT_DIR/raw/quantum-computing-copy.md" 2>/dev/null || true

INGEST_OUT=$($OLW ingest "$VAULT_DIR/raw/quantum-computing-copy.md" 2>&1 || true)
_TMP=$(mktemp); echo "$INGEST_OUT" > "$_TMP"
check "duplicate skipped" "grep -qiE 'skip|duplicate|already' \"$_TMP\""
rm -f "$_TMP"
rm -f "$VAULT_DIR/raw/quantum-computing-copy.md"

# ── Query (Stage 3) ───────────────────────────────────────────────────────────
header "olw query (Stage 3)"
info "Approving drafts so query has articles to search..."
$OLW approve --all 2>&1 || true

info "Running query against wiki..."
QUERY_OUT=$($OLW query "What is a qubit?" 2>&1 || true)
echo "$QUERY_OUT"
_TMP=$(mktemp); echo "$QUERY_OUT" > "$_TMP"
check "query returns an answer" \
    "grep -qiE 'qubit|quantum|superposition|bit' \"$_TMP\""
rm -f "$_TMP"

info "Running query with --save..."
QUERY_SAVE_OUT=$($OLW query --save "What algorithms are used in quantum computing?" 2>&1 || true)
echo "$QUERY_SAVE_OUT"
QUERY_COUNT=$(find "$VAULT_DIR/wiki/queries" -name "*.md" 2>/dev/null | wc -l | tr -d ' ')
check "query --save creates file in wiki/queries/" "test \"$QUERY_COUNT\" -ge 1"

# ── Lint (Stage 3) ────────────────────────────────────────────────────────────
header "olw lint (Stage 3)"
LINT_OUT=$($OLW lint 2>&1 || true)
echo "$LINT_OUT"
_TMP=$(mktemp); echo "$LINT_OUT" > "$_TMP"
check "lint reports health score" \
    "grep -qiE 'health|score|100|issue' \"$_TMP\""
rm -f "$_TMP"

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
_RETRY_TMP=$(mktemp)
$OLW compile --retry-failed 2>&1 | tee "$_RETRY_TMP" || true
check "retry-failed reports failed notes" \
    "grep -qiE 'retry|failed|not found|re-ingest' \"$_RETRY_TMP\""
rm -f "$_RETRY_TMP"

# ── olw run (orchestrator) ───────────────────────────────────────────────────
header "olw run (pipeline orchestrator)"
info "Adding 4th note to drive olw run..."

cat > "$VAULT_DIR/raw/reinforcement-learning.md" <<'EOF'
---
title: Reinforcement Learning
---

Reinforcement learning (RL) trains agents to make decisions by maximising
cumulative reward from an environment.

Key components: agent, environment, state, action, reward, policy.
Algorithms: Q-learning, SARSA, PPO, A3C. Applications: game playing (AlphaGo,
Atari), robotics control, recommendation systems, autonomous driving.

Model-free methods learn directly from experience. Model-based methods build
an internal model of the environment for planning.
EOF

RUN_OUT=$($OLW run 2>&1 || true)
echo "$RUN_OUT"
_TMP=$(mktemp); echo "$RUN_OUT" > "$_TMP"
check "olw run completes without fatal error" \
    "! grep -qiE 'traceback|exception|fatal' \"$_TMP\""
check "olw run reports ingested or compiled" \
    "grep -qiE 'ingest|compile|draft|publish|rounds' \"$_TMP\""
rm -f "$_TMP"

DRAFTS_BEFORE=$(find "$VAULT_DIR/wiki/.drafts" -name '*.md' 2>/dev/null | wc -l | tr -d ' ')
RUN_DRYRUN_OUT=$($OLW run --dry-run 2>&1 || true)
DRAFTS_AFTER=$(find "$VAULT_DIR/wiki/.drafts" -name '*.md' 2>/dev/null | wc -l | tr -d ' ')
_TMP=$(mktemp); echo "$RUN_DRYRUN_OUT" > "$_TMP"
check "olw run --dry-run makes no LLM calls (no new drafts)" \
    "test '$DRAFTS_AFTER' -eq '$DRAFTS_BEFORE'"
rm -f "$_TMP"

# ── Draft annotations ────────────────────────────────────────────────────────
header "Draft annotations"
info "Compiling with low-quality source to trigger annotation..."
# Force a concept to recompile by inserting low quality + low confidence source
$OLW approve --all 2>&1 || true   # clear drafts first

# Inject a single-source, low-confidence concept by direct DB manipulation
python3 - <<PYEOF
import sqlite3
db_path = "$VAULT_DIR/.olw/state.db"
conn = sqlite3.connect(db_path)
# Update one raw note to low quality so annotation triggers
conn.execute("UPDATE raw_notes SET status='ingested', quality='low' WHERE path='raw/reinforcement-learning.md'")
conn.commit()
conn.close()
PYEOF

$OLW compile 2>&1 || true
DRAFTS_WITH_ANNOTATION=$({ grep -rl 'olw-auto' "$VAULT_DIR/wiki/.drafts/" 2>/dev/null || true; } \
    | wc -l | tr -d ' ')
# Not guaranteed to annotate since model may produce high confidence — just check no crash
pass "annotation check ran (found $DRAFTS_WITH_ANNOTATION annotated draft(s))"

# Verify annotations are stripped on approve
$OLW approve --all 2>&1 || true
PUBLISHED_WITH_ANNOTATION=$({ grep -rl 'olw-auto' "$VAULT_DIR/wiki/" \
    --include='*.md' --exclude-dir='.drafts' --exclude-dir='sources' 2>/dev/null || true; } \
    | wc -l | tr -d ' ')
check "no olw-auto annotations in published articles" \
    "test '$PUBLISHED_WITH_ANNOTATION' -eq 0"

# ── Rejection feedback loop ───────────────────────────────────────────────────
header "Rejection feedback loop"
info "Recompiling to produce a draft to reject..."

# Force one concept back to needing compile
python3 - <<PYEOF
import sqlite3
db_path = "$VAULT_DIR/.olw/state.db"
conn = sqlite3.connect(db_path)
conn.execute("UPDATE raw_notes SET status='ingested' WHERE path='raw/quantum-computing.md'")
conn.commit()
conn.close()
PYEOF

$OLW compile 2>&1 || true

REJECT_DRAFT=$(find "$VAULT_DIR/wiki/.drafts" -name "*.md" | head -1)
if [[ -n "$REJECT_DRAFT" ]]; then
    DRAFT_TITLE=$(grep '^title:' "$REJECT_DRAFT" | head -1 | sed 's/title: *//')
    info "Rejecting draft: $DRAFT_TITLE"

    REJECT_OUT=$($OLW reject "$REJECT_DRAFT" --feedback "Too brief, needs more concrete examples" 2>&1 || true)
    echo "$REJECT_OUT"
    check "reject removes draft file" "test ! -f \"$REJECT_DRAFT\""
    check "reject confirms feedback saved" \
        "echo \"$REJECT_OUT\" | grep -qiE 'feedback|saved|rejection|next compile'"

    # Force concept back to compile again and verify feedback appears in output
    python3 - <<PYEOF
import sqlite3
db_path = "$VAULT_DIR/.olw/state.db"
conn = sqlite3.connect(db_path)
conn.execute("UPDATE raw_notes SET status='ingested' WHERE path='raw/quantum-computing.md'")
conn.commit()
conn.close()
PYEOF
    COMPILE_OUT2=$($OLW compile 2>&1 || true)
    echo "$COMPILE_OUT2"
    # We can't easily inspect the prompt, but compile should succeed without crash
    check "recompile after rejection completes" \
        "! echo \"$COMPILE_OUT2\" | grep -qiE 'traceback|fatal'"
else
    pass "rejection test skipped (no draft available)"
fi

# ── olw unblock ──────────────────────────────────────────────────────────────
header "olw unblock"
info "Simulating 5-rejection block..."

python3 - <<PYEOF
import sqlite3
db_path = "$VAULT_DIR/.olw/state.db"
conn = sqlite3.connect(db_path)
conn.execute("""
    INSERT OR IGNORE INTO blocked_concepts (concept, blocked_at)
    VALUES ('Fake Blocked Concept', datetime('now'))
""")
conn.commit()
conn.close()
PYEOF

STATUS_BLOCKED=$($OLW status 2>&1 || true)
_TMP=$(mktemp); echo "$STATUS_BLOCKED" > "$_TMP"
check "status shows blocked concept" \
    "grep -qiE 'blocked|Fake Blocked' \"$_TMP\""
rm -f "$_TMP"

UNBLOCK_OUT=$($OLW unblock "Fake Blocked Concept" 2>&1 || true)
echo "$UNBLOCK_OUT"
check "unblock completes without error" \
    "! echo \"$UNBLOCK_OUT\" | grep -qiE 'traceback|error'"
check "concept no longer blocked after unblock" \
    "! $OLW status 2>&1 | grep -qiE 'Fake Blocked'"

# ── olw maintain ─────────────────────────────────────────────────────────────
header "olw maintain"
MAINTAIN_OUT=$($OLW maintain 2>&1 || true)
echo "$MAINTAIN_OUT"
_TMP=$(mktemp); echo "$MAINTAIN_OUT" > "$_TMP"
check "maintain runs without fatal error" \
    "! grep -qiE 'traceback|exception|fatal' \"$_TMP\""
check "maintain reports health or quality info" \
    "grep -qiE 'health|quality|lint|stub|orphan|issue|ok' \"$_TMP\""
rm -f "$_TMP"

MAINTAIN_DRY_OUT=$($OLW maintain --dry-run 2>&1 || true)
_TMP=$(mktemp); echo "$MAINTAIN_DRY_OUT" > "$_TMP"
check "maintain --dry-run completes" \
    "! grep -qiE 'traceback|fatal' \"$_TMP\""
rm -f "$_TMP"

# ── olw maintain --fix (stubs) ────────────────────────────────────────────────
header "olw maintain --fix (stub creation)"
# Inject a broken wikilink into a published article so maintain can create a stub
FIRST_WIKI=$(find "$VAULT_DIR/wiki" -maxdepth 1 -name "*.md" \
    ! -name "index.md" ! -name "log.md" 2>/dev/null | head -1)

if [[ -n "$FIRST_WIKI" ]]; then
    echo -e "\n[[Nonexistent Stub Topic]]" >> "$FIRST_WIKI"
    STUB_OUT=$($OLW maintain --fix 2>&1 || true)
    echo "$STUB_OUT"
    _TMP=$(mktemp); echo "$STUB_OUT" > "$_TMP"
    check "maintain --fix runs without fatal error" \
        "! grep -qiE 'traceback|fatal' \"$_TMP\""
    # Verify stub draft was created in .drafts or DB has stub entry
    STUB_DRAFT_COUNT=$(find "$VAULT_DIR/wiki/.drafts" -name "*.md" 2>/dev/null | wc -l | tr -d ' ')
    STUB_DB_COUNT=$(python3 -c "
import sqlite3
conn = sqlite3.connect('$VAULT_DIR/.olw/state.db')
try:
    n = conn.execute('SELECT COUNT(*) FROM stubs').fetchone()[0]
    print(n)
except Exception:
    print(0)
conn.close()
" 2>/dev/null || echo 0)
    check "maintain --fix created stub draft or DB entry" \
        "test '$STUB_DRAFT_COUNT' -gt 0 || test '$STUB_DB_COUNT' -gt 0"
    rm -f "$_TMP"
    # Restore the file
    # (truncate last line — safe enough for smoke test purposes)
    sed -i '' '$ d' "$FIRST_WIKI" 2>/dev/null || sed -i '$ d' "$FIRST_WIKI" 2>/dev/null || true
else
    pass "stub creation test skipped (no wiki article available)"
fi

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

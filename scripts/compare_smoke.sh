#!/usr/bin/env bash
# compare_smoke.sh — end-to-end smoke for the simplified `olw compare`
#
# Creates a temporary vault, runs `olw compare` against that vault using the
# current config as baseline and an overridden challenger model, then checks
# that reports are generated and the active vault is unchanged outside
# `.olw/compare/`.

set -euo pipefail

FAST_MODEL="${FAST_MODEL:-gemma4:e4b}"
HEAVY_MODEL="${HEAVY_MODEL:-$FAST_MODEL}"
CHALLENGER_HEAVY_MODEL="${CHALLENGER_HEAVY_MODEL:-}"
PROVIDER="${PROVIDER:-ollama}"
PROVIDER_URL="${PROVIDER_URL:-}"
KEEP_OUT="${KEEP_OUT:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ -n "${OUT_DIR:-}" ]]; then
    KEEP_OUT=1
    mkdir -p "$OUT_DIR"
else
    OUT_DIR="$(mktemp -d)"
fi

VAULT_DIR="$OUT_DIR/vault"
COMPARE_DIR="$VAULT_DIR/.olw/compare"

if [[ -z "$PROVIDER_URL" ]]; then
    case "$PROVIDER" in
        ollama) PROVIDER_URL="${OLLAMA_URL:-http://localhost:11434}" ;;
        lm_studio) PROVIDER_URL="${LM_STUDIO_URL:-http://localhost:1234/v1}" ;;
        vllm) PROVIDER_URL="http://localhost:8000/v1" ;;
        *)
            echo "Unsupported PROVIDER without explicit PROVIDER_URL: $PROVIDER" >&2
            exit 1
            ;;
    esac
fi

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

check_json_file() {
    local desc="$1"
    local path="$2"
    if uv run python - <<'PY' "$path" > /dev/null 2>&1
import json
import sys

with open(sys.argv[1]) as f:
    json.load(f)
PY
    then
        pass "$desc"
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        fail "$desc"
    fi
}

cleanup() {
    if [[ "$KEEP_OUT" == "0" ]]; then
        rm -rf "$OUT_DIR"
    else
        info "Output retained at: $OUT_DIR"
    fi
}
trap cleanup EXIT

dir_hash() {
    local root="$1"
    uv run python - <<'PY' "$root"
import hashlib
import sys
from pathlib import Path

root = Path(sys.argv[1])
h = hashlib.sha256()
for p in sorted(root.rglob("*")):
    rel = p.relative_to(root)
    if rel.parts[:2] == (".olw", "compare"):
        continue
    h.update(str(rel).encode())
    h.update(b"\x00")
    if p.is_file():
        h.update(p.read_bytes())
    h.update(b"\x01")
print(h.hexdigest())
PY
}

resolve_loaded_model() {
    local model="$1"
    uv run python - <<'PY' "$PROVIDER" "$PROVIDER_URL" "$model"
import sys
import tempfile
import re
from pathlib import Path

from obsidian_llm_wiki.config import Config
from obsidian_llm_wiki.client_factory import build_client

provider, url, model = sys.argv[1:4]
def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())

with tempfile.TemporaryDirectory(prefix="compare-smoke-") as tmp:
    vault = Path(tmp)
    (vault / "raw").mkdir()
    (vault / "wiki").mkdir()
    (vault / ".olw").mkdir()
    (vault / "wiki.toml").write_text(
        f"[models]\nfast = \"{model}\"\nheavy = \"{model}\"\n\n"
        f"[provider]\nname = \"{provider}\"\nurl = \"{url}\"\n"
    )
    cfg = Config.from_vault(vault)
    client = build_client(cfg)
    try:
        client.require_healthy()
        models = client.list_models()
    finally:
        try:
            client.close()
        except Exception:
            pass
if model in models:
    print(model)
    raise SystemExit(0)

wanted = norm(model)
matches = [m for m in models if wanted == norm(m) or wanted in norm(m) or norm(m) in wanted]
if len(matches) == 1:
    print(matches[0])
    raise SystemExit(0)

raise SystemExit(
    f"Model {model!r} is not loaded in {provider} at {url}. Available: {models}"
)
PY
}

select_alternate_model() {
    local baseline="$1"
    uv run python - <<'PY' "$PROVIDER" "$PROVIDER_URL" "$baseline"
import sys
import tempfile
from pathlib import Path

from obsidian_llm_wiki.config import Config
from obsidian_llm_wiki.client_factory import build_client

provider, url, baseline = sys.argv[1:4]
with tempfile.TemporaryDirectory(prefix="compare-smoke-") as tmp:
    vault = Path(tmp)
    (vault / "raw").mkdir()
    (vault / "wiki").mkdir()
    (vault / ".olw").mkdir()
    (vault / "wiki.toml").write_text(
        f"[models]\nfast = \"{baseline}\"\nheavy = \"{baseline}\"\n\n"
        f"[provider]\nname = \"{provider}\"\nurl = \"{url}\"\n"
    )
    cfg = Config.from_vault(vault)
    client = build_client(cfg)
    try:
        client.require_healthy()
        models = client.list_models()
    finally:
        try:
            client.close()
        except Exception:
            pass

preferred = [
    "nvidia/nemotron-3-nano-4b",
    "zai-org/glm-4.6v-flash",
    "qwen/qwen3.5-9b",
]
for candidate in preferred:
    if candidate in models and candidate != baseline and "embed" not in candidate.lower():
        print(candidate)
        raise SystemExit(0)

for candidate in models:
    if candidate != baseline and "embed" not in candidate.lower():
        print(candidate)
        raise SystemExit(0)

raise SystemExit(f"No alternate loaded model available in {provider} at {url}.")
PY
}

cd "$REPO_DIR"

header "Setup temporary vault"
mkdir -p "$VAULT_DIR/raw" "$VAULT_DIR/wiki" "$VAULT_DIR/.olw"
cat > "$VAULT_DIR/raw/note-1.md" <<'EOF'
# Gradient Descent

Gradient descent is an optimization method used to minimize a loss function.
EOF

cat > "$VAULT_DIR/raw/note-2.md" <<'EOF'
# Chain Rule

The chain rule is used to compute derivatives of composed functions.
EOF

cat > "$VAULT_DIR/raw/note-3.md" <<'EOF'
# Backpropagation

Backpropagation applies the chain rule to train neural networks.
EOF

cat > "$OUT_DIR/queries.toml" <<'EOF'
[[query]]
id = "q1"
question = "What is backpropagation?"
expected_contains = ["chain rule"]
EOF

pass "temporary vault created"
PASS_COUNT=$((PASS_COUNT + 1))

header "Provider preflight"
FAST_MODEL_RESOLVED="$(resolve_loaded_model "$FAST_MODEL")"
HEAVY_MODEL_RESOLVED="$(resolve_loaded_model "$HEAVY_MODEL")"
pass "baseline model loaded: $FAST_MODEL_RESOLVED"
PASS_COUNT=$((PASS_COUNT + 1))
if [[ -n "$CHALLENGER_HEAVY_MODEL" ]]; then
    CHALLENGER_HEAVY_MODEL_RESOLVED="$(resolve_loaded_model "$CHALLENGER_HEAVY_MODEL")"
else
    CHALLENGER_HEAVY_MODEL_RESOLVED="$(select_alternate_model "$HEAVY_MODEL_RESOLVED")"
fi
if [[ "$CHALLENGER_HEAVY_MODEL_RESOLVED" == "$HEAVY_MODEL_RESOLVED" ]]; then
    fail "challenger model resolves to the same model as the baseline"
fi
pass "challenger model loaded: $CHALLENGER_HEAVY_MODEL_RESOLVED"
PASS_COUNT=$((PASS_COUNT + 1))

cat > "$VAULT_DIR/wiki.toml" <<EOF
[models]
fast = "$FAST_MODEL_RESOLVED"
heavy = "$HEAVY_MODEL_RESOLVED"

[provider]
name = "$PROVIDER"
url = "$PROVIDER_URL"
timeout = 600
fast_ctx = 16384
heavy_ctx = 32768

[pipeline]
auto_approve = false
auto_commit = false
auto_maintain = false
watch_debounce = 3.0
max_concepts_per_source = 8
ingest_parallel = false
EOF

HASH_BEFORE="$(dir_hash "$VAULT_DIR")"

header "Run olw compare"
if ! uv run olw compare \
    --vault "$VAULT_DIR" \
    --heavy-model "$CHALLENGER_HEAVY_MODEL_RESOLVED" \
    --queries "$OUT_DIR/queries.toml" \
    --format both \
    2>&1 | tee "$OUT_DIR/run.log"; then
    fail "compare command exited non-zero"
fi
pass "compare command exited 0"
PASS_COUNT=$((PASS_COUNT + 1))

RUN_DIR="$(find "$COMPARE_DIR" -mindepth 1 -maxdepth 1 -type d | head -1)"
[[ -n "$RUN_DIR" ]] || fail "no compare run directory created"
RESULTS_DIR="$RUN_DIR/results"

header "Report structure"
check "report.md exists" "[[ -s '$RESULTS_DIR/report.md' ]]"
check "report.json exists" "[[ -s '$RESULTS_DIR/report.json' ]]"
check "summary.json exists" "[[ -s '$RESULTS_DIR/summary.json' ]]"

REPORT_MD="$RESULTS_DIR/report.md"
for section in \
    "# olw compare" \
    "## Recommendation" \
    "## Next Steps" \
    "## Config Change" \
    "## Query Summary" \
    "## Vault Impact" \
    "## Structure And Reliability" \
    "## Representative Page Changes" \
    "## Operational Cost" \
    "## Caveats"; do
    check "report.md has '$section'" "grep -Fq '$section' '$REPORT_MD'"
done

REPORT_JSON="$RESULTS_DIR/report.json"
SUMMARY_JSON="$RESULTS_DIR/summary.json"
check_json_file "report.json parses" "$REPORT_JSON"
check_json_file "summary.json parses" "$SUMMARY_JSON"
check "summary.json has verdict" "grep -q '\"verdict\"' '$SUMMARY_JSON'"

header "Safety checks"
HASH_AFTER="$(dir_hash "$VAULT_DIR")"
[[ "$HASH_BEFORE" == "$HASH_AFTER" ]] || fail "active vault changed outside .olw/compare"
pass "active vault unchanged outside .olw/compare"
PASS_COUNT=$((PASS_COUNT + 1))

check "active wiki/queries not created" "[[ ! -d '$VAULT_DIR/wiki/queries' ]]"

header "Summary"
pass "$PASS_COUNT checks passed"

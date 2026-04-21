#!/usr/bin/env bash
# compare_smoke.sh — end-to-end smoke for `olw compare`
#
# Runs `olw compare --quick --self-test` against the built-in corpus with a
# real Ollama/OpenAI-compat backend, then asserts report structure (not
# values — quality judgement is out of scope for smoke).
#
# Usage:
#   ./scripts/compare_smoke.sh                               # Ollama, gemma4:e4b
#   PROVIDER=lm_studio ./scripts/compare_smoke.sh            # LM Studio
#   FAST_MODEL=llama3.2:latest ./scripts/compare_smoke.sh
#   OUT_DIR=/tmp/my-compare ./scripts/compare_smoke.sh       # keep output dir
#   SKIP_PULL=1 ./scripts/compare_smoke.sh                   # skip ollama pull
#
# Requirements:
#   - uv (https://docs.astral.sh/uv/)
#   - Ollama running (ollama serve)  — OR —  LM Studio running w/ model loaded

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
PROVIDER="${PROVIDER:-ollama}"

case "$PROVIDER" in
    ollama)
        PROVIDER_URL="${PROVIDER_URL:-${OLLAMA_URL:-http://localhost:11434}}"
        FAST_MODEL="${FAST_MODEL:-gemma4:e4b}"
        HEAVY_MODEL="${HEAVY_MODEL:-$FAST_MODEL}"
        ;;
    lm_studio)
        PROVIDER_URL="${PROVIDER_URL:-http://localhost:1234/v1}"
        FAST_MODEL="${FAST_MODEL:-google/gemma-4-e4b}"
        HEAVY_MODEL="${HEAVY_MODEL:-$FAST_MODEL}"
        ;;
    *)
        PROVIDER_URL="${PROVIDER_URL:-}"
        FAST_MODEL="${FAST_MODEL:-}"
        HEAVY_MODEL="${HEAVY_MODEL:-$FAST_MODEL}"
        if [[ -z "$PROVIDER_URL" || -z "$FAST_MODEL" ]]; then
            echo "ERROR: PROVIDER=$PROVIDER requires PROVIDER_URL and FAST_MODEL."
            exit 1
        fi
        ;;
esac

SKIP_PULL="${SKIP_PULL:-0}"
KEEP_OUT="${KEEP_OUT:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ -n "${OUT_DIR:-}" ]]; then
    KEEP_OUT=1
    mkdir -p "$OUT_DIR"
else
    OUT_DIR="$(mktemp -d)"
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
    if [[ "$KEEP_OUT" == "0" ]]; then
        rm -rf "$OUT_DIR"
    else
        info "Output retained at: $OUT_DIR"
    fi
}
trap cleanup EXIT

# ── Pre-flight ────────────────────────────────────────────────────────────────
header "Pre-flight"
info "Provider:   $PROVIDER"
info "Fast model: $FAST_MODEL"
info "Heavy model: $HEAVY_MODEL"
info "Output dir: $OUT_DIR"

if [[ "$PROVIDER" == "ollama" && "$SKIP_PULL" == "0" ]]; then
    info "Pulling $FAST_MODEL…"
    ollama pull "$FAST_MODEL" > /dev/null 2>&1 || fail "ollama pull $FAST_MODEL"
    if [[ "$HEAVY_MODEL" != "$FAST_MODEL" ]]; then
        info "Pulling $HEAVY_MODEL…"
        ollama pull "$HEAVY_MODEL" > /dev/null 2>&1 || fail "ollama pull $HEAVY_MODEL"
    fi
fi

cd "$REPO_DIR"

# ── Run compare --quick --self-test ───────────────────────────────────────────
header "Running olw compare --quick --self-test"

CONFIG_SPEC="smoke:fast=${FAST_MODEL},heavy=${HEAVY_MODEL}"
if [[ "$PROVIDER" != "ollama" ]]; then
    CONFIG_SPEC="${CONFIG_SPEC},provider=${PROVIDER},url=${PROVIDER_URL}"
fi

# --quick → 3 notes, 1 seed. --self-test duplicates config as two contestants.
# --skip-queries to shave runtime; report rendering still exercised.
if ! uv run olw compare \
    --config "$CONFIG_SPEC" \
    --quick \
    --self-test \
    --skip-queries \
    --out "$OUT_DIR" \
    --format both \
    --keep-artifacts \
    2>&1 | tee "$OUT_DIR/run.log"; then
    fail "compare command exited non-zero"
fi
pass "compare command exited 0"
PASS_COUNT=$((PASS_COUNT + 1))

# Resolve the one run_id subdir.
RUN_DIR="$(find "$OUT_DIR" -mindepth 1 -maxdepth 1 -type d | head -1)"
[[ -n "$RUN_DIR" ]] || fail "no run_id subdir created under $OUT_DIR"
RESULTS_DIR="$RUN_DIR/results"

# ── Report structure checks ───────────────────────────────────────────────────
header "Report structure"
check "report.md exists"   "[[ -s '$RESULTS_DIR/report.md' ]]"
check "report.json exists" "[[ -s '$RESULTS_DIR/report.json' ]]"

REPORT_MD="$RESULTS_DIR/report.md"
for section in \
    "# olw compare" \
    "## Contestants" \
    "## Overall verdict" \
    "## Per-dimension scores" \
    "## Advisory diagnostics" \
    "## Trade-off narrative" \
    "## Blind spots"; do
    check "report.md has '$section'" "grep -Fq '$section' '$REPORT_MD'"
done

# JSON has the version-pinning fields and both contestants.
REPORT_JSON="$RESULTS_DIR/report.json"
check "report.json parses"                  "uv run python -c 'import json,sys; json.load(open(\"$REPORT_JSON\"))'"
check "report.json has olw_version"         "grep -q '\"olw_version\"' '$REPORT_JSON'"
check "report.json has pipeline_prompt_hash" "grep -q '\"pipeline_prompt_hash\"' '$REPORT_JSON'"
check "report.json has 2 contestants (self-test)" \
    "[[ \$(uv run python -c 'import json;print(len(json.load(open(\"$REPORT_JSON\"))[\"contestants\"]))') == 2 ]]"

# Ephemeral vaults kept (--keep-artifacts).
check "artifact vault dir present" "[[ -d '$RUN_DIR/vaults' ]]"

# ── Summary ───────────────────────────────────────────────────────────────────
header "Summary"
pass "$PASS_COUNT checks passed"

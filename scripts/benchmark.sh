#!/usr/bin/env bash
# benchmark.sh — measure per-concept compile time on a real vault
#
# Requires a running Ollama instance.
# Usage:
#   bash scripts/benchmark.sh [--vault PATH] [--notes N]
#
# What it does:
#   1. Creates a temp vault with N synthetic raw notes
#   2. Runs olw ingest + olw compile, capturing timing output
#   3. Prints a summary: total time, avg per concept, slowest concepts
#
# Options:
#   --vault PATH   Use existing vault (skips setup; notes already ingested)
#   --notes N      Number of synthetic notes to generate (default: 5)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

VAULT=""
N_NOTES=5

while [[ $# -gt 0 ]]; do
    case "$1" in
        --vault) VAULT="$2"; shift 2 ;;
        --notes) N_NOTES="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

OLW="uv run --project $PROJECT_DIR olw"

# ── Setup ─────────────────────────────────────────────────────────────────────

if [[ -z "$VAULT" ]]; then
    VAULT="$(mktemp -d)"
    CLEANUP_VAULT=1
    echo "Created temp vault: $VAULT"

    $OLW init --vault "$VAULT" --yes 2>/dev/null || true
    mkdir -p "$VAULT/raw"

    TOPICS=("Machine Learning" "Neural Networks" "Reinforcement Learning"
            "Natural Language Processing" "Computer Vision" "Graph Theory"
            "Cryptography" "Quantum Computing" "Distributed Systems" "Databases")

    for i in $(seq 1 "$N_NOTES"); do
        TOPIC="${TOPICS[$((i % ${#TOPICS[@]}))]}"
        cat > "$VAULT/raw/note_${i}.md" <<EOF
---
title: Note $i — $TOPIC
---
# $TOPIC — Note $i

$TOPIC is a fundamental area of computer science. This note covers key concepts,
algorithms, and applications. Topics include core theory, practical implementations,
performance considerations, and recent advances in the field.

Key concepts: definitions, examples, comparisons, trade-offs.
Related areas: mathematics, statistics, software engineering.
EOF
    done
    echo "Generated $N_NOTES synthetic notes."
else
    CLEANUP_VAULT=0
fi

export OLW_VAULT="$VAULT"

# ── Ingest ────────────────────────────────────────────────────────────────────

echo ""
echo "── Ingest ───────────────────────────────────────────────────"
T_INGEST_START=$(date +%s%3N)
$OLW ingest 2>&1 | grep -E "ingested|Ingested|concept|Concept|chunk|Chunk" || true
T_INGEST_END=$(date +%s%3N)
T_INGEST=$(( (T_INGEST_END - T_INGEST_START) ))
echo "Ingest total: ${T_INGEST}ms"

# ── Compile ───────────────────────────────────────────────────────────────────

echo ""
echo "── Compile ──────────────────────────────────────────────────"
T_COMPILE_START=$(date +%s%3N)
COMPILE_OUT=$($OLW compile 2>&1)
T_COMPILE_END=$(date +%s%3N)
T_COMPILE=$(( (T_COMPILE_END - T_COMPILE_START) ))

# Print lines with timing info
echo "$COMPILE_OUT" | grep -E "\([0-9]+\.[0-9]+s\)|Draft written|Stub draft|compiled|Compiled" || echo "$COMPILE_OUT"

# ── Summary ───────────────────────────────────────────────────────────────────

echo ""
echo "── Summary ──────────────────────────────────────────────────"
echo "Ingest:  ${T_INGEST}ms"
echo "Compile: ${T_COMPILE}ms"

# Extract per-concept timings from compile output
TIMINGS=$(echo "$COMPILE_OUT" | grep -oE "[A-Za-z ]+: [0-9]+\.[0-9]+s\)" | \
          sed 's/: /\t/' | sed 's/)//' | sort -t$'\t' -k2 -rn 2>/dev/null || true)

if [[ -n "$TIMINGS" ]]; then
    echo ""
    echo "Slowest concepts (from log):"
    echo "$TIMINGS" | head -5 | while IFS=$'\t' read -r name secs; do
        printf "  %-40s %s\n" "$name" "$secs"
    done
fi

# Count drafts created
N_DRAFTS=$(find "$VAULT/wiki/.drafts" -name "*.md" 2>/dev/null | wc -l | tr -d ' ')
echo ""
echo "Drafts created: $N_DRAFTS"
echo "Notes processed: $N_NOTES"
if [[ $N_DRAFTS -gt 0 ]]; then
    AVG=$(( T_COMPILE / N_DRAFTS ))
    echo "Avg compile: ${AVG}ms/concept"
fi

# ── Cleanup ───────────────────────────────────────────────────────────────────

if [[ $CLEANUP_VAULT -eq 1 ]]; then
    rm -rf "$VAULT"
fi

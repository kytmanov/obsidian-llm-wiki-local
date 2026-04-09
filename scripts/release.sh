#!/usr/bin/env bash
# Step 1 — prepare release branch:  bash scripts/release.sh 0.2.0
# Step 2 — after PR merged:         bash scripts/release.sh 0.2.0 --tag
set -e

VERSION=${1:?Usage: bash scripts/release.sh <version> [--tag]}
TAG_ONLY=${2:-}

# Validate semver format
if ! [[ "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "Error: version must be X.Y.Z (got '$VERSION')"
  exit 1
fi

if [[ "$TAG_ONLY" == "--tag" ]]; then
  # Step 2: run after the release PR is merged
  git fetch origin master
  git checkout master
  git pull origin master

  git tag "v$VERSION"
  git push origin "v$VERSION"

  echo ""
  echo "Tag v$VERSION pushed — GitHub Actions will create the release automatically."
  echo "Track it at: https://github.com/kytmanov/obsidian-llm-wiki-local/actions"
  exit 0
fi

# Step 1: create release branch and bump version
git fetch origin master
git checkout -b "release/v$VERSION" origin/master

sed -i.bak "s/^version = \".*\"/version = \"$VERSION\"/" pyproject.toml && rm pyproject.toml.bak
echo "Bumped version to $VERSION"

git add pyproject.toml
git commit -m "chore: release v$VERSION"
git push origin "release/v$VERSION"

echo ""
echo "Next steps:"
echo "  1. Open a PR: release/v$VERSION → master, then merge it"
echo "  2. After merge run:  bash scripts/release.sh $VERSION --tag"

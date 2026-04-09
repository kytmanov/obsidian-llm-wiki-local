#!/usr/bin/env bash
# Usage: bash scripts/release.sh 0.2.0
set -e

VERSION=${1:?Usage: bash scripts/release.sh <version>}

# Validate semver-ish format
if ! [[ "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "Error: version must be X.Y.Z (got '$VERSION')"
  exit 1
fi

# Must be on master and up to date
BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [[ "$BRANCH" != "master" ]]; then
  echo "Error: must be on master (currently on '$BRANCH')"
  exit 1
fi

git fetch origin master
LOCAL=$(git rev-parse HEAD)
REMOTE=$(git rev-parse origin/master)
if [[ "$LOCAL" != "$REMOTE" ]]; then
  echo "Error: local master is not up to date with origin. Run: git pull"
  exit 1
fi

# Bump version in pyproject.toml
sed -i.bak "s/^version = \".*\"/version = \"$VERSION\"/" pyproject.toml && rm pyproject.toml.bak

echo "Bumped version to $VERSION"

# Commit and tag
git add pyproject.toml
git commit -m "chore: release v$VERSION"
git tag "v$VERSION"

# Push
git push origin master
git push origin "v$VERSION"

echo ""
echo "Released v$VERSION — GitHub Actions will create the release automatically."
echo "Track it at: https://github.com/kytmanov/obsidian-llm-wiki/actions"

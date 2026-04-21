# Release Process

This repo has a release helper script at `scripts/release.sh`, but the safest
release flow is still PR-first.

## Standard Flow

1. Start from up-to-date `master`.
2. Create a release branch named `release/vX.Y.Z`.
3. Bump version metadata:
   - `pyproject.toml`
   - `src/obsidian_llm_wiki/__init__.py`
4. Update `CHANGELOG.md` with a short human-facing entry.
5. Push `release/vX.Y.Z` and open a PR into `master`.
6. Merge the PR.
7. Run `bash scripts/release.sh X.Y.Z --tag` from a clean checkout.
8. Watch GitHub Actions create/publish the release.

## Important Notes

- Do not tag from a branch whose version files still point at the previous
  release. The `vX.Y.Z` tag must land on code that reports `X.Y.Z`.
- The helper script currently bumps only `pyproject.toml`; also update
  `src/obsidian_llm_wiki/__init__.py` manually.
- If branch protection blocks direct pushes to `master`, merge the release PR
  normally and only then run the `--tag` step.
- If your local `master` has diverged or contains local-only release commits,
  run the tag step from a fresh clone to avoid tagging the wrong commit.

## v0.6.0 Lessons Learned

- Keep the release branch name as `release/vX.Y.Z` from the start.
- A changelog-only release commit is not enough; version metadata must be part
  of the merged release state.
- Tagging after merge is correct, but only if the merged `master` already has
  the final version bump.
- Using a clean temporary clone for the final `--tag` step is a safe fallback
  when local branch state is messy.

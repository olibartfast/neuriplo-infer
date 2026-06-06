# Versioning and Changelog

## Overview

This project uses two files to track releases:

| File | Purpose |
|------|---------|
| `VERSION` | Single source of truth for the current version (read by CMake) |
| `CHANGELOG.md` | Human-readable history of notable changes per release |

## VERSION file

Contains a single line like `0.3.0` — a plain `MAJOR.MINOR.PATCH` semver, no suffix.

- CMake reads this file at configure time to set `project(neuriplo-infer VERSION X.Y.Z)`.
- After a release, `master` and `develop` carry the **same** `VERSION` (see Release workflow). It is bumped on the next release branch, never directly on `develop`.
- Follows [Semantic Versioning](https://semver.org/).

## CHANGELOG.md

Follows the [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format.

Sections per release:
- **Added** — new features
- **Changed** — changes to existing functionality
- **Fixed** — bug fixes
- **Removed** — removed features
- **Deprecated** — features marked for future removal

Unreleased work goes under the `[Unreleased]` heading at the top.

## Day-to-day workflow

When merging a PR into `develop`, add a line under `[Unreleased]` in the appropriate section. Example:

```markdown
## [Unreleased]

### Added
- Support for new model type `yolo26seg`
```

## Release workflow

This project follows a [Gitflow](https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow) model. After a release, `master` and `develop` point at the **same** tagged commit; `develop` then moves ahead again as features land. The sibling repos (`neuriplo-tasks`, `neuriplo`, `videocapture`) follow the same scheme and must be released first.

1. **Create a release branch** from `develop`:
   ```
   git checkout develop
   git checkout -b release/0.3.0
   ```

2. **Pin sibling refs and set VERSION** — run the release helper:
   ```
   scripts/cut_release.sh 0.3.0
   ```
   This sets `VERSION` to `0.3.0` and rewrites `versions.env`, pinning `neuriplo`, `neuriplo-tasks`, and `videocapture` each to its **own current release tag**. Siblings version independently — videocapture may stay at an older tag while the others advance. Every pin must be a concrete `vX.Y.Z` tag, never a branch like `master`/`develop`. `scripts/validate_release_pins.sh` enforces this, as do the `pre-push` hook and the `release-guard` workflow. Tag the sibling repos *before* cutting the neuriplo-infer release so their tags exist on the remote.

3. **Update CHANGELOG.md** — rename `[Unreleased]` to the new version with today's date, add a fresh empty `[Unreleased]`, and update the comparison links at the bottom.

4. **Commit, merge into `master`, and tag**:
   ```
   git commit -am "release: v0.3.0"
   git checkout master
   git merge --no-ff release/0.3.0
   git tag -a v0.3.0 -m "Release v0.3.0"
   ```

5. **Fast-forward `develop` to `master`** so both point at the release commit, then delete the release branch:
   ```
   git checkout develop
   git merge --ff-only master
   git branch -d release/0.3.0
   ```

6. **Push**:
   ```
   git push origin master develop v0.3.0
   ```

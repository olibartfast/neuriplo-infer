# Versioning and Changelog

## Overview

This project uses two files to track releases:

| File | Purpose |
|------|---------|
| `VERSION` | Single source of truth for the current version (read by CMake) |
| `CHANGELOG.md` | Human-readable history of notable changes per release |

## VERSION file

Contains a single line like `0.2.0-dev`.

- The `-dev` suffix indicates unreleased development work on `develop`.
- CMake reads this file at configure time and strips the suffix to set `project(vision-inference VERSION X.Y.Z)`.
- Follows [Semantic Versioning](https://semver.org/): `MAJOR.MINOR.PATCH`.

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

1. **Create a release branch** from `develop`:
   ```
   git checkout -b release/0.2.0 develop
   ```

2. **Update VERSION** — remove the `-dev` suffix:
   ```
   0.2.0
   ```

3. **Update CHANGELOG.md** — rename `[Unreleased]` to the new version with today's date, and add a fresh empty `[Unreleased]` section:
   ```markdown
   ## [Unreleased]

   ## [0.2.0] - 2026-04-15

   ### Added
   - ...
   ```
   Update the comparison links at the bottom:
   ```markdown
   [Unreleased]: https://github.com/olibartfast/vision-inference/compare/v0.2.0...HEAD
   [0.2.0]: https://github.com/olibartfast/vision-inference/compare/v0.1.0...v0.2.0
   ```

4. **Merge into `master`** and tag:
   ```
   git checkout master
   git merge release/0.2.0
   git tag v0.2.0
   git push origin master --tags
   ```

5. **Bump develop** — merge back and set the next dev version:
   ```
   git checkout develop
   git merge release/0.2.0
   ```
   Update `VERSION` to `0.3.0-dev`, commit, push.

# Feature Plan — Move app-surface tests from neuriplo-platform

Derived from `requirements.md`.

## Group 1 — Census (done in requirements)

1. Four dirs confirmed with full file lists; all path logic is
   `parents[2]`/`REPOS_ROOT`-relative and survives a verbatim move.
2. Platform references to update identified: `integration-tests/README.md`,
   `AGENTS.md` build commands; `check_platform.py` needs no change (it
   validates remaining dirs generically).

## Group 2 — Move + re-anchor (this repo)

3. Copy the four dirs verbatim to `integration-tests/`.
4. Re-anchor: smoke `run.py` — drop the `versions.yaml`/sibling-tag half,
   keep the runner dry-run half with `ROOT` = this repo; rewrite README usage
   strings (`From neuriplo-platform` → `From neuriplo-infer`,
   `../neuriplo-infer/...` → repo-relative, owning-repos blocks list this repo
   as owner).
5. Register cheap checks in `app/test/CMakeLists.txt` (or a root-level CTest
   include): scaffold/existence checks and dry-run paths only.

## Group 3 — Platform cleanup (separate commit, that repo)

6. Delete the four dirs; update `integration-tests/README.md` matrix;
   remove the smoke line from `AGENTS.md`. Verify with
   `scripts/check_platform.py`.

## Group 4 — Contract and verification

7. `CHANGELOG.md` entry; cite moved tests from `docs/BackendCoverage.md`
   Evidence where they gate a cell.
8. Run `validation.md`: default + test builds, full `ctest`, sync `--check`,
   moved dry-run checks, hyperlink checks, `check_platform.py` on the
   platform side.

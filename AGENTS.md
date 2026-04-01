# Review Instructions

## Repository workflow

- `develop` is the protected integration branch.
- `master` is the protected release-only branch.
- All normal feature, fix, refactor, docs, and chore work should land through pull requests into `develop`.
- Pull requests into `master` are release PRs only and should be treated as release-safety checks.
- Before creating a release, update both `VERSION` and `CHANGELOG.md`.
- After finishing a release, delete any temporary branches created ad hoc for that release.
- Do not suggest switching this repository to a `main`-centric trunk workflow.

## Review focus

Focus on:
- Correctness and edge cases
- Backward compatibility
- Performance regressions
- Missing tests
- Unsafe file, path, process, or network handling
- API consistency
- Build, packaging, and release safety

Avoid:
- Trivial style-only comments
- Major rewrites unless clearly justified
- Workflow suggestions that bypass `develop` as the integration branch

## C++ review focus

- Ownership and lifetime issues
- Thread safety and exception safety
- ABI or API changes and unnecessary copies
- Const-correctness

## ML and inference review focus

- Shape and dtype assumptions
- Device placement assumptions
- Latency regressions
- Memory copies and synchronization points
- Backend fallback behavior and logging

## Agentic maintenance assets

- Cluster-level metadata and runbooks live under `ops/`.
- Use `ops/CLUSTER_MAP.yaml` as the source of truth for repo ownership, dependency edges, and validation order.
- Use `ops/repo-meta/*.yaml` for repo-specific build, test, benchmark, and API-surface metadata.
- Use `ops/policies.yaml` before proposing or implementing automated changes; changes outside the allowed classes require human review.
- Use `ops/runbooks/` for the execution flow for CI triage and cross-repo API migrations.

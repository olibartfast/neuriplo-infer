# Review Instructions

## Repository workflow

- `develop` is the protected integration branch.
- `master` is the protected release-only branch.
- All normal feature, fix, refactor, docs, and chore work should land through pull requests into `develop`.
- Pull requests into `master` are release PRs only and should be treated as release-safety checks.
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

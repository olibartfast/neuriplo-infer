# Review Instructions

## System overview

`vision-inference` is the application-layer repo in the `vision-stack` cluster.

- It owns the CLI, app configuration, runtime wiring, visualization, and end-to-end execution flow.
- It consumes task contracts from `vision-core`.
- It consumes backend orchestration and runtime compatibility from `neuriplo`.
- It consumes source and video backend behavior from `videocapture`.

A sibling application repo, [vision-tracking](https://github.com/olibartfast/vision-tracking), handles detection + tracking pipelines using the same shared libraries. Another sibling, [tritonic](https://github.com/olibartfast/tritonic), is a Triton Inference Server client for CV tasks that also consumes vision-core. Both maintain their own ops control planes independently — vision-inference does not depend on them.

Treat `ops/CLUSTER_MAP.yaml` as the source of truth for repo roles, dependency edges, validation order, and coordinator/worker/verifier responsibilities.

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

## Standard workflow

When operating as an agent in this repo, follow this loop:

1. Observe the task, failing signal, or requested change.
2. Diagnose the owning repo, dependency edge, and allowed change class using `ops/CLUSTER_MAP.yaml` and `ops/policies.yaml`.
3. Act with the smallest reviewable change that fixes the issue without widening scope.
4. Verify using repo-local checks first, then downstream validation when a declared contract edge is affected.

Stop and escalate to a human if the required work falls into a forbidden change class or changes inference semantics rather than mechanical wiring.

## Repo-local entrypoints

Use the canonical repo-local commands from `ops/repo-meta/vision-inference.yaml`:

- Configure default build:
  - `cmake -S . -B build -DDEFAULT_BACKEND=OPENCV_DNN -DCMAKE_BUILD_TYPE=Release`
- Configure test build:
  - `cmake -S . -B build-test -DDEFAULT_BACKEND=OPENCV_DNN -DENABLE_APP_TESTS=ON -DCMAKE_BUILD_TYPE=Release`
- Build default target:
  - `cmake --build build`
- Build test target:
  - `cmake --build build-test`
- Run tests:
  - `ctest --test-dir build-test --output-on-failure`

Use the benchmark smoke command from `ops/repo-meta/vision-inference.yaml` only when the required weights are available.

## Operational constraints

- Preserve CLI compatibility unless the task is an explicitly reviewed contract change.
- Preserve output schema, backend fallback behavior, and latency-sensitive paths.
- Keep changes small and reviewable.
- For cross-repo contract work, validate in the declared order: repo-local checks first, then downstream integration, then performance/output checks.
- PRs produced by agents should include evidence consistent with `ops/PR_EVIDENCE_TEMPLATE.md`.
- Consult `docs/Roadmap.md` for project direction when evaluating whether a change aligns with planned work.

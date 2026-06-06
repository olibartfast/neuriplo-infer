# Review Instructions

## System overview

`neuriplo-infer` is the application-layer repo in the `vision-stack` cluster.

- It owns the CLI, app configuration, runtime wiring, visualization, and end-to-end execution flow.
- It consumes task contracts from `neuriplo-tasks`.
- It consumes backend orchestration and runtime compatibility from `neuriplo`.
- It consumes source and video backend behavior from `videocapture`.

A sibling application repo, [vision-tracking](https://github.com/olibartfast/vision-tracking), handles detection + tracking pipelines using the same shared libraries. Another sibling, [tritonic](https://github.com/olibartfast/tritonic), is a Triton Inference Server client for CV tasks that also consumes neuriplo-tasks. Both maintain their own ops control planes independently — neuriplo-infer does not depend on them.

Treat `ops/CLUSTER_MAP.yaml` as the source of truth for repo roles, dependency edges, validation order, and coordinator/worker/verifier responsibilities.

## Do-not-skip automated steps

Every agent taking ownership of this repo must know these easily-missed steps
(see the full checklist under "Documentation checklist when wiring a new task
type" and the always-on rule `.cursor/rules/new-task-type-checklist.mdc`):

1. **Supported-model-types docs are generated, not hand-written.** The
   `<!-- SUPPORTED_MODEL_TYPES -->` block in `README.md` and
   `docs/generated/supported-model-types.md` come from the neuriplo-tasks README via
   `python3 scripts/sync_supported_model_types.py [--neuriplo-tasks-readme <path>]`.
   `ci.yml` runs it with `--check`; a stale block fails CI. Run it (not a manual
   edit) whenever neuriplo-tasks adds/changes a task or model type.
2. **App task routing must match neuriplo-tasks.** `VisionApp::getTaskType`
   (`app/src/VisionAppTaskRouting.cpp`) must map each type string to the same
   `TaskType` that `neuriplo_tasks::TaskFactory` builds.

## Repository workflow

- `develop` is the protected integration branch.
- `master` is the protected release-only branch.
- All normal feature, fix, refactor, docs, and chore work should land through pull requests into `develop`.
- Pull requests into `master` are release PRs only and should be treated as release-safety checks.
- Before creating a release, update both `VERSION` and `CHANGELOG.md`.
- Before tagging a release, pin sibling refs in `versions.env` for reproducibility.
  - Tag the matching commit in each sibling repo (`neuriplo-tasks`, `neuriplo`, `videocapture`) with the same tag name (e.g. `v0.3.0`) first.
  - Then run `scripts/cut_release.sh <version>` (e.g. `0.3.0`) to bump `VERSION` and rewrite the `NEURIPLO_VERSION` / `VIDEOCAPTURE_VERSION` / `NEURIPLO_TASKS_VERSION` pins in `versions.env`. The script refuses to proceed if any sibling is missing the matching tag.
  - `scripts/validate_release_pins.sh <tag>` is the same check used by the pre-push hook and the `release-guard.yml` CI workflow; run it manually if you want to sanity-check before pushing.
  - Without these pins, checking out an old neuriplo-infer tag fetches sibling `master` at fetch time, which keeps moving — builds drift.
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

Use the canonical repo-local commands from `ops/repo-meta/neuriplo-infer.yaml`:

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

Use the benchmark smoke command from `ops/repo-meta/neuriplo-infer.yaml` only when the required weights are available.

## Documentation checklist when wiring a new task type

When a new task type is added end-to-end (neuriplo-tasks → neuriplo → neuriplo-infer), update **all** of the following before closing the work:

**neuriplo-tasks:**
1. `## Features` bullet list in `README.md` — one line for the new task.
2. `<!-- TASKFACTORY_MODEL_LIST:START/END -->` block in `README.md` — type strings, contract, backend requirements.
3. `export/<task_domain>/` directory — setup/download guide; update `export/README.md` tree and reference links (use absolute GitHub URLs).

**neuriplo-infer:**
4. `## Key Features` bullet in `README.md` — update the task list inline (not synced from neuriplo-tasks).
5. Run `python3 scripts/sync_supported_model_types.py --neuriplo-tasks-readme <path>` and commit the updated `README.md` and `docs/generated/supported-model-types.md`.

Missing any of these makes the task invisible to users reading the top-level READMEs.

---

## Skipping CI for docs-only commits

CI workflows have `paths-ignore` for `**.md` and `docs/**` — pure documentation pushes skip CI automatically.

For mixed commits (docs + code) where CI is still unnecessary, add `[skip ci]` to the commit message subject line.

## Operational constraints

- Preserve CLI compatibility unless the task is an explicitly reviewed contract change.
- Preserve output schema, backend fallback behavior, and latency-sensitive paths.
- Keep changes small and reviewable.
- For cross-repo contract work, validate in the declared order: repo-local checks first, then downstream integration, then performance/output checks.
- PRs produced by agents should include evidence consistent with `ops/PR_EVIDENCE_TEMPLATE.md`.
- Consult `docs/Roadmap.md` for project direction when evaluating whether a change aligns with planned work.

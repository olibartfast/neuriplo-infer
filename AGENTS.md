# Review Instructions

## System overview

`neuriplo-infer` is the application-layer repo in the `vision-stack` cluster.

- It owns the CLI, app configuration, runtime wiring, visualization, and end-to-end execution flow.
- It consumes task contracts from `neuriplo-tasks`.
- It consumes backend orchestration and runtime compatibility from `neuriplo`.
- It consumes source and video backend behavior from `videocapture`.

A sibling application repo, [neuriplo-track](https://github.com/olibartfast/neuriplo-track), handles detection + tracking pipelines using the same shared libraries. Another sibling, [tritonic](https://github.com/olibartfast/tritonic), is a Triton Inference Server client for CV tasks that also consumes neuriplo-tasks. Both maintain their own ops control planes independently — neuriplo-infer does not depend on them.

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
2. **App task routing must match neuriplo-tasks.** `getTaskTypeForModel`
   (`app/src/NeuriploInferTaskRouting.cpp`) must map each type string to the same
   `TaskType` that `neuriplo_tasks::TaskFactory` builds.
3. **The `## Key Features` bullet in README.md is manual, not synced.** When
   a new task **category** (not just a new model type within an existing category)
   is added in neuriplo-tasks, update the parenthesized task list under
   `## Key Features`. This is step 4 in the full documentation checklist below.

## Repository workflow (GitFlow — mandatory)

Agents must follow the [Atlassian GitFlow workflow](https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow).
See `.cursor/rules/gitflow-workflow.mdc` for branch naming and merge order. In this
repo GitFlow **`main`** is **`master`**; **`develop`** is the integration branch.

- **`feature/*`** — branch from `develop`; merge back to `develop` via PR. Never
  target `master`.
- **`release/*`** — branch from `develop` when preparing a version; only release
  fixes/docs (no new features). Merge to `master`, tag `vX.Y.Z`, then merge the
  same branch back into `develop` and delete it locally and on `origin`.
- **`hotfix/*`** — branch from `master` for production patches; merge to `master`
  (tag), then merge back into `develop` and delete it locally and on `origin`.
- Pull requests into `master` are release or hotfix merges only.
- **Back-merge immediately after every merge into `master`.** The merge commit
  created on `master` must be merged back into `develop` right away
  (`git checkout develop && git merge master && git push`), so `master`,
  `develop`, and the release tag converge on the same commit. `develop` being
  even one commit behind `master` is a workflow violation — verify with
  `git rev-list --left-right --count origin/develop...origin/master` (must be `0 0`).

Release prep on `release/<version>`:

- Update `CHANGELOG.md` and run `scripts/cut_release.sh <version>` (bumps
  `VERSION` and sibling pins in `versions.env`).
- Tag matching commits in sibling repos (`neuriplo-tasks`, `neuriplo`,
  `videocapture`, `neuriplo-kserve-client`) before cutting the release if pins
  must move.
- Run `scripts/validate_release_pins.sh vX.Y.Z` (same check as the pre-push hook
  and `release-guard.yml` CI).
- After pushing the tag, **Release Guard** validates pins; **Publish GitHub Release**
  CI (`.github/workflows/publish-github-release.yml`) then creates the GitHub
  Release from `CHANGELOG.md`. A pushed git tag alone does not appear on the
  Releases page.
- Without concrete pins, checking out an old neuriplo-infer tag fetches sibling
  `master` at fetch time — builds drift.

Do not suggest trunk-based or `main`-only workflows for this repository.

## Agent Commit Signing

Every commit produced by an AI agent MUST include a `Co-authored-by` trailer
that identifies the agent, the LLM model used, and the agent vendor. This makes
agent contributions visible in GitHub's contribution graph and `git shortlog`.

Format: `Co-Authored-By: <Agent> <Model> <<vendor-email>>`

| Agent | Vendor email | Example trailer |
|-------|-------------|-----------------|
| Cursor | `cursoragent@cursor.com` | `Co-authored-by: Cursor <cursoragent@cursor.com>` |
| Claude Code | `noreply@anthropic.com` | `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>` |
| Opencode | `agent@opencode.ai` | `Co-Authored-By: Opencode via DeepSeek V4 Pro <agent@opencode.ai>` |

The model name MUST match the LLM the agent is powered by (check the system
prompt). If the model changes across sessions, the trailer must reflect the
model used for that specific commit.

Place the trailer in the commit body (after the subject line and blank line),
not the subject.

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

## Standard workflow

When operating as an agent in this repo, follow this loop:

1. Observe the task, failing signal, or requested change.
2. Diagnose the owning repo, dependency edge, and allowed change class from the dependency graph and change policy.
3. Act with the smallest reviewable change that fixes the issue without widening scope.
4. Verify using repo-local checks first, then downstream validation when a declared contract edge is affected.

Stop and escalate to a human if the required work falls into a forbidden change class or changes inference semantics rather than mechanical wiring.

## Repo-local entrypoints

Use the canonical repo-local commands:

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

Run the benchmark smoke command only when the required weights are available.

## Hyperlink verification

When editing `README.md` or any documentation with hyperlinks:
- Verify all relative links resolve to existing files in the repo (`ls <path>`).
- Verify absolute GitHub URLs are reachable (use `curl -sI <url>` or a quick fetch).
- Prefer absolute GitHub blob/tree URLs over fragile cross-repo relative paths (e.g. `../../../neuriplo/docs/foo.md`).

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
- PRs produced by agents should include evidence of validation performed and any cross-repo impact.

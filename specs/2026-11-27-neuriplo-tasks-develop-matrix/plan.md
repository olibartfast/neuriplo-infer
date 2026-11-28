# Feature Plan — neuriplo-tasks develop matrix

Derived from `requirements.md`. Numbered groups, dependency order, each ending
in something observable. This packet produces evidence and a verdict; it ships
no code or pin change.

## Group 1 — Pin the moving target

1. Resolve `neuriplo-tasks@develop` to a commit SHA (`git ls-remote`), record
   it at the top of `validation.md`. All work below uses that SHA; if producer
   `develop` moves mid-packet, re-resolve once, note both SHAs, and re-run only
   the affected cells.
2. Fresh-dir configure against the SHA:
   `cmake -S . -B build-tasks-develop -DNEURIPLO_TASKS_VERSION=<sha> -DCMAKE_BUILD_TYPE=Release`.
   Observable: configure log shows
   `neuriplo-tasks: <sha> [source: ...]` and the build completes. Any fetch- or
   configure-shape failure is finding #1 (filed upstream).

## Group 2 — Contract parity (no hardware needed)

3. Point `scripts/sync_supported_model_types.py` at the develop checkout and
   regenerate to a scratch copy; diff against the committed
   `docs/generated/supported-model-types.md`. Observable: a type-string
   add/remove/rename list. No committed doc changes.
4. Walk every `--type` in the diff plus the full existing list through
   `getTaskTypeForModel` (`app/src/NeuriploInferTaskRouting.cpp`) against the
   `TaskFactory` built from the SHA. Observable: a routing-parity table
   (match / mismatch with upstream issue link). Mismatches stop the matrix for
   that task family until triaged — running gates on misrouted types would
   produce misleading passes.

## Group 3 — Build and dry-run matrix

5. Fresh-dir configure + build for each supported `-DDEFAULT_BACKEND`, plus
   the `integration-tests/local-inference-smoke` suite and the
   `e2e-example-emulation.yml` preset list at the develop SHA. Observable:
   per-backend build verdict and the smoke/DR checklist. This needs no GPU and
   runs anywhere, including CI-shaped runners.

## Group 4 — Live gates where hardware exists

6. On Jetson Orin (TRT 10.16.2) and CPU backends, re-run the representative
   preset per Task × Backend cell, graded to the `backend-coverage` bars (P /
   RUNTIME-PASS / DR). Reuse the pinned-release models/engines wherever the
   contract is unchanged; rebuild engines only where the export contract
   moved. Observable: per-cell verdict with `run_report.json`, render
   artifact, and SHA-pinned evidence notes in `validation.md` (scoreboard
   itself untouched).
7. Cluster-only cells (x86_64 TRT 11.2, MGX/ROCm) go through
   `scripts/slurm_backend_matrix.py` with the develop-built binaries; record
   `submitted.json` / `result.json` outcomes. DR bar only, per hardware
   reality.

## Group 5 — Report and file

8. File each failure upstream in `neuriplo-tasks` with repro (SHA, command,
   log excerpt); link every issue from `validation.md`. Observable: zero
   untracked failures.
9. Write the packet verdict: per-cell table (pass / fail+issue / untestable
   with reason), the resolved SHA, and a pin-bump recommendation (ready /
   blocked-by). Propose whether this becomes a recurring roadmap phase
   (requirements.md Open Questions).

> No branch commits for config/build scratch dirs, weights, engines, or
> tensors — evidence notes and verdicts only. Deviation 2026-11-27
> (maintainer direction: fix, don't file): routing shims, contract tests,
> and the pre-push AGENTS.md rule also landed on this branch; see
> validation.md deviations.

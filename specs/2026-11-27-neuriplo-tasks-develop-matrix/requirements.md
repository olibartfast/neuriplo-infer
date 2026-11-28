# Feature Requirements — neuriplo-tasks develop matrix

Roadmap phase: testing producer-`develop` (unphased; see Open Questions)
Branch: `feature/neuriplo-tasks-develop-matrix`

## Goal

Determine, with evidence, whether `neuriplo-infer` on `develop` works against
the `develop` tip of the producer repo `neuriplo-tasks` across the full
Task × Backend matrix — without changing any pin, claim, or shipped behavior.
The outcome is a verdict per cell (pass / fail with upstream issue) plus a
recorded `develop` SHA, so a future pin bump starts from known ground instead
of a blind version jump.

## In Scope

- Resolve `neuriplo-tasks@develop` to one recorded commit SHA; all runs in the
  packet use that SHA via `-DNEURIPLO_TASKS_VERSION=<sha>` (precedence:
  `cmake -D` over `versions.env`, per `cmake/versions.cmake`). Record the SHA
  in `validation.md`.
- Routing parity: every `--type` string in
  `docs/generated/supported-model-types.md` routes to the same `TaskType` in
  `app/src/NeuriploInferTaskRouting.cpp` as `neuriplo_tasks::TaskFactory`
  builds at that SHA. Regenerate (never hand-edit) via
  `python3 scripts/sync_supported_model_types.py` pointed at the develop
  checkout; report added/removed/renamed type strings as a diff, not a sync.
- Build matrix: default build plus each `-DDEFAULT_BACKEND=<BACKEND>` this
  repo supports, configured from fresh build dirs against the develop SHA.
- Live gates where hardware exists (Jetson Orin ARM64 TRT 10.16.2, CPU
  backends): re-run the representative preset per cell at the same evidence
  bar as `docs/BackendCoverage.md` (`P` criteria), reusing the pinned-release
  models/engines where the contract is unchanged.
- Dry-run gates (`DR` bar: `integration-tests/local-inference-smoke`,
  `e2e-example-emulation.yml`) for cells without local hardware (GPU-only
  x86_64 TRT 11.2, ROCm MGX).
- Every failure filed upstream in `neuriplo-tasks` first; issue links tracked
  in `validation.md`.

## Out of Scope

- Moving `NEURIPLO_TASKS_VERSION` in `versions.env` (testing only; a bump is a
  separate release-gated change via `scripts/cut_release.sh`).
- Promoting `docs/BackendCoverage.md` cells from develop runs. A passing result
  applies to the tested model *and* dependency set; the scoreboard tracks
  released pins only. Develop results live in this packet's `validation.md`.
- Code fixes in this repo for develop-only breaks, unless the fix is also
  correct against the pinned v0.8.5 (then it is a normal bug fix, split to its
  own branch).
- Fixes inside `neuriplo-tasks` itself; file them, do not fork them here.
- Release, tag, or `CHANGELOG.md` entry (no user-visible change ships).

## Decisions

- Version override (`-DNEURIPLO_TASKS_VERSION=<sha>`), not
  `FETCHCONTENT_SOURCE_*` to a sibling checkout: the override exercises the
  exact fetch path a future pin bump will use, so fetch-shape problems (tag vs
  SHA, submodules, generated headers) surface now rather than at bump time.
  (Maintainer interview 2026-11-27.)
- Full matrix, not a contract smoke test: a routing-only check would miss
  backend-facing contract drift (tensor names, dtypes, shapes), which is the
  historically observed failure mode at this boundary. (Interview 2026-11-27.)
- Failures go upstream first: task logic, pre/post-processing, and export
  contracts are owned by `neuriplo-tasks` per `AGENTS.md` system overview; this
  repo records and tracks, it does not absorb producer bugs.

## Constraints and Context

- Constitution rules: CLI compatibility preserved; output schema, backend
  fallback behavior, and latency-sensitive paths untouched (this packet runs
  the app, it does not change it). Cross-repo contract work validates in
  declared order: repo-local checks first, then downstream integration.
- Reuse: `scripts/sync_supported_model_types.py --check` (pointed at the
  develop checkout), `scripts/slurm_backend_matrix.py` for cluster cells,
  `docs/evidence/*.json` as the evidence format, `backend-coverage` skill
  evidence bars (P / RUNTIME-PASS / DR / B) for grading runs.
- Existing pattern for unpinned testing: `docs/KeypointMatching.md` and
  `docs/SlurmBackendCoverage.md` already describe sibling-checkout and Slurm
  flows; this packet systematizes them to the full matrix for one SHA.
- Contract surfaces touched by *observation* only: `--type` routing,
  generated model-type docs, `docs/BackendCoverage.md` (read, not written).

## Open Questions

- Should this become a roadmap phase with a cadence (per-release develop
  re-gate), or stay a one-off packet? Propose in the plan; maintainer decides
  at review.
- If producer `develop` renames a `--type` string, is the compat shim (if
  any) this repo's or upstream's? Surface before planning the response.
- None other. Interview held 2026-11-27: scope = full matrix, consume =
  version override, outcome = testing only, failures = upstream first.

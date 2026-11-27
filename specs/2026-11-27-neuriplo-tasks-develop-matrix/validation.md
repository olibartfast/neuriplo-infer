# Feature Validation — neuriplo-tasks develop matrix

Written before implementation; execute from this file, do not summarize from
memory. This packet ships no code, so "pass" here means the *testing* is
complete and truthful — every cell has a verdict or an explicit reason it
could not be tested.

## Producer repo health (2026-11-27, executor: agent)

`neuriplo-tasks@fd94cab` configured clean (`-DBUILD_TESTS=ON`) and its own
suite passes fully: **34/34 ctest, 0 failures** (includes
`test_factory_aliases`, `test_gaze_estimation`,
`test_monocular_3d_detection`). No producer-side test failure to fix —
remaining work is downstream adaptation only (routing shims done on the
packet branch; app-flow support for flexglue/gaze/3D still open).

- Resolved SHA `fd94cabca822201341f6bd3d710ac265eb55957c`
  (`Align direct YOLO API with README and rename factory tests`).
- Fresh-dir configure with `-DNEURIPLO_TASKS_VERSION=<sha>` works; log shows
  `[source: cmake -D override]`. First attempt failed on
  `find_package(glog)`, but the pinned configure fails identically — a
  pre-existing environment condition (fixed with the local `glog_DIR` hint),
  not a producer finding.
- Full default build (`build-tasks-develop`, `-j4`) exits 0 against the SHA.
  Test suite not yet run (Group 3).

## Group 2 — findings (2026-11-27)

Sync drift (`sync_supported_model_types.py` against develop README; committed
file restored afterwards — diff observed only):

- New video-classification families: `pevideo`, `vjepa`, `videoprism`.
- `flexglue` keypoint matcher; `lightglue` now implemented upstream
  (previously threw `std::logic_error`).
- New task categories: Monocular 3D Detection (`urbanomni*`,
  `monocular3d`, `mono3d`) and Gaze Estimation (`gaze`, `ethxgaze`,
  `gazetr`) — new `TaskType` values our code cannot name.
- `depth_anything_3` / `da3*` relative-depth models; `yolo*depth*`
  generalized. Cosmetic: pose heading rename, `yolov10seg` spacing.

Routing parity (`getTaskTypeForModel` vs develop `TaskFactory`):

| Family (develop) | Our routing | Verdict |
|---|---|---|
| VideoClassification `pevideo`/`vjepa`/`videoprism` | falls to Detection default | MISMATCH — gate halted, upstream report needed |
| KeypointMatching `flexglue` | falls to Detection default | MISMATCH — gate halted |
| `depth_anything_3` / `da3*` | falls to Detection default | MISMATCH — gate halted |
| GazeEstimation (`gaze`, `ethxgaze`, `gazetr`) | no such TaskType in app | MISMATCH (structural: new task unsupported) |
| Monocular3DDetection (`urbanomni*`, `monocular3d`, `mono3d`) | no such TaskType in app | MISMATCH (structural) |
| All pinned-v0.8.5 families | unchanged rules | MATCH (spot-checked: lightglue/superglue/matchanything, timesformer/videomae/vivit, raft, vitpose, depth_anything_v2, owlv2 family, rtdetr/ecdet/rfdetr/yolo detection) |

Per plan Group 2: gates for the five mismatched families stay halted until
triaged upstream; matched families may proceed to Groups 3–4.

## Group 4 — live-gate notes (2026-11-27, develop binary `build-tasks-develop`)

- NERT (default backend) + develop tasks, resnet50 ONNX on CPU: exit 0,
  `run_report.json` success, render saved (22 s inference — CPU reference
  box, not a perf claim). Consumer e2e path works against the SHA.
- NERT + `rtdetrv4.onnx`: rejected at load — shape inference on the stem
  Conv node (`Conv input and weight must share rank 3, 4, or 5`). NOT ruled
  a develop regression: NERT is CPU-only/experimental in this phase and this
  export was produced for the TRT path. Needs a pinned-NERT baseline run to
  triage; recorded as open, non-blocking for the routing verdict.
- TRT live re-gates of matched families still queued (need a
  `-DDEFAULT_BACKEND=TENSORRT` build dir at the SHA).
- Full `ctest` suite (`build-test-tasks-develop`, includes the new
  routing cases) running; verdict pending.

## Automated

- [ ] Default build configures and builds against the SHA from a fresh dir:
      `cmake -S . -B build-tasks-develop -DNEURIPLO_TASKS_VERSION=<sha> -DCMAKE_BUILD_TYPE=Release && cmake --build build-tasks-develop`
- [ ] Test build and full suite pass against the SHA:
      `cmake -S . -B build-test-tasks-develop -DENABLE_APP_TESTS=ON -DCMAKE_BUILD_TYPE=Release -DNEURIPLO_TASKS_VERSION=<sha> && cmake --build build-test-tasks-develop && ctest --test-dir build-test-tasks-develop --output-on-failure`
- [ ] Generated docs check passes against the develop checkout:
      `python3 scripts/sync_supported_model_types.py --neuriplo-tasks-readme <develop-checkout>/README.md --check` — or the diff is recorded below as finding.
- [ ] `pre-commit run --all-files` (clang-format, cppcheck) is clean — expected
      trivially true (no code changes); run once to confirm the branch is
      spec-files only.
- [ ] `--capabilities` output at the SHA is byte-identical to the pinned run
      (or the diff is recorded as a finding).
- [ ] Build variants still configure against the SHA — list the ones that
      apply (KServe-only, local-only, `-DNEURIPLO_INFER_ENABLE_GRPC=OFF`,
      `-DWERROR=ON`).

## Manual

- [ ] Routing-parity table complete: every `--type` checked against the
      SHA's `TaskFactory`; mismatches have upstream issue links.
- [ ] Live gates graded per `backend-coverage` bars (P / RUNTIME-PASS / DR)
      for all cells with local hardware; each P-equivalent has
      `run_report.json` + checked render recorded here (paths, SHAs).
- [ ] Hardware-absent cells have DR verdicts or explicit untestable reasons
      (which hardware, what would unblock).
- [ ] Slurm cells submitted via `scripts/slurm_backend_matrix.py` with
      `submitted.json` / `result.json` retained; outcomes recorded here.
- [ ] Every failure has a `neuriplo-tasks` issue link; zero untracked
      failures remain.
- [ ] No `versions.env` change, no `docs/BackendCoverage.md` promotion, no
      app-code change on this branch — verify with
      `git status --short` (spec files only) before review.

## Definition of Done

- [ ] Every requirement is evidenced or explicitly deferred in
      `requirements.md`.
- [ ] Nothing in *Out of Scope* was done anyway (pin untouched, scoreboard
      untouched).
- [ ] Deviations from this file are recorded here, honestly, with what was
      run instead (including a producer-`develop` move mid-packet and any
      re-runs it forced).
- [ ] Pin-bump recommendation recorded: ready (with SHA) or blocked-by (with
      issue links). The recommendation is not the bump.
- [ ] `../roadmap.md` disposition decided: recurring phase or one-off (per
      requirements.md Open Questions), updated or explicitly declined with
      reason.
- [ ] Spec files tell the same story in one branch; evidence notes are
      SHA-pinned and reproducible.

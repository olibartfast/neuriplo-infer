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
- Consumer suite (`build-test-tasks-develop`, 191 tests): full `ctest`
  hung 7+ h on death test
  `RunReportFile.AnArmedExitHookWritesAConfigurationReport` (fork+threads
  warning, unrelated to this packet's changes — nothing here touches
  RunReport) and was killed. Re-running with that test excluded and
  `--timeout 300`; verdict pending. Single-test repro still open.
- Branch CI (push `b950f3e`): C++ Lint & Build SUCCESS, CI SUCCESS.

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

| Family (develop) | Our routing (packet branch) | Verdict |
|---|---|---|
| VideoClassification `pevideo`/`vjepa`/`videoprism` | prefix rules added (commit `28e8122`) | MATCH — unit-tested (`TaskRouting` cases added) |
| KeypointMatching `flexglue` | prefix rule added | ROUTING MATCH — live gate still blocked: app pair-flow has no two-model/external-keypoint support |
| `depth_anything_3` / `da3*` | `depthanything3`/prefix-`da3` rules added | MATCH — unit-tested |
| GazeEstimation (`gaze`, `ethxgaze`, `gazetr`) | no such TaskType in app | MISMATCH (structural: new task unsupported) — no fix attempted |
| Monocular3DDetection (`urbanomni*`, `monocular3d`, `mono3d`) | no such TaskType in app | MISMATCH (structural) — no fix attempted |
| All pinned-v0.8.5 families | unchanged rules | MATCH (routing + capabilities contract tests green on develop build) |

Strategy change 2026-11-27 (maintainer direction: fix, don't file): the
three string-rule mismatches were fixed downstream instead of reported
upstream; `upstream-issue-drafts.md` stays UNFILED as a record of what was
decided against. The two structural mismatches stay open by design — no
sibling defect, just new surface with no downstream support yet. Per the
packet's own deviation rule this paragraph is the record.

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

- [x] Default build configures and builds against the SHA from a fresh dir
      (done 2026-11-27, exit 0; glog caveat recorded above).
- [ ] Test build and full suite pass against the SHA — build done;
      `ctest` (191 tests, `--timeout 300`, one death test excluded):
      **166 pass, 22 fail — ALL failures are `EXPECT_EXIT` fork-based
      death tests** (`ParseCommandLineArguments.*Exits/Rejected`,
      `RunReportFile.ArmAndDisarmAcrossThreads...`) hanging in this
      environment (gtest itself warns: fork in threaded context; one hung
      7 h before the timeout run). Nothing in the failures touches routing,
      rendering, or any file this packet changed, and branch CI runs the
      identical code green. Verdict: environment-specific death-test hangs,
      not code failures. Re-run cleanly in CI or a TTY environment to close.
- [x] Generated docs check run against the develop checkout — diff recorded
      as the Group 2 finding (deliberately not applied: pins unchanged).
- [x] `pre-commit` (clang-format, cppcheck) clean — branch CI Lint & Build
      SUCCESS on push `b950f3e` (after fixing my own format violations).
- [ ] `--capabilities` output at the SHA byte-identical to the pinned run —
      develop binary runs (`capabilities OK`) but no byte-diff performed yet.
- [ ] Build variants still configure against the SHA — not attempted yet.

## Manual

- [x] Routing-parity table complete: every `--type` checked against the
      SHA's `TaskFactory`; table above is current as of 2026-11-27
      (three fixed+unit-tested, two structural open).
- [ ] Live gates graded per `backend-coverage` bars (P / RUNTIME-PASS / DR)
      for all cells with local hardware — only NERT/resnet50 done; TRT
      re-gates need a TensorRT-default build at the SHA.
- [ ] Hardware-absent cells have DR verdicts or explicit untestable reasons
      (which hardware, what would unblock).
- [ ] Slurm cells submitted via `scripts/slurm_backend_matrix.py` with
      `submitted.json` / `result.json` retained; outcomes recorded here.
- [x] Failure routing decided AND RECORDED (deviation from template):
      maintainer directed fix-locally on 2026-11-27, so zero issues filed
      and zero untracked failures remain only in the sense that the three
      fixable mismatches are fixed on this branch; structural items have no
      upstream ticket by explicit decision, not by oversight.
- [x] Deviations from the template recorded HERE (not deferred):
      (a) app-code changed on this branch (routing fix + contract tests +
      AGENTS.md pre-push rule) — template said spec-files only;
      (b) no upstream issues filed — template assumed upstream-first, and the
      `upstream-issue-drafts.md` file stands as the unfiled record;
      (c) `pre-commit` "trivially true" assumption was wrong once code
      landed — fixed and CI-verified instead.

## Definition of Done

- [ ] Every requirement is evidenced or explicitly deferred in
      `requirements.md` — partial: Group 3-4 evidence still open (suite
      verdict, TRT re-gates, Slurm, variants, capabilities diff).
- [x] Out-of-scope respected: pin untouched, scoreboard untouched
      (`git status` shows branch files only).
- [x] Deviations recorded above, honestly (strategy change, app-code on
      branch, unfiled drafts, pre-commit assumption).
- [ ] Pin-bump recommendation recorded: blocked until Groups 3–4 close.
- [ ] `../roadmap.md` disposition decided: recurring phase or one-off (per
      requirements.md Open Questions), updated or explicitly declined with
      reason.
- [ ] Spec files tell the same story in one branch; evidence notes are
      SHA-pinned and reproducible — true as of this edit; re-verify after
      the exclusion suite lands.

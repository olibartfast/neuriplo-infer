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
packet branch; gaze and 3D support landed later in slice 2; FlexGlue
app-flow support remains open).

- Resolved SHA `fd94cabca822201341f6bd3d710ac265eb55957c`
  (`Align direct YOLO API with README and rename factory tests`).
- Fresh-dir configure with `-DNEURIPLO_TASKS_VERSION=<sha>` works; log shows
  `[source: cmake -D override]`. First attempt failed on
  `find_package(glog)`, but the pinned configure fails identically — a
  pre-existing environment condition (fixed with the local `glog_DIR` hint),
  not a producer finding.
- Full default build (`build-tasks-develop`, `-j4`) exits 0 — BUT the
  binary linked **pinned v0.8.5 headers, not develop**: a v0.8.5 sibling
  checkout at `../neuriplo-tasks` (kept for export tooling) shadows any
  `-DNEURIPLO_TASKS_VERSION` via `FETCHCONTENT_SOURCE_DIR_NEURIPLO-TASKS`
  (see CMakeLists "Prefer a sibling checkout"). The NERT/resnet50 e2e run
  below therefore proves the pinned path only. A version override needs
  `-DFETCHCONTENT_SOURCE_DIR_NEURIPLO-TASKS=<develop-checkout>` alongside
  it — packet Group 1 method corrected accordingly; build redone with both
  flags as `build-tasks-develop2`.
- Consumer suite history: first full run hung on a fork-based death test in
  this threaded binary; the threadsafe death-test default fixed it, and the
  final true-develop run completed 195/195.
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
| GazeEstimation (`gaze`, `ethxgaze`, `gazetr`) | real routing + renderer + capabilities under `NEURIPLO_TASKS_HAS_GAZE_MONO3D` (slice 2, 2026-11-28) | SUPPORTED — live-gated below on ex-novo media |
| Monocular3DDetection (`urbanomni*`, `monocular3d`, `mono3d`) | real routing + cuboid renderer + capabilities under macro (slice 2, 2026-11-28) | SUPPORTED — live-gated below on ex-novo media |
| All pinned-v0.8.5 families | unchanged rules | MATCH (routing + capabilities contract tests green on develop build) |

Strategy change 2026-11-27 (maintainer direction: fix, don't file): the
three string-rule mismatches were fixed downstream instead of reported
upstream; `upstream-issue-drafts.md` stays UNFILED as a record of what was
decided against. Gaze and monocular 3D were later implemented downstream (slice 2);
remaining unblocked work is FlexGlue's two-model app flow. Per the
packet's own deviation rule this paragraph is the record.

## Group 4 — live gates (2026-11-28, develop TRT binary
`build-tasks-develop-trt`, Jetson Orin TRT 10.16.2)

Slice-2 app support (routing + renderers + capabilities, all macro-gated)
is implemented on this branch and verified below with fresh ex-novo media
(`data/showcase/`, no usual detector images):

- Gaze (`ethxgaze`, ptgaze resnet18 ONNX via sibling export, engine built
  with local trtexec): exit 0, 13 ms inference, success report. Render
  checked: yellow arrow from face center points image-right, matching the
  subject's visible rightward gaze on the 224x224 face crop.
  Render SHA `e2a5d61…`; engine SHA `2d6cdb1…`.
- Monocular 3D (`urbanomni`, `urbanomnidetect_yolo11n-p2_640` ONNX via
  sibling Ultralytics export, engine built with local trtexec): exit 0,
  80 ms inference, success report. Render checked: two instances on
  Times Square taxis, blue 2D boxes + yellow cuboid wireframes (ground
  quad, roof quad, verticals) tracking the cars.
  Render SHA `b82685c…`; engine SHA `3adbdb2…`.
- Both gates meet the backend-coverage P bar (ran + output checked) but
  are recorded HERE, not in `docs/BackendCoverage.md`: the scoreboard
  tracks released pins, and the pin here is develop (`fd94cab`).
  Promotion awaits the sibling release + pin bump.
- Embedded 5x7 bitmap font (`app/inc/BitmapFont.hpp`) + `imageops::drawText`
  (2026-11-28): all renders now carry text — detection/seg/mono3D label bands
  show `LABEL 0.94`, ranked bands show the label strings, gaze shows a
  yaw/pitch readout. 3 new `AppImageOpsText` unit tests green. Full gates
  re-run on the develop TRT build; renders verified legible (dog: CAR,
  BICYCLE, DOG; gaze: YAW 0.60 PITCH -0.23; mono3D: CAR 0.76/0.71 with the
  `urbanomni_labels.txt` car/person/bike vocabulary).
- Showcase (2026-11-28): `data/showcase/showcase_reel.mp4` (47 s, 720p)
  assembles title card, 10 Ken Burns stills (text renders verified legible),
  an 8 s mono-3D video segment (62 Bangkok frames, cuboids + CAR labels),
  and a gaze close-up. Staged at `/tmp/showcase-bundle/` with all 12
  renders, 11 evidence JSONs, inputs, and labels.

Open items at release time: TRT re-gates of matched families still queued
as a matrix sweep; NERT `rtdetrv4.onnx` shape-inference rejection still
needs a pinned-NERT baseline to triage; Slurm cells, capabilities byte-diff,
and build variants remain unattempted. The earlier NERT/resnet50 e2e run
was from the shadowed build (pinned headers) and is not claimed for
develop. The consumer suite was later re-run on the true-develop build
(see Automated).

## Automated

- [x] Default build configures and builds from a fresh dir (done
      2026-11-27, exit 0; glog caveat recorded above) — the initial
      build linked pinned headers (sibling shadowing); the corrected
      true-develop build is `build-tasks-develop2` with both override flags.
- [x] Test build and full suite pass on the true-develop build
      (`build-test-tasks-develop2`, macro ON): **195/195, 0 fail**
      (1 expected skip), 12 s. Covers new `AppImageOpsText` (3),
      macro-gated routing/capabilities cases, and the death-test fix.
      (The earlier 191/191 ran against pinned headers — superseded.)
      History: full `ctest` first hung 7+ h on fork-based death tests;
      fixed by defaulting `death_test_style` to `threadsafe`.
- [x] Generated docs check run against the develop checkout — diff recorded
      as the Group 2 finding (deliberately not applied: pins unchanged).
- [x] Format and targeted lint clean locally; branch CI Lint & Build
      SUCCESS on push `b950f3e` for the original work, and the current
      changes need fresh CI validation after integration.
- [ ] `--capabilities` output at the SHA byte-identical to the pinned run —
      develop binary runs (`capabilities OK`) but no byte-diff performed yet.
- [ ] Build variants still configure against the SHA — not attempted yet.

## Manual

- [x] Routing-parity table complete: every `--type` checked against the
      SHA's `TaskFactory`; updated 2026-11-28 (three fixed+unit-tested,
      gaze/3D supported, FlexGlue routing-only).
- [ ] Live gates graded per `backend-coverage` bars (P / RUNTIME-PASS / DR)
      for all cells with local hardware — gaze and mono-3D are P-evidenced
      on the develop TRT build; the remaining matched families and
      hardware-absent cells still need their sweep.
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
- [ ] Spec files tell the same story in one branch after the latest
      true-develop suite run; evidence notes are SHA-pinned.

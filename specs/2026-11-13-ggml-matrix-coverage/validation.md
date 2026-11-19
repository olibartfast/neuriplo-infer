# Feature Validation — GGML matrix coverage

Written before implementation; execute from this file, do not summarize from memory. Pyramid scoring: worker runs each acceptance command exactly once per attempt and reports pass/fail; planner decides what happens next.

## Automated (2026-11-13 re-verification, `feature/ggml-matrix-coverage`)

- [x] Default build configures and builds (packet must not break NERT default):
      `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build`
      — configure OK (NERT default), build 16/16 OK.
- [x] Test build and full suite pass:
      `cmake -S . -B build-test -DENABLE_APP_TESTS=ON -DCMAKE_BUILD_TYPE=Release && cmake --build build-test && ctest --test-dir build-test --output-on-failure`
      — 100% tests passed, 0 failed out of 197 (21.56 s).
- [x] Generated docs in sync: `python3 scripts/sync_supported_model_types.py --check`
      — PASS against pinned `v0.8.5` README (`git show v0.8.5:README.md` from
      sibling checkout). Note: check against sibling `develop` HEAD fails
      (sibling is `v0.8.5-29-geeda711`, README changed 12+/5-); that is
      expected — CI fetches the pinned `v0.8.5` ref, not sibling HEAD.
- [x] `--capabilities` still validates (`capabilities_schema_contract` ctest).
      — `capabilities_schema_contract` Passed + `capabilities_cli_contract` Passed
      as part of the 197/197 run above.
- [x] `pre-commit run --all-files` (clang-format, cppcheck) clean on touched files.
      — `pre-commit` binary not installed in this environment; emulated:
      touched files are markdown-only (`CHANGELOG.md`,
      `docs/BackendCoverage.md` + 3 spec files), so clang-format/cppcheck
      (both binaries present) have nothing to check; trailing-whitespace
      grep clean, EOF newline present on all 5 files.
- [x] Every hyperlink added to `docs/BackendCoverage.md` resolves (`ls` for relative, `curl -sI` for absolute). New anchors: `#ev-ggml` / `E19`.
      — no new absolute URLs in the diff; 1x `<a id="ev-ggml">` anchor +
      4x `#ev-ggml` references, all internal.

## Manual (superseded)

The planned brev attempt was blocked at SSH transport; no remote command
executed. Local sibling-checkout gates superseded it in Phase 2 below, and
teardown was verified (`brev ls` → no instances).

## Matrix honesty (reviewer checks against `requirements.md` R-1..R-4)

- [x] R-1: `GGML` column exists in second gate table; legend disambiguates
      GGML from LLAMACPP. Phase 1 state was every cell `U`; Phase 2 promotes
      classification, `yolov8`, and `yoloseg` — see V-7/V-8/V-14.
- [x] R-2: `E19` records upstream builder + `77/77` engine tests; `E11` keeps LLaMA PASS with cross-ref. `Environments` GGML row + `Open items` follow-up present. Scoreboard counts unchanged.
- [x] R-3: no `neuriplo` inference-semantics change, no `versions.env` pin move, no CLI/`--capabilities`/schema/generated-docs change.
- [x] R-4: Phase 1 `CHANGELOG.md` described the docs-only column. Phase 2
      supersedes that wording: the entry now records the three local PASSes and
      still claims no CI job or released pin.

## Definition of Done

- [x] Every requirement implemented or explicitly deferred in `requirements.md`.
- [x] Nothing in *Out of Scope* implemented anyway.
- [x] Deviations recorded here with what ran instead.
- [x] `CHANGELOG.md` + `../roadmap.md` status consistent with shipped work (roadmap: no phase closed by this packet; Open-items pointer only).
- [ ] Spec, code, changelog, roadmap tell the same story in `feature/ggml-matrix-coverage` → `develop`.

## Deviations (2026-11-13, orchestrator record)

- The planned brev attempt never executed a remote command (SSH timeouts);
  it is superseded by the local Phase 2 gates below. Teardown was verified.
- V-2/V-3/V-4 source-level bounds were replaced by real local runs; no cell
  was promoted without a recorded run.

## Cross-repo follow-up (2026-11-13, `neuriplo` real-loader packet)

- `neuriplo` branch `feature/ggml-real-loader`: real head-only loader
  (pool + real FC through ggml), converter emitting NGGL, hardened tests —
  local gates green (77/77 ctest, golden + F2 orientation probe exact vs
  numpy).
- App-side wiring probe in this branch (no semantics change): `GGML` branch in
  `cmake/LinkBackend.cmake` + test mirror in `app/test/CMakeLists.txt` +
  `DEFAULT_BACKEND` help string. `cmake -S . -B build-ggml
  -DDEFAULT_BACKEND=GGML` configures OK (Backend: GGML) and
  `cmake --build build-ggml` links OK against the sibling working tree.
  The old V-2 expectation (configure fails on missing LinkBackend branch) is
  superseded: with the probe + SDK present, configure passes; without
  `GGML_DIR`, neuriplo's `validate_ggml` fails loudly as designed.

## Re-verification (2026-11-13, continuation session)

- Automated all green: default build 16/16; test build + `ctest` 197/197
  (incl. `capabilities_schema_contract`, `capabilities_cli_contract`);
  `sync_supported_model_types.py --check` PASS vs pinned `v0.8.5` README
  (fails vs sibling `develop` HEAD `v0.8.5-29-geeda711` — expected, CI uses
  the pinned fetch); pre-commit emulated clean (markdown-only diff);
  hyperlinks internal-only, anchor present.
- V-5 complete: `brev ls` → `No instances in org olibartfast-hq`.
- One DoD box left: the spec/code/changelog/roadmap story converges on the
  `feature/ggml-matrix-coverage` → `develop` PR itself — open the PR to close it.

## Phase 2 — app-surface gates (2026-11-14, local sibling)

Run from `feature/ggml-matrix-coverage` with the unreleased sibling
`neuriplo` `feature/ggml-yolo-family` @ `9ee2bde` and
`GGML_DIR=/home/oli/dependencies/ggml`. These runs supersede the blocked
2026-11-13 brev attempt for the covered representatives; they are local
sibling runs, not an isolated remote-fetch release gate.

**NGGL-era status.** V-7, V-8, and V-14 below were recorded against NGGL
artifacts, before the backend became strict-GGUF-only (`.gguf` via
`convert_onnx_to_gguf.py`; the ONNX-subset frontend and graph executor were
later removed). Their numbers describe that build. The GGUF re-gate
(V-16/V-17/V-18 below) supersedes them on the current backend.

- [x] V-6 build: `cmake --build build-ggml` (cache:
      `DEFAULT_BACKEND=GGML`, `GGML_DIR=/home/oli/dependencies/ggml`,
      `neuriplo_SOURCE_DIR=/home/oli/repos/neuriplo`) — app re-linked against
      the loader branch; `GGML model loaded: full ResNet50 ...` and
      `full YOLOv8n ... real weights` in the run logs.
- [x] V-7 classification app gate (NGGL-era, pre-GGUF): ONNX Model Zoo `resnet50-v1-12.onnx` →
      NGGL 299 tensors / 102,532,870 bytes →
      `neuriplo-infer --type=torchvisionclassifier --weights=resnet50.nggl
      --source=data/dog.jpg --input_sizes=1,3,224,224` — exit 0, 206 ms,
      top-5 `249 0.793273`, `248 0.11378`, `250 0.036304`, `537 0.024229`,
      `261 0.011177`; ORT 1.32.0 on the same preprocessed input gives the
      same indices and probabilities. Render inspected.
- [x] V-8 YOLOv8n detection app gate (NGGL-era, pre-GGUF): ultralytics 8.4.213
      `yolov8n.onnx` (opset 12, static 640) → NGGL 132 tensors / 12,782,274
      bytes → `neuriplo-infer --type=yolov8 --weights=yolov8n.nggl
      --source=data/dog.jpg --input_sizes=1,3,640,640
      --labels=labels/coco.names --min_confidence=0.4 --nms_threshold=0.5`
      — exit 0, 495 ms, bicycle 0.89 / dog 0.93 / car 0.61, identical to the
      DNN/ExecT/LiteRT/TF gates in E2. Render inspected.
- [x] V-9 matrix/docs: `docs/BackendCoverage.md` promotes only the GGML
      classification, `yolov8`, and `yoloseg` cells to `P`; E2/E5/E6/E19/
      Environments/Open items updated; `CHANGELOG.md` updated. Every other
      GGML cell stays `U` (`yolo26` is documented as BLOCKED at the
      converter; transformer rows still need sibling graph work).
- [x] V-10 GGML app test build: `cmake -S . -B build-ggml-test
      -DENABLE_APP_TESTS=ON -DDEFAULT_BACKEND=GGML
      -DGGML_DIR=/home/oli/dependencies/ggml -DCMAKE_BUILD_TYPE=Release`,
      `cmake --build build-ggml-test`, `ctest --test-dir build-ggml-test
      --output-on-failure` — configure reports `Backend: GGML`; build green;
      **197/197 tests passed** (including `capabilities_schema_contract` and
      `capabilities_cli_contract`).
- [x] V-11 default NERT build/test re-run after the doc/CMake help-string
      edits: `cmake --build build` (Backend: NERT, no work needed) and
      `cmake --build build-test && ctest --test-dir build-test
      --output-on-failure` — **197/197 passed**, zero failures.
- [x] V-12 `yolo26` attempt: exported `yolo26n.onnx` (ultralytics 8.4.213,
      opset 12, 368 nodes with attention `MatMul`/`Softmax`, head
      `model.23`, output `[1,84,8400]`); converter exits 1 with
      `YOLO file missing fixed DFL conv weight 'model.22.dfl.conv.weight'`.
      BLOCKED at the sibling converter/graph builder; owner `neuriplo`.
      GGML `yolo26` cell stays `U` (E1).
- [x] V-13 gate harness: `scripts/ggml_app_gate.py` converts, runs, and writes
      a JSON evidence record (commands, exit codes, hashes, timings,
      `run_report.json`, renders) without deciding PASS/BLOCKED.
      `scripts/test_ggml_app_gate.py` covers command construction, success,
      and failure; the test is wired into `.github/workflows/ci.yml`.
- [x] V-14 `yoloseg` app gate (NGGL-era, pre-GGUF): `yolov8n-seg.onnx` (ultralytics 8.4.213,
      opset 12, static 640) → NGGL 158 tensors / 13,793,302 bytes →
      `scripts/ggml_app_gate.py --model-type yoloseg ... --mask-threshold
      0.5` — exit 0, ~0.7 s (657–730 ms across runs), dog 0.87
      / bicycle 0.80 / car 0.45, masks
      rendered. Sibling backend parity: `GGMLInferTest` 21 tests (20 pass,
      1 intentional skip), including 7/7 `GGMLOpsParity`; full ctest 77/77;
      raw diffs 7.6e-04/1.5e-03/7.9e-04
      for `[1,116,8400]` and 1.6e-03/1.1e-03/1.7e-03 for `[1,32,160,160]` on
      gray/zeros/gradient. Cross-repo packet:
      `neuriplo/specs/2026-11-14-ggml-yoloseg/validation.md`.
      App-gate evidence: `/tmp/opencode/ggml-matrix/gate-yoloseg/evidence.json`;
      render: `/tmp/opencode/ggml-matrix/gate-yoloseg/run/data/output/processed_yoloseg_local.png`.
- [x] V-15 transformer/RT-DETR foundation: sibling packets
      `neuriplo/specs/2026-11-14-ggml-attention-ops`,
      `2026-11-15-ggml-graph-executor` and `2026-11-15-ggml-onnx-frontend`
      add `GGMLOps` helpers, a multi-input graph executor and a static-shape
      ONNX-subset frontend. Sibling `GGMLInferTest` is 57 tests
      (56 pass, 1 skip), including 29/29 ONNX-frontend tests and 7/7
      graph-executor tests; full ctest 77/77. The fixed-batch `rtdetrv4`
      graph now imports and matches ORT by label and box on all 300
      detections (max box L1 2.7e-3, max score diff 2.2e-5); 10 positions
      are near-tie swaps. No app gate yet and no GGML cell promoted; the
      remaining RT-DETR work is app wiring, sorted TopK order, and dynamic
      batch. Historical: the ONNX-subset frontend and graph-executor suites
      were later removed under strict-GGUF loading, so the 29/29 and 7/7
      counts above describe that build, not the current backend.

Artifacts (NGGL-era, pre-GGUF): `/tmp/opencode/ggml-matrix/{resnet50-v1-12.onnx,resnet50.nggl,
yolov8n.onnx,yolov8n.ggml,yolov8n-seg.onnx,run_app_verbose.log,run_yolo.log}`;
sibling backend artifact `/tmp/yolov8n-seg.ggml`; gate evidence under
`/tmp/opencode/ggml-matrix/gate-{classification,yolov8,yoloseg}/evidence.json`;
renders under the respective `run/data/output/` directories. `.gguf`
counterparts exist from sibling runs (`/tmp/{resnet18,resnet50,yolov8n,yolov8n-seg}.gguf`).

## Phase 2b — GGUF re-gate (2026-11-15, strict-GGUF-only backend)

Rerun from `feature/ggml-matrix-coverage` with sibling `neuriplo`
`feature/ggml-yolo-family` (strict-GGUF-only loader: only `.gguf` loads;
`.onnx`/`.ggml`/`.nggl` rejected) and `GGML_DIR=/home/oli/dependencies/ggml`
(v1.0.0). App rebuilt (`cmake --build build-ggml`). Each gate converts
ONNX→GGUF with `scripts/ggml_app_gate.py` (default
`convert_onnx_to_gguf.py`), then runs `neuriplo-infer`. Evidence under
`/tmp/opencode/ggml-matrix-gguf/gate-{classification,yolov8,yoloseg}/evidence.json`.

- [x] V-16 classification GGUF re-gate: `resnet50-v1-12.onnx` → GGUF 299
      tensors / 102,556,256 bytes → `--type=torchvisionclassifier
      --weights=resnet50-v1-12.gguf --source=data/dog.jpg
      --input_sizes=1,3,224,224` — convert exit 0 (583 ms), run exit 0
      (773 ms; inference 177.6 ms, model_load 313.7 ms), render
      `processed_torchvisionclassifier_local.png`, `run_report.json`
      status success.
- [x] V-17 YOLOv8n detection GGUF re-gate: `yolov8n.onnx` → GGUF 132
      tensors / 12,791,904 bytes → `--type=yolov8 ... --labels=labels/coco.names
      --min_confidence=0.4 --nms_threshold=0.5` — convert exit 0 (247 ms),
      run exit 0 (644 ms; inference 383.7 ms), render inspected:
      bicycle 0.89 / dog 0.93 / car 0.61, identical to the NGGL-era V-8
      detections.
- [x] V-18 YOLOv8n-seg GGUF re-gate: `yolov8n-seg.onnx` → GGUF 158
      tensors / 13,804,864 bytes → `--type=yoloseg ... --mask-threshold 0.5`
      — convert exit 0 (243 ms), run exit 0 (843 ms; inference 505.1 ms),
      `GGML model loaded: full YOLOv8n-seg ... real weights`, render
      inspected: dog 0.87 / bicycle 0.80 / car 0.45 plus masks, identical
      to the NGGL-era V-14 detections.
- [x] V-19 matrix/docs: GGML `torchvisionclassifier`, `yolov8`, `yoloseg`
      cells stay `P` — now on GGUF evidence. Every other GGML cell stays `U`
      (`yolo26` BLOCKED at the converter; transformer rows need sibling
      loader work).
- [x] V-20 regression re-gate (2026-11-16): same three gates rerun against
      the current stack (app `feature/ggml-matrix-coverage` @ `5905b8b`,
      sibling `feature/ggml-rfdetr` @ `0aee3ea` plus uncommitted RF-DETR
      backbone/encoder/decoder work) to prove the loader additions did not
      regress the shipped paths. Byte-identical GGUFs to V-16/17/18
      (converter unchanged for these models): all three `status=ok`,
      exit 0, renders inspected — `yolov8` bicycle 0.89 / dog 0.93 /
      car 0.61, `yoloseg` dog 0.87 / bicycle 0.80 / car 0.45 + masks,
      identical to V-17/V-18. Evidence under
      `/home/oli/opencode-gates/ggml-matrix-regate/gate-{classification,yolov8,yoloseg}/`
      (`/tmp` quota exhausted, evidence moved to persistent disk).

## Phase 2c — RF-DETR detection gate (2026-11-16, full loader graph)

Run from `feature/ggml-matrix-coverage` with sibling `neuriplo`
`feature/ggml-rfdetr` (full 3-layer deformable decoder + head in
`GGMLInfer.cpp`; app-facing outputs lead with `dets`/`labels` so the
stock `RfDetrPostprocessor` applies) and `GGML_DIR=/home/oli/dependencies/ggml`.
App rebuilt (`cmake --build build-ggml`).

- [x] V-21 RF-DETR app gate: `rfdetr.onnx` (op18 inventory, 114,871,649
      bytes) → GGUF 352 tensors / 107,591,104 bytes →
      `neuriplo-infer --type=rfdetr --weights=rfdetr.gguf
      --source=data/dog.jpg --input_sizes=1,3,560,560
      --labels=labels/coco.names --min_confidence=0.4 --nms-threshold 0.5`
      — convert exit 0 (676 ms), run exit 0 (2285 ms; inference 1515.4 ms,
      model_load 359.8 ms), `run_report.json` status success, render
      `processed_rfdetr_local.png` inspected: bicycle 0.94 / horse 0.92 /
      car 0.89 / motorbike 0.60. ORT reference on the identical
      app preprocessing (resize 768x576→560x560, /255, ImageNet norm, NCHW)
      gives bicycle 0.945 / horse 0.928 / car 0.884 / motorbike 0.610 —
      identical detections (the "horse" label is the model's own prediction,
      also in ORT; the `coco.names` index follows the task's -1 convention).
      Evidence: `/tmp/opencode/ggml-matrix-gguf/gate-rfdetr/evidence.json`
      + persistent copy under
      `/home/oli/opencode-gates/ggml-matrix-regate/gate-rfdetr/`
      (evidence.json, app.log, render, run_report.json).
- [x] V-22 matrix/docs: GGML `rfdetr` detection cell `U` → `P`;
      `docs/BackendCoverage.md` Environments/Known-limitations notes updated.
      Every other GGML cell unchanged (`yolo26` BLOCKED at the converter;
      transformer rows beyond RF-DETR and `rfdetrseg` need sibling work).

## Phase 2d — RF-DETR-seg gate (2026-11-16, full loader graph)

Run from `feature/ggml-matrix-coverage` with sibling `neuriplo`
`feature/ggml-rfdetr` (seg dispatch on `segmentation_head.bias`, 432px
geometry, 12-block backbone with global attention at 3/6/9, 200 queries,
4-layer decoder, mask branch with native bilinear Resize + exact
depthwise-conv3x3; app-facing outputs lead with `dets`/`masks`/`labels`
so the stock `RfDetrSegmentationPostprocessor` applies) and
`GGML_DIR=/home/oli/dependencies/ggml`. App rebuilt
(`cmake --build build-ggml`).

- [x] V-23 RF-DETR-seg app gate: `rfdetr_seg.onnx` (op18 inventory,
      132,354,643 bytes) → GGUF 436 tensors / 123,514,688 bytes →
      `neuriplo-infer --type=rfdetrseg --weights=rfdetr_seg.gguf
      --source=data/dog.jpg --input_sizes=1,3,432,432
      --labels=labels/coco.names --min_confidence=0.4 --nms-threshold 0.5
      --mask-threshold 0.5`
      — convert exit 0 (645 ms), run exit 0 (2926 ms; inference 1905.5 ms,
      model_load 411.2 ms, postprocess 45.4 ms), `run_report.json`
      status success, render `processed_rfdetrseg_local.png` inspected:
      bicycle 0.94 / horse 0.82 / car 0.61 / motorbike 0.59, each with a
      tight instance mask (dog, bike, car, mirror-bike all segmented).
      ORT reference on the identical app preprocessing (letterbox
      768x576→432x432, /255, ImageNet norm, NCHW) gives bicycle 0.938 /
      car 0.880 / horse 0.826 / motorbike 0.320 — same instances (render
      scores are post-NMS selections; raw top-1 slots agree GGML↔ORT to
      0.936/0.938, 0.868/0.880, 0.819/0.826, 0.320/0.320). Raw parity on
      the dog input with ORT's TopK order pinned: dets max `3.14e-02`,
      labels top-1 198/200, masks max 2.68 / mean 0.04, and mask IoU
      **1.0 on every kept instance** (bicycle, car, horse, motorbike,
      bench, truck). Evidence:
      `/tmp/opencode/ggml-matrix-gguf/gate-rfdetrseg/evidence.json` +
      persistent copy under
      `/home/oli/opencode-gates/ggml-matrix-regate/gate-rfdetrseg/`
      (evidence.json, app.log, render, run_report.json).
- [x] V-24 matrix/docs: GGML `rfdetrseg` cell `U` → `P`;
      `docs/BackendCoverage.md` Environments/Known-limitations notes updated.
      Every other GGML cell unchanged (`yolo26` BLOCKED at the converter;
      pose and other transformer rows need sibling work).
- [ ] V-25 next YOLO layouts probed (2026-11-17, local sibling `develop`
      8596a2e): `yolo11n.pt` exported (ultralytics 8.3.237, opset 12,
      static 640 → `yolo11n.onnx` 10,741,277 bytes, sha256
      `32920b66…5e1d`) and cache `yolov10n.onnx` (sha256
      `d9a367e9…f6981a), `yolo11n_seg.onnx`, `yolo11n_pose.onnx` all
      rejected by
      `neuriplo/scripts/convert_onnx_to_gguf.py` (`0aee3ea`) with exit 1:
      `YOLO file missing fixed DFL conv weight
      'model.22.dfl.conv.weight'`. All files carry the weight at
      `model.23.dfl.conv.weight` plus one attention subgraph
      (`/model.10/m/m.0/attn` C2PSA in the yolo11 family, `/model.10` PSA in
      yolov10n) the strict-GGUF converter does not map. Backbone/neck
      indices also differ from the hardcoded v8 path (v11:
      11/13/16/17/19/20 + head 23 vs v8: 12/15/16/18/19/21 + head 22),
      so this is a new sibling graph path, not a rename. BLOCKED at the
      sibling converter/graph builder, owner `neuriplo` — same class as
      the `yolo26` V-12 BLOCKED, actionable handoff: accept the
      `model.23` DFL path and map the single `/model.10` attention
      subgraph. No gate-table rows added and no cells changed until the
      sibling maps these layouts; `yolo11`/`yolov10` stay unrepresented.
- [x] V-25 follow-up (2026-11-19): sibling PR #79 merged to `develop`
      (`5b1ba85`) — `model.23` DFL path and single `/model.10` C2PSA
      subgraph mapped for **det/seg only**. `yolo11pose` and `yolov10n`
      have no sibling loader path (22 `pose` hits in `GGMLInfer.cpp` are
      all transpose/compose words; no `v10` handling) and stay
      converter-BLOCKED — pose needs a new sibling graph packet, v10 a
      layout probe. R-9 therefore gates det+seg now (V-26/V-27); the
      `yolov10` family stays unrepresented.
- [x] V-26 YOLO11n detection GGUF gate: `/tmp/yolo11/yolo11n.onnx`
      (10,741,277 bytes, sha256 `32920b66…` — same artifact as the V-25
      probe) → GGUF 180 tensors / 10,655,744 bytes →
      `--type=yolov8 ... --labels=labels/coco.names --min_confidence=0.4
      --nms-threshold=0.5` — convert exit 0, run exit 0 (474 ms;
      inference 335.5 ms), `GGML model loaded: full YOLO11`,
      `run_report.json` status success, 1 sample, render inspected:
      bicycle 0.93 / dog 0.93 / car 0.45. ORT reference (same
      letterbox-640 preprocessing, conf 0.4 / NMS 0.5): bicycle 0.94 /
      dog 0.93 / car 0.52 (+truck 0.49, same region) — same instances;
      boxes agree within ~1 px; the car-proposal score gap is the known
      NMS-selection fallout of the box-decode accumulation signature
      (sibling `ParityVsYolo11` green with max_cell_agree=1).
      Evidence under
      `/home/oli/opencode-gates/ggml-matrix-regate/gate-yolo11/`
      (evidence.json, app.log, render, run_report.json, gguf).
      Built from `feature/ggml-matrix-coverage` with sibling worktree
      `/home/oli/neuriplo-yolo11` (`6128849`, post-merge) via
      `FETCHCONTENT_SOURCE_DIR_NEURIPLO`; tasks pinned at a `v0.8.5`
      worktree (live checkout has moved past it and no longer provides
      `neuriplo-tasks::vision-opencv`).
- [x] V-27 YOLO11n-seg GGUF gate: `yolo11n-seg.onnx` (op-inventory cache,
      11,764,411 bytes, `model.23` DFL + `model.23.proto/cv4` layout) →
      GGUF 205 tensors / 11,668,576 bytes → `--type=yoloseg ...
      --mask-threshold 0.5` — convert exit 0, run exit 0,
      `run_report.json` status success, 1 sample, render inspected:
      bicycle 0.88 / dog 0.85 / car 0.59, each with a tight instance
      mask. ORT reference (116-ch + proto decode, same preprocessing):
      bicycle 0.89 / dog 0.85 / car 0.70, masks firing in-box
      (0.44/0.69/0.61) — same instances, boxes within ~1 px; car score
      gap is the same NMS-selection fallout (sibling
      `ParityVsYolo11Seg` green, proto max-cell ≤ 0.017). GGUF carries
      `model.10.m.0.attn.qkv.conv.weight`, so the YOLO11 graph executes
      (note: the loader log still prints "YOLOv8n-seg" — sibling
      cosmetic follow-up, correct graph verified by tensor names and
      output shapes). Evidence under
      `/home/oli/opencode-gates/ggml-matrix-regate/gate-yolo11seg/`.
- [x] V-28 matrix/docs: GGML `yolo11` det and `yolo11seg` cells `U` → `P`
      (new gate-table rows); `docs/BackendCoverage.md` Environments/
      Known-limitations/Remaining-coverage notes updated. Every other
      GGML cell unchanged (`yolo26` BLOCKED at the converter;
      `yolo11pose`/`yolov10n` BLOCKED for lack of a sibling loader path).
- [x] V-29 YOLO11-pose sibling loader + parity (2026-11-19, sibling branch
      `feature/ggml-pose-v10` from `develop` 5b1ba85): pose layout probed —
      same det backbone (`model.23` DFL + `/model.10` C2PSA) with a 51ch
      cv4 kpt head (3 scales, no proto) and `[1,56,8400]` output
      (box 4 + cls 1 + 17x3 kpts); box constants 13/14/16, kpt decode
      (raw*2 + anchor) * stride with anchor/stride Constant_23/24.
      Converter needed no change (generic initializers + DFL check already
      accept `model.23`). Loader additions: `is_yolo11_pose_` dispatch
      (cv4 without proto + 51ch guard), `[1,56,8400]` shapes, 64-wide
      1-class cls branch, kpt decode + interleave to x0,y0,c0...,
      `yconst` accepts `{2,8400}`, seg completeness check relaxed to
      proto-without-cv4, log labels (`YOLO11n-pose`, plus `YOLO11n-seg`
      mislabel fixed). New `ParityVsYolo11Pose` gate (max_cell_agree=1,
      box/kpt 10.0, cls 1e-2): gray box 1.88/kpt 1.50, zeros 2.10/1.87,
      gradient 0.71/0.50, all agree=1. Debugging note: an early reshape
      `[8400,1,51,1]`→`[3,17,8400]` silently transposed (column-major
      reshape cannot move the anchor axis first); fixed with a strided
      `[8400,17,3]` view + explicit per-keypoint interleave. Full GGML
      suite: 25 passed, 0 failed (det/seg/v8 parity unchanged).
- [x] V-30 YOLO11n-pose GGUF gate: `yolo11n-pose.onnx` (op-inventory cache,
      11,764,411 bytes) → GGUF 199 tensors / 11,760,224 bytes →
      `--type=yolo11pose --source=data/bus.jpg ... --min_confidence=0.4
      --nms-threshold 0.5` — convert exit 0, run exit 0, `GGML model
      loaded: full YOLO11n-pose`, `run_report.json` status success,
      render inspected: 3 persons with correct 17-kpt skeletons on
      bodies. ORT reference (same letterbox-640 preprocessing, conf 0.4
      / NMS 0.5): same 3 persons (0.90/0.88/0.88, 9–16/17 visible kpts),
      nose/ankle coords consistent with the render. (`data/dog.jpg`
      correctly yields 0 persons — no person in frame; evidence kept as
      the negative control.) Evidence under
      `/home/oli/opencode-gates/ggml-matrix-regate/gate-yolo11pose/`
      (evidence.json, app.log, render, run_report.json, gguf). Built
      from `feature/ggml-matrix-coverage` against sibling
      `feature/ggml-pose-v10` worktree via `FETCHCONTENT_SOURCE_DIR_NEURIPLO`
      (tasks pinned at a `v0.8.5` worktree as in V-26).
- [x] V-31 matrix/docs: GGML `yolo11pose` cell `U` → `P` (new gate-table
      row); `docs/BackendCoverage.md` Known-limitations/Remaining-coverage
      notes updated. `yolov10n` stays BLOCKED (in-graph TopK decode needs
      mapping — separate packet); `yolo26` unchanged.

## Review (2026-11-14)

- Read-only reviewer pass over both repos: **no blocking issues**. Addressed
  the non-blocking findings: corrected the `output0` channel map in the
  sibling requirements, added an F32 output-type assertion before every
  float read, documented the ConvTranspose copy as an intentional
  reversed-dims coincidence, added gate-harness negative-path tests
  (missing input, converter failure, `--skip-convert`), and added a
  two-run determinism check to the seg parity test.

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

## Manual (brev live attempt — evidence, not aspiration)

- [ ] Brev instance created cheap CPU-only: `brev create ggml-matrix-coverage --type cpu-e2.4vcpu-16gb` (~$0.14/hr, 4 vCPU / 16 GB / x86_64). Record `brev exec ggml-matrix-coverage "nproc && free -g && uname -m"` output. V-1.
- [ ] Configure attempt on brev: `cmake -S . -B build-ggml -DDEFAULT_BACKEND=GGML -DCMAKE_BUILD_TYPE=Release`. Record exact stdout/stderr tail + exit code. If it fails, the failure text is the evidence (expected: missing `LinkBackend.cmake` GGML branch). V-2.
- [ ] If V-2 fails: smallest wiring probe applied, configure re-run, verbatim result recorded. No other build files touched. V-3.
- [ ] If configure passes: `cmake --build build-ggml` result + single-shape run (`3x224x224 F32 → 1000`) stdout/stderr, exit code, output values recorded. Zeros output = `BLOCKED`/placeholder confirmed, not `PASS`. No cell promoted above its proof. V-4.
- [x] Brev teardown: `brev delete ggml-matrix-coverage` executed; `brev ls` no longer lists it. Record both outputs. V-5 (required even if V-2/V-4 fail — no orphan instances).
      — 2026-11-13 re-verified: `brev ls` → `No instances in org olibartfast-hq`.
      Teardown complete, nothing billable remains.

## Matrix honesty (reviewer checks against `requirements.md` R-1..R-4)

- [x] R-1: `GGML` column exists in second gate table; legend disambiguates
      GGML from LLAMACPP. Phase 1 state was every cell `U`; Phase 2 promotes
      classification, `yolov8`, and `yoloseg` — see V-7/V-8/V-14.
- [x] R-2: `E19` records upstream builder + `77/77` engine tests AND the brev attempt (V-2..V-4 excerpts); `E11` keeps LLaMA PASS with cross-ref. `Environments` GGML row + `Open items` follow-up present. Scoreboard counts unchanged.
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
- [x] No running brev instance remains (`brev ls` verified post-teardown).

## Deviations (2026-11-13, orchestrator record)

- V-1 partial: instance `ggml-matrix-coverage` (`cpu-e2.4vcpu-16gb`, nebius,
  4 vCPU / 16 GB / x86_64, ~$0.14/hr) was created and reached
  HEALTHY/READY, but `nproc/free/uname` verification never returned —
  all `brev exec` calls timed out at SSH (connection timed out, retries
  exhausted; ephemeral host IP omitted). No remote command executed.
- V-2/V-3/V-4 not executed remotely: configure, build, and single-shape run
  never started on brev. Substituted with source-level bounds recorded in
  E19 (`GGMLInfer.cpp:91,94,97-98` placeholder graph, `:77-89` no model
  bytes parsed, `:61-70` CPU-forced, `:179-185` zeros return), which already
  prove no cell can promote above `U` without a sibling-side loader.
- V-5 in-flight: `brev delete ggml-matrix-coverage` issued and acknowledged,
  but `brev ls` still listed the instance RUNNING minutes later (delete is
  async: "can take a few minutes"). Teardown must be re-verified with
  `brev ls` before merge; do not merge with the instance still present.
- Automated checks (default build, test suite, `sync_supported_model_types
  --check` against the pinned `v0.8.5` README, `capabilities_schema_contract`,
  `pre-commit`) not yet run — docs-only diff so far, code untouched. Run
  them before opening the PR.
- Superseded 2026-11-14: V-2/V-3/V-4 were not executed remotely, but the
  equivalent app-surface gates ran locally against the sibling loader; see
  Phase 2 below. The blocked brev attempt stays recorded as the reason for
  the local rerun, not as a PASS.

## Cross-repo follow-up (2026-11-13, `neuriplo` real-loader packet)

- `neuriplo` branch `feature/ggml-real-loader`: real head-only loader
  (pool + real FC through ggml), converter emitting NGGL, hardened tests —
  local gates green (77/77 ctest, golden + F2 orientation probe exact vs
  numpy). Brev live gate in progress on `ggml-real-loader` (cpu-e2.4vcpu-16gb);
  teardown required before any merge.
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

- [x] V-6 build: `cmake --build build-ggml` (cache:
      `DEFAULT_BACKEND=GGML`, `GGML_DIR=/home/oli/dependencies/ggml`,
      `neuriplo_SOURCE_DIR=/home/oli/repos/neuriplo`) — app re-linked against
      the loader branch; `GGML model loaded: full ResNet50 ...` and
      `full YOLOv8n ... real weights` in the run logs.
- [x] V-7 classification app gate: ONNX Model Zoo `resnet50-v1-12.onnx` →
      NGGL 299 tensors / 102,532,870 bytes →
      `neuriplo-infer --type=torchvisionclassifier --weights=resnet50.nggl
      --source=data/dog.jpg --input_sizes=1,3,224,224` — exit 0, 206 ms,
      top-5 `249 0.793273`, `248 0.11378`, `250 0.036304`, `537 0.024229`,
      `261 0.011177`; ORT 1.32.0 on the same preprocessed input gives the
      same indices and probabilities. Render inspected.
- [x] V-8 YOLOv8n detection app gate: ultralytics 8.4.213
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
- [x] V-14 `yoloseg` app gate: `yolov8n-seg.onnx` (ultralytics 8.4.213,
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
- [x] V-15 transformer/RT-DETR foundation: sibling packet
      `neuriplo/specs/2026-11-14-ggml-attention-ops` adds `GGMLOps` helpers
      and 7/7 op-level/tiny-block parity tests; sibling `GGMLInferTest` is
      21 tests (20 pass, 1 skip) and full ctest 77/77. No app gate: the
      available `rtdetrv4.onnx` is opset 18, 1928 nodes, 2 inputs
      (FP32 image + INT64 sizes), 3 outputs, with `GridSample`/`TopK`/
      `GatherElements`/dynamic shapes. `GGMLInfer` is still a
      single-FP32-input hand-written topology loader, so RT-DETR is BLOCKED
      before load (E3) until multi-input/INT64 and a generic graph importer
      land. No GGML cell promoted.

Artifacts: `/tmp/opencode/ggml-matrix/{resnet50-v1-12.onnx,resnet50.nggl,
yolov8n.onnx,yolov8n.ggml,yolov8n-seg.onnx,run_app_verbose.log,run_yolo.log}`;
sibling backend artifact `/tmp/yolov8n-seg.ggml`; gate evidence under
`/tmp/opencode/ggml-matrix/gate-{classification,yolov8,yoloseg}/evidence.json`;
renders under the respective `run/data/output/` directories.

## Review (2026-11-14)

- Read-only reviewer pass over both repos: **no blocking issues**. Addressed
  the non-blocking findings: corrected the `output0` channel map in the
  sibling requirements, added an F32 output-type assertion before every
  float read, documented the ConvTranspose copy as an intentional
  reversed-dims coincidence, added gate-harness negative-path tests
  (missing input, converter failure, `--skip-convert`), and added a
  two-run determinism check to the seg parity test.

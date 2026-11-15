# Backend coverage — task models × backends

One gated representative per task family, one status per backend, no cell
claiming more than its evidence. Style follows the NERT
[coverage matrix](https://github.com/olibartfast/nert/blob/main/specs/2026-10-17-task-models/coverage.md).

**Last updated: 2026-11-14.** Baseline evidence uses neuriplo-infer
0.11.6, neuriplo 0.11.3 and neuriplo-tasks 0.8.5. Newer local corrections and
unreleased backend fixes are scoped in the Environments table and Open items.
The [gated table](#gate-table) summarizes coverage by task and backend. Build
versions and test totals are under [Environments](#environments); remaining
gaps are under [Open items](#open-items). Matrix coverage is always reported
here — never only in an external harness's working files. OpenCV DNN
coverage is 5.x only; 4.x is not considered.

Main limitations:

- **TorchScript input metadata:** since neuriplo 0.11.3, TorchScript files
  must record their input datatypes. Files exported without the metadata
  helper are rejected at load time and must be re-exported. Re-exported
  DETR files use the supported PASS routes; the RF-DETR TorchScript
  architecture limitation remains distinct from its passing AOTI route.
- **Legacy incorrect artifacts:** old EdgeCrafter and generic-`onnx2tf`
  RF-DETR TFLite files must be replaced. EdgeCrafter now passes locally and
  over HTTP with a reconverted model and the local XNNPACK reshaping fix; RF-DETR uses its native LiteRT exporter. TensorFlow ResNet-50
  passes with the corrected exporter. The LiteRT backend fix is local
  and has not yet been released or pinned.
- **Incomplete test coverage:** OpenVINO and LLaMA native unit-test binaries
  do not link.
- **Backend and transport blockers:** apt OpenCV 4.6 cannot load current
  models — the DNN column below runs OpenCV 5.0.0 source-built (see
  [Environments](#environments)); KServe HTTP has live evidence, while the
  gRPC failure remains blocked server-side.
- **OpenVINO status (evidence recorded 2026-11-04, P-LNX-EDGE-CPU):**
  `owlv2` / `owlvit` PASS; `videomae` / `vivit` / `timesformer`
  RUNTIME-PASS (inference only, no video export); `rtdetrul` PASS on the
  current local stack; `tensorflow-classifier` remains KNOWN-BAD and was
  not re-gated for OpenVINO. Slurm results use a derived validation image
  with missing runtime libraries added; they do not certify the original
  image or a fresh source build. See Open items.

See [Open items](#open-items) for remaining validation and fixes.

## How to read this

- **PASS** — a live end-to-end run is recorded with its expected output
  (detections, labels, or round-trip log).
- **RUNTIME-PASS** — a live run completed with positive sample count and
  inference timing, but output correctness or rendering is not fully gated.
  This does not satisfy a full PASS.
- **DRY-RUN** — an e2e preset exists for this task × backend and the
  `e2e-example-emulation.yml` dry-run is green. Commands are well-formed;
  no live inference is claimed.
- **BUILDS** — CI `ci.yml` compiles this backend, but no preset exercises
  this task on it. It links; it has not run this task.
- **BLOCKED** — a documented loader/export blocker stops this combination
  before inference (cited, not guessed).
- **KNOWN-BAD** — it loads and runs but the outputs are numerically wrong;
  the fault location is cited. Do not serve it.
- **UNTESTED** — none of the above. No build, preset, or live evidence.
  The combination could work; it just hasn't been tried.
- **N/A** — the backend cannot serve this task by construction (e.g. LLaMA
  is GGUF + `mmproj` only, so it never applies to detection/segmentation).
  Not a coverage gap. Maps to the matrix's NOT_APPLICABLE.

`BUILDS` means the `ci.yml` job for that backend is green (`NERT`, `OPENCV_DNN`,
`ONNX_RUNTIME`, `LIBTORCH`, `LIBTENSORFLOW`, `TENSORRT`, `OPENVINO`); there is
no `LITERT` / `EXECUTORCH` / `LLAMACPP` / `MIGRAPHX` / `GGML` build job. `DRY-RUN` means the preset
is listed by `--list-presets` and green in `e2e-example-emulation.yml`
(dry-run validates command construction, not numerics).

**Matrix-to-table rule:** who enters the Matrix enters the table — every
explicit TaskFactory architecture with a terminal matrix result gets a
gate-table row. No matrix-covered architecture stays table-less.

Node counts, sha256 digests, and per-tensor tolerances belong to the
exporting repo (`neuriplo-tasks`) and the engine repo (`neuriplo`), not here.
This matrix gates the app seam: routed, run, rendered, tested.

## Gate table

The same tasks appear in both tables. Backends are split into groups to
keep each table narrow. [Scope and caveats](#scope-and-caveats) explains the
NERT gates and the marked cells.

`DNN` = OPENCV_DNN · `ORT` = ONNX_RUNTIME · `TRT` = TENSORRT ·
`TRC` = LIBTORCH · `OV` = OPENVINO · `TF` = LIBTENSORFLOW.
`ExecT` = EXECUTORCH · `LLaMA` = LLAMACPP · `KServe` = remote `KserveEngine`.
`MGX` = MIGraphX, the independent `MIGRAPHX` backend in neuriplo, distinct from
ONNX Runtime's MIGraphX execution provider.
`GGML` = GGML, the independent `GGML` backend in neuriplo, distinct from
the `LLAMACPP` (llama.cpp) column.
NERT is the first-party Neuriplo Engine Runtime. Representative presets
appear in parentheses.

Table key: **P** = PASS · **U** = UNTESTED · **B** = BUILDS · **DR** = DRY-RUN.
Some cells use a different member of the row family than the preset
(RT-DETRv4-s for `rtdetr`, YOLO11m-seg for `yoloseg`, ResNet-50 for
`torchvisionclassifier`), or include local unreleased fixes; the Environments
table and Open items record the artifact and scope.

### Local inference backends

| Task (representative) | NERT | DNN | ORT | TRT | TRC | OV | TF |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Object detection (`yolo26`, `yolo26s_tflite`) | P | P | P | P | P | P | B |
| Object detection (`yolov8`, `yolov8_executorch`) | P | P | DR | B | B | B | P |
| Object detection (`rtdetr`, `rtdetrv4`) | P | P | DR | DR | P | B | B |
| Object detection (`rtdetrul`, Ultralytics RT-DETR) | U | U | P | U | U | P | U |
| Object detection (`ecdet`, `edgecrafter_det`) | P | U | P | P | B | B | B |
| Object detection (`rfdetr`, `rfdetr-medium`) | P | P | P | P | P | P | B |
| Instance segmentation (`yoloseg`, `yoloseg`) | P | P | P | P | B | B | B |
| Instance segmentation (`ecseg`, `edgecrafter_seg`) | P | U | DR | B | B | B | B |
| Instance segmentation (`rfdetrseg`, `rfdetrseg-medium`) | P | P | P | P | P | P | B |
| Classification (`torchvisionclassifier`, `torchvision_classifier`) | P | U | DR | B | P | B | P |
| Video classification (`videomae`, `videomae`) | P | U | DR | B | B | RUNTIME-PASS | B |
| Optical flow (`raft`, `raft`) | P | U | DR | B | B | B | B |
| Pose estimation (`vitpose`, `vitpose`) | P | P | DR | B | B | B | B |
| Pose estimation (`rfdetrpose`, `rfdetr_keypoint`) | P | P | P | P | P | P | B |
| Depth estimation (`depth_anything_v2`, `depth_anything_v2`) | P | P | P | B | B | P | B |
| Open-vocabulary detection (`owlv2`, `owlv2`) | P | P | P | B | B | P | B |
| Image understanding (`gemma4`, `gemma4`) | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Gaussian splatting (`lgm`/`grm`, no preset) | N/A | U | U | U | B | B | U |
| Keypoint matching (`SuperPoint + LightGlue`, no app preset) | U | U | P | P | U | U | U |

### Additional runtimes and remote execution

| Task (representative) | MGX | GGML | LiteRT | ExecT | LLaMA | KServe |
| --- | --- | --- | --- | --- | --- | --- |
| Object detection (`yolo26`, `yolo26s_tflite`) | U | U | P | U | N/A | P |
| Object detection (`yolov8`, `yolov8_executorch`) | U | P | P | P | N/A | U |
| Object detection (`rtdetr`, `rtdetrv4`) | U | U | U | U | N/A | U |
| Object detection (`rtdetrul`, Ultralytics RT-DETR) | U | U | U | U | N/A | U |
| Object detection (`ecdet`, `edgecrafter_det`) | U | U | P | U | N/A | P |
| Object detection (`rfdetr`, `rfdetr-medium`) | U | U | P | P | N/A | P |
| Instance segmentation (`yoloseg`, `yoloseg`) | U | P | U | U | N/A | P |
| Instance segmentation (`ecseg`, `edgecrafter_seg`) | U | U | U | U | N/A | U |
| Instance segmentation (`rfdetrseg`, `rfdetrseg-medium`) | U | U | P | P | N/A | P |
| Classification (`torchvisionclassifier`, `torchvision_classifier`) | U | P | U | U | N/A | U |
| Video classification (`videomae`, `videomae`) | U | U | U | U | N/A | U |
| Optical flow (`raft`, `raft`) | U | U | U | U | N/A | U |
| Pose estimation (`vitpose`, `vitpose`) | U | U | P | U | N/A | P |
| Pose estimation (`rfdetrpose`, `rfdetr_keypoint`) | U | U | P | P | N/A | P |
| Depth estimation (`depth_anything_v2`, `depth_anything_v2`) | U | U | P | U | N/A | P |
| Open-vocabulary detection (`owlv2`, `owlv2`) | U | U | U | U | N/A | P |
| Image understanding (`gemma4`, `gemma4`) | N/A | U | N/A | N/A | P | U |
| Gaussian splatting (`lgm`/`grm`, no preset) | U | U | U | U | N/A | U |
| Keypoint matching (`SuperPoint + LightGlue`, no app preset) | U | U | U | U | N/A | U |

### Scope and caveats

The MGX column tracks the app validation gap. Existing upstream CPU and
AMD GPU contract tests are recorded in the upstream contract suite; they do not promote
any task cell to PASS, RUNTIME-PASS or BUILDS. No app build or preset gate is
recorded for this backend. Its ONNX-only input format cannot serve the GGUF
`gemma4` representative, so that cell is N/A.

The GGML column records three local app-surface PASSes — classification,
`yolov8` detection, and `yoloseg` — plus the upstream builder and
engine-test evidence. All runs used the unreleased sibling
loader (`neuriplo` `feature/ggml-yolo-family` @ `9ee2bde` plus the unreleased
`2026-11-14-ggml-yoloseg` work), because the pinned `neuriplo v0.11.3` predates it. No CI
build job and no preset gate are claimed. The loader implements ResNet-18/50
and the YOLOv8n detection/segmentation graphs only; every other GGML task cell
stays `U` until sibling graph work lands. `gemma4` stays `U`: GGML serves no
VLM, and the VLM evidence belongs to the `LLaMA` column.

The `TRC` column covers TorchScript. AOTInductor is a second runtime inside
the same LibTorch backend, selected when the model path ends in `.pt2`. It is
not covered by these cells yet — except RF-DETR, whose TRC PASS rests on
AOTInductor because TorchScript tracing is broken for that architecture; do
not read any other TRC cell as AOTInductor evidence.

**NERT scope:** `PASS` records the upstream full-model reference gate
on CPU, CUDA and OpenCL from the
[NERT coverage matrix](https://github.com/olibartfast/nert/blob/main/specs/2026-10-17-task-models/coverage.md)
(2026-10-24). It does not assert an app CLI/render gate or validate the app's
released dependency pins. Neuriplo exposes CPU and opt-in CUDA; OpenCL and HIP
are engine capabilities without consumer selection. Full-catalog AMD/HIP
validation remains incomplete.

## Environments

How each column was built and tested. App gates are CPU except TRT;
the upstream MIGraphX contract suite has separate CPU and MI300X GPU results.
Rows without a local-fix qualification use the 2026-11-01/02 pinned-stack
evidence (neuriplo-infer 0.11.7-dev + neuriplo 0.11.3 + tasks 0.8.4).
DNN and ExecT include the newer 2026-11-04 local builds; their fixes
are not released dependency pins.

| Backend | Version / artifact | Build flags | Tests | Notes |
|---|---|---|---|---|
| DNN | OpenCV 5.0.0 source-built (`~/dependencies/opencv-5.0.0`, `scripts/setup_opencv.sh`) | `OPENCV_DNN` | app 191/191; backend 80/80 | Named/typed graph inputs; explicit sizes only for unresolved dynamic dimensions; apt 4.6 legacy compatibility is separate |
| ORT | 1.19.2 CPU (local SDK; pin 1.28.0 unfetchable upstream — `setup_dependencies.sh` handles the renamed `gpu_cuda<major>` archives) | `ONNX_RUNTIME` | 179/179 | Full inference runs, renders inspected |
| TRT | 11.2.1.2 (NGC `tensorrt:26.08` base, `docker/Dockerfile.tensorrt`) | `TENSORRT` | image green in CI 2026-11-02 | Trio live-gated 2026-11-01 (det 109 ms, seg 380 ms + masks, keypoint 292 ms, pose 0.850754). 11.2 `trtexec` dropped the `--fp16` shorthand; engines rebuilt from the same `.onnx` |
| Torch | libtorch 2.14 CPU | `LIBTORCH`, `NEURIPLO_INFER_ENABLE_KSERVE=OFF` | 134/134 | TorchScript cells, except RF-DETR where it rests on AOTInductor (`.pt2`) because TorchScript tracing is broken for that arch. KServe client unlinked (its gRPC protobuf clashes with libtorch's bundled protobuf); first `cmake` seeds the CUDA cache, a second run is needed (nvcc 12.0 vs driver-reported 13.4). Since neuriplo 0.11.3, TorchScript archives must carry input-datatype metadata; legacy files are rejected |
| OVINO | 2026.3.0 (pinned; `docker/Dockerfile.openvino`) | `OPENVINO`, tests OFF | — (not covered) | Trio live-gated 2026-11-01 (det 784 ms, seg 1184 ms + masks, keypoint 1566 ms, pose 0.850784). Native `runUnitTests` link vs `openvino::runtime` fails; CI builds tests OFF too |
| TF | 2.19 (`LIBTENSORFLOW`; 2.21 C++ runtime unloadable — protobuf ABI) | `LIBTENSORFLOW` | 179/179 | `--weights` accepts directories since 2026-10-23 (was file-only) |
| LiteRT | `~/dependencies/litert` (LiteRT 2.19.0 pinned) | `LITERT` | 192/192 | Vendored gtest/gmock headers pruned from the SDK (they shadowed system gtest and crashed every death test) |
| ExecT | v1.2.0 optimized SDK `/tmp/blocked-et-sdk` | `EXECUTORCH`; KServe=ON, gRPC=ON | app 191/191; backend 77/77 | XNNPACK + combined optimized registration; compiled transport is not live transport validation |
| MGX | Recorded upstream environment: 2.12.0.60401-83, ROCm 6.4.1, MI300X `gfx942` | upstream `MIGRAPHX`; app build unvalidated | backend contracts 6/6: 3 CPU + 3 GPU, zero skips | Synthetic typed fixtures only; no app or learned-model gate |
| GGML | Local CPU (AMD Ryzen 7 5700U); unreleased sibling `neuriplo` `feature/ggml-yolo-family` @ `9ee2bde` + unreleased `2026-11-14-ggml-yoloseg`/`2026-11-14-ggml-attention-ops`; `GGML_DIR=/home/oli/dependencies/ggml` | `-DDEFAULT_BACKEND=GGML` app build; no CI job | app 197/197 in the GGML test build; classification + `yolov8` + `yoloseg` runs; upstream `ctest` 77/77 in the builder image; backend `GGMLInferTest` 21 tests (20 pass, 1 intentional skip), including 7/7 `GGMLOpsParity` | ResNet-50 top-5 exact vs ORT; YOLOv8n detection/segmentation match ORT within 1e-2; transformer/RT-DETR remains blocked on the graph importer; other GGML task cells remain `U` |
| LLaMA | llama.cpp b9085 | `LLAMACPP`, tests OFF | — (not covered) | Native `runUnitTests` link misses ggml transitive libs (same class as the OVINO gap) |
| KServe | `neuriplo-kserve-runtime` `real-onnx` / `grpc-onnx` | client in default build | suites run manually | HTTP green incl. per-task gates; gRPC `ModelInfer` 13 INTERNAL server-side |

## Scoreboard

| Status | Count (task rows) | Rows |
|---|---|---|
| PASS (≥1 backend live-gated) | 14 | `yolo26`, `yolov8`, `rtdetr`, `rtdetrul`, `ecdet`, `rfdetr`, `owlv2`, `gemma4`, `yoloseg`, `rfdetrseg`, `torchvisionclassifier`, pose (`vitpose` via `rfdetr_keypoint` alias, `rfdetrpose`), depth (`yolo26depth` alias) |
| Legacy KNOWN-BAD artifacts (supported replacements available) | 2 | `rfdetrseg` on LiteRT-`onnx2tf` (converter fault, stands; native route PASS); generic-`onnx2tf` `rfdetr` (native-exporter route is PASS) |
| RUNTIME-PASS best | 1 | `videomae` (OVINO inference; video export unvalidated) |
| DRY-RUN best | 2 | `ecseg`, `raft` |
| BUILDS / UNTESTED best | 1 | Gaussian splatting (no preset; Torch/OVINO BUILDS-only) |
| **Total** | **18** | 10 task families; detection/segmentation/pose split by representative |

## Open items

1. **Live-gate every DRY-RUN cell.** Closest first: `rtdetrv4` on TRT;
   `raft` / `vitpose` / `videomae` on ORT.
   Export or obtain the required artifacts before live validation.
2. **Record Torch / OpenVINO `--backend` override runs as repeatable gates.**
   The DETR Torch run is a manual CLI invocation on `rtdetr.torchscript`, not
   a preset. Add repeatable preset gates using the typed TorchScript exports.
3. **OPENCV_DNN scope (5.0.0).** Named typed input binding resolves
   RT-DETR and OWLv2 locally; v4 needs its compatible export. GroundingDINO
   has not been re-gated, and EdgeCrafter exact-GELU remains an importer-level
   open issue. Parallel repaired Slurm validation awaits worker availability.
   Still unprobed on 5.x: `ecseg`, torchvision ResNet, `vitpose`,
   `depth_anything_v2`.
4. **KServe per-task gates.** Done for servable local models (`real-onnx`,
   HTTP): pose, depth, open-vocab — plus YOLO26-`litert` PASS and `ecdet`
   PASS with the local backend patch. gRPC transport BLOCKED server-side. Still open: classification,
   video, flow, VLM serving.
5. **Gaussian-splatting representative.** No preset exists. Export one `lgm`
   artifact, add a preset, then gate it.
6. **MIGraphX app gates.** Validate an app build with the independent
   backend and run ONNX task representatives on a supported AMD/ROCm worker.
   Record expected outputs and inspected renders before promoting the
   UNTESTED cells; the upstream synthetic suite is already covered.
7. **GGML app gate.** Classification, `yolov8`, and `yoloseg` now pass
   locally against the unreleased sibling loader. `yolo26` is
   BLOCKED at the converter: its `model.23` head needs attention ops,
   so that row stays `U` until the sibling converter/graph work lands.
   `rtdetr` is BLOCKED before load on multi-input/INT64 and a generic
   graph importer; the op-level attention helpers in
   `neuriplo/specs/2026-11-14-ggml-attention-ops` are the foundation, not a
   model gate. The other transformer representatives need the same importer
   work. Re-gate here after that lands. These local runs are
   sibling-checkout evidence, not isolated remote-release gates.

## Integration-test evidence

This section owns all app-surface test documentation. Each suite below
lives under `integration-tests/` as a runner (README + executable entry
point); coverage claims for gate-table cells are made here, not in the
suite READMEs. Live paths need local binaries, model artifacts, or a GPU
and stay manual; what runs in routine CI is noted per suite.

DRY-RUN cells everywhere rest on the same two automated legs: the
`local-inference-smoke` runner pattern (preset lists, one preset
dry-runs) plus the `e2e-example-emulation.yml` matrix. CTest names are
from `app/test/CMakeLists.txt`.

| Suite | Gates which cells | CI | Live (manual) | Runner docs |
|---|---|---|---|---|
| `local-inference-smoke` | DRY-RUN shape for every preset row (lists presets, dry-runs one) | `local_inference_smoke_runner` (`run.py --preset owlv2`) | — (dry-run only by design) | [README](../integration-tests/local-inference-smoke/README.md), [run.py](../integration-tests/local-inference-smoke/run.py) |
| `kserve-runtime-yolo-e2e` | `yolo26` KServe PASS (YOLO26 `.tflite` over `litert`) | — (needs runtime binary + model) | `run.py` — starts `neuriplo-kserve-runtime`, readiness + metadata, app-binary HTTP infer, refreshed render, metrics 1/0 | [README](../integration-tests/kserve-runtime-yolo-e2e/README.md), [run.py](../integration-tests/kserve-runtime-yolo-e2e/run.py) |
| `kserve-runtime-edgecrafter-e2e` | `ecdet` ORT/TRT PASS + OpenVINO/ExecT matrix; `ecdet` LiteRT/KServe HTTP replacement PASS with local backend patch ( separate manual gate) | — (needs per-backend runtime binaries + model repository + GPU for TRT) | `run.py --backend {onnx_runtime,tensorrt,openvino,executorch} [--transport {http,grpc}] [--mode {kserve,local}]`; `prepare_model_repository.py` lays out `ecdet_s_<backend>/1/` | [README](../integration-tests/kserve-runtime-edgecrafter-e2e/README.md), [BENCHMARK.md](../integration-tests/kserve-runtime-edgecrafter-e2e/BENCHMARK.md) |
| `rfdetr-pose-local-vs-kserve` | Pose row scaffold for `rfdetr_keypoint` (evidence tracking, not a live gate) | `rfdetr_pose_evidence_scaffold` (`run.py` — baselines + report exist) | Attested checks in [evidence.yaml](../integration-tests/rfdetr-pose-local-vs-kserve/evidence.yaml) (`kserve_grpc_smoke` still `skipped`) | [README](../integration-tests/rfdetr-pose-local-vs-kserve/README.md), [evidence.yaml](../integration-tests/rfdetr-pose-local-vs-kserve/evidence.yaml), [reports/latest.md](../integration-tests/rfdetr-pose-local-vs-kserve/reports/latest.md) |
| `tpu-kserve` | Remote-TPU scaffold for `tensorflow-classifier` over KServe V2 (evidence tracking, not a live gate) | `tpu_kserve_dry_run` (`run.py --dry-run` — scaffold files exist, live command prints, no network) | `run.py --live` against an operator-supplied TPU endpoint; attested checks in [evidence.yaml](../integration-tests/tpu-kserve/evidence.yaml) (both live checks `skipped`) | [README](../integration-tests/tpu-kserve/README.md), [run.py](../integration-tests/tpu-kserve/run.py), [evidence.yaml](../integration-tests/tpu-kserve/evidence.yaml) |

KServe protocol-level legs (not app-surface suites, but they back the
KServe column): `kserve_integration_dry_run` and
`kserve_input_datatypes_e2e_dry_run` in CTest (`app/test/kserve_integration.sh`,
`app/test/kserve_input_datatypes_e2e.sh`; `--live` is opt-in) — full matrix
in [KserveCompatibility.md](KserveCompatibility.md).

## Non-kernel limits

- **Opset / export policy lives upstream.** This repo consumes what
  `neuriplo-tasks` exports; conversion fidelity (e.g. `onnx2tf` for LiteRT)
  is judged at this seam but fixed upstream.
- **Task processing stays on the client by default.** `neuriplo-tasks` owns
  NMS, mask assembly, flow/depth resize, and tokenization semantics; this app
  wires those contracts into the pipeline. KServe input and postprocess modes
  determine which processing is delegated to the server.
- **One binary, one local backend.** Each image carries the backend it was
  built with (`DEFAULT_BACKEND`); switching means rebuilding. KServe mode is
  the only multi-backend workflow in one binary.
- **CI builds ≠ runs.** A green `ci.yml` build job puts BUILDS in the column;
  only a recorded live output promotes a cell to PASS.

## Sources

- Backends and artifacts: [Deployment.md — Backends](Deployment.md#backends),
  `docker/Dockerfile.*`, `docker_run_inference_e2e_example.sh --list-presets`.
- Emulation evidence: `.github/workflows/e2e-example-emulation.yml` matrix.
- Build evidence: `.github/workflows/ci.yml` jobs.
- KServe evidence: [KserveCompatibility.md](KserveCompatibility.md) (live
  `ecdet` numbers, YOLO26-`litert` note, Identity round-trips).
- Task inventory: [generated/supported-model-types.md](generated/supported-model-types.md),
  `app/src/Capabilities.cpp` (`taskCapabilities`), `app/src/NeuriploInferTaskRouting.cpp`.
- TorchScript input metadata:
  [torchscript_metadata.py](https://github.com/olibartfast/neuriplo/blob/master/scripts/lib/torchscript_metadata.py)
  (writes the datatype sidecar read by the libtorch backend).
- App-surface integration tests: [Integration-test evidence](#integration-test-evidence)
  above owns them (moved from `neuriplo-platform`; live paths stay manual
  per each suite's README).
- Style reference:
  [NERT coverage matrix](https://github.com/olibartfast/nert/blob/main/specs/2026-10-17-task-models/coverage.md).

# Backend coverage — task models × backends

One gated representative per task family, one status per backend, no cell
claiming more than its evidence. Style follows the NERT
[coverage matrix](https://github.com/olibartfast/nert/blob/main/specs/2026-10-17-task-models/coverage.md).

**Last updated: 2026-10-23.** The [gate table](#gate-table) summarizes
coverage by task and backend. Recorded results are listed under
[Evidence](#evidence); build versions and test totals are under
[Environments](#environments).

Main limitations:

- **Incorrect outputs:** TensorFlow ResNet-50 classification and EdgeCrafter
  TFLite conversion are **KNOWN-BAD**; do not serve these combinations.
- **Incomplete test coverage:** LiteRT has 21 failing death-tests;
  OpenVINO and LLaMA native unit-test binaries do not link.
- **Backend and transport blockers:** apt OpenCV 4.6 cannot load current
  models — the DNN column below runs OpenCV 5.0.0 source-built (see
  [Environments](#environments)); KServe HTTP has live evidence, while the
  gRPC failure described in E1 remains blocked server-side.

See [Open items](#open-items) for remaining validation and fixes.

## How to read this

- **PASS** — a live end-to-end run is recorded with its expected output
  (detections, labels, or round-trip log) in the task's evidence section.
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

`BUILDS` means the `ci.yml` job for that backend is green (`OPENCV_DNN`,
`ONNX_RUNTIME`, `LIBTORCH`, `LIBTENSORFLOW`, `TENSORRT`, `OPENVINO`); there is
no `LITERT` / `EXECUTORCH` / `LLAMACPP` build job. `DRY-RUN` means the preset
is listed by `--list-presets` and green in `e2e-example-emulation.yml`
(dry-run validates command construction, not numerics).

Node counts, sha256 digests, and per-tensor tolerances belong to the
exporting repo (`neuriplo-tasks`) and the engine repo (`neuriplo`), not here.
This matrix gates the app seam: routed, run, rendered, tested.

## Gate table

Columns: `DNN` = OPENCV_DNN · `ORT` = ONNX_RUNTIME · `TRT` = TENSORRT ·
`Torch` = LIBTORCH · `OVINO` = OPENVINO · `TF` = LIBTENSORFLOW ·
`LiteRT` = LITERT · `ExecT` = EXECUTORCH · `LLaMA` = LLAMACPP ·
`KServe` = remote `KserveEngine`. Representative preset in parentheses.
Evidence column points to the task's section below.

| Task (representative) | DNN | ORT | TRT | Torch | OVINO | TF | LiteRT | ExecT | LLaMA | KServe | Evidence |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Object detection (`yolo26`, `yolo26s_tflite`) | PASS | PASS | PASS | PASS | PASS | BUILDS | PASS | UNTESTED | UNTESTED | PASS | [E1](#ev-yolo26) |
| Object detection (`yolov8`, `yolov8_executorch`) | PASS | DRY-RUN | BUILDS | BUILDS | BUILDS | BUILDS | UNTESTED | PASS | UNTESTED | UNTESTED | [E2](#ev-yolov8) |
| Object detection (`rtdetr`, `rtdetrv4`) | UNTESTED | DRY-RUN | DRY-RUN | BUILDS | BUILDS | BUILDS | UNTESTED | UNTESTED | UNTESTED | UNTESTED | [E3](#ev-rtdetr) |
| Object detection (`ecdet`, `edgecrafter_det`) | UNTESTED | PASS | PASS | BUILDS | BUILDS | BUILDS | KNOWN-BAD | UNTESTED | UNTESTED | KNOWN-BAD | [E4](#ev-ecdet) |
| Instance segmentation (`yoloseg`, `yoloseg`) | PASS | PASS | PASS | BUILDS | BUILDS | BUILDS | UNTESTED | UNTESTED | UNTESTED | PASS | [E5](#ev-seg) |
| Instance segmentation (`ecseg`, `edgecrafter_seg`) | UNTESTED | DRY-RUN | BUILDS | BUILDS | BUILDS | BUILDS | UNTESTED | UNTESTED | UNTESTED | UNTESTED | [E5](#ev-seg) |
| Classification (`torchvisionclassifier`, `torchvision_classifier`) | UNTESTED | DRY-RUN | BUILDS | PASS | BUILDS | KNOWN-BAD | UNTESTED | UNTESTED | UNTESTED | UNTESTED | [E6](#ev-classification) |
| Video classification (`videomae`, `videomae`) | UNTESTED | DRY-RUN | BUILDS | BUILDS | BUILDS | BUILDS | UNTESTED | UNTESTED | UNTESTED | UNTESTED | [E7](#ev-video) |
| Optical flow (`raft`, `raft`) | UNTESTED | DRY-RUN | BUILDS | BUILDS | BUILDS | BUILDS | UNTESTED | UNTESTED | UNTESTED | UNTESTED | [E7](#ev-video) |
| Pose estimation (`vitpose`, `vitpose`) | PASS | DRY-RUN | BUILDS | BUILDS | BUILDS | BUILDS | UNTESTED | UNTESTED | UNTESTED | PASS | [E8](#ev-pose) |
| Depth estimation (`depth_anything_v2`, `depth_anything_v2`) | PASS | DRY-RUN | BUILDS | BUILDS | BUILDS | BUILDS | UNTESTED | UNTESTED | UNTESTED | PASS | [E9](#ev-depth) |
| Open-vocabulary detection (`owlv2`, `owlv2`) | BLOCKED | PASS | BUILDS | BUILDS | BUILDS | BUILDS | UNTESTED | UNTESTED | UNTESTED | PASS | [E10](#ev-owlv2) |
| Image understanding (`gemma4`, `gemma4`) | UNTESTED | UNTESTED | UNTESTED | UNTESTED | UNTESTED | UNTESTED | UNTESTED | UNTESTED | PASS | UNTESTED | [E11](#ev-gemma4) |
| Gaussian splatting (`lgm`/`grm`, no preset) | UNTESTED | UNTESTED | UNTESTED | BUILDS | BUILDS | UNTESTED | UNTESTED | UNTESTED | UNTESTED | UNTESTED | [E12](#ev-splatting) |

## Evidence

Live runs below are all 2026-10-23 on `data/dog.jpg` (`person.jpg` for pose)
with rendered outputs inspected, unless noted otherwise.

### <a id="ev-yolo26"></a>E1 — Object detection (`yolo26`)

- **DNN PASS** (5.0.0 source-built, CPU, 378 ms): @ 0.4/0.5 — bicycle 0.95,
  dog 0.94, car 0.52, needs `--input_sizes=3,640,640` (the 5.0 backend
  requires explicit input sizes). apt OpenCV 4.6 stays BLOCKED on this
  model (`Split` parse error) — use the 5.x build from
  `scripts/setup_opencv.sh`.
- **ORT PASS** (CPU, 1137 ms): @ 0.4/0.5 — bicycle 0.95, dog 0.94, car 0.52.
- **TRT PASS** (10.14.1.48 FP16 engine from `yolo26s.onnx`, 251 ms):
  identical outputs. The pre-existing `rfdetr-keypoint-preview.engine`
  (built with 10.13) does NOT deserialize on 10.14 — engines were rebuilt
  with 10.14 `trtexec`.
- **Torch PASS** (CPU, 1083 ms): `yolo26s.torchscript` (ultralytics export)
  with `--input_sizes=1,3,640,640` — identical outputs. The TorchScript
  graph declares dynamic inputs, and the backend uses provided sizes
  verbatim, so the batch dim must be included explicitly.
- **OpenVINO PASS** (CPU, 267 ms): `yolo26s.onnx` read directly.
- **LiteRT PASS** (CPU, 888 ms): `yolo26s.tflite` (`1x640x640x3`).
- **KServe PASS** (HTTP): YOLO26 `.tflite` over `litert`, validated locally
  2026-06-13 — [KserveCompatibility.md](KserveCompatibility.md#model-formats);
  live path is `kserve-runtime-yolo-e2e`
  ([§ Integration-test evidence](#integration-test-evidence)).
- **KServe gRPC BLOCKED (server-side).** Same `grpc-onnx` server + model
  answers HTTP 200, but gRPC `ModelInfer` fails 13 INTERNAL — reproduced
  with the sibling `kserve-client-conformance` oracle. Fault lives in
  `neuriplo-kserve-runtime`'s gRPC respond path, not here.

### <a id="ev-yolov8"></a>E2 — Object detection (`yolov8`)

- **DNN PASS** (5.0.0, CPU, 306 ms): `yolov8n.onnx` @ 0.4/0.5 — bicycle 0.89,
  dog 0.93, car 0.61. The 4.6 `Unsqueeze` blocker is gone on 5.x.
- **ExecT PASS** (CPU, 320 ms): `yolov8n.pte` @ 0.4/0.5 — bicycle 0.89,
  dog 0.93, car 0.61. No CI build job for ExecT.

### <a id="ev-rtdetr"></a>E3 — Object detection (`rtdetr`)

- **ORT/TRT DRY-RUN.** TRT is the default preset; `ci.yml` builds the TRT
  image; emulation dry-run green. No live log cited — not PASS.

### <a id="ev-ecdet"></a>E4 — Object detection (`ecdet`)

- **ORT + TRT PASS.** C++ e2e `data/dog.jpg` @ 0.5 — bicycle 0.93, dog 0.92,
  car 0.74 — [KserveCompatibility.md](KserveCompatibility.md#known-bad-conversion-edgecrafter-ecdet--tflite).
  Live multi-backend path is `kserve-runtime-edgecrafter-e2e`
  ([§ Integration-test evidence](#integration-test-evidence),
  [BENCHMARK.md](../integration-tests/kserve-runtime-edgecrafter-e2e/BENCHMARK.md)).
- **LiteRT / KServe KNOWN-BAD.** Same input over `litert` gives top ≈0.19,
  random labels — conversion fault (`onnx2tf`), not runtime. Do not serve
  `ecdet` as TFLite.

### <a id="ev-seg"></a>E5 — Instance segmentation (`yoloseg`, `ecseg`)

- **DNN PASS** (5.0.0, CPU, 494 ms): `yolov8n-seg.onnx` — dog 0.87,
  bicycle 0.80, car 0.45, masks verified. The 4.6 `Unsqueeze` blocker is
  gone on 5.x.
- **ORT PASS** (CPU, 464 ms): @ 0.4/0.5 — dog 0.87, bicycle 0.80, car 0.45; masks verified; rendered `data/output/processed_yoloseg_local.png`.
- **TRT PASS** (10.14.1.48 FP16 engine from `yolov8n-seg.onnx`, 430 ms GPU): masks + boxes rendered, identical output shapes (1x116x8400 + 1x32x160x160).
- **KServe PASS** (`real-onnx` runtime, HTTP, model `seg` v1): exit 0, `neuriplo_http_infer_requests_success_total{model="seg",version="1"} 1`, failure 0; render `data/output/processed_yoloseg_kserve-ort.png` (>10 KB).
- `ecseg`: preset + emulation green; no live numbers recorded (unlike `ecdet`).

### <a id="ev-classification"></a>E6 — Classification (`torchvisionclassifier`)

- **Torch PASS** (CPU, 689 ms): traced `resnet50` (`torchvision`
  IMAGENET1K_V2, `--input_sizes=1,3,224,224`) — malamute 0.44.
- **TF KNOWN-BAD** (CPU, 2600 ms): ResNet-50 SavedModel (neuriplo-tasks
  export guide) loads and runs but labels `tiger shark`. The app's generic
  RGB `[0,1]` preprocessing lacks ResNet-50's caffe-mode preprocessing
  (Python cross-check: app-style input → class 111 @ 0.12, proper
  preprocessing → class 249 @ 0.55). Backend is bit-correct; do not serve
  TF classification until neuriplo-tasks preprocessing covers it.
  `tensorflow-classifier` / `vit-classifier` aliases have no preset —
  same cells, no separate row.

### <a id="ev-video"></a>E7 — Video classification (`videomae`), optical flow (`raft`)

- Preset + emulation green (ORT DRY-RUN). `vivit` / `timesformer` aliases
  ride the same task path; no separate preset. No live runs yet.

### <a id="ev-pose"></a>E8 — Pose estimation (`vitpose`)

Representative `vitpose` unrun everywhere below; `rfdetr_keypoint` gated
as alias (see Open item 2 for the emulation gap).

- **DNN PASS via alias** (5.0.0, CPU, 3582 ms): `rfdetr_keypoint` renders
  skeleton, pose score 0.85. (Apt 4.6 failed this model on `BOOL` dtype —
  gone on 5.x.) `vitpose` itself unrun.
- **ORT DRY-RUN + alias live** (CPU, 2817 ms): `rfdetr_keypoint` on
  `person.jpg` renders skeleton, pose score 0.85.
- **TRT alias live** (10.14.1.48 FP16 engine, 246 ms): same skeleton,
  score 0.85. Cell stays BUILDS for the unrun representative.
- **KServe PASS** (`real-onnx` runtime, HTTP): same score 0.85 — task gated
  via alias.

### <a id="ev-depth"></a>E9 — Depth estimation (`depth_anything_v2`)

Representative `depth_anything_v2` unrun everywhere below; `yolo26depth`
gated as alias.

- **DNN PASS via alias** (5.0.0, CPU, 838 ms): `yolo26depth` renders sane
  depth map, min/max 1.26/5.41. (Apt 4.6 failed it on the YOLO `Split`
  error — gone on 5.x.) `depth_anything_v2` itself unrun.
- **ORT DRY-RUN + alias live** (CPU, 618 ms): sane depth map, min/max
  1.26/5.41.
- **KServe PASS** (`real-onnx` runtime, HTTP): same sane map — task gated
  via alias.

### <a id="ev-owlv2"></a>E10 — Open-vocabulary detection (`owlv2`)

- **DNN BLOCKED** (5.0.0 importer loads both models, but multi-input
  inference fails): `owlv2.onnx` and `groundingdino.onnx` parse cleanly —
  the 4.6 `Mul`/`Identity` errors are gone — then fail in
  `setMainGraphInput` (`net_impl2.cpp:1032`, needs single input). Apt 4.6
  never got past the importer.
- **ORT PASS** (CPU, 14.3 s; CLIP vocab/merges from
  `openai/clip-vit-base-patch32`): prompts `cat;dog;bus` — dog 0.28; alias
  `groundingdino` also live (CPU, 28 s) — dog 0.68, van-as-bus 0.31.
  `owlvit` / `groundingdino` aliases same cells.
- **KServe PASS** (`real-onnx` runtime, HTTP, dog 0.28). Needs
  `--kserve_timeout_ms=240000` — the 30 s default aborts slow CPU runs.

### <a id="ev-gemma4"></a>E11 — Image understanding (`gemma4`)

- **LLaMA PASS** (CPU, 64 s): `gemma-4-E2B-it-Q4_K_M.gguf` + `mmproj-F16.gguf`
  (HF `unsloth/gemma-4-E2B-it-GGUF`), preset prompt — accurate description.
  Only LLaMA applies (GGUF + mmproj). No CI build job.

### <a id="ev-splatting"></a>E12 — Gaussian splatting (`lgm`/`grm`)

- Routed (`Capabilities.cpp`), no e2e preset, no emulation. Torch/OVINO
  cells are BUILDS-only. See Open item 6.

## Environments

How each column was built and tested on 2026-10-23 (all CPU except TRT).

| Backend | Version / artifact | Build flags | Tests | Notes |
|---|---|---|---|---|
| DNN | OpenCV 5.0.0 source-built (`~/dependencies/opencv-5.0.0`, `scripts/setup_opencv.sh`) | `OPENCV_DNN` | 179/179 | New graph engine; `--input_sizes` required; apt 4.6 kept as fallback when the 5.x dir is absent |
| ORT | 1.19.2 CPU | `ONNX_RUNTIME` | 179/179 | Full inference runs, renders inspected |
| TRT | 10.14.1.48 FP16 | `TENSORRT` | 194/194 | Engines rebuilt from the same `.onnx`; stale `build-trt` dir (pinned videocapture v0.5.0) reconfigured to v0.6.0 |
| Torch | libtorch 2.14 CPU | `LIBTORCH`, `NEURIPLO_INFER_ENABLE_KSERVE=OFF` | 134/134 | KServe client unlinked (its gRPC protobuf clashes with libtorch's bundled protobuf); first `cmake` seeds the CUDA cache, a second run is needed (nvcc 12.0 vs driver-reported 13.4) |
| OVINO | 2025.2.0 | `OPENVINO`, tests OFF | — (not covered) | Native `runUnitTests` link vs `openvino::runtime` fails; CI builds tests OFF too (`Dockerfile.openvino`) |
| TF | 2.19 | `LIBTENSORFLOW` | 179/179 | `--weights` accepts directories since 2026-10-23 (was file-only); see E6 |
| LiteRT | `~/dependencies/litert` | `LITERT` | 158/179 | 21 death-tests crash at subprocess exit (SEGV / `free(): invalid pointer`); live CLI exits 0 — see Open item 8 |
| ExecT | `~/dependencies/executorch` | `EXECUTORCH`, `NEURIPLO_INFER_ENABLE_KSERVE=OFF` | 134/134 | Same protobuf clash as Torch |
| LLaMA | llama.cpp b9085 | `LLAMACPP`, tests OFF | — (not covered) | Native `runUnitTests` link misses ggml transitive libs (same class as the OVINO gap) |
| KServe | `neuriplo-kserve-runtime` `real-onnx` / `grpc-onnx` | client in default build | suites run manually | HTTP green incl. per-task gates; gRPC `ModelInfer` 13 INTERNAL server-side (see E1) |

## Scoreboard

| Status | Count (task rows) | Rows |
|---|---|---|
| PASS (≥1 backend live-gated) | 9 | `yolo26`, `yolov8`, `torchvisionclassifier`, `ecdet`, `owlv2`, `gemma4`, `yoloseg`, pose (`rfdetr_keypoint` alias), depth (`yolo26depth` alias) |
| KNOWN-BAD recorded | 2 | `ecdet` on LiteRT / KServe-`litert`; `torchvisionclassifier` on LIBTENSORFLOW |
| DRY-RUN best | 4 | `rtdetr`, `ecseg`, `videomae`, `raft` |
| BUILDS / UNTESTED best | 1 | Gaussian splatting (no preset; Torch/OVINO BUILDS-only) |
| **Total** | **14** | 10 task families; detection/segmentation/pose split by representative |

## Open items

1. **Live-gate every DRY-RUN cell.** Closest first: `rtdetrv4` on TRT;
   `raft` / `vitpose` / `depth_anything_v2` / `videomae` on ORT.
   Needs model exports that don't exist locally yet.
2. **`rfdetr_keypoint` missing from emulation.** The preset exists in
   `docker_run_inference_e2e_example.sh --list-presets` but has no job in
   `e2e-example-emulation.yml`. Add it (one-line matrix entry).
3. **Record Torch / OpenVINO `--backend` override runs as repeatable gates.**
   Both gated live (`yolo26`, plus `torchvisionclassifier` on Torch) but only
   via manual CLI invocations, not presets.
4. **OPENCV_DNN scope (now 5.0.0).** The YOLO `Split`/`Unsqueeze` blockers,
   the `rfdetr` `BOOL` dtype failure, and the `owlv2`/`groundingdino`
   importer errors are all gone on the 5.x importer. Remaining 5.x gap:
   multi-input inference (`owlv2`, `groundingdino` fail in
   `setMainGraphInput`). Still unprobed on 5.x: `rtdetr`, `ecdet`, `ecseg`,
   torchvision ResNet, `raft`, `vitpose`, `depth_anything_v2`.
5. **KServe per-task gates.** Done for servable local models (`real-onnx`,
   HTTP): pose, depth, open-vocab — plus YOLO26-`litert` PASS and `ecdet`
   KNOWN-BAD. gRPC transport BLOCKED server-side. Still open: classification,
   video, flow, VLM serving.
6. **Gaussian-splatting representative.** No preset exists. Export one `lgm`
   artifact, add a preset, then gate it.
7. **LIBTENSORFLOW classification needs ResNet preprocessing.** 2026-10-23:
   the CLI seam is fixed (`--weights` accepts directories); the ResNet-50
   SavedModel loads and runs, but the app's generic RGB `[0,1]` preprocessing
   does not satisfy ResNet-50's caffe-mode requirement (KNOWN-BAD). Fix belongs
   in neuriplo-tasks classification preprocessing, then re-gate.
8. **LiteRT-linked `runUnitTests`: 21 death-tests crash at exit.** 158/179
   pass; 21 death-style tests SEGV / `free(): invalid pointer` in the forked
   child at teardown. Live CLI exits 0. Bisect with `ctest -R` on a minimal
   link before trusting LiteRT CI.

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
| `kserve-runtime-edgecrafter-e2e` | `ecdet` ORT/TRT PASS + OpenVINO/ExecT matrix; `ecdet` LiteRT/KServe KNOWN-BAD (conversion fault, not runtime) | — (needs per-backend runtime binaries + model repository + GPU for TRT) | `run.py --backend {onnx_runtime,tensorrt,openvino,executorch} [--transport {http,grpc}] [--mode {kserve,local}]`; `prepare_model_repository.py` lays out `ecdet_s_<backend>/1/` | [README](../integration-tests/kserve-runtime-edgecrafter-e2e/README.md), [BENCHMARK.md](../integration-tests/kserve-runtime-edgecrafter-e2e/BENCHMARK.md) |
| `rfdetr-pose-local-vs-kserve` | Pose row scaffold for `rfdetr_keypoint` (covers the Open-items emulation gap with evidence tracking, not a live gate) | `rfdetr_pose_evidence_scaffold` (`run.py` — baselines + report exist) | Attested checks in [evidence.yaml](../integration-tests/rfdetr-pose-local-vs-kserve/evidence.yaml) (`kserve_grpc_smoke` still `skipped`) | [README](../integration-tests/rfdetr-pose-local-vs-kserve/README.md), [evidence.yaml](../integration-tests/rfdetr-pose-local-vs-kserve/evidence.yaml), [reports/latest.md](../integration-tests/rfdetr-pose-local-vs-kserve/reports/latest.md) |

KServe protocol-level legs (not app-surface suites, but they back the
KServe column): `kserve_integration_dry_run` and
`kserve_input_datatypes_e2e_dry_run` in CTest (`app/test/kserve_integration.sh`,
`app/test/kserve_input_datatypes_e2e.sh`; `--live` is opt-in) — full matrix
in [KserveCompatibility.md](KserveCompatibility.md).

## Non-kernel limits

- **Opset / export policy lives upstream.** This repo consumes what
  `neuriplo-tasks` exports; conversion fidelity (e.g. `onnx2tf` for LiteRT)
  is judged at this seam but fixed upstream.
- **Postprocessing stays here.** NMS, mask assembly, flow/depth resize, and
  tokenization are app-side and outside any backend gate — same as NERT's
  host-side rule.
- **One binary, one local backend.** Each image carries the backend it was
  built with (`DEFAULT_BACKEND`); switching means rebuilding. KServe mode is
  the only multi-backend workflow in one binary.
- **CI builds ≠ runs.** A green `ci.yml` build job puts BUILDS in the column;
  only a recorded live output promotes a cell to PASS.

## Sources

- Backends and artifacts: [Deployment.md § Backends](Deployment.md#backends),
  `docker/Dockerfile.*`, `docker_run_inference_e2e_example.sh --list-presets`.
- Emulation evidence: `.github/workflows/e2e-example-emulation.yml` matrix.
- Build evidence: `.github/workflows/ci.yml` jobs.
- KServe evidence: [KserveCompatibility.md](KserveCompatibility.md) (live
  `ecdet` numbers, YOLO26-`litert` note, Identity round-trips).
- Task inventory: [generated/supported-model-types.md](generated/supported-model-types.md),
  `app/src/Capabilities.cpp` (`taskCapabilities`), `app/src/NeuriploInferTaskRouting.cpp`.
- App-surface integration tests: [§ Integration-test evidence](#integration-test-evidence)
  above owns them (moved from `neuriplo-platform`; live paths stay manual
  per each suite's README).
- Style reference:
  [NERT coverage matrix](https://github.com/olibartfast/nert/blob/main/specs/2026-10-17-task-models/coverage.md).

# Backend coverage — task models × backends

Date: 2026-10-23. Branch `feature/backend-coverage` merged `origin/develop`
at `6951525` (no-op — already at tip) and re-validated: `ctest` 177/177,
sync `--check` clean, all 15 script presets `--dry-run` green. DNN column
live-probed 2026-10-23 on apt OpenCV 4.6.0 (`OPENCV_DNN` build, `ctest`
179/179): 5 loader probes via `--export_metadata` (full pipeline build incl.
model load, no inference) — see Evidence cells and Open item 4. ORT column
live-gated 2026-10-23 on ORT 1.19.2 CPU (`ONNX_RUNTIME` build, `ctest`
179/179): full inference runs on `data/dog.jpg` (`person.jpg` for pose) with
rendered outputs inspected — see Evidence cells. TRT column live-gated
2026-10-23 on TRT 10.14.1.48 FP16 (`TENSORRT` build, `ctest` 194/194):
engines built with 10.14 `trtexec` from the same `.onnx` files; pre-existing
`rfdetr-keypoint-preview.engine` (built with 10.13) does NOT deserialize on
10.14 and was rebuilt. Torch column live-gated 2026-10-23 on libtorch 2.14
CPU (`LIBTORCH` build with `NEURIPLO_INFER_ENABLE_KSERVE=OFF` — the KServe
gRPC protobuf clashes with libtorch's bundled protobuf at link;
`ctest` 134/134): TorchScript models need `--input_sizes` WITH batch dim
(`1,3,H,W` — the backend uses provided sizes verbatim for dynamic graph
inputs). OpenVINO column: native test build is NOT covered — CI builds
OpenVINO with tests OFF (`Dockerfile.openvino`) and the native
`runUnitTests` link against `openvino::runtime` fails; live gate 2026-10-23
(`OPENVINO` build, tests OFF) via `yolo26s.onnx` read directly. TF column
probed 2026-10-23 (`LIBTENSORFLOW` build, `ctest` 179/179): builds and links.
CLI seam fixed same day (`--weights` now accepts directories, mirroring
`--source` — was file-only, which no SavedModel can satisfy). Live run
2026-10-23 (CPU, 2600 ms): ResNet-50 SavedModel via the neuriplo-tasks export
guide on `data/dog.jpg` runs end-to-end but labels `tiger shark` — the app's
generic RGB `[0,1]` preprocessing lacks ResNet-50's caffe-mode preprocessing
(Python check: app-style input → class 111 @ 0.12; proper preprocessing →
class 249 @ 0.55). Backend is bit-correct; fix belongs in neuriplo-tasks
preprocessing. TF classification cell is KNOWN-BAD: do not serve. LiteRT column live-gated 2026-10-23 (`LITERT` build):
`yolo26s.tflite` runs live; unit suite 158/179 — 21 death-tests crash at
subprocess exit in the LiteRT-linked binary (live CLI exits 0; see Open
items). ExecuTorch column live-gated 2026-10-23 (`EXECUTORCH` build with
`NEURIPLO_INFER_ENABLE_KSERVE=OFF`, `ctest` 134/134). LLaMA column live-gated
2026-10-23 (`LLAMACPP` build, tests OFF — native `runUnitTests` link misses
ggml transitive libs, same class as the OpenVINO test gap):
`gemma-4-E2B-it-Q4_K_M.gguf` + `mmproj-F16.gguf` (HF
`unsloth/gemma-4-E2B-it-GGUF`) on `data/dog.jpg` — coherent description
(dog, red bicycle, white van, yellow railing), 64 s CPU. Style follows the NERT
[coverage matrix](https://github.com/olibartfast/nert/blob/main/specs/2026-10-17-task-models/coverage.md):
one gated representative per task family, one status per backend, no cell
claiming more than its evidence.

## Definitions

- **PASS** — a live end-to-end run is recorded with its expected output
  (detections, labels, or round-trip log) cited in Evidence.
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

Node counts, sha256 digests, and per-tensor tolerances belong to the
exporting repo (`neuriplo-tasks`) and the engine repo (`neuriplo`), not here.
This matrix gates the app seam: routed, run, rendered, tested.

## Backend capability table

| Backend | Flag / image | Model file | Notes |
|---------|--------------|------------|-------|
| OPENCV_DNN | `OPENCV_DNN` (default, no image) | `.onnx` static shapes; Darknet `.weights` + `.cfg` | Needs nothing beyond OpenCV. |
| ONNX_RUNTIME | `ONNX_RUNTIME` / `neuriplo-infer:onnxruntime` | `.onnx` | General ONNX backend; most presets default here. |
| TENSORRT | `TENSORRT` / `neuriplo-infer:tensorrt` | serialized `.engine` (built from ONNX via `trtexec`) | NVIDIA GPU only; engine loads only on the TensorRT version that built it. |
| LIBTORCH | `LIBTORCH` / `neuriplo-infer:libtorch` | TorchScript `.pt` | CPU unless built with `COMPUTE_PLATFORM`. |
| OPENVINO | `OPENVINO` / `neuriplo-infer:openvino` | IR `.xml` + `.bin`, or `.onnx` | Intel path; no preset defaults here. |
| LIBTENSORFLOW | `LIBTENSORFLOW` / `neuriplo-infer:libtensorflow` | SavedModel directory | Existing TensorFlow models. CLI `--weights` accepts directories since 2026-10-23 (was file-only); classification live run labels wrong — ResNet preprocessing gap, see KNOWN-BAD cell. |
| LITERT | `LITERT` / `neuriplo-infer:litert` | `.tflite` | Ultralytics TFLite export, or `onnx2tf` lowering from ONNX. |
| EXECUTORCH | `EXECUTORCH` (no CI image job) | `.pte` | Ultralytics `export(format=executorch)`. |
| LLAMACPP | `LLAMACPP` / `neuriplo-infer:llamacpp` | `.gguf` model + `.gguf` projector | VLM only; text-only without `--mmproj`. |
| KSERVE | `client_server` workflow, no local build | server-side artifact (ONNX, IR, TorchScript, TFLite, …) | `KserveEngine` speaks KServe V2 over HTTP/gRPC; the server's backend decides the format. See [KserveCompatibility.md](KserveCompatibility.md). |

Source: [Deployment.md § Backends](Deployment.md#backends). Backend SDK
versions are owned by `neuriplo` and never copied here.

## Gate table

Columns: `DNN` = OPENCV_DNN · `ORT` = ONNX_RUNTIME · `TRT` = TENSORRT ·
`Torch` = LIBTORCH · `OVINO` = OPENVINO · `TF` = LIBTENSORFLOW ·
`LiteRT` = LITERT · `ExecT` = EXECUTORCH · `LLaMA` = LLAMACPP ·
`KServe` = remote `KserveEngine`. Representative preset in parentheses.

| Task (representative) | DNN | ORT | TRT | Torch | OVINO | TF | LiteRT | ExecT | LLaMA | KServe | Evidence |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Object detection (`yolo26`, preset `yolo26s_tflite`) | BLOCKED | PASS | PASS | PASS | PASS | BUILDS | PASS | UNTESTED | UNTESTED | PASS | DNN BLOCKED: apt OpenCV 4.6 rejects Ultralytics exports (`Split`, `Unsqueeze`) — [Deployment.md](Deployment.md#backends) + roadmap candidate; re-confirmed live 2026-10-23 (`yolo26s.onnx` → `Split` parse error, `slice_layer.cpp:235`). ORT PASS live 2026-10-23 (CPU, 1137 ms): `data/dog.jpg` @ 0.4/0.5 — bicycle 0.95, dog 0.94, car 0.52, rendered `data/output/processed_yolo26_local.png`. TRT PASS live 2026-10-23 (10.14.1.48 FP16 engine from `yolo26s.onnx`, 251 ms): identical outputs — bicycle 0.95, dog 0.94, car 0.52. Torch PASS live 2026-10-23 (CPU, 1083 ms): `yolo26s.torchscript` (ultralytics export) with `--input_sizes=1,3,640,640` — identical outputs. OpenVINO PASS live 2026-10-23 (CPU, 267 ms): `yolo26s.onnx` read directly — identical outputs. LiteRT PASS live 2026-10-23 (CPU, 888 ms): `yolo26s.tflite` (`1x640x640x3`) — identical outputs. KServe PASS: YOLO26 `.tflite` serves correctly over `litert`, validated locally 2026-06-13 — [KserveCompatibility.md](KserveCompatibility.md#model-formats); live path is [§ Integration-test evidence](#integration-test-evidence) `kserve-runtime-yolo-e2e`. |
| Object detection (`yolov8`, preset `yolov8_executorch`) | BLOCKED | DRY-RUN | BUILDS | BUILDS | BUILDS | BUILDS | UNTESTED | PASS | UNTESTED | UNTESTED | Same DNN blocker (YOLOv8n `Unsqueeze` at opset 12). ExecT PASS live 2026-10-23 (CPU, 320 ms): `yolov8n.pte` on `data/dog.jpg` @ 0.4/0.5 — bicycle 0.89, dog 0.93, car 0.61 (`data/output/processed_yolov8_local.png`). No CI build job for ExecT. |
| Object detection (`rtdetr`, preset `rtdetrv4`) | UNTESTED | DRY-RUN | DRY-RUN | BUILDS | BUILDS | BUILDS | UNTESTED | UNTESTED | UNTESTED | UNTESTED | TRT default preset; `ci.yml` builds the TRT image; emulation dry-run green. No live log cited — not PASS. |
| Object detection (`ecdet`, preset `edgecrafter_det`) | UNTESTED | PASS | PASS | BUILDS | BUILDS | BUILDS | KNOWN-BAD | UNTESTED | UNTESTED | KNOWN-BAD | PASS: C++ e2e `data/dog.jpg` @ 0.5 — bicycle 0.93, dog 0.92, car 0.74 on ORT + TRT — [KserveCompatibility.md](KserveCompatibility.md#known-bad-conversion-edgecrafter-ecdet--tflite). Live multi-backend path is [§ Integration-test evidence](#integration-test-evidence) `kserve-runtime-edgecrafter-e2e` ([BENCHMARK.md](../integration-tests/kserve-runtime-edgecrafter-e2e/BENCHMARK.md)). LiteRT/KServe KNOWN-BAD: same input over `litert` gives top ≈0.19, random labels — conversion fault (`onnx2tf`), not runtime. Do not serve `ecdet` as TFLite. |
| Instance segmentation (`yoloseg`, preset `yoloseg`) | BLOCKED | DRY-RUN | BUILDS | BUILDS | BUILDS | BUILDS | UNTESTED | UNTESTED | UNTESTED | UNTESTED | YOLO-seg shares the Ultralytics/DNN blocker. ORT DRY-RUN via preset. |
| Instance segmentation (`ecseg`, preset `edgecrafter_seg`) | UNTESTED | DRY-RUN | BUILDS | BUILDS | BUILDS | BUILDS | UNTESTED | UNTESTED | UNTESTED | UNTESTED | Preset + emulation green; no live numbers recorded (unlike `ecdet`). |
| Classification (`torchvisionclassifier`, preset `torchvision_classifier`) | UNTESTED | DRY-RUN | BUILDS | PASS | BUILDS | KNOWN-BAD | UNTESTED | UNTESTED | UNTESTED | UNTESTED | Preset + emulation green. Torch PASS live 2026-10-23 (CPU, 689 ms): traced `resnet50` (`torchvision` IMAGENET1K_V2, `--input_sizes=1,3,224,224`) on `data/dog.jpg` — malamute 0.44 (`data/output/processed_torchvisionclassifier_local.png`). TF KNOWN-BAD live 2026-10-23 (CPU, 2600 ms): ResNet-50 SavedModel (neuriplo-tasks export guide) runs but labels `tiger shark` — app RGB `[0,1]` preprocessing vs ResNet-50 caffe-mode requirement (Python: app-style → class 111 @ 0.12, proper → class 249 @ 0.55). Backend bit-correct; do not serve TF classification until neuriplo-tasks preprocessing covers it. `tensorflow-classifier` / `vit-classifier` aliases have no preset — same cells, no separate row. |
| Video classification (`videomae`, preset `videomae`) | UNTESTED | DRY-RUN | BUILDS | BUILDS | BUILDS | BUILDS | UNTESTED | UNTESTED | UNTESTED | UNTESTED | Preset + emulation green. `vivit` / `timesformer` aliases ride the same task path; no separate preset. |
| Optical flow (`raft`, preset `raft`) | UNTESTED | DRY-RUN | BUILDS | BUILDS | BUILDS | BUILDS | UNTESTED | UNTESTED | UNTESTED | UNTESTED | Preset + emulation green. |
| Pose estimation (`vitpose`, preset `vitpose`) | UNTESTED | DRY-RUN | BUILDS | BUILDS | BUILDS | BUILDS | UNTESTED | UNTESTED | UNTESTED | UNTESTED | Preset + emulation green. `yolo*pose` shares the DNN/Ultralytics blocker on DNN only; `ecpose`/`rfdetr_keypoint` same ORT DRY-RUN level (see Open items for the `rfdetr_keypoint` emulation gap; scaffold evidence is [§ Integration-test evidence](#integration-test-evidence) `rfdetr-pose-local-vs-kserve`). DNN loader probe 2026-10-23: `rfdetr_keypoint` (`rfdetr-keypoint-preview.onnx`) BLOCKED — `Unsupported data type: BOOL` in `onnx_graph_simplifier.cpp:842`; `vitpose` artifact itself not probed, cell stays UNTESTED. ORT live 2026-10-23 (CPU, 2817 ms): `rfdetr_keypoint` on `data/person.jpg` renders skeleton, pose score 0.85 (`data/output/processed_rfdetr-keypoint_local.png`); `vitpose` artifact itself unrun, cell stays DRY-RUN. TRT live 2026-10-23 (10.14.1.48 FP16 engine, 246 ms): `rfdetr_keypoint` renders skeleton, pose score 0.85; `vitpose` artifact itself unrun, cell stays BUILDS. |
| Depth estimation (`depth_anything_v2`, preset `depth_anything_v2`) | UNTESTED | DRY-RUN | BUILDS | BUILDS | BUILDS | BUILDS | UNTESTED | UNTESTED | UNTESTED | UNTESTED | Preset + emulation green. `yolo*depth` alias same cells. DNN loader probe 2026-10-23: `yolo26depth` (`yolo26n-depth.onnx`) BLOCKED — same Ultralytics `Split` parse error as detection; `depth_anything_v2` artifact itself not probed, cell stays UNTESTED. ORT live 2026-10-23 (CPU, 618 ms): `yolo26depth` on `data/dog.jpg` renders sane depth map, min/max 1.26/5.41 (`data/output/processed_yolo26depth_local.png`); `depth_anything_v2` artifact itself unrun, cell stays DRY-RUN. |
| Open-vocabulary detection (`owlv2`, preset `owlv2`) | BLOCKED | PASS | BUILDS | BUILDS | BUILDS | BUILDS | UNTESTED | UNTESTED | UNTESTED | UNTESTED | DNN BLOCKED (loader probe 2026-10-23, apt 4.6.0): `owlv2.onnx` → `Mul` broadcast-with-constant parse error (`onnx_importer.cpp:2221`); alias `groundingdino.onnx` also fails (`Identity_12597`, `layer.cpp:246`). ORT PASS live 2026-10-23 (CPU, 14.3 s, CLIP vocab/merges from `openai/clip-vit-base-patch32`): `data/dog.jpg` prompts `cat;dog;bus` — dog 0.28 (`data/output/processed_owlv2_local.png`); alias `groundingdino` also live (CPU, 28 s) — dog 0.68, van-as-bus 0.31. `owlvit` / `groundingdino` aliases same cells. |
| Image understanding (`gemma4`, preset `gemma4`) | UNTESTED | UNTESTED | UNTESTED | UNTESTED | UNTESTED | UNTESTED | UNTESTED | UNTESTED | PASS | UNTESTED | Only LLaMA applies (GGUF + mmproj). LLaMA PASS live 2026-10-23 (CPU, 64 s): preset prompt on `data/dog.jpg` returns accurate description. Preset + emulation green, no CI build job. |
| Gaussian splatting (`lgm`/`grm`, no preset) | UNTESTED | UNTESTED | UNTESTED | BUILDS | BUILDS | UNTESTED | UNTESTED | UNTESTED | UNTESTED | UNTESTED | Routed (`Capabilities.cpp`), no e2e preset, no emulation. Torch/OVINO cells are BUILDS-only. |

`BUILDS` means the `ci.yml` job for that backend is green (`OPENCV_DNN`,
`ONNX_RUNTIME`, `LIBTORCH`, `LIBTENSORFLOW`, `TENSORRT`, `OPENVINO`); there is
no `LITERT` / `EXECUTORCH` / `LLAMACPP` build job. `DRY-RUN` means the preset
is listed by `--list-presets` and green in `e2e-example-emulation.yml`
(dry-run validates command construction, not numerics).

## Scoreboard

| Status | Count (task rows) | Rows |
|---|---|---|
| PASS (≥1 backend live-gated) | 6 | `yolo26` detection (ORT + TRT + Torch + OVINO + LiteRT; KServe `litert`), `yolov8` detection (ExecT), `torchvisionclassifier` classification (Torch), `ecdet` detection (ORT + TRT), `owlv2` open-vocab (ORT, + `groundingdino` alias live), `gemma4` image-understanding (LLaMA) |
| KNOWN-BAD recorded | 2 | `ecdet` on LiteRT / KServe-`litert`; `torchvisionclassifier` on LIBTENSORFLOW (preprocessing, not backend) |
| DRY-RUN best | 7 | All preset rows except splatting and the 6 PASS rows |
| BUILDS / UNTESTED best | 1 | Gaussian splatting (no e2e preset, Torch/OVINO BUILDS-only) |
| **Total** | **14** | 10 task families; detection/segmentation/pose split by representative |

## Open items

1. **Live-gate every DRY-RUN cell.** Run each preset for real (Docker, GPU
   where needed) and promote cells to PASS/FAIL with the observed output
   pasted or linked. Closest first: `rtdetrv4` on TRT, `yoloseg` / `raft` /
   `vitpose` / `depth_anything_v2` / `videomae` on ORT. Done 2026-10-23:
   ORT CPU (`yolo26`, `owlv2` + `groundingdino` alias; alias-level
   `yolo26depth`, `rfdetr_keypoint`); TRT 10.14 FP16 (`yolo26`,
   `rfdetr_keypoint` alias); Torch CPU (`yolo26`, `torchvisionclassifier`);
   OVINO CPU (`yolo26`); LiteRT CPU (`yolo26`); ExecT CPU (`yolov8`);
   LLaMA CPU (`gemma4`).
2. **`rfdetr_keypoint` missing from emulation.** The preset exists in
   `docker_run_inference_e2e_example.sh --list-presets` but has no job in
   `e2e-example-emulation.yml`. Add it (one-line matrix entry) so the pose
   row's second representative is covered.
3. **Fill the LIBTORCH / OPENVINO / LIBTENSORFLOW columns.** Partially done
   2026-10-23: Torch live (`yolo26`, `torchvisionclassifier`) and OpenVINO
   live (`yolo26`) — both need preset/`--backend` override runs recorded as
   repeatable gates. LIBTENSORFLOW classification ran live but is KNOWN-BAD
   (ResNet preprocessing gap, see item 7).
4. **OPENCV_DNN scope.** The Ultralytics `Split`/`Unsqueeze` blocker is
   recorded for YOLO families. Partially probed 2026-10-23 (apt 4.6.0,
   `--export_metadata` loader probes): `yolo26depth` shares the YOLO
   `Split` blocker; `rfdetr_keypoint` fails differently (`BOOL` dtype);
   `owlv2`/`groundingdino` fail differently again (`Mul` broadcast,
   `Identity`). Still unprobed on DNN: `rtdetr`, `ecdet`, `yoloseg`,
   `ecseg`, torchvision ResNet, `raft`, `vitpose`, `depth_anything_v2`.
5. **KServe per-task gates.** Today's KServe evidence is protocol-level
   (Identity round-trip, dry-run) plus YOLO26-`litert` and the `ecdet`
   KNOWN-BAD. Gate one model per task through `neuriplo-kserve-runtime`
   and record backend + transport per cell (feeds roadmap Phase 5).
6. **Gaussian-splatting representative.** No preset exists. Export one `lgm`
   artifact, add a preset, then gate it.
7. **LIBTENSORFLOW classification needs ResNet preprocessing.** 2026-10-23:
   the CLI seam is fixed (`--weights` accepts directories); the ResNet-50
   SavedModel (neuriplo-tasks export guide) loads and runs, but the app's
   generic RGB `[0,1]` preprocessing does not satisfy ResNet-50's caffe-mode
   requirement, so labels are wrong (KNOWN-BAD). Fix belongs in
   neuriplo-tasks classification preprocessing (model-specific), then re-gate.
8. **LiteRT-linked `runUnitTests`: 21 death-tests crash at exit.**
   2026-10-23 `LITERT` build: 158/179 pass; 21 death-style tests
   (`RunReportFile`, `ParseCommandLineArguments` exits) SEGV / `free():
   invalid pointer` in the forked child at teardown. Live CLI runs exit 0,
   so this is test-harness-only until proven otherwise — but a heap bug in
   the LiteRT-linked binary is not ruled out. Bisect with
   `ctest -R` on a minimal link before trusting LiteRT CI.

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
| `kserve-runtime-edgecrafter-e2e` | `ecdet` ORT/TRT PASS + OpenVINO/ExecT matrix; `ecdet` LiteRT/KServe KNOWN-BAD (conversion fault, not runtime) | — (needs per-backend runtime binaries + model repository + GPU for TRT) | `run.py --backend {onnx_runtime,tensorrt,openvino,executorch} [--transport {http,grpc}] [--mode {kserve,local}]`; `run_openvino_matrix.py`, `run_executorch_matrix.py`; `prepare_model_repository.py` lays out `ecdet_s_<backend>/1/` | [README](../integration-tests/kserve-runtime-edgecrafter-e2e/README.md), [BENCHMARK.md](../integration-tests/kserve-runtime-edgecrafter-e2e/BENCHMARK.md) (local TRT ~90 ms vs HTTP/JSON ~1375 ms vs gRPC ~84–94 ms vs HTTP+binary ~142 ms), [run.py](../integration-tests/kserve-runtime-edgecrafter-e2e/run.py) |
| `rfdetr-pose-local-vs-kserve` | Pose row scaffold for `rfdetr_keypoint` (covers the Open-items emulation gap with evidence tracking, not a live gate) | `rfdetr_pose_evidence_scaffold` (`run.py` — baselines + report exist) | Attested checks in [evidence.yaml](../integration-tests/rfdetr-pose-local-vs-kserve/evidence.yaml) (GPU/C++ tooling; `kserve_grpc_smoke` still `skipped`) | [README](../integration-tests/rfdetr-pose-local-vs-kserve/README.md), [evidence.yaml](../integration-tests/rfdetr-pose-local-vs-kserve/evidence.yaml), [reports/latest.md](../integration-tests/rfdetr-pose-local-vs-kserve/reports/latest.md), baselines [`local.json`](../integration-tests/rfdetr-pose-local-vs-kserve/baselines/local.json) / [`kserve-grpc.json`](../integration-tests/rfdetr-pose-local-vs-kserve/baselines/kserve-grpc.json) |

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
- App-surface integration tests: [§ Integration-test evidence](#integration-test-evidence) above owns them — the runner
  dry-run, YOLO/EdgeCrafter KServe E2E, and RF-DETR pose scaffold under
  `../integration-tests/` (moved from `neuriplo-platform`; live paths stay
  manual per each suite's README).
- Style reference:
  [NERT coverage matrix](https://github.com/olibartfast/nert/blob/main/specs/2026-10-17-task-models/coverage.md).

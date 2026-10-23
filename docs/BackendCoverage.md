# Backend coverage — task models × backends

Date: 2026-10-23. Branch `feature/backend-coverage` merged `origin/develop`
at `6951525` (no-op — already at tip) and re-validated: `ctest` 177/177,
sync `--check` clean, all 15 script presets `--dry-run` green. Style follows the NERT
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
| LIBTENSORFLOW | `LIBTENSORFLOW` / `neuriplo-infer:libtensorflow` | SavedModel directory | Existing TensorFlow models. |
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
| Object detection (`yolo26`, preset `yolo26s_tflite`) | BLOCKED | DRY-RUN | BUILDS | BUILDS | BUILDS | BUILDS | DRY-RUN | UNTESTED | UNTESTED | PASS | DNN BLOCKED: apt OpenCV 4.6 rejects Ultralytics exports (`Split`, `Unsqueeze`) — [Deployment.md](Deployment.md#backends) + roadmap candidate. LiteRT local DRY-RUN (preset + emulation green). KServe PASS: YOLO26 `.tflite` serves correctly over `litert`, validated locally 2026-06-13 — [KserveCompatibility.md](KserveCompatibility.md#model-formats). |
| Object detection (`yolov8`, preset `yolov8_executorch`) | BLOCKED | DRY-RUN | BUILDS | BUILDS | BUILDS | BUILDS | UNTESTED | DRY-RUN | UNTESTED | UNTESTED | Same DNN blocker (YOLOv8n `Unsqueeze` at opset 12). ExecT DRY-RUN (preset + emulation green, no CI build job). |
| Object detection (`rtdetr`, preset `rtdetrv4`) | UNTESTED | DRY-RUN | DRY-RUN | BUILDS | BUILDS | BUILDS | UNTESTED | UNTESTED | UNTESTED | UNTESTED | TRT default preset; `ci.yml` builds the TRT image; emulation dry-run green. No live log cited — not PASS. |
| Object detection (`ecdet`, preset `edgecrafter_det`) | UNTESTED | PASS | PASS | BUILDS | BUILDS | BUILDS | KNOWN-BAD | UNTESTED | UNTESTED | KNOWN-BAD | PASS: C++ e2e `data/dog.jpg` @ 0.5 — bicycle 0.93, dog 0.92, car 0.74 on ORT + TRT — [KserveCompatibility.md](KserveCompatibility.md#known-bad-conversion-edgecrafter-ecdet--tflite). LiteRT/KServe KNOWN-BAD: same input over `litert` gives top ≈0.19, random labels — conversion fault (`onnx2tf`), not runtime. Do not serve `ecdet` as TFLite. |
| Instance segmentation (`yoloseg`, preset `yoloseg`) | BLOCKED | DRY-RUN | BUILDS | BUILDS | BUILDS | BUILDS | UNTESTED | UNTESTED | UNTESTED | UNTESTED | YOLO-seg shares the Ultralytics/DNN blocker. ORT DRY-RUN via preset. |
| Instance segmentation (`ecseg`, preset `edgecrafter_seg`) | UNTESTED | DRY-RUN | BUILDS | BUILDS | BUILDS | BUILDS | UNTESTED | UNTESTED | UNTESTED | UNTESTED | Preset + emulation green; no live numbers recorded (unlike `ecdet`). |
| Classification (`torchvisionclassifier`, preset `torchvision_classifier`) | UNTESTED | DRY-RUN | BUILDS | BUILDS | BUILDS | BUILDS | UNTESTED | UNTESTED | UNTESTED | UNTESTED | Preset + emulation green. `tensorflow-classifier` / `vit-classifier` aliases have no preset — same cells, no separate row. |
| Video classification (`videomae`, preset `videomae`) | UNTESTED | DRY-RUN | BUILDS | BUILDS | BUILDS | BUILDS | UNTESTED | UNTESTED | UNTESTED | UNTESTED | Preset + emulation green. `vivit` / `timesformer` aliases ride the same task path; no separate preset. |
| Optical flow (`raft`, preset `raft`) | UNTESTED | DRY-RUN | BUILDS | BUILDS | BUILDS | BUILDS | UNTESTED | UNTESTED | UNTESTED | UNTESTED | Preset + emulation green. |
| Pose estimation (`vitpose`, preset `vitpose`) | UNTESTED | DRY-RUN | BUILDS | BUILDS | BUILDS | BUILDS | UNTESTED | UNTESTED | UNTESTED | UNTESTED | Preset + emulation green. `yolo*pose` shares the DNN/Ultralytics blocker on DNN only; `ecpose`/`rfdetr_keypoint` same ORT DRY-RUN level (see Open items for the `rfdetr_keypoint` emulation gap). |
| Depth estimation (`depth_anything_v2`, preset `depth_anything_v2`) | UNTESTED | DRY-RUN | BUILDS | BUILDS | BUILDS | BUILDS | UNTESTED | UNTESTED | UNTESTED | UNTESTED | Preset + emulation green. `yolo*depth` alias same cells. |
| Open-vocabulary detection (`owlv2`, preset `owlv2`) | UNTESTED | DRY-RUN | BUILDS | BUILDS | BUILDS | BUILDS | UNTESTED | UNTESTED | UNTESTED | UNTESTED | Preset + emulation green. `owlvit` / `groundingdino` aliases same cells. |
| Image understanding (`gemma4`, preset `gemma4`) | UNTESTED | UNTESTED | UNTESTED | UNTESTED | UNTESTED | UNTESTED | UNTESTED | UNTESTED | DRY-RUN | UNTESTED | Only LLaMA applies (GGUF + mmproj). Preset + emulation green, no CI build job. |
| Gaussian splatting (`lgm`/`grm`, no preset) | UNTESTED | UNTESTED | UNTESTED | BUILDS | BUILDS | UNTESTED | UNTESTED | UNTESTED | UNTESTED | UNTESTED | Routed (`Capabilities.cpp`), no e2e preset, no emulation. Torch/OVINO cells are BUILDS-only. |

`BUILDS` means the `ci.yml` job for that backend is green (`OPENCV_DNN`,
`ONNX_RUNTIME`, `LIBTORCH`, `LIBTENSORFLOW`, `TENSORRT`, `OPENVINO`); there is
no `LITERT` / `EXECUTORCH` / `LLAMACPP` build job. `DRY-RUN` means the preset
is listed by `--list-presets` and green in `e2e-example-emulation.yml`
(dry-run validates command construction, not numerics).

## Scoreboard

| Status | Count (task rows) | Rows |
|---|---|---|
| PASS (≥1 backend live-gated) | 1 | `ecdet` detection (ORT + TRT); KServe detection (YOLO26 `litert`) |
| KNOWN-BAD recorded | 1 | `ecdet` on LiteRT / KServe-`litert` |
| DRY-RUN best | 11 | All preset rows except VLM/splatting |
| BUILDS / UNTESTED best | 2 | Image understanding (LLaMA DRY-RUN only), Gaussian splatting |
| **Total** | **14** | 10 task families; detection/segmentation/pose split by representative |

## Open items

1. **Live-gate every DRY-RUN cell.** Run each preset for real (Docker, GPU
   where needed) and promote cells to PASS/FAIL with the observed output
   pasted or linked. Closest first: `rtdetrv4` on TRT, `owlv2` / `yoloseg` /
   `raft` / `vitpose` / `depth_anything_v2` / `videomae` on ORT,
   `yolo26s_tflite` on LiteRT, `yolov8_executorch` on ExecT, `gemma4` on
   LLAMACPP.
2. **`rfdetr_keypoint` missing from emulation.** The preset exists in
   `docker_run_inference_e2e_example.sh --list-presets` but has no job in
   `e2e-example-emulation.yml`. Add it (one-line matrix entry) so the pose
   row's second representative is covered.
3. **Fill the LIBTORCH / OPENVINO / LIBTENSORFLOW columns.** They build in CI
   but no preset targets them. Either add `--backend` override runs that are
   known to work, or record per-task BLOCKED with the loader error.
4. **OPENCV_DNN scope.** The Ultralytics `Split`/`Unsqueeze` blocker is
   recorded for YOLO families. Confirm cell-by-cell whether non-YOLO tasks
   (`raft`, `vitpose`, torchvision ResNet) load on apt OpenCV 4.6 or belong
   in BLOCKED too.
5. **KServe per-task gates.** Today's KServe evidence is protocol-level
   (Identity round-trip, dry-run) plus YOLO26-`litert` and the `ecdet`
   KNOWN-BAD. Gate one model per task through `neuriplo-kserve-runtime`
   and record backend + transport per cell (feeds roadmap Phase 5).
6. **Gaussian-splatting representative.** No preset exists. Export one `lgm`
   artifact, add a preset, then gate it.

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
- Style reference:
  [NERT coverage matrix](https://github.com/olibartfast/nert/blob/main/specs/2026-10-17-task-models/coverage.md).

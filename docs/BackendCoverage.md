# Backend coverage — task models × backends

One gated representative per task family, one status per backend, no cell
claiming more than its evidence. Style follows the NERT
[coverage matrix](https://github.com/olibartfast/nert/blob/main/specs/2026-10-17-task-models/coverage.md).

**Last updated: 2026-11-01.** Coverage was refreshed against neuriplo-infer
0.11.6, neuriplo 0.11.3 and neuriplo-tasks 0.8.4. The
[gated table](#gate-table) summarizes coverage by task and backend. Recorded
results are listed under [Evidence](#evidence); build versions and test totals
are under [Environments](#environments). Matrix coverage is always reported
here — never only in an external harness's working files. OpenCV DNN
coverage is 5.x only; 4.x is not considered.

Main limitations:

- **TorchScript input metadata:** since neuriplo 0.11.3, TorchScript files
  must record their input datatypes. Files exported without the metadata
  helper are rejected at load time and must be re-exported; some Torch cells
  below are BLOCKED for that reason, while the re-exported DETR files are
  PASS.
- **Incorrect outputs:** TensorFlow ResNet-50 classification and EdgeCrafter
  TFLite conversion are **KNOWN-BAD**; do not serve these combinations.
- **Incomplete test coverage:** OpenVINO and LLaMA native unit-test binaries
  do not link (LiteRT was in this list until the vendored-gtest prune;
  see Open item 8).
- **Backend and transport blockers:** apt OpenCV 4.6 cannot load current
  models — the DNN column below runs OpenCV 5.0.0 source-built (see
  [Environments](#environments)); KServe HTTP has live evidence, while the
  gRPC failure described in E1 remains blocked server-side.
- **Full cell matrix (2026-11-01):** the external release cell-matrix
  harness ran every explicit TaskFactory architecture × backend on the 0.11.6
  stack: 176 RUNTIME_PASS / 22 FAILED / 12 TIMEOUT / 10 KNOWN_BAD /
  159 EXPORT_BLOCKED / 51 NOT_APPLICABLE, with 13 triage decisions. On
  2026-11-01 the RF-DETR trio (detection `rfdetr`, segmentation
  `rfdetrseg`, pose `rfdetrpose`) was re-run on four gates, all
  RUNTIME_PASS with statuses unchanged: opencv5-only gate (binary
  `49737e78`, OpenCV 5.0.0; runs `3557889f` / `bdd99c4e` / `2d038303`),
  onnx_runtime (`793e80c7` / `54475a05` / `1c2a8aee`), openvino
  (`cd6f3259` / `66d8b5db` / `62265dd2`) and tensorrt (`25cd93dd` /
  `b8553c97` / `80bb106c`). Render spot-check: bicycle/person/sheep
  correct with masks; the dog is labeled `horse` ~0.94 on every backend
  (model-accuracy note, same weights — not a harness failure). Slurm
  session `2026-11-01-full-coverage-1` rebuilt the non-GenAI backend
  images in parallel from the same pins (`libtorch`, `libtensorflow`,
  `openvino`, `litert`, `tensorrt` BUILD_OK; `executorch` FAILED at link
  (`ld: cannot find -lxnnpack_backend`, owner infer Dockerfile linkage);
  `tensorrt` running); GenAI backends
  (`llamacpp`, `cactus`, `ggml`) are out of coverage scope, and
  `tvm`/`migraphx`/`dali` have no matrix pipeline. New
  blockers recorded there and not re-claimed here: OpenCV DNN single-input
  limit (DETR two-input, OWL three-input, `groundingdino` four-input),
  OpenCV DNN importer SIGSEGV on EdgeCrafter exact-GELU graphs, OpenVINO
  CPU SIGSEGV on 5D video inference, and NERT/OpenVINO setup-contract
  rejections plus CPU-timeout video cells.

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
  The combination could work; it just hasn't been tried.
- **N/A** — the backend cannot serve this task by construction (e.g. LLaMA
  is GGUF + `mmproj` only, so it never applies to detection/segmentation).
  Not a coverage gap. Maps to the matrix's NOT_APPLICABLE.

`BUILDS` means the `ci.yml` job for that backend is green (`OPENCV_DNN`,
`ONNX_RUNTIME`, `LIBTORCH`, `LIBTENSORFLOW`, `TENSORRT`, `OPENVINO`); there is
no `LITERT` / `EXECUTORCH` / `LLAMACPP` build job. `DRY-RUN` means the preset
is listed by `--list-presets` and green in `e2e-example-emulation.yml`
(dry-run validates command construction, not numerics).

**Matrix-to-table rule:** who enters the Matrix enters the table — every
explicit TaskFactory architecture with a terminal matrix result gets a
gate-table row. No matrix-covered architecture stays table-less.

Node counts, sha256 digests, and per-tensor tolerances belong to the
exporting repo (`neuriplo-tasks`) and the engine repo (`neuriplo`), not here.
This matrix gates the app seam: routed, run, rendered, tested.

## Gate table

Columns: `DNN` = OPENCV_DNN · `ORT` = ONNX_RUNTIME · `TRT` = TENSORRT ·
`Torch` = LIBTORCH · `OVINO` = OPENVINO · `TF` = LIBTENSORFLOW ·
`LiteRT` = LITERT · `ExecT` = EXECUTORCH · `LLaMA` = LLAMACPP ·
`KServe` = remote `KserveEngine`. Representative preset in parentheses.
Evidence column points to the task's section below.

The `Torch` column covers TorchScript. AOTInductor is a second runtime inside
the same LibTorch backend, selected when the model path ends in `.pt2`. It is
not covered by these cells yet — except E13, where RF-DETR's Torch PASS rests
on AOTInductor because TorchScript tracing is broken for that architecture
(see E13); do not read any other Torch cell as AOTInductor evidence.

| Task (representative) | DNN | ORT | TRT | Torch | OVINO | TF | LiteRT | ExecT | LLaMA | KServe | Evidence |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Object detection (`yolo26`, `yolo26s_tflite`) | PASS | PASS | PASS | BLOCKED | PASS | BUILDS | PASS | UNTESTED | N/A | PASS | [E1](#ev-yolo26) |
| Object detection (`yolov8`, `yolov8_executorch`) | PASS | DRY-RUN | BUILDS | BUILDS | BUILDS | PASS | UNTESTED | PASS | N/A | UNTESTED | [E2](#ev-yolov8) |
| Object detection (`rtdetr`, `rtdetrv4`) | BLOCKED | DRY-RUN | DRY-RUN | PASS | BUILDS | BUILDS | UNTESTED | UNTESTED | N/A | UNTESTED | [E3](#ev-rtdetr) |
| Object detection (`ecdet`, `edgecrafter_det`) | UNTESTED | PASS | PASS | BUILDS | BUILDS | BUILDS | KNOWN-BAD | UNTESTED | N/A | KNOWN-BAD | [E4](#ev-ecdet) |
| Object detection (`rfdetr`, `rfdetr`) | PASS | PASS | PASS | PASS | PASS | BUILDS | KNOWN-BAD | BLOCKED | N/A | PASS | [E13](#ev-rfdetr) |
| Instance segmentation (`yoloseg`, `yoloseg`) | PASS | PASS | PASS | BUILDS | BUILDS | BUILDS | UNTESTED | UNTESTED | N/A | PASS | [E5](#ev-seg) |
| Instance segmentation (`ecseg`, `edgecrafter_seg`) | UNTESTED | DRY-RUN | BUILDS | BUILDS | BUILDS | BUILDS | UNTESTED | UNTESTED | N/A | UNTESTED | [E5](#ev-seg) |
| Instance segmentation (`rfdetrseg`, `rfdetrseg`) | PASS | PASS | PASS | KNOWN-BAD | PASS | BUILDS | UNTESTED | BLOCKED | N/A | PASS | [E13](#ev-rfdetr) |
| Classification (`torchvisionclassifier`, `torchvision_classifier`) | UNTESTED | DRY-RUN | BUILDS | BLOCKED | BUILDS | KNOWN-BAD | UNTESTED | UNTESTED | N/A | UNTESTED | [E6](#ev-classification) |
| Video classification (`videomae`, `videomae`) | UNTESTED | DRY-RUN | BUILDS | BUILDS | BUILDS | BUILDS | UNTESTED | UNTESTED | N/A | UNTESTED | [E7](#ev-video) |
| Optical flow (`raft`, `raft`) | UNTESTED | DRY-RUN | BUILDS | BUILDS | BUILDS | BUILDS | UNTESTED | UNTESTED | N/A | UNTESTED | [E7](#ev-video) |
| Pose estimation (`vitpose`, `vitpose`) | PASS | DRY-RUN | BUILDS | BUILDS | BUILDS | BUILDS | UNTESTED | UNTESTED | N/A | PASS | [E8](#ev-pose) |
| Pose estimation (`rfdetrpose`, `rfdetr_keypoint`) | PASS | PASS | PASS | PASS | PASS | BUILDS | UNTESTED | BLOCKED | N/A | PASS | [E13](#ev-rfdetr) |
| Depth estimation (`depth_anything_v2`, `depth_anything_v2`) | PASS | DRY-RUN | BUILDS | BUILDS | PASS | BUILDS | UNTESTED | UNTESTED | N/A | PASS | [E9](#ev-depth) |
| Open-vocabulary detection (`owlv2`, `owlv2`) | BLOCKED | PASS | BUILDS | BUILDS | BUILDS | BUILDS | UNTESTED | UNTESTED | N/A | PASS | [E10](#ev-owlv2) |
| Image understanding (`gemma4`, `gemma4`) | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | PASS | UNTESTED | [E11](#ev-gemma4) |
| Gaussian splatting (`lgm`/`grm`, no preset) | UNTESTED | UNTESTED | UNTESTED | BUILDS | BUILDS | UNTESTED | UNTESTED | UNTESTED | N/A | UNTESTED | [E12](#ev-splatting) |

## Evidence

Live runs below were recorded on 2026-10-23 on `data/dog.jpg` (`person.jpg`
for pose) with rendered outputs inspected, unless a bullet gives another date.
The 2026-10-30 runs used the neuriplo-infer 0.11.5 release stack (neuriplo
0.11.3, neuriplo-tasks 0.8.4); the 2026-10-31 cell matrix ran on the 0.11.6
stack and is recorded in the external cell-matrix report (REPORT.md +
13 decisions).

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
- **Torch BLOCKED (2026-10-30).** The 2026-10-23 `yolo26s.torchscript`
  (ultralytics export) ran PASS in 1083 ms with
  `--input_sizes=1,3,640,640`, but that file predates the input-datatype
  metadata requirement in neuriplo 0.11.3. It now stops at load with
  `TorchScript input 'x.1' has no datatype metadata; re-export with the typed
  input metadata helper`. Re-export with the helper to restore the cell; the
  other backends are unaffected.
- **OpenVINO PASS** (CPU, 267 ms): `yolo26s.onnx` read directly.
- **LiteRT PASS** (CPU, 888 ms): `yolo26s.tflite` (`1x640x640x3`).
  Re-verified 2026-11-02 after the vendored-gtest prune (915 ms):
  bicycle 0.95, dog 0.94, car 0.52 — no regression.
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
- **TF PASS** (2026-11-01, TF 2.19 runtime `build-tensorflow`, CPU,
  1423 ms): `yolov8n_saved_model` (ultralytics SavedModel export) @ 0.4/0.5
  — bicycle 0.89, dog 0.93, car 0.61 — no regression. Render inspected
  (`data/output/processed_yolo_local.png`).
- **TF 2.21 BLOCKED at load (protobuf ABI, requested-version probe).**
  Same SavedModel under a 2.21-linked binary dies before `main`:
  `libtensorflow_framework.so.2` binds system `libprotobuf.so.32` and
  segfaults in `Arena` during static init (gdb: `KernelDefBuilder` ←
  `_GLOBAL__sub_I_dataset.cc`). Same bug family as the Torch/KServe
  protobuf clash. The pinned 2.19 PASS above stands; 2.21 needs a
  matching-protobuf environment, not a model fix.

### <a id="ev-rtdetr"></a>E3 — Object detection (`rtdetr`)

- **ORT/TRT DRY-RUN.** TRT is the default preset; `ci.yml` builds the TRT
  image; emulation dry-run green. No live log cited — not PASS.
- **Torch PASS (2026-10-30, CPU, 15.8 s inference).** A typed
  `rtdetr.torchscript` export (image plus original-size input, FP32/INT64)
  loaded on the 0.11.5 stack and rendered dog, bicycle and car correctly on
  `data/dog.jpg` at 0.25/0.45. The file records its input datatypes; the
  pre-contract export is rejected. `rtdetrv4` on Torch is still unrun.
- **DNN BLOCKED (2026-10-30).** `rtdetr` and `rtdetrv4` take two inputs (image
  and original size). OpenCV DNN accepts a single input only — the run stops
  with `OpenCV DNN backend currently supports only single input models, got 2
  inputs`. Same limitation as E10.

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

- **Torch BLOCKED (2026-10-30).** The traced `resnet50` (`torchvision`
  IMAGENET1K_V2) ran PASS in 689 ms with `--input_sizes=1,3,224,224` —
  malamute 0.44 — but the file has no input-datatype metadata and is now
  rejected at load with the same error as E1. Re-export with the helper to
  restore the cell.
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

Representative `depth_anything_v2` live-gated on ORT below; `yolo26depth`
additionally gated as alias.

- **ORT PASS** (2026-11-02, 1.19.2 CPU, 1310 ms): `depth_anything_v2.onnx`
  on `data/dog.jpg` renders a sane map (near yellow dog/bicycle, far blue
  background), min/max -0.31/7.64 — render inspected
  (`data/output/processed_depth-anything-v2_local.png`).

- **DNN PASS via alias** (5.0.0, CPU, 838 ms): `yolo26depth` renders sane
  depth map, min/max 1.26/5.41. (Apt 4.6 failed it on the YOLO `Split`
  error — gone on 5.x.) `depth_anything_v2` itself unrun.
- **ORT DRY-RUN + alias live** (CPU, 618 ms): sane depth map, min/max
  1.26/5.41.
- **OVINO PASS via alias** (2026-11-01, pinned 2026.3.0 runtime on
  optiplex-3040, CPU, 434 ms): `yolo26depth` renders the same sane map,
  min/max 1.26/5.41 — the promo-video memory was right; the BUILDS cell
  was stale.
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

### <a id="ev-rfdetr"></a>E13 — RF-DETR trio (`rfdetr`, `rfdetrseg`, `rfdetrpose`)

- **Matrix PASS 2026-11-01 (0.11.6 stack).** Full cell matrix ran every
  explicit TaskFactory architecture × backend; the trio was re-run on four
  gates, all RUNTIME_PASS: opencv5-only gate (binary `49737e78`, OpenCV
  5.0.0; runs `3557889f` / `bdd99c4e` / `2d038303`), onnx_runtime
  (`793e80c7` / `54475a05` / `1c2a8aee`), openvino (`cd6f3259` /
  `66d8b5db` / `62265dd2`), tensorrt (`25cd93dd` / `b8553c97` /
  `80bb106c`). Render spot-check: bicycle/person/sheep correct with masks;
  the dog is labeled `horse` ~0.94 on every backend (model-accuracy note,
  same weights — not a harness failure).
- **Pose alias live (see E8).** `rfdetr_keypoint` (alias of `rfdetrpose`)
  renders skeleton pose 0.85 on DNN/ORT/TRT plus KServe HTTP; the new pose
  row reuses that KServe PASS.
- **KServe PASS via `real-onnx` HTTP (2026-11-02, `neuriplo-kserve-runtime`
  `build/real-onnx`, CPU ORT).** Detection `rfdetr-medium` (13.5 s):
  bicycle 0.92, horse 0.94, car 0.87 — identical to every local backend;
  `neuriplo_http_infer_requests_success_total{model="det"} 2`, failure 0;
  render `data/output/processed_rfdetr_kserve-ort.png`. Seg
  `rfdetrseg-medium` (18.8 s): same trio with masks plus the `motorbike
  0.57` background artifact; success 1, failure 0; render
  `data/output/processed_rfdetrseg_kserve-ort.png`. Pose was already PASS
  via the `rfdetr_keypoint` alias (E8); the trio is now fully gated on
  KServe HTTP.
- **Presets + emulation.** `rfdetr` / `rfdetrseg` / `rfdetr_keypoint`
  presets exist in `docker_run_inference_e2e_example.sh --list-presets` and
  are green in `e2e-example-emulation.yml` (dry-run validates command
  construction, not numerics); LiteRT/ExecT/LLaMA N/A or UNTESTED per row.
- **Torch PASS via AOTInductor (2026-11-01, neuriplo-infer 0.11.7-dev +
  neuriplo 0.11.4-dev, LibTorch CPU, torch 2.14.0 AOTI packages).**
  Exception to the AOTI policy above: TorchScript tracing is broken for this
  architecture (see next bullet), so the Torch cells below rest on `.pt2`.
  Detection `rfdetr-medium` (576, `--input_sizes=1,3,576,576`, 1026 ms):
  bicycle 0.92, dog-as-`horse` 0.94, car 0.87 — same model-accuracy note as
  the matrix. Pose `rfdetr-keypoint-preview-xlarge` (576,
  `--input_sizes=1,3,576,576`, 2327 ms): skeleton + pose score 0.850784,
  matching the E8 alias score. Renders inspected
  (`data/output/processed_rfdetr_local.png`,
  `data/output/processed_rfdetr-keypoint_local.png`).
- **TorchScript (`.torchscript`) BLOCKED for this architecture.**
  `torch.jit.trace` completes and loads, but inference fails in-interpreter
  (`Dimension out of range` in the traced windowed-DINOv2 backbone) — trace
  freezes a data-dependent branch. `torch.export` captures the same model
  cleanly (detection 1220 nodes, seg 1611, keypoint 2250), hence the `.pt2`
  route above. Do not serve RF-DETR as TorchScript.
- **Seg Torch KNOWN-BAD (same stack).** `rfdetrseg-medium` (432,
  `--input_sizes=1,3,432,432`): exit 0, outputs `(1,200,4)` + `(1,200,91)` +
  `(1,200,108,108)`, 1296 ms — but the render is empty. Fault (tasks-side):
  `RfDetrSegmentationPostprocessor::findOutputIndices`
  (`neuriplo-tasks/src/instance_segmentation/rfdetr_segmentation_postprocessor.cpp`)
  resolves outputs by name with a digit heuristic; the Torch backend's
  synthetic names (`output0/1/2`) collapse `masks` and `labels` onto index 2,
  so classification reads the masks tensor. Named (ONNX) outputs are
  unaffected. Do not serve `rfdetrseg` on Torch until tasks maps by shape
  or position for synthetic names.
- **OpenVINO app-surface confirmation (pinned 2026.3.0 runtime,
  `build-openvino`, no override).** Detection `rfdetr-medium` (784 ms):
  bicycle 0.92, horse 0.94, car 0.87. Seg `rfdetrseg-medium` (1184 ms):
  bicycle 0.94, horse 0.91, car 0.86, all with masks, plus the same spurious
  `motorbike 0.57` background hit at 0.4. Keypoint (1566 ms): skeleton,
  pose 0.850784 — identical across OVINO, Torch-AOTI and the E8 alias runs.
  (An earlier pass on the unpinned 2025.2.0 runtime gave identical labels;
  only the pinned run is claimed.)
- **TensorRT app-surface confirmation (pinned 11.2.1.2, `build-trt`
  rebuilt against 11.2 headers, default-build engines — 11.2 `trtexec`
  dropped the `--fp16` shorthand under strongly-typed networks).**
  Detection `rfdetr-medium` (109 ms): bicycle 0.92, horse 0.94, car 0.86.
  Seg `rfdetrseg-medium` (380 ms): same trio with masks + the `motorbike`
  artifact. Keypoint (292 ms): skeleton, pose **0.850754** — 3e-5 below the
  0.850784 every other backend reports (FP tactic selection, rounds to the
  same 0.85).
- **LiteRT KNOWN-BAD (conversion fault, not runtime).** `rfdetr-nano.onnx`
  → `onnx2tf` float32 `.tflite` converts cleanly and runs exit 0 in
  `build-litert` (2445 ms) — but the render keeps a single wrong
  `person 0.75` where ONNX finds bicycle/horse/car. Same failure class as
  `ecdet` TFLite (E4): the converter breaks this architecture's numerics.
  Do not serve RF-DETR on LiteRT.
- **ExecuTorch BLOCKED (all three; runtime lacks `aten::linear.out`).**
  `rfdetr-nano` / `rfdetrseg-medium` / `rfdetr-keypoint-preview-xlarge`
  export cleanly to `.pte` (`format="executorch"`, `backend="xnnpack"`,
  rfdetr 1.11.2 / executorch 1.3.1) and load in `build-executorch` — then
  `forward()` fails: `Missing operator: aten::linear.out` (28 instructions
  detection/seg, 29 keypoint). Since rfdetr 1.9.1 the exporter recombines
  undelegated `addmm` into `aten::linear`; the local runtime
  (ExecuTorch v1.2.0, portable kernels, no `optimized_native_cpu_ops_lib`)
  does not register it — the exact case the reference
  `rf-detr-cpp-inference` docs flag (`EXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON`
  rebuild). Fix path: rebuild the runtime with optimized kernels and wire
  the lib into `neuriplo` (no such wiring exists today); re-test all three
  `.pte` then. Export-side note: 1.11.2 appends the delegate without an
  explicit `output_name` (`rfdetr-medium_xnnpack.pte`), pass it explicitly.

## Environments

How each column was built and tested (all CPU except TRT).
Live app-surface runs below are from 2026-11-01/02 on the pinned stack
(neuriplo-infer 0.11.7-dev + neuriplo 0.11.3 + neuriplo-tasks 0.8.4);
older dates are kept only where no newer run exists.

| Backend | Version / artifact | Build flags | Tests | Notes |
|---|---|---|---|---|
| DNN | OpenCV 5.0.0 source-built (`~/dependencies/opencv-5.0.0`, `scripts/setup_opencv.sh`) | `OPENCV_DNN` | 179/179 | New graph engine; `--input_sizes` required; apt 4.6 kept as fallback when the 5.x dir is absent |
| ORT | 1.19.2 CPU (local SDK; pin 1.28.0 unfetchable upstream — `setup_dependencies.sh` handles the renamed `gpu_cuda<major>` archives) | `ONNX_RUNTIME` | 179/179 | Full inference runs, renders inspected |
| TRT | 11.2.1.2 (NGC `tensorrt:26.08` base, `docker/Dockerfile.tensorrt`) | `TENSORRT` | image green in CI 2026-11-02 | Trio live-gated 2026-11-01 (det 109 ms, seg 380 ms + masks, keypoint 292 ms, pose 0.850754) — see E13. 11.2 `trtexec` dropped the `--fp16` shorthand; engines rebuilt from the same `.onnx` |
| Torch | libtorch 2.14 CPU | `LIBTORCH`, `NEURIPLO_INFER_ENABLE_KSERVE=OFF` | 134/134 | TorchScript cells, except E13 where RF-DETR rests on AOTInductor (`.pt2`) because TorchScript tracing is broken for that arch. KServe client unlinked (its gRPC protobuf clashes with libtorch's bundled protobuf); first `cmake` seeds the CUDA cache, a second run is needed (nvcc 12.0 vs driver-reported 13.4). Since neuriplo 0.11.3, TorchScript archives must carry input-datatype metadata; legacy files are rejected |
| OVINO | 2026.3.0 (pinned; `docker/Dockerfile.openvino`) | `OPENVINO`, tests OFF | — (not covered) | Trio live-gated 2026-11-01 (det 784 ms, seg 1184 ms + masks, keypoint 1566 ms, pose 0.850784) — see E13. Native `runUnitTests` link vs `openvino::runtime` fails; CI builds tests OFF too |
| TF | 2.19 (`LIBTENSORFLOW`; 2.21 C++ runtime unloadable — protobuf ABI, see E2) | `LIBTENSORFLOW` | 179/179 | `--weights` accepts directories since 2026-10-23 (was file-only); see E6 |
| LiteRT | `~/dependencies/litert` (LiteRT 2.19.0 pinned) | `LITERT` | 192/192 | Vendored gtest/gmock headers pruned from the SDK (they shadowed system gtest and crashed every death test) — see Open item 8 |
| ExecT | `~/dependencies/executorch` | `EXECUTORCH`, `NEURIPLO_INFER_ENABLE_KSERVE=OFF` | 134/134 | Same protobuf clash as Torch |
| LLaMA | llama.cpp b9085 | `LLAMACPP`, tests OFF | — (not covered) | Native `runUnitTests` link misses ggml transitive libs (same class as the OVINO gap) |
| KServe | `neuriplo-kserve-runtime` `real-onnx` / `grpc-onnx` | client in default build | suites run manually | HTTP green incl. per-task gates; gRPC `ModelInfer` 13 INTERNAL server-side (see E1) |

## Scoreboard

| Status | Count (task rows) | Rows |
|---|---|---|
| PASS (≥1 backend live-gated) | 12 | `yolo26`, `yolov8`, `rtdetr`, `ecdet`, `rfdetr`, `owlv2`, `gemma4`, `yoloseg`, `rfdetrseg`, pose (`vitpose` via `rfdetr_keypoint` alias, `rfdetrpose`), depth (`yolo26depth` alias) |
| KNOWN-BAD recorded | 4 | `ecdet` on LiteRT / KServe-`litert`; `torchvisionclassifier` on LIBTENSORFLOW; `rfdetrseg` on LIBTORCH; `rfdetr` on LiteRT |
| DRY-RUN best | 4 | `torchvisionclassifier`, `ecseg`, `videomae`, `raft` |
| BUILDS / UNTESTED best | 1 | Gaussian splatting (no preset; Torch/OVINO BUILDS-only) |
| **Total** | **17** | 10 task families; detection/segmentation/pose split by representative |

## Open items

1. **Live-gate every DRY-RUN cell.** Closest first: `rtdetrv4` on TRT;
   `raft` / `vitpose` / `depth_anything_v2` / `videomae` on ORT.
   Needs model exports that don't exist locally yet.
2. **Closed — RF-DETR presets in emulation.** `rfdetr`, `rfdetrseg`, and
   `rfdetr_keypoint` are listed by
   `docker_run_inference_e2e_example.sh --list-presets` and have jobs in
   `e2e-example-emulation.yml` (dry-run green).
3. **Record Torch / OpenVINO `--backend` override runs as repeatable gates.**
   The DETR Torch run is a manual CLI invocation on `rtdetr.torchscript`, not
   a preset. Add presets and re-gate once the remaining TorchScript exports
   are refreshed (item 9).
4. **OPENCV_DNN scope (now 5.0.0).** The YOLO `Split`/`Unsqueeze` blockers,
   the `rfdetr` `BOOL` dtype failure, and the `owlv2`/`groundingdino`
   importer errors are all gone on the 5.x importer. Remaining 5.x gap:
   multi-input inference (`owlv2`, `groundingdino` and the two-input
   `rtdetr`/`rtdetrv4` exports fail in `setMainGraphInput`). EdgeCrafter
   exact-GELU graphs are a separate importer SIGSEGV, not the input-count
   limit. `groundingdino` is confirmed 4-input (2026-10-31). Still unprobed on
   5.x: `ecseg`, torchvision ResNet, `vitpose`,
   `depth_anything_v2`.
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
8. **LiteRT-linked `runUnitTests`: fixed (2026-11-02), 192/192 on the pinned
   stack.** Root cause was not the backend: `setup_litert.sh` harvested the
   TensorFlow build tree's vendored `gtest`/`gmock` headers into the SDK
   include tree, and they shadowed the system gtest headers in every consumer
   built with `-I$LITERT_DIR/include`. The vendored headers carry an older
   `testing::internal::RE` layout (144 vs 168 bytes), so test binaries mixed
   layouts with the system `libgtest` — an ODR violation that crashed all
   21 death tests pre-fork in matcher construction. The script now prunes
   `include/{gtest,gmock}` and `lib/libgtest*`/`libgmock*` on every run
   (including no-op runs against an existing install).
9. **Re-export the remaining TorchScript representatives.** neuriplo 0.11.3
   requires input-datatype metadata in the TorchScript archive. `yolo26` and
   `torchvisionclassifier` were gated on pre-contract files and are now
   BLOCKED until re-exported with
   [torchscript_metadata.py](https://github.com/olibartfast/neuriplo/blob/master/scripts/lib/torchscript_metadata.py);
   the DETR family has typed exports and was re-gated 2026-10-30.

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
- TorchScript input metadata:
  [torchscript_metadata.py](https://github.com/olibartfast/neuriplo/blob/master/scripts/lib/torchscript_metadata.py)
  (writes the datatype sidecar read by the libtorch backend).
- App-surface integration tests: [§ Integration-test evidence](#integration-test-evidence)
  above owns them (moved from `neuriplo-platform`; live paths stay manual
  per each suite's README).
- Style reference:
  [NERT coverage matrix](https://github.com/olibartfast/nert/blob/main/specs/2026-10-17-task-models/coverage.md).

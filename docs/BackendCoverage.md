# Backend coverage — task models × backends

One gated representative per task family, one status per backend, no cell
claiming more than its evidence. Style follows the NERT
[coverage matrix](https://github.com/olibartfast/nert/blob/main/specs/2026-10-17-task-models/coverage.md).

**Last updated: 2026-11-07.** Baseline evidence uses neuriplo-infer
0.11.6, neuriplo 0.11.3 and neuriplo-tasks 0.8.5. Newer local corrections
are scoped in their evidence sections; unreleased backend fixes are marked ‡ or §. The
[gated table](#gate-table) summarizes coverage by task and backend. Recorded
results are listed under [Evidence](#evidence); build versions and test totals
are under [Environments](#environments). Matrix coverage is always reported
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
  over HTTP with a reconverted model and the local XNNPACK reshaping fix
  (E4); RF-DETR uses its native LiteRT exporter (E13). TensorFlow ResNet-50
  passes with the corrected exporter (E6). The LiteRT backend fix is local
  and has not yet been released or pinned.
- **Incomplete test coverage:** OpenVINO and LLaMA native unit-test binaries
  do not link.
- **Backend and transport blockers:** apt OpenCV 4.6 cannot load current
  models — the DNN column below runs OpenCV 5.0.0 source-built (see
  [Environments](#environments)); KServe HTTP has live evidence, while the
  gRPC failure described in E1 remains blocked server-side.
- **OpenVINO status (evidence recorded 2026-11-04, P-LNX-EDGE-CPU):**
  `owlv2` / `owlvit` PASS; `videomae` / `vivit` / `timesformer`
  RUNTIME-PASS (inference only, no video export); `rtdetrul` PASS on the
  current local stack; `tensorflow-classifier` remains KNOWN-BAD and was
  not re-gated for OpenVINO. Slurm results use a derived validation image
  with missing runtime libraries added; they do not certify the original
  image or a fresh source build. See [E14](#ev-ov-slurm).

See [Open items](#open-items) for remaining validation and fixes.

## How to read this

- **PASS** — a live end-to-end run is recorded with its expected output
  (detections, labels, or round-trip log) in the task's evidence section.
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
no `LITERT` / `EXECUTORCH` / `LLAMACPP` / `MIGRAPHX` build job. `DRY-RUN` means the preset
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
keep each table narrow. Each evidence link points to
the task's recorded results. [Scope and caveats](#scope-and-caveats) explain
the NERT gates and the marked cells.

`DNN` = OPENCV_DNN · `ORT` = ONNX_RUNTIME · `TRT` = TENSORRT ·
`TRC` = LIBTORCH · `OV` = OPENVINO · `TF` = LIBTENSORFLOW.
`ExecT` = EXECUTORCH · `LLaMA` = LLAMACPP · `KServe` = remote `KserveEngine`.
`MGX` = MIGraphX, the independent `MIGRAPHX` backend in neuriplo, distinct from
ONNX Runtime's MIGraphX execution provider.
NERT is the first-party Neuriplo Engine Runtime. Representative presets
appear in parentheses.

Table key: **P** = PASS · **U** = UNTESTED · **B** = BUILDS · **DR** = DRY-RUN.
Footnote markers retain the scope explained below.

### Local inference backends

| Task (representative) | NERT | DNN | ORT | TRT | TRC | OV | TF | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Object detection (`yolo26`, `yolo26s_tflite`) | P | P | P | P | P | P | B | [E1](#ev-yolo26) |
| Object detection (`yolov8`, `yolov8_executorch`) | P | P | DR | B | B | B | P | [E2](#ev-yolov8) |
| Object detection (`rtdetr`, `rtdetrv4`) | P* | P§ | DR | DR | P | B | B | [E3](#ev-rtdetr) |
| Object detection (`rtdetrul`, Ultralytics RT-DETR) | U | U | P | U | U | P | U | [E14](#ev-ov-slurm) |
| Object detection (`ecdet`, `edgecrafter_det`) | P | U | P | P | B | B | B | [E4](#ev-ecdet) |
| Object detection (`rfdetr`, `rfdetr`) | P | P | P | P | P | P | B | [E13](#ev-rfdetr) |
| Instance segmentation (`yoloseg`, `yoloseg`) | P* | P | P | P | B | B | B | [E5](#ev-seg) |
| Instance segmentation (`ecseg`, `edgecrafter_seg`) | P | U | DR | B | B | B | B | [E5](#ev-seg) |
| Instance segmentation (`rfdetrseg`, `rfdetrseg`) | P | P | P | P | P | P | B | [E13](#ev-rfdetr) |
| Classification (`torchvisionclassifier`, `torchvision_classifier`) | P* | U | DR | B | P | B | P | [E6](#ev-classification) |
| Video classification (`videomae`, `videomae`) | P | U | DR | B | B | RUNTIME-PASS† | B | [E7](#ev-video) |
| Optical flow (`raft`, `raft`) | P | U | DR | B | B | B | B | [E7](#ev-video) |
| Pose estimation (`vitpose`, `vitpose`) | P | P | DR | B | B | B | B | [E8](#ev-pose) |
| Pose estimation (`rfdetrpose`, `rfdetr_keypoint`) | P | P | P | P | P | P | B | [E13](#ev-rfdetr) |
| Depth estimation (`depth_anything_v2`, `depth_anything_v2`) | P | P | P | B | B | P | B | [E9](#ev-depth) |
| Open-vocabulary detection (`owlv2`, `owlv2`) | P | P§ | P | B | B | P† | B | [E10](#ev-owlv2) |
| Image understanding (`gemma4`, `gemma4`) | N/A | N/A | N/A | N/A | N/A | N/A | N/A | [E11](#ev-gemma4) |
| Gaussian splatting (`lgm`/`grm`, no preset) | N/A | U | U | U | B | B | U | [E12](#ev-splatting) |
| Keypoint matching (`SuperPoint + LightGlue`, no app preset) | U | U | P¶ | P¶ | U | U | U | [E17](#ev-keypoint-matching) |

### Additional runtimes and remote execution

**MGX backend contract tests: P, 6/6 (3 CPU + 3 MI300X GPU, zero skips).**
These backend-level results are recorded in [E16](#ev-migraphx) and are the
basis for the MGX `P` cells below. No per-task app run is recorded.

| Task (representative) | MGX | LiteRT | ExecT | LLaMA | KServe | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| Object detection (`yolo26`, `yolo26s_tflite`) | P | P | U | N/A | P | [E1](#ev-yolo26) |
| Object detection (`yolov8`, `yolov8_executorch`) | P | P | P | N/A | U | [E2](#ev-yolov8) |
| Object detection (`rtdetr`, `rtdetrv4`) | P | U | U | N/A | U | [E3](#ev-rtdetr) |
| Object detection (`rtdetrul`, Ultralytics RT-DETR) | P | U | U | N/A | U | [E14](#ev-ov-slurm) |
| Object detection (`ecdet`, `edgecrafter_det`) | P | P‡ | U | N/A | P‡ | [E4](#ev-ecdet) |
| Object detection (`rfdetr`, `rfdetr`) | P | P | P§ | N/A | P | [E13](#ev-rfdetr) |
| Instance segmentation (`yoloseg`, `yoloseg`) | P | U | U | N/A | P | [E5](#ev-seg) |
| Instance segmentation (`ecseg`, `edgecrafter_seg`) | P | U | U | N/A | U | [E5](#ev-seg) |
| Instance segmentation (`rfdetrseg`, `rfdetrseg`) | P | P | P§ | N/A | P | [E13](#ev-rfdetr) |
| Classification (`torchvisionclassifier`, `torchvision_classifier`) | P | U | U | N/A | U | [E6](#ev-classification) |
| Video classification (`videomae`, `videomae`) | P | U | U | N/A | U | [E7](#ev-video) |
| Optical flow (`raft`, `raft`) | P | U | U | N/A | U | [E7](#ev-video) |
| Pose estimation (`vitpose`, `vitpose`) | P | P | U | N/A | P | [E8](#ev-pose) |
| Pose estimation (`rfdetrpose`, `rfdetr_keypoint`) | P | P | P§ | N/A | P | [E13](#ev-rfdetr) |
| Depth estimation (`depth_anything_v2`, `depth_anything_v2`) | P | P | U | N/A | P | [E9](#ev-depth) |
| Open-vocabulary detection (`owlv2`, `owlv2`) | P | U | U | N/A | P | [E10](#ev-owlv2) |
| Image understanding (`gemma4`, `gemma4`) | N/A | N/A | N/A | P | U | [E11](#ev-gemma4) |
| Gaussian splatting (`lgm`/`grm`, no preset) | P | U | U | N/A | U | [E12](#ev-splatting) |
| Keypoint matching (`SuperPoint + LightGlue`, no app preset) | P | U | U | N/A | U | [E17](#ev-keypoint-matching) |

### Scope and caveats

The MGX `P` cells record the upstream backend contract suite (6/6,
[E16](#ev-migraphx)); no app task run, app build or preset gate is recorded
for this backend. Its ONNX-only input format cannot serve the GGUF
`gemma4` representative, so that cell is N/A.

The OV cells marked **†** use the retained 0.11.6 campaign executable
in a derived validation image on P-LNX-EDGE-CPU. They do not certify the
current source pins or the original image packaging; see [E14](#ev-ov-slurm).

The `TRC` column covers TorchScript. AOTInductor is a second runtime inside
the same LibTorch backend, selected when the model path ends in `.pt2`. It is
not covered by these cells yet — except E13, where RF-DETR's TRC PASS rests
on AOTInductor because TorchScript tracing is broken for that architecture
(see E13); do not read any other TRC cell as AOTInductor evidence.

**NERT scope:** `PASS` records the upstream full-model reference gate
on CPU, CUDA and OpenCL from the
[NERT coverage matrix](https://github.com/olibartfast/nert/blob/main/specs/2026-10-17-task-models/coverage.md)
(2026-10-24). It does not assert an app CLI/render gate or validate the app's
released dependency pins. Neuriplo exposes CPU and opt-in CUDA; OpenCL and HIP
are engine capabilities without consumer selection. Full-catalog AMD/HIP
validation remains incomplete.

`PASS*` uses a different representative within the row's family:
RT-DETRv4-s for `rtdetr`, YOLO11m-seg for `yoloseg`, and ResNet-50 for
`torchvisionclassifier`. The ResNet-50 artifact has an untrained replacement
head: its gate proves numerical equivalence, not ImageNet class correctness.
RF-DETR, EdgeCrafter and YOLO26 results retain the artifact-specific task
and numerical deviations documented upstream; the seg-nano counterfactual
is not a deployable substitute. No result is inferred for `rtdetrul`.

‡ EdgeCrafter LiteRT and KServe HTTP PASS require the reconverted artifact
and the unreleased neuriplo XNNPACK subgraph-reshaping fix
[`9bde6c0`](https://github.com/olibartfast/neuriplo/commit/9bde6c0fd07ab88b70e1076f5fc667c20be41d7d).
The app's released dependency pin does not contain that fix.

§ Local 2026-11-04 fixes, not released pins: OpenCV 5 named/typed graph
inputs, RT-DETRv4's prepared vector-MatMul export, and ExecuTorch v1.2.0
rebuilt with optimized registrations. Identical-input ORT parity and renders
are validated locally. Repaired parallel Slurm reruns are **pending worker
availability**; these are not Slurm PASS claims. See [E15](#ev-blocked-slurm).

## Evidence

Live runs below were recorded on 2026-10-23 on `data/dog.jpg` (`person.jpg`
for pose) with rendered outputs inspected, unless a bullet gives another date.
The 2026-10-30 runs used the neuriplo-infer 0.11.5 release stack (neuriplo
0.11.3, neuriplo-tasks 0.8.4); the 2026-10-31 cell matrix ran on the 0.11.6
stack and is recorded in the external cell-matrix report (REPORT.md +
13 decisions). Develop-stack re-gates from 2026-11-02/03 ran on sibling
checkouts (neuriplo-infer `develop` c7dd16b + neuriplo-tasks `develop`
ea05454, which carries the RF-DETR synthetic-name mapping fix) and are
labeled as such. Seg-LiteRT uses tasks `v0.8.5` tag content. Later
unreleased corrections have explicit ‡/§ scope notes and do not certify
the app's released dependency pins.

### <a id="ev-yolo26"></a>E1 — Object detection (`yolo26`)

- **DNN PASS** (5.0.0 source-built, CPU, 378 ms): @ 0.4/0.5 — bicycle 0.95,
  dog 0.94, car 0.52, needs `--input_sizes=3,640,640` (the 5.0 backend
  requires explicit input sizes). apt OpenCV 4.6 is BLOCKED on this model
  (`Split` parse error); use the 5.x build from
  `scripts/setup_opencv.sh`.
- **ORT PASS** (CPU, 1137 ms): @ 0.4/0.5 — bicycle 0.95, dog 0.94, car 0.52.
- **TRT PASS** (10.14.1.48 FP16 engine from `yolo26s.onnx`, 251 ms):
  identical outputs; engines use 10.14 `trtexec`.
- **Torch PASS (pinned 2026-11-03).** The typed resave of
  `yolo26s.torchscript` (input-datatype metadata verified with
  `has_torchscript_metadata`) loads and runs: 1242 ms,
  `--input_sizes=1,3,640,640` — bicycle 0.95, dog 0.94, car 0.52,
  exact match with DNN/ORT. Render inspected
  (`data/output/processed_yolo26_local.png`).
- **OpenVINO PASS** (CPU, 267 ms): `yolo26s.onnx` read directly.
- **LiteRT PASS** (CPU, 888 ms): `yolo26s.tflite` (`1x640x640x3`).
  Re-verified 2026-11-02 after the vendored-gtest prune (915 ms):
  bicycle 0.95, dog 0.94, car 0.52.
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
  dog 0.93, car 0.61.
- **ExecT PASS** (CPU, 320 ms): `yolov8n.pte` @ 0.4/0.5 — bicycle 0.89,
  dog 0.93, car 0.61. No CI build job for ExecT.
- **LiteRT PASS** (2026-11-02, `build-litert`, CPU, 471 ms):
  `yolov8n` float32 `.tflite` via `onnx2tf` @ 0.4/0.5 — bicycle 0.89,
  dog 0.93, car 0.61, identical to DNN/ORT. Render inspected
  (`data/output/processed_yolov8_local.png`).
- **TF PASS** (2026-11-01, TF 2.19 runtime `build-tensorflow`, CPU,
  1423 ms): `yolov8n_saved_model` (ultralytics SavedModel export) @ 0.4/0.5
  — bicycle 0.89, dog 0.93, car 0.61. Render inspected
  (`data/output/processed_yolo_local.png`).
- **TF 2.21 BLOCKED at load (protobuf ABI, requested-version probe).**
  Same SavedModel under a 2.21-linked binary cannot start:
  `libtensorflow_framework.so.2` binds system `libprotobuf.so.32` and
  crashes during static initialization. The pinned 2.19 PASS above stands;
  2.21 needs a matching-protobuf environment, not a model fix.

### <a id="ev-rtdetr"></a>E3 — Object detection (`rtdetr`)

- **ORT/TRT DRY-RUN.** TRT is the default preset; `ci.yml` builds the TRT
  image; emulation dry-run green. No live log cited — not PASS.
- **Torch PASS (2026-10-30, CPU, 15.8 s inference).** A typed
  `rtdetr.torchscript` export (image plus original-size input, FP32/INT64)
  loaded on the 0.11.5 stack and rendered dog, bicycle and car correctly on
  `data/dog.jpg` at 0.25/0.45. The file records its input datatypes; the
  typed input metadata is required. `rtdetrv4` on Torch is still unrun.
- **DNN PASS§ (2026-11-04, local OpenCV 5).** The adapter binds the
  real FP32 image and INT64 size inputs by name and preserves integer labels.
  RT-DETR agrees with ORT (labels exact, max score error 1.84e-6, boxes
  within 0.004 pixels). RT-DETRv4 additionally requires the prepared
  vector-MatMul export; the original artifact still hits OpenCV's matrix-only
  operator restriction. Use CLI type `rtdetr` for both files. See
  [E15](#ev-blocked-slurm) for scope and pending Slurm validation.

### <a id="ev-ecdet"></a>E4 — Object detection (`ecdet`)

- **ORT + TRT PASS.** C++ e2e `data/dog.jpg` @ 0.5 — bicycle 0.93, dog 0.92,
  car 0.74 — [KserveCompatibility.md](KserveCompatibility.md#known-bad-conversion-edgecrafter-ecdet--tflite).
  Live multi-backend path is `kserve-runtime-edgecrafter-e2e`
  ([§ Integration-test evidence](#integration-test-evidence),
  [BENCHMARK.md](../integration-tests/kserve-runtime-edgecrafter-e2e/BENCHMARK.md)).
- **LiteRT + KServe HTTP PASS (2026-11-04, local backend patch).**
  Reconversion with `onnx2tf 2.6.9` removes the old custom GridSample op.
  The remaining score drift was isolated with byte-identical app inputs:
  Python LiteRT's builtin path reproduces C++'s dog 0.7985 / bicycle
  0.7153 / car 0.6192; Python XNNPACK matches ONNX. The installed C++
  runtime creates XNNPACK automatically but cannot delegate the dynamic
  graph with its default options. Enabling XNNPACK subgraph reshaping in
  `neuriplo` fixes the C++ run: dog **0.929990**, bicycle **0.927981**,
  car **0.780462** (plus the same motorbike/truck/cup background hits as
  ORT at 0.4). Scores differ from ORT by at most **4.49e-6** over all
  300 queries; the six high-confidence labels and boxes agree. Two low-score
  query ties swap labels/boxes, so full ordered box equality is not claimed.
  CLI local inference: 3005 ms. Live KServe HTTP returns identical raw
  tensors to local LiteRT and renders successfully (17139 ms end to end;
  single-run timings, not a benchmark). The server loads the patched local
  `libneuriplo.so`; this is not evidence for the unchanged released pin.
  Old TFLite files remain unsupported and must be reconverted.
  [Reproduction and provenance](../specs/2026-11-04-known-bad-backends/validation.md).

### <a id="ev-seg"></a>E5 — Instance segmentation (`yoloseg`, `ecseg`)

- **DNN PASS** (5.0.0, CPU, 494 ms): `yolov8n-seg.onnx` — dog 0.87,
  bicycle 0.80, car 0.45, masks verified.
- **ORT PASS** (CPU, 464 ms): @ 0.4/0.5 — dog 0.87, bicycle 0.80, car 0.45; masks verified; rendered `data/output/processed_yoloseg_local.png`.
- **TRT PASS** (10.14.1.48 FP16 engine from `yolov8n-seg.onnx`, 430 ms GPU): masks + boxes rendered, identical output shapes (1x116x8400 + 1x32x160x160).
- **KServe PASS** (`real-onnx` runtime, HTTP, model `seg` v1): exit 0, `neuriplo_http_infer_requests_success_total{model="seg",version="1"} 1`, failure 0; render `data/output/processed_yoloseg_kserve-ort.png` (>10 KB).
- `ecseg`: preset + emulation green; no live numbers recorded (unlike `ecdet`).

### <a id="ev-classification"></a>E6 — Classification (`torchvisionclassifier`)

- **Torch PASS (pinned 2026-11-03).** The typed resave of the
  `resnet50` (`torchvision` IMAGENET1K_V2) archive (metadata verified with
  `has_torchscript_metadata`)
  loads and runs: 687 ms, `--input_sizes=1,3,224,224` — top class 249 @
  0.44 (proper preprocessing gives 0.55, so the app seam under-reports but
  ranks correctly). Render
  inspected (`data/output/processed_torchvision-classifier_local.png`).
  TensorFlow status is listed separately below.
- **TF PASS (2026-11-04, CPU, TF 2.19 C++ runtime).** Re-exporting
  pretrained Keras ResNet50 with the corrected neuriplo-tasks exporter gives
  class **249 @ 0.532603**, versus Python TF 2.21 **249 @ 0.532377** on
  `data/dog.jpg`; all five top classes agree. CLI inference completed in
  2850 ms and rendered `processed_tensorflow-classifier_local.png`.
  The task sends raw BGR Float32 pixels; the SavedModel subtracts BGR means
  and emits logits, so the task applies softmax once. The fix is upstream
  commit `75cc7ac`, already included in the pinned `neuriplo-tasks v0.8.5`.
  Unwrapped SavedModels are unsupported; re-export them using the
  [supported recipe](https://github.com/olibartfast/neuriplo-tasks/blob/v0.8.5/export/classification/TensorflowClassification.md).
  This run used local sibling checkouts, with no differences from the pins
  in the classification/exporter and TensorFlow backend paths; it is not an
  isolated remote-fetch release gate. Exact provenance and reproduction:
  [validation](../specs/2026-11-04-tensorflow-classification-regate/validation.md).
  `tensorflow-classifier` / `vit-classifier` aliases still have no preset.

### <a id="ev-video"></a>E7 — Video classification (`videomae`), optical flow (`raft`)

- Preset + emulation green (ORT DRY-RUN). `vivit` / `timesformer` aliases
  ride the same task path; no separate preset.
- **OVINO RUNTIME-PASS† (2026-11-04, Slurm job 35).** `videomae`, `vivit`
  and `timesformer` completed on real moving clips with 16 / 32 / 8 frames:
  one sample each, positive inference timing (9091 / 24992 / 9736 ms).
  The build has `NEURIPLO_INFER_WITH_VIDEOWRITER=OFF`; removing the optional
  `--output_video` argument enabled inference. Exported-video rendering and
  numerical classification correctness remain unvalidated. See [E14](#ev-ov-slurm)
  for image provenance and the per-model evidence. This does not cover `raft`.

### <a id="ev-pose"></a>E8 — Pose estimation (`vitpose`)

Representative `vitpose` unrun everywhere below; `rfdetr_keypoint` gated
as alias. RF-DETR presets are covered by emulation.

- **DNN PASS via alias** (5.0.0, CPU, 3582 ms): `rfdetr_keypoint` renders
  skeleton, pose score 0.85. `vitpose` itself unrun.
- **ORT DRY-RUN + alias live** (CPU, 2817 ms): `rfdetr_keypoint` on
  `person.jpg` renders skeleton, pose score 0.85.
- **TRT alias live** (10.14.1.48 FP16 engine, 246 ms): same skeleton,
  score 0.85. Cell stays BUILDS for the unrun representative.
- **LiteRT alias live** (CPU, 6540 ms): `rfdetr_keypoint` float32 `.tflite`
  via `onnx2tf` renders skeleton, pose score 0.850784 — bit-identical to
  every other backend. Render inspected
  (`data/output/processed_rfdetr-keypoint_local.png`). Re-verified
  2026-11-02 on the develop stack (6898 ms, same 0.850784).
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
  depth map, min/max 1.26/5.41. `depth_anything_v2` itself unrun.
- **ORT DRY-RUN + alias live** (CPU, 618 ms): sane depth map, min/max
  1.26/5.41.
- **OVINO PASS via alias** (2026-11-01, pinned 2026.3.0 runtime on
  optiplex-3040, CPU, 434 ms): `yolo26depth` renders the same sane map,
  min/max 1.26/5.41.
- **LiteRT PASS** (2026-11-02, `build-litert`, CPU, 133 s):
  `depth_anything_v2` float32 `.tflite` via `onnx2tf` renders the same sane
  map, min/max -0.31/7.64 (identical to ORT to 5 decimals). Slow on CPU
  XNNPACK, but numerically intact. Render inspected
  (`data/output/processed_depth-anything-v2_local.png`).
- **KServe PASS** (`real-onnx` runtime, HTTP): same sane map — task gated
  via alias.

### <a id="ev-owlv2"></a>E10 — Open-vocabulary detection (`owlv2`)

- **OVINO PASS† (2026-11-04, Slurm job 34).** Prompts
  `a dog;a bicycle;a car`, confidence 0.25: `owlv2` rendered bicycle 0.34
  and dog 0.27; `owlvit` rendered bicycle 0.39. Both renders were inspected;
  inference stages were 30058 / 2788 ms. See [E14](#ev-ov-slurm) for the
  derived-image limitation. No other open-vocabulary model was rerun.
- **DNN PASS§ (2026-11-04, local OpenCV 5).** Native graph metadata
  provides the FP32 image and INT64 token/attention-mask inputs. The original
  OWLv2 ONNX file passes ORT parity and renders bicycle 0.34 / dog 0.27.
  This does not re-gate GroundingDINO or apt OpenCV 4.6. See
  [E15](#ev-blocked-slurm).
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
- **GGML is an independent backend, not the LLaMA column.** `neuriplo`
  ships `docker/Dockerfile.ggml` (builder: `-DDEFAULT_BACKEND=GGML` +
  engine tests, `docker/run_ggml_tests.sh`). Slurm-orchestrated builder
  coverage dispatched 2026-11-02 (job 29, zenbook worker,
  `neuriplo@develop` clone): builder image present
  (`neuriplo:ggml-2026-11-02-ggml-coverage`, 995 MB) but no test log.
  Re-gated 2026-11-03 (job 30, zenbook, `neuriplo@develop` `06b6903`):
  `BUILD_OK` + builder-image `ctest` **77/77 pass** (`CTEST_EXIT 0`).
  Builder + engine-test leg green; no app-surface gate claimed (GGML has
  no matrix column).

### <a id="ev-splatting"></a>E12 — Gaussian splatting (`lgm`/`grm`)

- Routed (`Capabilities.cpp`), no e2e preset, no emulation. Torch/OVINO
  cells are BUILDS-only. See [Open items](#open-items).

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
  Use the passing AOTInductor `.pt2` route above; do not serve RF-DETR as
  TorchScript.
- **Seg Torch PASS (pinned 2026-11-03).** `rfdetrseg-medium` `.pt2`
  (432, `--input_sizes=1,3,432,432`, 1370 ms, tasks `v0.8.5` tag content,
  `build-libtorch`): bicycle 0.94, horse 0.91, car 0.86, all with masks,
  plus the usual `motorbike 0.57` artifact — exact trio match with
  ORT/TRT/OVINO. Detection agrees on the same stack
  (`rfdetr-medium` 1004 ms: bicycle 0.92, horse 0.94, car 0.87).
  Renders inspected (`data/output/processed_rfdetrseg_local.png`,
  `data/output/processed_rfdetr_local.png`).
- **OpenVINO app-surface confirmation (pinned 2026.3.0 runtime,
  `build-openvino`, no override).** Detection `rfdetr-medium` (784 ms):
  bicycle 0.92, horse 0.94, car 0.87. Seg `rfdetrseg-medium` (1184 ms):
  bicycle 0.94, horse 0.91, car 0.86, all with masks, plus the same spurious
  `motorbike 0.57` background hit at 0.4. Keypoint (1566 ms): skeleton,
  pose 0.850784 — identical across OVINO, Torch-AOTI and the E8 alias runs.
  The pinned runtime is used for this status.
- **TensorRT app-surface confirmation (pinned 11.2.1.2, `build-trt`
  rebuilt against 11.2 headers with default-build engines).**
  Detection `rfdetr-medium` (109 ms): bicycle 0.92, horse 0.94, car 0.86.
  Seg `rfdetrseg-medium` (380 ms): same trio with masks + the `motorbike`
  artifact. Keypoint (292 ms): skeleton, pose **0.850754** — 3e-5 below the
  0.850784 every other backend reports (FP tactic selection, rounds to the
  same 0.85).
- **LiteRT detection: PASS via upstream's native exporter; legacy generic
  `onnx2tf` artifacts are KNOWN-BAD.** `rfdetr-medium` through rfdetr's own `format="litert"`
  route (`torch.export` + `litert-torch`, no ONNX/TF step; 576,
  `--input_sizes=1,3,576,576`, 3207 ms): bicycle 0.92, horse 0.94,
  car 0.87 at 0.4 — exact match with every other backend (same dog-as-horse
  note). Render inspected (`data/output/processed_rfdetr_local.png`).
  Upstream lists LiteRT as supported through its native exporter. App presets
  use that route.
- **LiteRT seg: native route PASS (pinned tasks `v0.8.5`); legacy
  generic-`onnx2tf` artifacts are KNOWN-BAD.** Native `format="litert"` file
  validated 2026-11-02 on the develop stack (3670 ms) and 2026-11-03 on
  the pinned stack (tasks `v0.8.5` tag content,
  `build-litert`, 4151 ms): bicycle 0.94, horse 0.91, car 0.86, all with
  masks, plus the usual `motorbike 0.57` artifact — exact trio match with
  ORT/TRT/OVINO. Render inspected
  (`data/output/processed_rfdetrseg_local.png`).
- **Pose LiteRT PASS (same stack).** `rfdetr-keypoint` float32 `.tflite`
  (576, `--input_sizes=1,3,576,576`, 6540 ms): skeleton, pose score
  **0.850784** — identical to every other backend (upstream notes keypoint
  is unsupported by `litert-torch`, so the `onnx2tf` file stands).
- **ExecuTorch PASS§ (2026-11-04, local rebuilt v1.2.0 SDK).** All
  three original native XNNPACK PTE artifacts execute with the existing
  optimized CPU kernels registered. No runtime version or model weights
  changed. Nano detection, medium segmentation and keypoint raw tensors
  agree with their same-model ORT references; selected detections/masks/pose
  are checked in [E15](#ev-blocked-slurm). This gate uses optimized CPU
  kernels; explicit portable delegate mode is not covered.

- **Supported export recipes (2026-11-04).** Both `rfdetr` and
  `rfdetrseg` app presets now select native `format="litert"` with
  `--backend litert` (RF-DETR 1.11.2), producing the exact file consumed by
  inference. They do not route through ONNX conversion. Local C++ re-gates
  against ONNX agree on detections; RF-DETR seg raw mask signs agree on
  **99.999914%** of elements. Legacy generic RF-DETR exports remain
  KNOWN-BAD; use the native artifacts. See the
  [native export helper](../scripts/export_rfdetr_litert.py) and
  [validation](../specs/2026-11-04-known-bad-backends/validation.md).

### <a id="ev-ov-slurm"></a>E14 — OpenVINO coverage status

Current status from evidence recorded 2026-11-04 on profile
`P-LNX-EDGE-CPU`, Slurm jobs 34 and 35:

| Model | Current OpenVINO status | Remaining validation |
|---|---|---|
| `owlv2` | PASS — inference and rendered output inspected | None for this recorded route |
| `owlvit` | PASS — inference and rendered output inspected | None for this recorded route |
| `videomae` | RUNTIME-PASS — inference completed | Classification correctness and video export/rendering |
| `vivit` | RUNTIME-PASS — inference completed | Classification correctness and video export/rendering |
| `timesformer` | RUNTIME-PASS — inference completed | Classification correctness and video export/rendering |
| `tensorflow-classifier` | KNOWN-BAD; OpenVINO not re-gated | Re-evaluate OpenVINO correctness |
| `rtdetrul` | PASS — same ONNX passes on the current local stack | Broader Ultralytics export coverage |

The Slurm models ran sequentially with two container CPUs, 3 GiB memory and
a 300-second timeout per model. All input files were checked against a
SHA-256 manifest.
The [tracked evidence summary](evidence/openvino-slurm-2026-11-04.json)
records image and binary hashes, fixture hashes, arguments, metrics and
artifact hashes. Single-run timings are not benchmark claims.

The separate LIBTENSORFLOW PASS for `tensorflow-classifier` is recorded in
E6; it does not change this OpenVINO status. These model-specific results do
not change the representative
`torchvisionclassifier` or `rtdetr` cells. A fresh 2026-11-04 C++ re-gate of
the same `rtdetrul.onnx` agrees
between ORT and OpenVINO: bicycle 0.962947, dog 0.942013, truck 0.73777,
car 0.55753; maximum raw output difference 1.06e-5. The current decoder
already handles normalized center/size boxes with explicit class IDs. The
result does not claim correctness for every Ultralytics export.

The release image `neuriplo-infer:openvino-2026-11-01-full-coverage-1` is
**NOT READY**: it lacks OpenCV and glog shared libraries. The validation image
added `libopencv-highgui406`,
`libopencv-imgcodecs406`, `libopencv-videoio406` and `libgoogle-glog0v6t64`.
The executable SHA-256 remained
`a31997f20450056a8c0c2378a693db8d33d0284542829262117e1eec311a1a2c`;
the derived image ID is
`sha256:cab51feff475b76d284012924d093fd7948489a97e68753d7b3e86a581f0fcd7`.
Source-image filesystem layers were verified after import. These results
apply to the derived validation image, not the release image or a fresh
source build. Packaging the missing runtime libraries remains open.

### <a id="ev-blocked-slurm"></a>E15 — Local backend status; Slurm rerun pending workers

The five representative cells have local PASS routes (§), covering six
model/backend checks. The
[evidence JSON](evidence/blocked-slurm-2026-11-04.json) records exact binaries,
libraries, models, SDK source and identical-input numerical comparisons. The backend fix is
[neuriplo 042bcef](https://github.com/olibartfast/neuriplo/commit/042bcefc8a20771d7c40d54c86f8ea588fcde1b8);
the export adapter is
[neuriplo-tasks d38e8b3](https://github.com/olibartfast/neuriplo-tasks/commit/d38e8b3984386c93a053215a1a88964e6e054e74).
The [pending Slurm plan](evidence/blocked-slurm-plan-2026-11-04.json) contains
all six exact commands and worker selections.

| Backend / model | Current local status | Slurm status |
|---|---|---|
| OpenCV 5 / RT-DETR | PASS; labels exact; max scores 1.84e-6, boxes 0.003861 pixels | Pending worker availability |
| OpenCV 5 / RT-DETRv4 | PASS; detections agree | Pending worker availability |
| OpenCV 5 / OWLv2 | PASS; max logits 0.001067, boxes 0.000134 normalized | Pending worker availability |
| ExecuTorch / RF-DETR nano | PASS; max boxes 0.000067, logits 0.000303 | Pending worker availability |
| ExecuTorch / RF-DETR seg medium | PASS; max boxes 0.000503, logits 0.007473; mask sign agreement 99.9997428% | Pending worker availability |
| ExecuTorch / RF-DETR keypoint | PASS; person.jpg max keypoints 0.000645; pose score 0.850784 | Pending worker availability |

RT-DETRv4's prepared graph preserves external names, shapes and dtypes.
Original versus prepared ORT differs by at most 0.000092 pixels / 9.54e-7
score. Original vector-MatMul weights remain unsupported in OpenCV.
The paired segmentation files in this re-gate give bicycle 0.943308, horse
0.916267 and car 0.801956 on both runtimes; archived artifact scores in E13
are separate evidence. The person.jpg pose renders show score 0.850784 on
both ORT and ExecuTorch.
Segmentation mask logits have maximum error 0.21214 and mean error 0.00138;
the mask agreement claim is a tolerance check, not bitwise equality.

Local checks: neuriplo CTest 80/80 on OpenCV 4.6 and 5, 77/77 on ExecuTorch
(including a real RF-DETR parity fixture); app CTest 191/191 on each changed
backend; tasks CTest 31/31; export adapter tests 3/3; Slurm worker tests 4/4.
No new model type or task contract was added.

**Slurm:** controller `zenbook-ux425uaz` is UP; `optiplex-3040` and
`jetson-nano` are DOWN (checked 2026-11-05). Repaired concurrent submissions
are pending worker availability; no repaired job IDs or concurrent PASS are
claimed. Accounting is disabled. The
[Slurm runner guide](SlurmBackendCoverage.md) and
[validation packet](../specs/2026-11-04-blocked-slurm-backends/validation.md)
provide the staged manifest and resume commands. Two workers are selected
for the repaired checks to permit overlap. Release pins, KServe gRPC,
TensorFlow 2.21 ABI and RF-DETR TorchScript status are listed above.

#### Input tensors used for parity

Names for RF-DETR below describe their image role; the PTE runtime supplies
its own positional metadata names. The hashes are over preprocessed bytes,
not JPEG files. Text inputs correspond to the three recorded OWLv2 prompts.

| Case | Input index / role | Dtype | Shape | Bytes | SHA-256 |
|---|---|---|---|---:|---|
| `rtdetr-dnn` | 0 / images | FP32 | 1×3×640×640 | 4,915,200 | `00db0764f4922d1b560e7fb18e770b65941fcb2fe7afa0aa46f5d685e143e0e0` |
| `rtdetr-dnn` | 1 / orig_target_sizes | INT64 | 1×2 | 16 | `9ae2a7b11ec45092ee082faa14507373560986cb09e909d8186a2eddaca3c1ca` |
| `rtdetrv4-dnn` | 0 / images | FP32 | 1×3×640×640 | 4,915,200 | `00db0764f4922d1b560e7fb18e770b65941fcb2fe7afa0aa46f5d685e143e0e0` |
| `rtdetrv4-dnn` | 1 / orig_target_sizes | INT64 | 1×2 | 16 | `9ae2a7b11ec45092ee082faa14507373560986cb09e909d8186a2eddaca3c1ca` |
| `owlv2-dnn` | 0 / pixel_values | FP32 | 1×3×960×960 | 11,059,200 | `bdcca8d9ca8d008165d8d6104907869085aa56d012c10a91003bbc80ec5f1e66` |
| `owlv2-dnn` | 1 / input_ids | INT64 | 3×16 | 384 | `4a3d5d9211841a7a77662ba96103b3919d2cd48e12e2083d15cc25f2a853dee1` |
| `owlv2-dnn` | 2 / attention_mask | INT64 | 3×16 | 384 | `ff665388dfa173d26115f43949f342315cd246196dd5e8b27d48b90f643fd3c4` |
| `rfdetr-executorch` | 0 / image | FP32 | 1×3×384×384 | 1,769,472 | `ba7e20ff2cf918a43b62f8996569b5f7b05852b74c8282d973f1e2cf1c613a8e` |
| `rfdetrseg-executorch` | 0 / image | FP32 | 1×3×432×432 | 2,239,488 | `9e3e5e953cb4b8b20fe1973e64841e7778f468d68fcd788f0baae9a502bf9160` |
| `rfdetr_keypoint-executorch` | 0 / image | FP32 | 1×3×576×576 | 3,981,312 | `d126bdd063cc86e69819506bd3068b3de4220f982e8112aae9159fd4d99a76da` |

#### Complete tensor comparison results

Every pair below received byte-identical preprocessed inputs, verified by SHA-256.
RT-DETR and RT-DETRv4 labels are INT64 and match exactly; their comparison
probe serializes label values as float32 without loss for these class IDs.
Other compared outputs are float32. Shape, element count, maximum and mean
absolute errors cover each complete output tensor, not just selected detections.
`dog.jpg` is the input for detection/segmentation/open-vocabulary checks;
keypoint results use `person.jpg`. Exact input/output buffer hashes remain in
the linked evidence JSON.

| Case | Output index / meaning | Shape | Elements | Max absolute error | Mean absolute error |
|---|---|---|---:|---:|---:|
| `rtdetr-dnn` | 0 / labels | 1×300 | 300 | 0 | 0 |
| `rtdetr-dnn` | 1 / boxes (pixels) | 1×300×4 | 1,200 | 0.00386047363 | 3.7179987e-05 |
| `rtdetr-dnn` | 2 / scores | 1×300 | 300 | 1.84029341e-06 | 2.16973319e-07 |
| `rtdetrv4-dnn` | 0 / labels | 1×300 | 300 | 0 | 0 |
| `rtdetrv4-dnn` | 1 / boxes (pixels) | 1×300×4 | 1,200 | 0.00355529785 | 3.47328169e-05 |
| `rtdetrv4-dnn` | 2 / scores | 1×300 | 300 | 5.54323196e-06 | 3.50363564e-07 |
| `owlv2-dnn` | 0 / logits | 1×3600×3 | 10,800 | 0.00106716156 | 4.55425507e-05 |
| `owlv2-dnn` | 1 / objectness_logits | 1×3600 | 3,600 | 0.000742912292 | 2.91300112e-05 |
| `owlv2-dnn` | 2 / pred_boxes (normalized) | 1×3600×4 | 14,400 | 0.000133812428 | 6.70462498e-07 |
| `rfdetr-executorch` | 0 / dets (normalized) | 1×300×4 | 1,200 | 6.67870045e-05 | 1.06222581e-06 |
| `rfdetr-executorch` | 1 / labels (logits) | 1×300×91 | 27,300 | 0.000302314758 | 7.89297337e-06 |
| `rfdetrseg-executorch` | 0 / dets (normalized) | 1×200×4 | 800 | 0.000502377748 | 9.09557093e-06 |
| `rfdetrseg-executorch` | 1 / labels (logits) | 1×200×91 | 18,200 | 0.00747203827 | 9.01507156e-05 |
| `rfdetrseg-executorch` | 2 / masks (logits) | 1×200×108×108 | 2,332,800 | 0.212135315 | 0.00137989956 |
| `rfdetr_keypoint-executorch` | 0 / dets (normalized) | 1×100×4 | 400 | 9.15527344e-05 | 6.84075076e-07 |
| `rfdetr_keypoint-executorch` | 1 / labels (logits) | 1×100×2 | 200 | 2.7179718e-05 | 2.90632238e-06 |
| `rfdetr_keypoint-executorch` | 2 / keypoints | 1×100×34×8 | 27,200 | 0.000644683838 | 2.70229248e-06 |

RF-DETR segmentation has six sign disagreements among 2,332,800 mask-logit
elements (agreement 99.999742798%). This is sign agreement at logit zero,
not mask IoU or dataset accuracy. All other recorded output sign agreements
are 100%; sign agreement alone is not a classification correctness criterion.
The ORT and ExecuTorch person-image pose PNGs have identical SHA-256:
`cc0d29f911cfaf5d5e8d13119ea6f3c8b73f5c4c7449857b7a62713e7df0e73d`.
The inspected segmentation PNG hash is
`cc5f0da28740560dd9ba6b52eb50d8b410ce1d64cc2f8fd9ef8de7bafa39f085`.

#### Local execution samples and checks

All E15 inference runs used CPU on Intel Core i5-11400H. A detected RTX 3060
was not selected (`--use-gpu=false`). These are individual CLI inference-stage
samples, not benchmarks, warm-run statistics, end-to-end latency or Slurm timings.

| Backend / model | Source | CLI inference (ms) | Observed result |
|---|---|---:|---|
| OpenCV 5 / RT-DETR | `dog.jpg` | 2784 | dog/bicycle/car rendered; full ORT comparison above |
| OpenCV 5 / prepared RT-DETRv4 | `dog.jpg` | 561 | bicycle/dog/truck rendered; full ORT comparison above |
| OpenCV 5 / OWLv2 | `dog.jpg` | 23407 | prompts `a dog;a bicycle;a car`; bicycle 0.34, dog 0.27 |
| ExecuTorch / RF-DETR nano | `dog.jpg` | 1430 | detections agree with paired nano ONNX |
| ExecuTorch / RF-DETR seg medium | `dog.jpg` | 2632 | bicycle/horse/car and masks; scores above |
| ExecuTorch / RF-DETR keypoint | `person.jpg` | 4777 | skeleton, pose 0.850784; paired ORT sample 1818 ms |

| Validation scope | Result | Qualification |
|---|---|---|
| neuriplo / OpenCV 4.6 | 80/80 CTest entries | legacy compatibility; no 4.6 multi-input model claim |
| neuriplo / OpenCV 5 | 80/80 CTest entries | includes real mixed FP32/INT64 fixture and integer raw-output checks |
| neuriplo / ExecuTorch | 77/77 CTest entries | real nano fixture enabled; every output checked within 0.001 |
| app / OpenCV 5 | 191/191 CTest entries | local build, KServe and gRPC compiled in |
| app / ExecuTorch | 191/191 CTest entries | local build, KServe and gRPC compiled in; no live transport gate implied |
| neuriplo-tasks | 31/31 CTest entries | no C++ task contract or model-type changes |
| Vector-MatMul adapter | 3/3 Python tests | parity, unchanged I/O, shared weights, collision avoidance, idempotence and exclusions |
| Slurm worker | 4/4 Python tests | real-process success/failure/timeout and missing-artifact cases |
| Source quality | PASS | full-tree clang-format and cppcheck; owning-repo hooks also ran the act cppcheck job |
| Model-type inventory | PASS | generated page checked against pinned tasks v0.8.5 README |

#### Exact source, SDK and artifact provenance

| Component | Revision / configuration |
|---|---|
| Application C++ base | `26d2e59ddfe553f6eb2b2ee8179c6559d5ee1687`; later app changes only add orchestration/docs |
| neuriplo adapter fix | `042bcefc8a20771d7c40d54c86f8ea588fcde1b8` |
| neuriplo-tasks export fix | `d38e8b3984386c93a053215a1a88964e6e054e74`; C++ task implementation unchanged |
| ExecuTorch SDK | `v1.2.0`, source `0b0e2c5cdd67c8b4396a46ea1d1aa72ffb0128d7`, isolated prefix `/tmp/blocked-et-sdk` |
| SDK build | optimized CPU kernels + XNNPACK, PIC, module/named-data-map/tensor/data-loader extensions; tests/examples OFF |
| SDK logging | OFF in this isolated validation build; setup script enables logging; tests inherit the SDK definitions |
| Toolchain | CMake 3.28.3; clang-format 18.1.3; Python codegen PyTorch 2.14.0+cpu |
| Slurm protocol | local client 23.11.4; controller/workers 25.11.2; submissions made on controller over SSH |

These builds use local sibling checkouts and an isolated SDK; they do not
certify a released image or an isolated remote-fetch release. Source fixes
are published on feature branches and release pins are unchanged. The combined
optimized registration archive SHA-256 is
`9c21120daa36e372b9c08ce0e28797481a27376f90794cbb05856fbefe1990bd`.

Artifact paths below are relative to local staging root
`/tmp/blocked-slurm-20261104`; the worker root is
`/home/oli/slurm-blocked-20261104`. Full buffer hashes, original reference
paths and exact command arguments are retained in the evidence JSON/plan.

| Staged artifact | Bytes | SHA-256 |
|---|---:|---|
| `dnn/neuriplo-infer` | 2,477,768 | `6ca4c681008a66c75eef71fa31b64a1c4d7d7b59cddd331b049f11aeac291a3b` |
| `dnn/lib/libneuriplo.so` | 246,696 | `854d676cff93891f25589ffdca1ef2220f25feaa7559076294de0f222e5548dc` |
| `dnn/models/owlv2.onnx` | 618,774,288 | `7f176ee3e1e1bdb7d920c31775e665bdc041699adbbe052e552bc1ad04a2a384` |
| `dnn/models/rtdetr.onnx` | 80,466,628 | `c114037232a26c67f1986505213358ef7bf3628b22714107bd333679b4495547` |
| `dnn/models/rtdetrv4-opencv.onnx` | 41,760,455 | `09d861b58deaa325ec0fe140d88b2869c20a77a8cfb9eb28bc3a78f83a1a044e` |
| `dnn/models/rtdetrv4.onnx` | 41,759,466 | `6b26f17e3de3cea3d5acb9b1eab6a4a40cfe03c8be1986215bba299220160590` |
| `executorch/neuriplo-infer` | 2,461,384 | `85ff3c209d565b8467468dba36617fd46492dca047a2b8882bf9d02e1776dfeb` |
| `executorch/lib/libneuriplo.so` | 13,976,952 | `51d079ffa46dfb120affe455b74f2c577798a1f7398d1e6f04433f1629bac6b3` |
| `executorch/models/rfdetr-keypoint.pte` | 145,583,624 | `7cb8c7279d59b0a4a7a9a65ad06018da6c57da69f3996c6a121661ced1408882` |
| `executorch/models/rfdetr-nano.pte` | 111,490,668 | `9181bd2d2b82494effcc23e8eee0d1223d208984c178b46ccc3ffbb7af57a7d7` |
| `executorch/models/rfdetrseg-medium.pte` | 132,500,332 | `79313c4c79b24d18728b7f79ad1b7f7beb409d5852b54dd2737482f60838a3aa` |

| Paired ONNX reference | SHA-256 |
|---|---|
| `rfdetr-nano.onnx` | `24eab284d5dffcb176219b8c386fdfc51450277145c8b95e15aeb6b26d4736ce` |
| `rfdetrseg-medium.onnx` | `8923e09dfdf12f474d9d679d8894d1385705ace45f4f7207798d42e2c99abe55` |
| `rfdetr-keypoint.onnx` | `daf251cb94d5001aeb08380c48986b21e12db616f41a3182ee0030eda2f0527b` |

| Source image | SHA-256 |
|---|---|
| `data/dog.jpg` | `5a9522051c3cec2bbd2f6323fccba32e8fbf3ddcc2b3e2fd46b04c720bc6f866` |
| `data/person.jpg` | `cdcbab947e46110fc2b77784ac54ddbbab2640f1e44cb5e91fc8984a9793a7d1` |

#### Slurm status and remaining gate

Accounting (`sacct`) is disabled. Cluster memory is advertised as a 1 MiB
placeholder; its allocation record is not a real memory measurement. This
availability snapshot is from the recorded attempt, not a fresh cluster probe.

The pending six-case plan assigns DNN to `optiplex-3040` and ExecuTorch to
`zenbook-ux425uaz`. Each case requests two CPUs, a 20-minute allocation, a
300-second command timeout and an 8192 MiB address-space limit. Both thread
variables `OMP_NUM_THREADS` and `OPENBLAS_NUM_THREADS` are 2. The address-space
limit includes mapped files and is not an RSS measurement or a Slurm memory
accounting result. Manifest SHA-256:
`1a19693fb882b35cd049a67509fd402506d60d724c7c25dc6395a345eed27765`.

Once workers return, synchronize the repaired capsule, person-image
fixture, runner and manifest as described in the validation packet. Retain
each job ID, node, start/end, exit code, runtime log and artifact hashes;
check rendered task results and verify a real cross-backend interval overlap.
Until then, repaired Slurm coverage is **PENDING_WORKER_AVAILABILITY** and
`parallel_overlap_verified=false`. No new live KServe gRPC, TensorFlow 2.21,
GPU or released-image result is implied.

### <a id="ev-migraphx"></a>E16 — MIGraphX coverage status

- **Upstream contract suite: 6/6 PASS (recorded 2026-10-29).** The retained
  `neuriplo` source snapshot `06b6903` records three CPU-reference and three
  MI300X `gfx942` GPU cases, with zero skips. They validate mixed
  FP32/INT32/INT64 input values, INT64 output values, host/device copies,
  model input metadata, and invalid buffer/count rejection. See the
  [recorded validation](https://github.com/olibartfast/neuriplo/blob/06b69034750a298f3fe62fc30d145bca9e59b107/specs/2026-10-29-runtime-input-contracts/validation.md).
- **App task cells: `P`, from the backend contract suite.** The retained Slurm
  campaigns and local results contain no MIGraphX app executable gate,
  learned-model output, or inspected render. Build logs that list `MIGRAPHX`
  among available backends are backend inventory, not evidence that MIGraphX
  was built or selected. The synthetic suite above therefore establishes
  backend datatype-contract coverage only; it does not validate this app's
  release pins or any task representative.
- **Artifact scope:** the independent backend consumes
  [ONNX files](https://github.com/olibartfast/neuriplo/blob/06b69034750a298f3fe62fc30d145bca9e59b107/docs/DEPENDENCY_MANAGEMENT.md#migraphx). The
  ONNX Runtime MIGraphX execution provider is a separate configuration and
  has no recorded gate here.

### <a id="ev-keypoint-matching"></a>E17 — Keypoint matching application coverage

**ORT CPU: PASS¶. TensorRT FP16: PASS¶.** Both backends execute the
`neuriplo-infer` CLI with task preprocessing, backend inference, task decoding,
and app rendering of the two original images. The confidence filter is
strictly **score > 0.2**.

| Pair | ORT CPU matches | TRT FP16 matches |
| --- | --- | --- |
| United States Capitol | 77 | 74 |
| Sacré-Cœur | 45 | 49 |

- **Model:** `superpoint_lightglue_pipeline.onnx`, LightGlue-ONNX v2.0;
  SHA256 `228994cea8c010146fa2aef933baa3ffaa4bcdc522bc8aa560087fcff8134526`.
  Input `images`: FP32 `[2,1,512,512]`. The upstream task owns resize,
  grayscale conversion, pair packing and per-image coordinate scaling.
- **Dispatch:** ORT runs on `zenbook-ux425uaz`, Slurm job 66; TRT runs on
  Jetson Orin through the Slurm 25.11.2 Docker worker, final integration job 67. The worker
  launches the native ARM application over host SSH with the installed CUDA
  and TensorRT libraries. Engine build provenance: job 56, TensorRT 10.16.2,
  FP16, SHA256 `7dfc23f4dd0ead6aba7fd74473073efdeaf36efd174caf459c2480af06b78d41`.
- **¶ Source scope:** application LightGlue integration, neuriplo-tasks `f867d5a`,
  and neuriplo `07ebeae` (the TRT patch rebased onto current history).
  These PASS cells certify the tested source builds; the current release
  pins do not include all these changes. FP16 results are not numerically
  identical to ORT.
- **Evidence and reproduction:** see the
  [application validation](../specs/2026-11-07-lightglue-app-e2e/validation.md)
  and [CLI guide](KeypointMatching.md). Both image pairs are in `data/`.

## Environments

How each column was built and tested. App gates are CPU except TRT;
the upstream MIGraphX contract suite has separate CPU and MI300X GPU results.
Rows without an E15 qualification use the 2026-11-01/02 pinned-stack
evidence (neuriplo-infer 0.11.7-dev + neuriplo 0.11.3 + tasks 0.8.4).
DNN and ExecT include the newer 2026-11-04 local E15 builds; their fixes
are not released dependency pins.

| Backend | Version / artifact | Build flags | Tests | Notes |
|---|---|---|---|---|
| DNN | OpenCV 5.0.0 source-built (`~/dependencies/opencv-5.0.0`, `scripts/setup_opencv.sh`) | `OPENCV_DNN` | app 191/191; backend 80/80 (E15) | Named/typed graph inputs; explicit sizes only for unresolved dynamic dimensions; apt 4.6 legacy compatibility is separate |
| ORT | 1.19.2 CPU (local SDK; pin 1.28.0 unfetchable upstream — `setup_dependencies.sh` handles the renamed `gpu_cuda<major>` archives) | `ONNX_RUNTIME` | 179/179 | Full inference runs, renders inspected |
| TRT | 11.2.1.2 (NGC `tensorrt:26.08` base, `docker/Dockerfile.tensorrt`) | `TENSORRT` | image green in CI 2026-11-02 | Trio live-gated 2026-11-01 (det 109 ms, seg 380 ms + masks, keypoint 292 ms, pose 0.850754) — see E13. 11.2 `trtexec` dropped the `--fp16` shorthand; engines rebuilt from the same `.onnx` |
| Torch | libtorch 2.14 CPU | `LIBTORCH`, `NEURIPLO_INFER_ENABLE_KSERVE=OFF` | 134/134 | TorchScript cells, except E13 where RF-DETR rests on AOTInductor (`.pt2`) because TorchScript tracing is broken for that arch. KServe client unlinked (its gRPC protobuf clashes with libtorch's bundled protobuf); first `cmake` seeds the CUDA cache, a second run is needed (nvcc 12.0 vs driver-reported 13.4). Since neuriplo 0.11.3, TorchScript archives must carry input-datatype metadata; legacy files are rejected |
| OVINO | 2026.3.0 (pinned; `docker/Dockerfile.openvino`) | `OPENVINO`, tests OFF | — (not covered) | Trio live-gated 2026-11-01 (det 784 ms, seg 1184 ms + masks, keypoint 1566 ms, pose 0.850784) — see E13. Native `runUnitTests` link vs `openvino::runtime` fails; CI builds tests OFF too |
| TF | 2.19 (`LIBTENSORFLOW`; 2.21 C++ runtime unloadable — protobuf ABI, see E2) | `LIBTENSORFLOW` | 179/179 | `--weights` accepts directories since 2026-10-23 (was file-only); see E6 |
| LiteRT | `~/dependencies/litert` (LiteRT 2.19.0 pinned) | `LITERT` | 192/192 | Vendored gtest/gmock headers pruned from the SDK (they shadowed system gtest and crashed every death test) |
| ExecT | v1.2.0 optimized SDK `/tmp/blocked-et-sdk` (E15) | `EXECUTORCH`; E15 KServe=ON, gRPC=ON | app 191/191; backend 77/77 | XNNPACK + combined optimized registration; compiled transport is not live transport validation |
| MGX | Recorded upstream environment: 2.12.0.60401-83, ROCm 6.4.1, MI300X `gfx942` (E16) | upstream `MIGRAPHX`; app build unvalidated | backend contracts 6/6: 3 CPU + 3 GPU, zero skips | Synthetic typed fixtures only; no app or learned-model gate |
| LLaMA | llama.cpp b9085 | `LLAMACPP`, tests OFF | — (not covered) | Native `runUnitTests` link misses ggml transitive libs (same class as the OVINO gap) |
| KServe | `neuriplo-kserve-runtime` `real-onnx` / `grpc-onnx` | client in default build | suites run manually | HTTP green incl. per-task gates; gRPC `ModelInfer` 13 INTERNAL server-side (see E1) |

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
   RT-DETR and OWLv2 locally (§); v4 needs its compatible export. GroundingDINO
   has not been re-gated, and EdgeCrafter exact-GELU remains an importer-level
   open issue. Parallel repaired Slurm validation awaits worker availability.
   Still unprobed on 5.x: `ecseg`, torchvision ResNet, `vitpose`,
   `depth_anything_v2`.
4. **KServe per-task gates.** Done for servable local models (`real-onnx`,
   HTTP): pose, depth, open-vocab — plus YOLO26-`litert` PASS and `ecdet`
   PASS with the local backend patch (E4). gRPC transport BLOCKED server-side. Still open: classification,
   video, flow, VLM serving.
5. **Gaussian-splatting representative.** No preset exists. Export one `lgm`
   artifact, add a preset, then gate it.
6. **MIGraphX app gates.** Validate an app build with the independent
   backend and run ONNX task representatives on a supported AMD/ROCm worker.
   Record expected outputs and inspected renders to add app-level evidence on
   top of the contract `P`; the upstream synthetic suite in E16 is already
   covered.

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
| `kserve-runtime-edgecrafter-e2e` | `ecdet` ORT/TRT PASS + OpenVINO/ExecT matrix; `ecdet` LiteRT/KServe HTTP replacement PASS with local backend patch (E4; separate manual gate) | — (needs per-backend runtime binaries + model repository + GPU for TRT) | `run.py --backend {onnx_runtime,tensorrt,openvino,executorch} [--transport {http,grpc}] [--mode {kserve,local}]`; `prepare_model_repository.py` lays out `ecdet_s_<backend>/1/` | [README](../integration-tests/kserve-runtime-edgecrafter-e2e/README.md), [BENCHMARK.md](../integration-tests/kserve-runtime-edgecrafter-e2e/BENCHMARK.md) |
| `rfdetr-pose-local-vs-kserve` | Pose row scaffold for `rfdetr_keypoint` (evidence tracking, not a live gate) | `rfdetr_pose_evidence_scaffold` (`run.py` — baselines + report exist) | Attested checks in [evidence.yaml](../integration-tests/rfdetr-pose-local-vs-kserve/evidence.yaml) (`kserve_grpc_smoke` still `skipped`) | [README](../integration-tests/rfdetr-pose-local-vs-kserve/README.md), [evidence.yaml](../integration-tests/rfdetr-pose-local-vs-kserve/evidence.yaml), [reports/latest.md](../integration-tests/rfdetr-pose-local-vs-kserve/reports/latest.md) |

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

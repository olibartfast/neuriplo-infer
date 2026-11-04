# Backend coverage — task models × backends

One gated representative per task family, one status per backend, no cell
claiming more than its evidence. Style follows the NERT
[coverage matrix](https://github.com/olibartfast/nert/blob/main/specs/2026-10-17-task-models/coverage.md).

**Last updated: 2026-11-04.** Baseline evidence uses neuriplo-infer
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
  `tvm`/`migraphx`/`dali` have no matrix pipeline. Slurm session
  `2026-11-03-backend-dispatch` (jobs 30/31/32, one backend per worker,
  `neuriplo@develop` `06b6903`): `ggml` on zenbook BUILD_OK + `ctest`
  77/77, `openvino` on optiplex-3040 BUILD_OK, `executorch` on
  jetson-nano (aarch64) BUILD_OK — the ExecT baseline before the
  optimized-kernel rebuild. New
  historical blockers recorded there: OpenCV DNN single-input
  limit (DETR two-input, OWL three-input, `groundingdino` four-input;
  local RT-DETR/OWLv2 fixes are recorded in E15),
  OpenCV DNN importer SIGSEGV on EdgeCrafter exact-GELU graphs, OpenVINO
  CPU SIGSEGV on 5D video inference, and NERT/OpenVINO setup-contract
  rejections plus CPU-timeout video cells. The later OpenVINO retest below
  succeeds on a different CPU environment; the historical failures remain
  scoped to their original environments.
- **OpenVINO Slurm retest (2026-11-04, P-LNX-EDGE-CPU):** only the seven
  non-passing, runnable cells were retried; 28 prior runtime passes were
  skipped. `owlv2` / `owlvit` passed inference and render inspection;
  `videomae` / `vivit` / `timesformer` passed inference without video-file
  export. The two existing KNOWN-BAD decisions were retained. This uses
  a derived validation image with missing runtime libraries added, not a
  fresh source build or a pass of the original image; see [E14](#ev-ov-slurm).

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

The OVINO cells marked **†** use the retained 0.11.6 campaign executable
in a derived validation image on P-LNX-EDGE-CPU. They do not certify the
current source pins or the original image packaging; see [E14](#ev-ov-slurm).

The `Torch` column covers TorchScript. AOTInductor is a second runtime inside
the same LibTorch backend, selected when the model path ends in `.pt2`. It is
not covered by these cells yet — except E13, where RF-DETR's Torch PASS rests
on AOTInductor because TorchScript tracing is broken for that architecture
(see E13); do not read any other Torch cell as AOTInductor evidence.

| Task (representative) | DNN | ORT | TRT | Torch | OVINO | TF | LiteRT | ExecT | LLaMA | KServe | Evidence |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Object detection (`yolo26`, `yolo26s_tflite`) | PASS | PASS | PASS | PASS | PASS | BUILDS | PASS | UNTESTED | N/A | PASS | [E1](#ev-yolo26) |
| Object detection (`yolov8`, `yolov8_executorch`) | PASS | DRY-RUN | BUILDS | BUILDS | BUILDS | PASS | PASS | PASS | N/A | UNTESTED | [E2](#ev-yolov8) |
| Object detection (`rtdetr`, `rtdetrv4`) | PASS§ | DRY-RUN | DRY-RUN | PASS | BUILDS | BUILDS | UNTESTED | UNTESTED | N/A | UNTESTED | [E3](#ev-rtdetr) |
| Object detection (`rtdetrul`, Ultralytics RT-DETR) | UNTESTED | PASS | UNTESTED | UNTESTED | PASS | UNTESTED | UNTESTED | UNTESTED | N/A | UNTESTED | [E14](#ev-ov-slurm) |
| Object detection (`ecdet`, `edgecrafter_det`) | UNTESTED | PASS | PASS | BUILDS | BUILDS | BUILDS | PASS‡ | UNTESTED | N/A | PASS‡ | [E4](#ev-ecdet) |
| Object detection (`rfdetr`, `rfdetr`) | PASS | PASS | PASS | PASS | PASS | BUILDS | PASS | PASS§ | N/A | PASS | [E13](#ev-rfdetr) |
| Instance segmentation (`yoloseg`, `yoloseg`) | PASS | PASS | PASS | BUILDS | BUILDS | BUILDS | UNTESTED | UNTESTED | N/A | PASS | [E5](#ev-seg) |
| Instance segmentation (`ecseg`, `edgecrafter_seg`) | UNTESTED | DRY-RUN | BUILDS | BUILDS | BUILDS | BUILDS | UNTESTED | UNTESTED | N/A | UNTESTED | [E5](#ev-seg) |
| Instance segmentation (`rfdetrseg`, `rfdetrseg`) | PASS | PASS | PASS | PASS | PASS | BUILDS | PASS | PASS§ | N/A | PASS | [E13](#ev-rfdetr) |
| Classification (`torchvisionclassifier`, `torchvision_classifier`) | UNTESTED | DRY-RUN | BUILDS | PASS | BUILDS | PASS | UNTESTED | UNTESTED | N/A | UNTESTED | [E6](#ev-classification) |
| Video classification (`videomae`, `videomae`) | UNTESTED | DRY-RUN | BUILDS | BUILDS | RUNTIME-PASS† | BUILDS | UNTESTED | UNTESTED | N/A | UNTESTED | [E7](#ev-video) |
| Optical flow (`raft`, `raft`) | UNTESTED | DRY-RUN | BUILDS | BUILDS | BUILDS | BUILDS | UNTESTED | UNTESTED | N/A | UNTESTED | [E7](#ev-video) |
| Pose estimation (`vitpose`, `vitpose`) | PASS | DRY-RUN | BUILDS | BUILDS | BUILDS | BUILDS | PASS | UNTESTED | N/A | PASS | [E8](#ev-pose) |
| Pose estimation (`rfdetrpose`, `rfdetr_keypoint`) | PASS | PASS | PASS | PASS | PASS | BUILDS | PASS | PASS§ | N/A | PASS | [E13](#ev-rfdetr) |
| Depth estimation (`depth_anything_v2`, `depth_anything_v2`) | PASS | PASS | BUILDS | BUILDS | PASS | BUILDS | PASS | UNTESTED | N/A | PASS | [E9](#ev-depth) |
| Open-vocabulary detection (`owlv2`, `owlv2`) | PASS§ | PASS | BUILDS | BUILDS | PASS† | BUILDS | UNTESTED | UNTESTED | N/A | PASS | [E10](#ev-owlv2) |
| Image understanding (`gemma4`, `gemma4`) | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | PASS | UNTESTED | [E11](#ev-gemma4) |
| Gaussian splatting (`lgm`/`grm`, no preset) | UNTESTED | UNTESTED | UNTESTED | BUILDS | BUILDS | UNTESTED | UNTESTED | UNTESTED | N/A | UNTESTED | [E12](#ev-splatting) |

‡ EdgeCrafter LiteRT and KServe HTTP PASS require the reconverted artifact
and the unreleased neuriplo XNNPACK subgraph-reshaping fix
[`9bde6c0`](https://github.com/olibartfast/neuriplo/commit/9bde6c0fd07ab88b70e1076f5fc667c20be41d7d).
The app's released dependency pin does not contain that fix.

§ Local 2026-11-04 fixes, not released pins: OpenCV 5 named/typed graph
inputs, RT-DETRv4's prepared vector-MatMul export, and ExecuTorch v1.2.0
rebuilt with optimized registrations. Identical-input ORT parity and renders
are validated locally. Repaired parallel Slurm reruns are **pending controller
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
labeled as such. The seg-LiteRT flip used tasks `v0.8.5` tag content. Later
unreleased corrections have explicit ‡/§ scope notes and do not certify
the app's released dependency pins.

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
- **Torch PASS (pinned re-gate 2026-11-03).** The typed resave of
  `yolo26s.torchscript` (input-datatype metadata verified with
  `has_torchscript_metadata`) loads and runs: 1242 ms,
  `--input_sizes=1,3,640,640` — bicycle 0.95, dog 0.94, car 0.52,
  exact match with DNN/ORT. Render inspected
  (`data/output/processed_yolo26_local.png`). Cell flipped
  BLOCKED → PASS.
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
- **LiteRT PASS** (2026-11-02, `build-litert`, CPU, 471 ms):
  `yolov8n` float32 `.tflite` via `onnx2tf` @ 0.4/0.5 — bicycle 0.89,
  dog 0.93, car 0.61, identical to DNN/ORT. Render inspected
  (`data/output/processed_yolov8_local.png`).
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
  bicycle 0.80, car 0.45, masks verified. The 4.6 `Unsqueeze` blocker is
  gone on 5.x.
- **ORT PASS** (CPU, 464 ms): @ 0.4/0.5 — dog 0.87, bicycle 0.80, car 0.45; masks verified; rendered `data/output/processed_yoloseg_local.png`.
- **TRT PASS** (10.14.1.48 FP16 engine from `yolov8n-seg.onnx`, 430 ms GPU): masks + boxes rendered, identical output shapes (1x116x8400 + 1x32x160x160).
- **KServe PASS** (`real-onnx` runtime, HTTP, model `seg` v1): exit 0, `neuriplo_http_infer_requests_success_total{model="seg",version="1"} 1`, failure 0; render `data/output/processed_yoloseg_kserve-ort.png` (>10 KB).
- `ecseg`: preset + emulation green; no live numbers recorded (unlike `ecdet`).

### <a id="ev-classification"></a>E6 — Classification (`torchvisionclassifier`)

- **Torch PASS (pinned re-gate 2026-11-03).** The typed resave of the
  `resnet50` (`torchvision` IMAGENET1K_V2) archive (metadata verified with
  `has_torchscript_metadata`)
  loads and runs: 687 ms, `--input_sizes=1,3,224,224` — top class 249 @
  0.44 (same score as the pre-contract gate; proper preprocessing gives
  0.55, so the app seam under-reports but ranks correctly). Render
  inspected (`data/output/processed_torchvision-classifier_local.png`).
  Cell flipped BLOCKED → PASS. TensorFlow was re-gated separately below.
- **TF PASS (re-gate 2026-11-04, CPU, TF 2.19 C++ runtime).** Re-exporting
  pretrained Keras ResNet50 with the corrected neuriplo-tasks exporter gives
  class **249 @ 0.532603**, versus Python TF 2.21 **249 @ 0.532377** on
  `data/dog.jpg`; all five top classes agree. CLI inference completed in
  2850 ms and rendered `processed_tensorflow-classifier_local.png`.
  The task sends raw BGR Float32 pixels; the SavedModel subtracts BGR means
  and emits logits, so the task applies softmax once. The fix is upstream
  commit `75cc7ac`, already included in the pinned `neuriplo-tasks v0.8.5`.
  Old unwrapped SavedModels retain the preprocessing/softmax mismatch and
  must be re-exported using the
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
as alias (see Open item 2 for the emulation gap).

- **DNN PASS via alias** (5.0.0, CPU, 3582 ms): `rfdetr_keypoint` renders
  skeleton, pose score 0.85. (Apt 4.6 failed this model on `BOOL` dtype —
  gone on 5.x.) `vitpose` itself unrun.
- **ORT DRY-RUN + alias live** (CPU, 2817 ms): `rfdetr_keypoint` on
  `person.jpg` renders skeleton, pose score 0.85.
- **TRT alias live** (10.14.1.48 FP16 engine, 246 ms): same skeleton,
  score 0.85. Cell stays BUILDS for the unrun representative.
- **LiteRT alias live** (CPU, 6540 ms): `rfdetr_keypoint` float32 `.tflite`
  via `onnx2tf` renders skeleton, pose score 0.850784 — bit-identical to
  every other backend. Render inspected
  (`data/output/processed_rfdetr-keypoint_local.png`). Re-verified
  2026-11-02 on the develop stack (6898 ms, same 0.850784) — the tasks
  mapping fix causes no regression here.
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
- **Seg Torch PASS (pinned re-gate 2026-11-03).** `rfdetrseg-medium` `.pt2`
  (432, `--input_sizes=1,3,432,432`, 1370 ms, tasks `v0.8.5` tag content,
  `build-libtorch`): bicycle 0.94, horse 0.91, car 0.86, all with masks,
  plus the usual `motorbike 0.57` artifact — exact trio match with
  ORT/TRT/OVINO. The PR #22 shape-first fix resolves the old empty
  render. Detection regression clean on the same stack
  (`rfdetr-medium` 1004 ms: bicycle 0.92, horse 0.94, car 0.87).
  Renders inspected (`data/output/processed_rfdetrseg_local.png`,
  `data/output/processed_rfdetr_local.png`). Cell flipped
  KNOWN-BAD → PASS.
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
- **LiteRT detection: PASS via upstream's native exporter; legacy generic
  `onnx2tf` artifacts remain unsupported.** `rfdetr-medium` through rfdetr's own `format="litert"`
  route (`torch.export` + `litert-torch`, no ONNX/TF step; 576,
  `--input_sizes=1,3,576,576`, 3207 ms): bicycle 0.92, horse 0.94,
  car 0.87 at 0.4 — exact match with every other backend (same dog-as-horse
  note). Render inspected (`data/output/processed_rfdetr_local.png`).
  Upstream lists LiteRT as supported and it is — provided their exporter
  is used: it rewrites the deformable-attention `grid_sample` ops that
  generic `onnx2tf` mangles. The generic path stays KNOWN-BAD
  (`rfdetr-nano` float32 `.tflite`: single wrong `person 0.75`, no masks).
- **LiteRT seg: native route PASS (pinned tasks `v0.8.5`); legacy
  generic-`onnx2tf` artifacts remain unsupported.** Generic float32 (432):
  exit 0, masks render, bicycle 0.94 / horse 0.91 / `motorbike 0.57`
  match — but the car originally read `truck 0.34`; with the local backend
  patch it reads car 0.80, still below the native reference: do not serve. Native
  `format="litert"` file gated 2026-11-02 on the develop stack (3670 ms)
  and re-gated 2026-11-03 on the pinned stack (tasks `v0.8.5` tag content,
  `build-litert`, 4151 ms): bicycle 0.94, horse 0.91, car 0.86, all with
  masks, plus the usual `motorbike 0.57` artifact — exact trio match with
  ORT/TRT/OVINO. The pre-fix empty render was the same tasks-side
  name-mapping gap as the Torch cell (positional
  `serving_default_output_<i>_output` names), fixed by PR #22. Render
  inspected (`data/output/processed_rfdetrseg_local.png`).
- **Pose LiteRT PASS (same stack).** `rfdetr-keypoint` float32 `.tflite`
  (576, `--input_sizes=1,3,576,576`, 6540 ms): skeleton, pose score
  **0.850784** — identical to every other backend (upstream notes keypoint
  is unsupported by `litert-torch`, so the `onnx2tf` file stands).
- **ExecuTorch PASS§ (2026-11-04, local rebuilt v1.2.0 SDK).** All
  three original native XNNPACK PTE artifacts execute with the existing
  optimized CPU kernels registered. No runtime version or model weights
  changed. Nano detection, medium segmentation and keypoint raw tensors
  agree with their same-model ORT references; selected detections/masks/pose
  are checked in [E15](#ev-blocked-slurm). Older portable-only SDKs still
  lack `aten::linear.out` and must be rebuilt. Explicit portable delegate
  mode retains portable kernels and is not this gate.

- **Supported export recipes fixed (2026-11-04).** Both `rfdetr` and
  `rfdetrseg` app presets now select native `format="litert"` with
  `--backend litert` (RF-DETR 1.11.2), producing the exact file consumed by
  inference. They do not route through ONNX conversion. Local C++ re-gates
  against ONNX agree on detections; RF-DETR seg raw mask signs agree on
  **99.999914%** of elements. The old generic nano detection file still
  renders a wrong person; the old generic seg file still drifts on the car.
  Replace those artifacts rather than treating a backend patch as a repair
  for every conversion. See the
  [native export helper](../scripts/export_rfdetr_litert.py) and
  [validation](../specs/2026-11-04-known-bad-backends/validation.md).

### <a id="ev-ov-slurm"></a>E14 — OpenVINO Slurm retest of non-passing cells

Recorded 2026-11-04 on profile `P-LNX-EDGE-CPU`, jobs 34 and 35. Models ran
sequentially with two container CPUs, 3 GiB memory and a 300-second timeout
per model. All input files were checked against a SHA-256 manifest.
The [tracked evidence summary](evidence/openvino-slurm-2026-11-04.json)
records image and binary hashes, fixture hashes, arguments, metrics and
artifact hashes. Single-run timings are not benchmark claims.

| Model | Previous cell | Archived Slurm result | Slurm job | Latest local follow-up |
|---|---|---|---|---|
| `owlv2` | FAILED | PASS (runtime + inspected render) | 34 | — |
| `owlvit` | FAILED | PASS (runtime + inspected render) | 34 | — |
| `videomae` | FAILED | RUNTIME-PASS (no exported video) | 35 | — |
| `vivit` | FAILED | RUNTIME-PASS (no exported video) | 35 | — |
| `timesformer` | FAILED | RUNTIME-PASS (no exported video) | 35 | — |
| `tensorflow-classifier` | KNOWN-BAD | KNOWN-BAD retained | 34 | LIBTENSORFLOW PASS (E6); OpenVINO not re-gated |
| `rtdetrul` | KNOWN-BAD | KNOWN-BAD retained | 34 | OpenVINO PASS (same ONNX, current local stack) |

Both models labeled KNOWN-BAD in that historical run completed inference,
but their prior correctness decisions were not re-evaluated. These model-specific decisions do not
change the representative `torchvisionclassifier` or `rtdetr` cells.
A fresh 2026-11-04 C++ re-gate of the same `rtdetrul.onnx` now agrees
between ORT and OpenVINO: bicycle 0.962947, dog 0.942013, truck 0.73777,
car 0.55753; maximum raw output difference 1.06e-5. The current decoder
already handles normalized center/size boxes with explicit class IDs. The
historical job-34 label is retained above as an archived result, not a
current failure. This does not claim correctness for every Ultralytics export.

Twenty-eight prior runtime passes, seven missing exports and one
non-applicable cell were skipped.

The retained image `neuriplo-infer:openvino-2026-11-01-full-coverage-1`
could not launch its executable because OpenCV and glog shared libraries
were missing. A derived image added `libopencv-highgui406`,
`libopencv-imgcodecs406`, `libopencv-videoio406` and `libgoogle-glog0v6t64`.
The executable SHA-256 remained
`a31997f20450056a8c0c2378a693db8d33d0284542829262117e1eec311a1a2c`;
the derived image ID is
`sha256:cab51feff475b76d284012924d093fd7948489a97e68753d7b3e86a581f0fcd7`.
Source-image filesystem layers were verified after import. This proves
execution in the derived environment, not release-image readiness or a
new source build. Packaging the missing runtime libraries remains open.

Job 34 first rejected all three video commands because `--output_video`
requires a VideoWriter-enabled build. Job 35 retried only those models
without file export; weights, clips, frame counts and inference parameters
were preserved. Job 35 completed with Slurm exit `0:0`. The historical
video SIGSEGV and OWL setup failures in the earlier environment are not
claimed fixed across all hardware by these successful runs.

### <a id="ev-blocked-slurm"></a>E15 — Blocked cells: local fixes, Slurm rerun pending

The five formerly BLOCKED representative cells now have local passing routes
(§), covering six model/backend checks. The
[evidence JSON](evidence/blocked-slurm-2026-11-04.json) records exact binaries,
libraries, models, SDK source and identical-input numerical comparisons. The backend fix is
[neuriplo 042bcef](https://github.com/olibartfast/neuriplo/commit/042bcefc8a20771d7c40d54c86f8ea588fcde1b8);
the export adapter is
[neuriplo-tasks d38e8b3](https://github.com/olibartfast/neuriplo-tasks/commit/d38e8b3984386c93a053215a1a88964e6e054e74).
The [pending Slurm plan](evidence/blocked-slurm-plan-2026-11-04.json) contains
all six exact commands and worker selections.

| Backend / model | Local result versus same-model ORT | Required fix |
|---|---|---|
| OpenCV 5 / RT-DETR | labels exact; max scores 1.84e-6, boxes 0.003861 pixels | named FP32/INT64 inputs and integer outputs |
| OpenCV 5 / RT-DETRv4 | detections agree; complete errors in JSON | same adapter plus prepared Unsqueeze/MatMul/Squeeze export |
| OpenCV 5 / OWLv2 | max logits 0.001067, boxes 0.000134 normalized | image, token IDs and attention mask bound with graph metadata |
| ExecuTorch / RF-DETR nano | max boxes 0.000067, logits 0.000303 | v1.2.0 optimized CPU registration with XNNPACK |
| ExecuTorch / RF-DETR seg medium | max boxes 0.000503, logits 0.007473; mask sign agreement 99.9997428% | same SDK rebuild |
| ExecuTorch / RF-DETR keypoint | person.jpg max keypoints 0.000645; pose score 0.850784 | same SDK rebuild |

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

**Slurm:** existing-controller jobs 39 and 40 reproduced the DNN input and
ExecuTorch operator failures. They ran sequentially and are not parallel
validation evidence. The controller then went offline before repaired
submissions; no repaired job IDs or concurrent PASS are claimed. Accounting
is disabled, so baseline states were captured with `scontrol`. The
[Slurm runner guide](SlurmBackendCoverage.md) and
[validation packet](../specs/2026-11-04-blocked-slurm-backends/validation.md)
provide the staged manifest and resume commands. Two workers are selected
for the repaired checks to permit overlap. Release pins, KServe gRPC,
TensorFlow 2.21 ABI and historical RF-DETR TorchScript limitations are unchanged.

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
| PASS (≥1 backend live-gated) | 14 | `yolo26`, `yolov8`, `rtdetr`, `rtdetrul`, `ecdet`, `rfdetr`, `owlv2`, `gemma4`, `yoloseg`, `rfdetrseg`, `torchvisionclassifier`, pose (`vitpose` via `rfdetr_keypoint` alias, `rfdetrpose`), depth (`yolo26depth` alias) |
| Legacy KNOWN-BAD artifacts (supported replacements available) | 2 | `rfdetrseg` on LiteRT-`onnx2tf` (converter fault, stands; native route PASS); generic-`onnx2tf` `rfdetr` (native-exporter route is PASS) |
| RUNTIME-PASS best | 1 | `videomae` (OVINO inference; video export unvalidated) |
| DRY-RUN best | 2 | `ecseg`, `raft` |
| BUILDS / UNTESTED best | 1 | Gaussian splatting (no preset; Torch/OVINO BUILDS-only) |
| **Total** | **18** | 10 task families; detection/segmentation/pose split by representative |

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
4. **OPENCV_DNN scope (5.0.0).** Named typed input binding resolves
   RT-DETR and OWLv2 locally (§); v4 needs its compatible export. GroundingDINO
   has not been re-gated, and EdgeCrafter exact-GELU remains a separate importer
   SIGSEGV. Parallel repaired Slurm validation awaits the existing controller.
   Still unprobed on 5.x: `ecseg`, torchvision ResNet, `vitpose`,
   `depth_anything_v2`.
5. **KServe per-task gates.** Done for servable local models (`real-onnx`,
   HTTP): pose, depth, open-vocab — plus YOLO26-`litert` PASS and `ecdet`
   PASS with the local backend patch (E4). gRPC transport BLOCKED server-side. Still open: classification,
   video, flow, VLM serving.
6. **Gaussian-splatting representative.** No preset exists. Export one `lgm`
   artifact, add a preset, then gate it.
7. **LIBTENSORFLOW classification: resolved by re-export (2026-11-04).**
   The corrected upstream ResNet50 export embeds mean subtraction and returns
   logits. Local C++ and Python correctness now agree (E6). Old SavedModels
   require re-export; KServe classification remains unvalidated.
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
9. **Closed — TorchScript re-exports re-gated.** `yolo26` and
   `torchvisionclassifier` now load typed archives (metadata verified)
   and were re-gated PASS 2026-11-03 on the pinned stack (yolo26
   1242 ms exact trio; classifier top class 249 @ 0.44); the DETR family
   was re-gated 2026-10-30. E13's TorchScript exception (broken trace
   for RF-DETR, AOTI route instead) stands.

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

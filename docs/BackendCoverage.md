# Backend coverage

Tested model and backend combinations in `neuriplo-infer`.
**Results last updated: 2026-11-25.** Some results use local, unreleased
fixes; see [tested environments](#environments) and [known limitations](#known-limitations).

Each binary uses the local backend selected at build time (`DEFAULT_BACKEND`).
Use KServe to run models on a remote server. Setup instructions are in
[Deployment](Deployment.md#backends).

## Reading the tables

| Status | Meaning |
| --- | --- |
| **P** — Pass | Inference ran successfully and the expected output was checked. |
| **RUNTIME-PASS** | Inference ran, but output correctness or rendering still needs checking. |
| **DR** — Dry run | The example command passed automated checks; inference was not run. |
| **B** — Builds | The backend builds in CI; this task has not been tested on it. |
| **U** — Untested | No validated result for this combination. |
| **N/A** | The backend does not support this kind of task. |

A passing result applies to the tested model and environment, not every model
in the family. Some tests use RT-DETRv4-s, YOLO11m-seg, or ResNet-50 instead
of the example preset shown.

<a id="gate-table"></a>

## Coverage by task

The tables show the same tasks across two groups of backends. Parentheses
contain the model type and, when different, the example preset.

- **NERT**: Neuriplo Engine Runtime.
- **DNN**: OpenCV DNN; **ORT**: ONNX Runtime; **TRT**: TensorRT.
- **TRC**: LibTorch; **OV**: OpenVINO; **TF**: LibTensorFlow.
- **MGX**: the standalone MIGraphX backend (separate from ONNX Runtime's MIGraphX provider).
- **GGML**: the standalone GGML backend; **LLaMA**: llama.cpp (`LLAMACPP`).
- **ExecT**: ExecuTorch; **KServe**: remote inference.

### Local inference backends

| Task (representative) | NERT | DNN | ORT | TRT | TRC | OV | TF |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Object detection (`yolo26`, `yolo26s_tflite`) | P | P | P | P | P | P | B |
| Object detection (`yolov8`, `yolov8_executorch`) | P | P | DR | B | B | B | P |
| Object detection (`rtdetr`, `rtdetrv4`) | P | P | DR | DR | P | P | B |
| Object detection (`rtdetrul`, Ultralytics RT-DETR) | U | U | P | U | U | P | U |
| Object detection (`ecdet`, `edgecrafter_det`) | P | U | P | P | B | B | B |
| Object detection (`rfdetr`, `rfdetr-medium`) | P | P | P | P | P | P | B |
| Instance segmentation (`yoloseg`) | P | P | P | P | B | B | B |
| Instance segmentation (`ecseg`, `edgecrafter_seg`) | P | U | DR | B | B | B | B |
| Instance segmentation (`rfdetrseg`, `rfdetrseg-medium`) | P | P | P | P | P | P | B |
| Classification (`torchvisionclassifier`, `torchvision_classifier`) | P | U | DR | B | P | B | P |
| Video classification (`videomae`) | P | U | DR | B | B | RUNTIME-PASS | B |
| Optical flow (`raft`) | P | U | DR | B | B | B | B |
| Pose estimation (`vitpose`) | P | P | DR | B | B | B | B |
| Pose estimation (`rfdetrpose`, `rfdetr_keypoint`) | P | P | P | P | P | P | B |
| Depth estimation (`depth_anything_v2`) | P | P | P | B | B | P | B |
| Open-vocabulary detection (`owlv2`) | P | P | P | B | B | P | B |
| Image understanding (`gemma4`) | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Gaussian splatting (`lgm`/`grm`, no preset) | N/A | U | U | U | B | B | U |
| Keypoint matching (`SuperPoint + LightGlue`, no app preset) | U | U | P | P | U | U | U |

### Additional runtimes and remote execution

| Task (representative) | MGX | GGML | LiteRT | ExecT | LLaMA | KServe |
| --- | --- | --- | --- | --- | --- | --- |
| Object detection (`yolo26`, `yolo26s_tflite`) | U | U | P | U | N/A | P |
| Object detection (`yolov8`, `yolov8_executorch`) | U | P | P | P | N/A | U |
| Object detection (`yolo11`) | U | P | U | U | N/A | U |
| Object detection (`rtdetr`, `rtdetrv4`) | U | U | U | U | N/A | U |
| Object detection (`rtdetrul`, Ultralytics RT-DETR) | U | U | U | U | N/A | U |
| Object detection (`ecdet`, `edgecrafter_det`) | U | U | P | U | N/A | P |
| Object detection (`rfdetr`, `rfdetr-medium`) | U | P | P | P | N/A | P |
| Instance segmentation (`yoloseg`) | U | P | U | U | N/A | P |
| Instance segmentation (`yolo11seg`) | U | P | U | U | N/A | U |
| Instance segmentation (`ecseg`, `edgecrafter_seg`) | U | U | U | U | N/A | U |
| Instance segmentation (`rfdetrseg`, `rfdetrseg-medium`) | U | P | P | P | N/A | P |
| Classification (`torchvisionclassifier`, `torchvision_classifier`) | U | P | U | U | N/A | U |
| Video classification (`videomae`) | U | U | U | U | N/A | U |
| Optical flow (`raft`) | U | U | U | U | N/A | U |
| Pose estimation (`vitpose`) | U | U | P | U | N/A | P |
| Pose estimation (`yolo11pose`) | U | P | U | U | N/A | U |
| Pose estimation (`rfdetrpose`, `rfdetr_keypoint`) | U | U | P | P | N/A | P |
| Depth estimation (`depth_anything_v2`) | U | U | P | U | N/A | P |
| Open-vocabulary detection (`owlv2`) | U | U | U | U | N/A | P |
| Image understanding (`gemma4`) | N/A | U | N/A | N/A | P | U |
| Gaussian splatting (`lgm`/`grm`, no preset) | U | U | U | U | N/A | U |
| Keypoint matching (`SuperPoint + LightGlue`, no app preset) | U | U | U | U | N/A | U |

## Environments

Most app tests ran on CPU; TensorRT tests used a GPU. The recorded baseline
uses neuriplo-infer 0.11.6, neuriplo 0.11.3, and neuriplo-tasks 0.8.5.
The November 1–2 runs used neuriplo-infer 0.11.7-dev, neuriplo 0.11.3,
and neuriplo-tasks 0.8.4. OpenCV DNN and ExecuTorch include November 4
local builds with unreleased fixes. GGML also uses unreleased code. Its loader
accepts `.gguf` weights only, written by the sibling `convert_onnx_to_gguf.py`;
`scripts/ggml_app_gate.py` defaults to that converter.

Test counts below are from those runs, not fresh results for the current branch.

| Backend | Tested version | Recorded checks | Notes |
| --- | --- | --- | --- |
| DNN | OpenCV 5.0.0, built from source | App 191/191; backend 80/80 | Coverage is for 5.x. Distribution-provided 4.6 cannot load the current test models. |
| ORT | 1.19.2 CPU | 179/179 | Used a local SDK because the pinned 1.28.0 download was unavailable. Rendered outputs checked. |
| TRT | 11.2.1.2, NGC `tensorrt:26.08` | CI image build passed; detection, segmentation, and keypoint runs passed | Engines rebuilt from ONNX; `trtexec` 11.2 no longer accepts `--fp16`. |
| TRC | LibTorch 2.14 CPU | 134/134 | RF-DETR uses AOTInductor (`.pt2`); other results use TorchScript. KServe disabled because of a protobuf conflict. |
| OV | OpenVINO 2026.3.0 | Live inference passed; unit tests unavailable | Unit-test executable fails to link; CI also builds with tests disabled. `rtdetr` gated 2026-11-25 on the x86_64 OpenVINO worker (exit 0, render-checked dog/bicycle/car) — see `docs/evidence/ov-rtdetr-2026-11-25.json`. |
| TF | TensorFlow 2.19 | 179/179 | The 2.21 C++ runtime failed to load because of a protobuf ABI conflict. |
| LiteRT | 2.19.0 | 192/192 | SDK's bundled gtest/gmock headers removed to avoid test crashes. EdgeCrafter requires an unreleased backend fix. |
| ExecT | ExecuTorch 1.2.0, optimized SDK | App 191/191; backend 77/77 | XNNPACK enabled. KServe and gRPC compiled, but live transport was not tested in this build. |
| MGX | MIGraphX 2.12.0.60401-83, ROCm 6.4.1, MI300X | Backend 6/6: 3 CPU + 3 GPU | Synthetic backend tests only; no app build or trained model tested. |
| GGML | Local CPU, AMD Ryzen 7 5700U; unreleased neuriplo changes | App 197/197; backend GGMLInferTest 35 passed / 8 skipped / 0 failed (Yolo/Ort50 parity SKIPs without their `/tmp` artifacts) and GGMLOpsParity 13/13 (including GridSample, tie-aware SoftmaxRows, sine-embed, TopK tie-break, bilinear Resize, exact depthwise-conv3x3, Gguf* parser/loader fixtures); the 77/77 upstream CTest predates strict-GGUF loading (ONNX-frontend and graph-executor suites removed) | ResNet-50 top-5 matches ORT exactly; YOLOv8n detection and segmentation match within 0.01 on `.gguf` with identical detections (V-16/V-17/V-18). RF-DETR detection gated 2026-11-16 on `.gguf` with detections identical to ORT (V-21: bicycle 0.94 / horse 0.92 / car 0.89 / motorbike 0.60). RF-DETR-seg gated 2026-11-16 on `.gguf` with ORT-identical instances and mask IoU 1.0 on every kept detection (V-23: bicycle 0.94 / horse 0.82 / car 0.61 / motorbike 0.59 + masks). No CI build job. |
| LLaMA | llama.cpp b9085 | Unit tests unavailable | Unit-test executable fails to link because of missing ggml libraries. |
| KServe | `neuriplo-kserve-runtime`, `real-onnx` / `grpc-onnx` | Manual HTTP tests passed | gRPC `ModelInfer` returns server error `13 INTERNAL`. |

GGML results used unreleased neuriplo code: strict-GGUF-only loading is
committed, and sibling branch `feature/ggml-rfdetr` (past `0aee3ea`, plus
uncommitted decoder/head/seg work) adds parser hardening, native
TopK/GridSample/softmax-rows/sine-embed/bilinear-resize/depthwise-conv3x3
helpers, the converter RF-DETR head rule, the complete RF-DETR detection
loader (S-1 through S-3 plus the V-21 app gate; the `rfdetr` cell is
promoted), and the complete RF-DETR-seg loader (S-4: 12-block backbone,
4-layer decoder, mask/proto branch plus the V-23 app gate; the `rfdetrseg`
cell is promoted). These fixes are not included
in the released dependency pins. Classification, `yolov8`, and `yoloseg`
are gated on `.gguf` (strict-GGUF-only loader) with identical detections —
see `specs/2026-11-13-ggml-matrix-coverage`
V-16/V-17/V-18.

## Known limitations

- **TorchScript:** neuriplo 0.11.3 and later require input datatype metadata.
  Re-export older files with the
  [metadata helper](https://github.com/olibartfast/neuriplo/blob/master/scripts/lib/torchscript_metadata.py).
  RF-DETR's TorchScript tracing remains broken; use its tested AOTInductor export.
- **Older exports:** replace old EdgeCrafter and generic `onnx2tf` RF-DETR
  TFLite files, which produced incorrect results. EdgeCrafter passes locally
  and over HTTP with a reconverted model and an unreleased LiteRT XNNPACK fix.
  RF-DETR detection and segmentation pass with the native LiteRT exporter.
  TensorFlow ResNet-50 passes with the corrected exporter.
- **OpenVINO:** `owlv2`, `owlvit`, and `rtdetrul` pass on the tested local stack.
  `videomae`, `vivit`, and `timesformer` run, but video export is untested.
  `tensorflow-classifier` still has known incorrect output and has not been
  retested. Slurm runs used an image with extra runtime libraries added;
  the original image and a fresh source build remain unverified.
- **OpenCV DNN:** RT-DETRv4 needs a compatible export. EdgeCrafter's exact-GELU
  operation remains an import problem; GroundingDINO has not been retested.
- **GGML:** only `.gguf` weights load; the ONNX-subset frontend was removed. YOLO26's attention head is still
  unsupported at the converter. YOLO11 det/seg/pose are gated on `.gguf`
  (V-26/V-27/V-29: `model.23` DFL path + `/model.10` C2PSA subgraph from
  sibling PR #79, pose kpt head from sibling `feature/ggml-pose-v10`); RF-DETR detection now has a complete loader
  graph (backbone + projector + neck + 3-layer deformable decoder + head,
  native GridSample/TopK/softmax-rows/sine-embed helpers) with an app gate
  on `.gguf` (V-21); `rfdetrseg` is now also gated (V-23) and its cell is
  promoted; pose and other transformer rows still need
  sibling work, so their table entries remain untested.
- **KServe:** HTTP has passing results; gRPC is blocked by a server error.
  Classification, video, optical flow, and image-understanding serving still
  need testing. See [KServe compatibility](KserveCompatibility.md).

## Remaining coverage

- Run the combinations marked **DR**, including RT-DETRv4 on TensorRT and
  RAFT, ViTPose, and VideoMAE on ONNX Runtime.
- ARM64 TensorRT gates need prebuilt `.engine` plans: the TRT backend
  deserializes the weights path and does not build from ONNX in-process
  (yolov8n/rtdetr `.onnx` fail with a serialization magic-tag error).
- Add repeatable Torch and OpenVINO examples for the manual `--backend` runs.
- Test EdgeCrafter segmentation and torchvision ResNet on OpenCV DNN 5.x.
- Add a Gaussian-splatting example; no preset exists yet.
- Build and test the app with MIGraphX on supported AMD/ROCm hardware.
- Test more GGML models once the GGUF converter and loader support them.
  YOLO11 det/seg/pose are gated (V-26/V-27/V-29); `yolov10n` stays
  converter-BLOCKED (no sibling loader path yet — in-graph TopK decode
  needs mapping, separate packet), as does YOLO26's attention head. Other
  transformer rows still need sibling work.

<a id="integration-test-evidence"></a>

## Integration tests

CI checks example commands without running inference. Live tests are manual
and require the relevant binaries, model files, and sometimes a GPU.
Each linked README explains how to run its suite.

| Suite | What it checks | How it runs |
| --- | --- | --- |
| [Local inference smoke](../integration-tests/local-inference-smoke/README.md) | Lists presets and checks an example command | CI, dry run only |
| [KServe YOLO](../integration-tests/kserve-runtime-yolo-e2e/README.md) | YOLO26 LiteRT inference over HTTP, rendered output, and metrics | Manual |
| [KServe EdgeCrafter](../integration-tests/kserve-runtime-edgecrafter-e2e/README.md) | Local and remote runs across backends; ORT and TRT passed | Manual; LiteRT/HTTP passed separately with a local patch |
| [RF-DETR pose](../integration-tests/rfdetr-pose-local-vs-kserve/README.md) | Compares recorded local and remote pose results | CI checks report files; live gRPC check still skipped |
| [TPU KServe](../integration-tests/tpu-kserve/README.md) | Prepares a TensorFlow classifier request to a supplied TPU endpoint | CI dry run; live checks still skipped |

The [example workflow](../.github/workflows/e2e-example-emulation.yml)
checks preset commands. Additional HTTP/gRPC protocol tests are described in
[KServe compatibility](KserveCompatibility.md).

## Further reading

- [Deployment and backend setup](Deployment.md#backends)
- [Supported model types](generated/supported-model-types.md)
- [Model export instructions](ExportInstructions.md)
- [Backend build workflow](../.github/workflows/ci.yml)

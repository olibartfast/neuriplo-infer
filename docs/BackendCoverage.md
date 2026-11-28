# Backend coverage

Tested model and backend combinations in `neuriplo-infer`.
**Results last updated: 2026-11-28.** Some results use local, unreleased
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
| Object detection (`yolov8`, `yolov8_executorch`) | P | P | DR | P | B | B | P |
| Object detection (`rtdetr`, `rtdetrv4`) | P | P | DR | P | P | P | B |
| Object detection (`rtdetrul`, Ultralytics RT-DETR) | U | U | P | P | U | P | U |
| Object detection (`ecdet`, `edgecrafter_det`) | P | U | P | P | B | B | B |
| Object detection (`rfdetr`, `rfdetr-medium`) | P | P | P | P | P | P | B |
| Instance segmentation (`yoloseg`) | P | P | P | P | B | B | B |
| Instance segmentation (`ecseg`, `edgecrafter_seg`) | P | U | DR | P | B | B | B |
| Instance segmentation (`rfdetrseg`, `rfdetrseg-medium`) | P | P | P | P | P | P | B |
| Classification (`torchvisionclassifier`, `torchvision_classifier`) | P | U | DR | P | P | B | P |
| Video classification (`videomae`) | P | U | DR | P | B | RUNTIME-PASS | B |
| Optical flow (`raft`) | P | U | DR | P | B | B | B |
| Pose estimation (`vitpose`) | P | P | DR | P | B | B | B |
| Pose estimation (`rfdetrpose`, `rfdetr_keypoint`) | P | P | P | P | P | P | B |
| Depth estimation (`depth_anything_v2`) | P | P | P | P | B | P | B |
| Gaze estimation (`ethxgaze`) | U | U | U | P | U | U | U |
| Monocular 3D detection (`urbanomni`) | U | U | U | P | U | U | U |
| Open-vocabulary detection (`owlv2`) | P | P | P | P | B | P | B |
| Image understanding (`gemma4`) | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Gaussian splatting (`lgm`/`grm`, no preset) | N/A | U | U | U | B | B | U |
| Keypoint matching (`SuperPoint + LightGlue`, no app preset) | U | U | P | P | U | U | U |

### Additional runtimes and remote execution

| Task (representative) | MGX | GGML | LiteRT | ExecT | LLaMA | KServe |
| --- | --- | --- | --- | --- | --- | --- |
| Object detection (`yolo26`, `yolo26s_tflite`) | U | U | P | U | N/A | P |
| Object detection (`yolov8`, `yolov8_executorch`) | U | U | P | P | N/A | U |
| Object detection (`yolo11`) | U | U | U | U | N/A | U |
| Object detection (`rtdetr`, `rtdetrv4`) | U | U | U | U | N/A | U |
| Object detection (`rtdetrul`, Ultralytics RT-DETR) | U | U | U | U | N/A | U |
| Object detection (`ecdet`, `edgecrafter_det`) | U | U | P | U | N/A | P |
| Object detection (`rfdetr`, `rfdetr-medium`) | U | U | P | P | N/A | P |
| Instance segmentation (`yoloseg`) | U | U | U | U | N/A | P |
| Instance segmentation (`yolo11seg`) | U | U | U | U | N/A | U |
| Instance segmentation (`ecseg`, `edgecrafter_seg`) | U | U | U | U | N/A | U |
| Instance segmentation (`rfdetrseg`, `rfdetrseg-medium`) | U | U | P | P | N/A | P |
| Classification (`torchvisionclassifier`, `torchvision_classifier`) | U | U | U | U | N/A | U |
| Video classification (`videomae`) | U | U | U | U | N/A | U |
| Optical flow (`raft`) | U | U | U | U | N/A | U |
| Pose estimation (`vitpose`) | U | U | P | U | N/A | P |
| Pose estimation (`yolo11pose`) | U | U | U | U | N/A | U |
| Pose estimation (`rfdetrpose`, `rfdetr_keypoint`) | U | U | P | P | N/A | P |
| Depth estimation (`depth_anything_v2`) | U | U | P | U | N/A | P |
| Gaze estimation (`ethxgaze`) | U | U | U | U | N/A | U |
| Monocular 3D detection (`urbanomni`) | U | U | U | U | N/A | U |
| Open-vocabulary detection (`owlv2`) | U | U | U | U | N/A | P |
| Image understanding (`gemma4`) | N/A | U | N/A | N/A | P | U |
| Gaussian splatting (`lgm`/`grm`, no preset) | U | U | U | U | N/A | U |
| Keypoint matching (`SuperPoint + LightGlue`, no app preset) | U | U | U | U | N/A | U |

## Environments

Most app tests ran on CPU; TensorRT tests used a GPU. The recorded baseline
uses neuriplo-infer 0.11.6, neuriplo 0.11.3, and neuriplo-tasks 0.8.5.
The November 1–2 runs used neuriplo-infer 0.11.7-dev, neuriplo 0.11.3,
and neuriplo-tasks 0.8.4. OpenCV DNN and ExecuTorch include November 4
local builds with unreleased fixes.

Test counts below are from those runs, not fresh results for the current branch.

| Backend | Tested version | Recorded checks | Notes |
| --- | --- | --- | --- |
| DNN | OpenCV 5.0.0, built from source | App 191/191; backend 80/80 | Coverage is for 5.x. Distribution-provided 4.6 cannot load the current test models. |
| ORT | 1.19.2 CPU | 179/179 | Used a local SDK because the pinned 1.28.0 download was unavailable. Rendered outputs checked. |
| TRT | 11.2.1.2, NGC `tensorrt:26.08` | CI image build passed; detection, segmentation, and keypoint runs passed | Engines rebuilt from ONNX; `trtexec` 11.2 no longer accepts `--fp16`. Jetson Orin ARM64 (Ubuntu 24.04, TRT 10.16.2, JetPack R39) gated 2026-11-26: `rtdetrv4`, `rtdetr`, `rtdetr-l`, `yolov8n`, `resnet50`, `depth_anything_v2`, `owlv2`, `vitpose`, `raft_large` (static 1x3x520x960 export; dynamic-axes ONNX fails TRT 10.16 tactic selection), and `videomae-base` (P 2026-11-27 after the video-classification output fix: per-window logging + render save; top-1 grooming dog on a dog clip), and `ecseg_s` (native EdgeCrafter export; exact-GELU converts fine on TRT) engines built with local `trtexec` 10.16.2 — detection runs render-checked dog/bicycle/car, classifier cross-checked top-1 malamute on dog.jpg — see `docs/evidence/trt-rtdetrv4-2026-11-26.json`, `trt-rtdetr-2026-11-26.json`, `trt-yolov8-2026-11-26.json`, `trt-resnet50-2026-11-26.json`, `trt-rtdetrul-2026-11-26.json`, `trt-depth-2026-11-26.json`, `trt-owlv2-2026-11-26.json`, `trt-vitpose-2026-11-26.json`, `trt-raft-2026-11-26.json`, `trt-videomae-2026-11-26.json`, `trt-ecseg-2026-11-26.json`. Gaze (ptgaze ETH-XGaze resnet18) and monocular 3D (UrbanOmniDetect yolo11n-p2) gated 2026-11-28 on the same Orin stack: render-checked arrow/readout and cuboid wireframes on fresh media; evidence in `specs/2026-11-27-neuriplo-tasks-develop-matrix/validation.md`. |
| TRC | LibTorch 2.14 CPU | 134/134 | RF-DETR uses AOTInductor (`.pt2`); other results use TorchScript. KServe disabled because of a protobuf conflict. |
| OV | OpenVINO 2026.3.0 | Live inference passed; unit tests unavailable | Unit-test executable fails to link; CI also builds with tests disabled. `rtdetr` gated 2026-11-25 on the x86_64 OpenVINO worker (exit 0, render-checked dog/bicycle/car) — see `docs/evidence/ov-rtdetr-2026-11-25.json`. |
| TF | TensorFlow 2.19 | 179/179 | The 2.21 C++ runtime failed to load because of a protobuf ABI conflict. |
| LiteRT | 2.19.0 | 192/192 | SDK's bundled gtest/gmock headers removed to avoid test crashes. EdgeCrafter requires an unreleased backend fix. |
| ExecT | ExecuTorch 1.2.0, optimized SDK | App 191/191; backend 77/77 | XNNPACK enabled. KServe and gRPC compiled, but live transport was not tested in this build. |
| MGX | MIGraphX 2.12.0.60401-83, ROCm 6.4.1, MI300X | Backend 6/6: 3 CPU + 3 GPU | Synthetic backend tests only; no app build or trained model tested. |
| GGML | Local CPU, AMD Ryzen 7 5700U; unreleased neuriplo changes | Not claimed for this release | The app and backend gates in the GGML packet pass only with unreleased neuriplo fixes (strict-GGUF-only loader, parser helpers, converter rules). The pinned neuriplo v0.11.3 does not contain them, so every GGML cell is U here. Evidence stays in the GGML coverage packet, not in the released scoreboard. |
| LLaMA | llama.cpp b9085 | Unit tests unavailable | Unit-test executable fails to link because of missing ggml libraries. |
| KServe | `neuriplo-kserve-runtime`, `real-onnx` / `grpc-onnx` | Manual HTTP tests passed | gRPC `ModelInfer` returns server error `13 INTERNAL`. |

GGML is **not part of this release**. The GGML packet records passing
classification, YOLOv8n detection/segmentation, RF-DETR detection, and
RF-DETR segmentation gates, but every one of them ran against unreleased
`neuriplo` changes (strict-GGUF-only loader plus parser/op helpers). The
pin shipped here is neuriplo v0.11.3, which predates those fixes, so the
GGML column is U across the board and no GGML support is claimed. The
column is restored when a `neuriplo` release carries the fixes and the pin
moves.

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
- **GGML (not in this release):** the strict-GGUF-only loader and the parser
  helpers exist only on unreleased `neuriplo` branches, so the pinned build
  does not run GGML models and every GGML cell is untested here. The GGML
  packet still records what works with those fixes (classification, YOLOv8n
  det/seg, RF-DETR det/seg) and the converter limits found there (YOLO26
  attention head, YOLO10 decode); the column returns when a `neuriplo`
  release carries the fixes.

- **KServe:** HTTP has passing results; gRPC is blocked by a server error.
  Classification, video, optical flow, and image-understanding serving still
  need testing. See [KServe compatibility](KserveCompatibility.md).

## Remaining coverage

- Run the combinations marked **DR**, including RAFT, ViTPose, and VideoMAE on ONNX Runtime. RT-DETRv4 on TensorRT is now gated on Jetson Orin ARM64 (TRT 10.16.2); the x86_64 NGC path remains dry-run only.
- ARM64 TensorRT gates need prebuilt `.engine` plans: the TRT backend
  deserializes the weights path and does not build from ONNX in-process
  (yolov8n/rtdetr `.onnx` fail with a serialization magic-tag error). `rtdetrv4`, `rtdetr`, `rtdetr-l`, `yolov8n`, `resnet50`, `depth_anything_v2`, `owlv2`, `vitpose`, and `raft_large`, and `ecseg_s` are now gated on Jetson Orin (TRT 10.16.2); all local-table TRT cells with available exports are now covered — remaining: x86_64 NGC TRT 11.2 live runs.
- Add repeatable Torch and OpenVINO examples for the manual `--backend` runs.
- Test EdgeCrafter segmentation and torchvision ResNet on OpenCV DNN 5.x.
- Add a Gaussian-splatting example; no preset exists yet.
- Build and test the app with MIGraphX on supported AMD/ROCm hardware.
- Release the GGML loader/parser fixes in `neuriplo`, move the pin, and then
  re-gate the GGML column before claiming any GGML support.
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

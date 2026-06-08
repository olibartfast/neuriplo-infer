# neuriplo-infer

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![C++20](https://img.shields.io/badge/C++-20-blue.svg)](https://isocpp.org/std/the-standard)

Local inference application for computer vision tasks, supporting multiple task types and deep learning backends.

> 🚧 Status: Under Development — expect frequent updates.
## Key Features

- **Multiple Computer Vision Tasks**: Supported via [neuriplo-tasks library](https://github.com/olibartfast/neuriplo-tasks/) (Object Detection, Open-Vocabulary Detection, Classification, Instance Segmentation, Video Classification, Optical Flow, Pose Estimation, Depth Estimation, Gaussian Splatting, Image Understanding / VLM)
- **Switchable Inference Backends**: OpenCV DNN, ONNX Runtime, TensorRT, Libtorch, OpenVINO, Libtensorflow (via [neuriplo library](https://github.com/olibartfast/neuriplo/))
- **Real-time Video Processing**: Multiple video backends via [VideoCapture library](https://github.com/olibartfast/videocapture/) (OpenCV, GStreamer, FFmpeg)
- **Docker Deployment Ready**: Multi-backend container support
- **Remote KServe Runtime Mode**: Send preprocessed tensors to a KServe V2 endpoint, including `neuriplo-kserve-runtime`, while keeping task preprocessing, postprocessing, and rendering in this app

## Requirements

### Core Dependencies
- CMake (≥ 3.24)
- C++20 compiler
- OpenCV (≥ 4.6)
  ```bash
  apt install libopencv-dev
  ```
- Google Logging (glog)
  ```bash
  apt install libgoogle-glog-dev
  ```

### Dependency Management

This project automatically fetches:
1. [neuriplo-tasks](https://github.com/olibartfast/neuriplo-tasks) - Contains pre/post-processing and model logic.
2. [neuriplo](https://github.com/olibartfast/neuriplo) - Provides inference backend abstractions and version management.
3. [videocapture](https://github.com/olibartfast/videocapture) - Handles video I/O.

## Development Workflow

`develop` is the integration branch; `master` is release-only. Use short-lived topic branches (`feat/...`, `fix/...`, `docs/...`, `chore/...`) and open PRs into `develop`. See [`AGENTS.md`](AGENTS.md) for the full workflow.

## Setup

Install dependencies for the backends you need:

```bash
./scripts/setup_dependencies.sh --backend <onnx_runtime|tensorrt|libtorch|openvino|tensorflow|all>
```

LibTorch also accepts `--compute-platform cpu` or `--compute-platform cuda` (CUDA version is read from `versions.neuriplo.env`). See [docs/DependencyManagement.md](docs/DependencyManagement.md) for details.

## Building
```bash
# Default build
cmake -S . -B build -DDEFAULT_BACKEND=OPENCV_DNN -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

Replace `<backend>` with one of the supported inference backends (see [Dependency Management Guide](docs/DependencyManagement.md)).

KServe HTTP client support is built into the app. gRPC client support is optional and is enabled only when Protobuf and gRPC are available at configure time; otherwise the build falls back to HTTP.

### Video Backend Support

The VideoCapture library picks a video backend by priority: **FFmpeg** (`-DUSE_FFMPEG=ON`, widest codec support) > **GStreamer** (`-DUSE_GSTREAMER=ON`) > **OpenCV** (default). Add the desired flag(s) at configure time, e.g.:

```bash
cmake -S . -B build -DDEFAULT_BACKEND=<backend> -DUSE_FFMPEG=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

### Test Build
```bash
cmake -S . -B build-test -DDEFAULT_BACKEND=OPENCV_DNN -DENABLE_APP_TESTS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-test
ctest --test-dir build-test --output-on-failure
```

## End-to-End Examples

The runnable local Docker E2E script remains app-owned:

```bash
bash docker_run_inference_e2e_example.sh --preset owlv2 --dry-run
```

Platform-level scenario ownership, compatibility sets, and cross-repo validation expectations live in `neuriplo-platform/examples/e2e-local-inference/README.md`.

## App Usage

### Command Line Options

```bash
./neuriplo-infer \
  [--help | -h] \
  --type=<model_type> \
  --source=<input_source> \
  [--weights=<model_weights>] \
  [--kserve_endpoint=<url>] \
  [--kserve_model_name=<name>] \
  [--kserve_model_version=<version>] \
  [--kserve_transport=<http|grpc>] \
  [--kserve_timeout_ms=<milliseconds>] \
  [--labels=<labels_file>] \
  [--text_prompts='<prompt_a;prompt_b;...>'] \
  [--prompt='<freeform_prompt>'] \
  [--output_format=<text|json>] \
  [--sample_stride=<n>] \
  [--max_frames=<n>] \
  [--tokenizer_vocab=<vocab_json_path>] \
  [--tokenizer_merges=<merges_txt_path>] \
  [--min_confidence=<threshold>] \
  [--nms_threshold=<threshold>] \
  [--mask_threshold=<threshold>] \
  [--batch|-b=<batch_size>] \
  [--input_sizes|-is='<input_sizes>'] \
  [--use-gpu] \
  [--warmup] \
  [--benchmark] \
  [--export_metadata] \
  [--no_gif] \
  [--iterations=<number>]
```

#### Required Parameters

- `--type=<model_type>`: Specifies the type of vision model to use. Supported categories:
  <!-- SUPPORTED_MODEL_TYPES:START -->
`TaskFactory` routes model type strings through a **built-in, compile-time**
registration table in `src/core/task_factory.cpp`. New built-in tasks require
editing that table and the README list below. **Third-party or runtime task
plugins are not supported**; if plugin extension becomes a product goal, add a
separate explicit extension registry rather than growing the internal table
indefinitely.

<!-- TASKFACTORY_MODEL_LIST:START -->
The TaskFactory supports the following model type strings. Matching normalizes strings by lowercasing and stripping whitespace, hyphens, and underscores, so `YOLO-V8`, `yolo_v8`, and ` yolo v8 ` route identically. Specific segmentation and pose aliases are checked before generic detection aliases.

**Object Detection:**

- `"yolo"`, `"yolov7e2e"`, `"yolov10"`, `"yolo26"`, `"yolov4"` - YOLO-based variants
- `"yolonas"` - YOLO-NAS
- `"rtdetr"` - RT-DETR family (RT-DETR v1, v2, and v4; excludes v3; includes D-FINE and DEIM v1/v2)
- `"rtdetrul"`, `"rtdetrultralytics"` - RT-DETR (Ultralytics implementation)
- `"rfdetr"` - RF-DETR
- `"ecdet"` - EdgeCrafter detection (any string starting with `ecdet`)
- `"edgecrafter"`, `"edgecrafter-det"` - EdgeCrafter detection unless the normalized string contains `seg` or `pose`

**Instance Segmentation:**
- `"ecseg"` - EdgeCrafter segmentation (any string starting with `ecseg` or `edgecrafter` and containing `seg`)
- `"yoloseg"`, `"yolo-seg"`, `"yolov8-seg"` - YOLOv5/YOLOv8/YOLO11-style segmentation
- `"yolov10seg"`- YOLOv10
- `"yolo26seg"` - YOLO26
- `"rfdetrseg"` - RF-DETR

**Classification:**
- `"torchvision-classifier"` - Torchvision models (ResNet, EfficientNet, etc.)
- `"tensorflow-classifier"` - TensorFlow/Keras models
- `"vit-classifier"` - Vision Transformers

Any model type starting with `resnet` (e.g. `resnet50`) or containing `tensorflow` also routes to classification.

**Video Classification:**
- `"videomae"` - VideoMAE
- `"vivit"` - ViViT
- `"timesformer"` - TimeSformer

**Optical Flow:**
- `"raft"` - RAFT optical flow

**Pose Estimation:**
- `"yolov8pose"`, `"yolov8-pose"` - YOLOv8 pose (single-stage, returns bbox + keypoints)
- `"yolo11pose"`, `"yolo11-pose"` - YOLO11 pose
- `"yolo26pose"`, `"yolo26-pose"` - YOLO26 pose
- `"yolov5pose"`, `"yolov5-pose"` - YOLOv5 pose
- `"vitpose"` - ViTPose (top-down, heatmap-based)
- `"ecpose"` - EdgeCrafter pose estimation (any string starting with `ecpose`, or `edgecrafter` and containing `pose`)

**Depth Estimation:**
- `"depth_anything_v2"`, `"depth-anything-v2"` - Depth Anything V2

**Open-Vocabulary Detection:**
- `"owlv2"` - OWLv2 open-vocabulary detection
- `"owlvit"` - OWL-ViT compatible open-vocabulary detection
- `"groundingdino"` - Grounding DINO text-conditioned detection

Open-vocabulary models use text prompts supplied at runtime through `TaskConfig::text_prompts`. Tokenizer assets can be passed either as file paths (`tokenizer_vocab_path`, `tokenizer_merges_path`) or preloaded text blobs (`tokenizer_vocab_json`, `tokenizer_merges_text`).

The expected ONNX contract is:
- Inputs: `pixel_values`, `input_ids`, `attention_mask`
- Outputs: `logits`, `pred_boxes`, and optional `objectness_logits`

Results are returned as `OpenVocabDetection` entries containing `bbox`, `score`, `prompt_index`, and resolved `label`.

For export details, see [export/open_vocab_detection/OWLv2.md](https://github.com/olibartfast/neuriplo-tasks/blob/master/export/open_vocab_detection/OWLv2.md).

**Image Understanding (VLM):**
- `"gemma4"`, `"gemma"`, `"llama"`, `"llamacpp"`, `"imageunderstanding"` - Vision-language model image captioning / Q&A via llama.cpp backend

Input contract: `preprocess()` returns two tensors — `[0]` UTF-8 prompt bytes, `[1]` raw RGB pixels with an 8-byte header `[uint32 width LE][uint32 height LE][H×W×3 bytes]`. When no image is provided only tensor `[0]` is returned (text-only mode). Output is a UTF-8 string returned as float-encoded bytes (one `float` per byte value).

Requires the llama.cpp `LLAMACPP` backend with an mmproj (vision projector) GGUF.

For model download and setup details, see [export/image_understanding/ImageUnderstanding.md](https://github.com/olibartfast/neuriplo-tasks/blob/master/export/image_understanding/ImageUnderstanding.md).

**Gaussian Splatting:**
- `"lgm"`, `"lgm-mini"` - LGM (Large Gaussian Model)
- `"grm"` - GRM
- `"gaussiansplatting"`, any string containing `"splat"` - generic alias


EdgeCrafter export and tensor contract details live in the task-specific docs:

- [EdgeCrafter Detection](https://github.com/olibartfast/neuriplo-tasks/blob/master/export/detection/edgecrafter/README.md)
- [EdgeCrafter Segmentation](https://github.com/olibartfast/neuriplo-tasks/blob/master/export/segmentation/edgecrafter/README.md)
- [EdgeCrafter Pose Estimation](https://github.com/olibartfast/neuriplo-tasks/blob/master/export/pose_estimation/edgecrafter/README.md)

<!-- TASKFACTORY_MODEL_LIST:END -->

Canonical copy: [docs/generated/supported-model-types.md](docs/generated/supported-model-types.md).
<!-- SUPPORTED_MODEL_TYPES:END -->
  App-specific routing and validation in `neuriplo-infer` still define the end-to-end supported subset for this repo.

- `--source=<input_source>`: Input image, video file, or stream URL (e.g. `rtsp://...`). Omit for text-only image-understanding tasks and for `--export_metadata`.
- `--weights=<path>`: Path to local model weights. Required for local backend execution and `--export_metadata`; not required when `--kserve_endpoint` is provided.
- `--labels=<path>`: Class-labels file, one label per line. Optional, for fixed-label models.
- `--text_prompts='<a;b;...>'`: Semicolon-separated prompts. Required for open-vocabulary detection (OWLv2).
- `--tokenizer_vocab=<vocab.json>`, `--tokenizer_merges=<merges.txt>`: Tokenizer assets. Required for OWLv2.
- `--prompt='<text>'`: Freeform prompt for image-understanding / VLM tasks.
- `--output_format=<text|json>`: Output hint; use `json` for parseable multimodal responses.
- `--sample_stride=<n>`, `--max_frames=<n>`: Frame-sampling stride and cap for multimodal video tasks.

#### KServe Runtime Parameters

Use these flags to run preprocessing/postprocessing in `neuriplo-infer` while sending inference tensors to a remote KServe V2 runtime. This is the intended path for `neuriplo-kserve-runtime` integration.

- `--kserve_endpoint=<url>`: Base KServe V2 endpoint, for example `http://127.0.0.1:19090`. A path prefix is allowed when the runtime is behind a gateway. `https://` (HTTP client) and `grpcs://` (gRPC client) select TLS; the server certificate is verified against the system CA roots (or `KSERVE_CA_CERT`). HTTPS requires a build with OpenSSL (`NEURIPLO_INFER_ENABLE_KSERVE_TLS`, on by default when OpenSSL is found); otherwise an `https://` endpoint fails fast with a clear error.
- `--kserve_model_name=<name>`: Model name served by the endpoint. Defaults to `--type` when omitted.
- `--kserve_model_version=<version>`: KServe model version to call. Defaults to `1`.
- `--kserve_transport=<http|grpc>`: Transport selection. Defaults to `http` (the validated path). `grpc` requires a build with Protobuf/gRPC available.
- `--kserve_timeout_ms=<milliseconds>`: Request timeout. Must be greater than zero; default is `30000`.

Tensor datatypes are taken from the server's model metadata (no longer hardcoded to `FP32`), so models with `UINT8`/`INT*`/`FP16`/etc. inputs work against KServe, Triton Inference Server, and OpenVINO Model Server. Set a bearer token via the `KSERVE_BEARER_TOKEN` environment variable to authenticate (sent as `Authorization: Bearer …` on HTTP and as gRPC call metadata).

Security/TLS environment variables (secrets are sourced from env/file, never the command line):

- `KSERVE_BEARER_TOKEN`: bearer token sent as `Authorization: Bearer …` (HTTP) / gRPC call metadata.
- `KSERVE_BEARER_TOKEN_FILE`: path to a file holding the bearer token, used when `KSERVE_BEARER_TOKEN` is unset (trailing whitespace/newline trimmed).
- `KSERVE_CA_CERT`: path to a PEM CA bundle used to verify the server certificate for `https://` / `grpcs://`. Defaults to the system CA roots when unset.
- `KSERVE_CLIENT_CERT` / `KSERVE_CLIENT_KEY`: PEM client certificate and private key. Providing both enables mutual TLS (mTLS); providing only one is an error.

Before loading model metadata the client issues a KServe V2 readiness probe (HTTP `/v2/models/{name}/ready`, gRPC `ModelReady`). If the endpoint is reachable but the model is not loaded/ready the run fails fast with a clear message instead of a confusing metadata error; an unreachable endpoint still surfaces as a connection error.

Tested server/transport/datatype combinations (kept green by CI via `app/test/kserve_integration.sh`) are documented in [docs/KserveCompatibility.md](docs/KserveCompatibility.md).

Build gating:
- `-DNEURIPLO_INFER_ENABLE_KSERVE=OFF` produces a pure local-only build that compiles no KServe code (and needs neither Protobuf nor gRPC).
- `-DNEURIPLO_INFER_ENABLE_LOCAL_BACKENDS=OFF` produces a **KServe-only** build that does **not** fetch or build `neuriplo` (nor any external contract library): the inference contract comes from the app-local headers in `app/inc/contract/`, `setup_inference_engine` is compiled out, and `--kserve_endpoint` becomes mandatory. Still uses OpenCV + `neuriplo-tasks`.
- At least one of `ENABLE_KSERVE` / `ENABLE_LOCAL_BACKENDS` must be `ON` (enforced at configure time).
- `-DNEURIPLO_INFER_ENABLE_GRPC=OFF` keeps the HTTP KServe client but drops the gRPC transport.

Example KServe-only build (no neuriplo fetch):
```bash
cmake -B build -DNEURIPLO_INFER_ENABLE_LOCAL_BACKENDS=OFF -DNEURIPLO_INFER_ENABLE_KSERVE=ON
```

See [docs/KserveRoadmap.md](docs/KserveRoadmap.md) for the production roadmap and server-compatibility matrix.

#### Optional Parameters

- `--min_confidence=<v>`: Minimum detection confidence (default `0.25`).
- `--nms_threshold=<v>`: IoU threshold for NMS in YOLO detectors/segmenters (default `0.45`).
- `--mask_threshold=<v>`: Mask binarization threshold for instance segmentation (default `0.50`).
- `--batch | -b=<n>`: Batch size (default `1`; batches > 1 not currently supported).
- `--input_sizes | -is='<CHW;...>'`: Input sizes for models with dynamic axes or backends that can't report input shapes (e.g. OpenCV DNN). E.g. `'3,224,224'`, or `'3,640,640;2'` for RT-DETR/D-FINE/DEIM.
- `--use-gpu`: Enable GPU inference (default `false`).
- `--warmup`: GPU warmup before inference; image sources only (default `false`).
- `--benchmark` / `--iterations=<n>`: Run repeated inference and report average time; image sources only (default `10` iterations).
- `--export_metadata`: Print model type, routed task, and input/output layers, then exit. Requires `--weights`, not `--source`.
- `--no_gif`: Reserved output flag; current paths emit no GIFs (default `false`).

### To check all available options:

```bash
./neuriplo-infer --help
```

### Common Use Case Examples

```bash
# Object Detection - YOLOv8 ONNX Runtime image processing
./neuriplo-infer \
  --type=yolo \
  --source=image.png \
  --weights=models/yolov8s.onnx \
  --labels=data/coco.names

# Object Detection - RT-DETR video processing
./neuriplo-infer \
  --type=rtdetr \
  --source=video.mp4 \
  --weights=models/rtdetr-l.onnx \
  --labels=data/coco.names \
  --min_confidence=0.4

# Classification - Image classification
./neuriplo-infer \
  --type=torchvisionclassifier \
  --source=image.png \
  --weights=models/resnet50.onnx \
  --labels=data/imagenet_labels.txt

# Instance Segmentation - YOLO segmentation
./neuriplo-infer \
  --type=yoloseg \
  --source=video.mp4 \
  --weights=models/yolov8s-seg.onnx \
  --labels=data/coco.names \
  --min_confidence=0.4 \
  --nms_threshold=0.5 \
  --mask_threshold=0.5 \
  --use-gpu

# Optical Flow - RAFT model
./neuriplo-infer \
  --type=raft \
  --source=video.mp4 \
  --weights=models/raft-small.onnx

# Open-vocabulary detection - OWLv2 image processing
./neuriplo-infer \
  --type=owlv2 \
  --source=image.png \
  --weights=models/owlv2.onnx \
  --text_prompts='cat;dog;bus' \
  --tokenizer_vocab=models/owlv2/vocab.json \
  --tokenizer_merges=models/owlv2/merges.txt \
  --min_confidence=0.2

# Remote KServe runtime - YOLO served by neuriplo-kserve-runtime over HTTP
./neuriplo-infer \
  --type=yolo26 \
  --source=data/dog.jpg \
  --labels=labels/coco.names \
  --kserve_endpoint=http://127.0.0.1:19090 \
  --kserve_model_name=yolo \
  --kserve_transport=http \
  --min_confidence=0.25

# Model metadata inspection without an input source
./neuriplo-infer \
  --type=yolo \
  --weights=models/yolov8s.onnx \
  --export_metadata
```

*Check the [`.vscode folder`](.vscode/launch.json) for other examples.*

## Documentation

- [`AGENTS.md`](AGENTS.md): workflow, review focus, and repo-local entrypoints
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md): ownership boundaries and runtime flow
- [`docs/DependencyManagement.md`](docs/DependencyManagement.md): dependency responsibilities and version sources
- [`docs/Versioning.md`](docs/Versioning.md): release/version workflow for `VERSION` and `CHANGELOG.md`
- [`docs/DetectorArchitectures.md`](docs/DetectorArchitectures.md): object-detection architecture guide
- [`docs/ExportInstructions.md`](docs/ExportInstructions.md) and [neuriplo-tasks export tools](https://github.com/olibartfast/neuriplo-tasks/tree/main/export): model export guides
- [`docs/VLMImageUnderstanding.md`](docs/VLMImageUnderstanding.md): running VLMs via the llama.cpp backend
- [`docs/generated/supported-model-types.md`](docs/generated/supported-model-types.md): generated model-type inventory
- [`scripts/check_code_quality.sh`](scripts/check_code_quality.sh): optional format / static-analysis / sanitizer helper

Cross-repo control-plane docs (cluster map, policies, repo-meta) live in `neuriplo-platform/ops/`.

## Docker Deployment

### Building Images
Inside the project, in the [Dockerfiles folder](docker), there will be a dockerfile for each inference backend (currently onnxruntime, libtorch, tensorrt, openvino)
```bash
# Build for specific backend
docker build --rm -t neuriplo-infer:<backend_tag> \
    -f docker/Dockerfile.<backend_tag> .
```

### Running Containers
Replace the wildcards with your desired options and paths:
```bash
docker run --rm \
    -v<path_host_data_folder>:/app/data \
    -v<path_host_weights_folder>:/weights \
    -v<path_host_labels_folder>:/labels \
    neuriplo-infer:<backend_tag> \
    --type=<model_type> \
    --weights=<weight_according_your_backend> \
    --source=/app/data/<image_or_video> \
    --labels=/labels/<labels_file>
```


For GPU support, add `--gpus all` to the docker run command.

### End-to-End Example Script

[`docker_run_inference_e2e_example.sh`](docker_run_inference_e2e_example.sh) provides preset-driven export-plus-inference workflows (OWLv2, YOLO26s TFLite, EdgeCrafter detection/segmentation/pose, and more). Most presets are self-contained and need no `neuriplo-tasks` checkout.

```bash
bash docker_run_inference_e2e_example.sh --list-presets        # list presets
bash docker_run_inference_e2e_example.sh --preset owlv2 --dry-run   # preview without running
bash docker_run_inference_e2e_example.sh --preset yolo26s_tflite    # run a preset
```

The OWLv2 dry-run is also exposed through CTest:

```bash
ctest --output-on-failure -R docker_run_inference_e2e_owlv2_dry_run
```

## ⚠️ Known Limitations
- Windows builds not currently supported
- Some model/backend combinations may require specific export configurations
- KServe HTTP mode is validated against `neuriplo-kserve-runtime`; gRPC support is built only when Protobuf/gRPC are available and has not yet been validated against every server. FP16/BF16 inputs over gRPC are not yet supported (use HTTP); see [docs/KserveRoadmap.md](docs/KserveRoadmap.md).
- KServe TLS is supported on both transports: HTTPS for the HTTP client (requires an OpenSSL-enabled build) and `grpcs://` for the gRPC client, with optional mTLS via `KSERVE_CLIENT_CERT`/`KSERVE_CLIENT_KEY`. A build without OpenSSL still works over plaintext `http://`, but `https://` endpoints fail fast with a clear error.

## 🙏 Acknowledgments
- [OpenCV YOLO detection with DNN module](https://github.com/opencv/opencv/blob/4.x/samples/dnn/yolo_detector.cpp)
- [TensorRTx](https://github.com/wang-xinyu/tensorrtx)
- [RT-DETR Deploy](https://github.com/CVHub520/rtdetr-onnxruntime-deploy)

 ## References
 - https://paperswithcode.co/tasks
 - https://leaderboard.roboflow.com

## Support

- Open an [issue](https://github.com/olibartfast/neuriplo-infer/issues) for bug reports or feature requests: contributions, corrections, and suggestions are welcome to keep this repository relevant and useful.
- Check existing issues for solutions to common problems

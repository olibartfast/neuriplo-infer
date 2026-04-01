# Vision Inference Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![C++20](https://img.shields.io/badge/C++-20-blue.svg)](https://isocpp.org/std/the-standard)

C++ application for computer vision inference, supporting multiple vision tasks and deep learning backends.

> 🚧 Status: Under Development — expect frequent updates.
## Key Features

- **Multiple Computer Vision Tasks**: Supported via [vision-core library](https://github.com/olibartfast/vision-core/) (Object Detection, Open-Vocabulary Detection, Classification, Instance Segmentation, Video Classification, Optical Flow, Pose Estimation, Depth Estimation)
- **Switchable Inference Backends**: OpenCV DNN, ONNX Runtime, TensorRT, Libtorch, OpenVINO, Libtensorflow (via [neuriplo library](https://github.com/olibartfast/neuriplo/))
- **Real-time Video Processing**: Multiple video backends via [VideoCapture library](https://github.com/olibartfast/videocapture/) (OpenCV, GStreamer, FFmpeg)
- **Docker Deployment Ready**: Multi-backend container support

## Requirements

### Core Dependencies
- CMake (≥ 3.15)
- C++17 compiler (GCC ≥ 8.0)
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
1. [vision-core](https://github.com/olibartfast/vision-core) - Contains pre/post-processing and model logic.
2. [neuriplo](https://github.com/olibartfast/neuriplo) - Provides inference backend abstractions and version management.
3. [videocapture](https://github.com/olibartfast/videocapture) - Handles video I/O.

When a sibling checkout exists at `../vision-core`, the build uses that local source tree instead of fetching from GitHub. This is the recommended setup for integrating current `vision-core` `develop` changes into `vision-inference`.

## Development Workflow

- `develop` is the integration branch for normal feature and fix work.
- `master` is release-only and should only receive release PRs and tagged releases.
- Use short-lived topic branches such as `feat/...`, `fix/...`, `refactor/...`, `docs/...`, and `chore/...`.
- Open normal pull requests into `develop`.
- Open release pull requests into `master`, then cut tags from `master`.

## Agentic Operations

This repository includes an agent-operable maintenance layer under `ops/`.

- `ops/README.md` defines the control-plane intent for the repo cluster.
- `ops/CLUSTER_MAP.yaml` declares repo ownership, dependency edges, validation order, and agent roles.
- `ops/repo-meta/vision-inference.yaml` provides repo-local entrypoints for configure, build, test, and benchmark flows.
- `ops/policies.yaml` defines which automated change classes are allowed and which changes require human review.
- `ops/runbooks/` encodes repeatable maintenance workflows such as CI triage and cross-repo API migration.

The intended maintenance loop is:

1. Observe the failure, request, or contract change.
2. Diagnose ownership and allowed change scope from `ops/`.
3. Act with the smallest reviewable repo-local change.
4. Verify repo-local and downstream impact in the declared validation order.

This makes the repository not just buildable by humans, but operable by coding agents working within explicit ownership, validation, and release-safety constraints.


## Setup
For the selected inference backends, set up the required dependencies first:

- **ONNX Runtime**:
  ```bash
  ./scripts/setup_dependencies.sh --backend onnx_runtime
  ```

- **TensorRT**:
  ```bash
  ./scripts/setup_dependencies.sh --backend tensorrt
  ```

- **LibTorch (CPU only)**:
  ```bash
  ./scripts/setup_dependencies.sh --backend libtorch --compute-platform cpu
  ```

- **LibTorch with GPU support**:
  ```bash
  ./scripts/setup_dependencies.sh --backend libtorch --compute-platform cuda
  # Note: Automatically set CUDA version from `versions.neuriplo.env`
  ```

- **OpenVINO**:
  ```bash
  ./scripts/setup_dependencies.sh --backend openvino
  ```

- **TensorFlow**:
  ```bash
  ./scripts/setup_dependencies.sh --backend tensorflow
  ```

- **All backends**:
  ```bash
  ./scripts/setup_dependencies.sh --backend all
  ```

## Building
```bash
mkdir build && cd build
# <backend> must be one between OPENCV_DNN, ONNX_RUNTIME, LIBTORCH, TENSORRT, OPENVINO, LIBTENSORFLOW
cmake -DDEFAULT_BACKEND=<backend> -DCMAKE_BUILD_TYPE=Release ..
cmake --build .
```

#### Enabling Video Backend Support

The VideoCapture library supports multiple video processing backends with the following priority:
1. **FFmpeg** (if `USE_FFMPEG=ON`) - Maximum format/codec compatibility
2. **GStreamer** (if `USE_GSTREAMER=ON`) - Advanced pipeline capabilities
3. **OpenCV** (default) - Simple and reliable

```bash
# Enable GStreamer support
cmake -DDEFAULT_BACKEND=<backend>  -DUSE_GSTREAMER=ON -DCMAKE_BUILD_TYPE=Release ..
cmake --build .

# Enable FFmpeg support
cmake -DDEFAULT_BACKEND=<backend>  -DUSE_FFMPEG=ON -DCMAKE_BUILD_TYPE=Release ..
cmake --build .

# Enable both (FFmpeg takes priority)
cmake -DDEFAULT_BACKEND=<backend>  -DUSE_GSTREAMER=ON -DUSE_FFMPEG=ON -DCMAKE_BUILD_TYPE=Release ..
cmake --build .
```

### Inference Backend Options
Replace `<backend>` with one of the supported options. See [Dependency Management Guide](docs/DependencyManagement.md) for complete list and details.

### Test Build
```bash
cmake -DENABLE_APP_TESTS=ON ..
```

## App Usage

### Command Line Options

```bash
./vision-inference \
  [--help | -h] \
  --type=<model_type> \
  --source=<input_source> \
  --weights=<model_weights> \
  [--labels=<labels_file>] \
  [--text_prompts='<prompt_a;prompt_b;...>'] \
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
  [--iterations=<number>]
```

#### Required Parameters

- `--type=<model_type>`: Specifies the type of vision model to use. Supported categories:
  <!-- SUPPORTED_MODEL_TYPES:START -->
The TaskFactory supports the following model type strings:

**Object Detection:**

- `"yolo"`, `"yolov7e2e"`, `"yolov10"`, `"yolo26"`, `"yolov4"` - YOLO-based variants
- `"yolonas"` - YOLO-NAS
- `"rtdetr"` - RT-DETR family (RT-DETR v1, v2, and v4; excludes v3; includes D-FINE and DEIM v1/v2)
- `"rtdetrul"` - RT-DETR (Ultralytics implementation)
- `"rfdetr"` - RF-DETR
- `"owlv2"` - OWLv2 open-vocabulary detection

**Instance Segmentation:**
- `"yoloseg"` - YOLOv5/YOLOv8/YOLO11
- `"yolov10seg"`- YOLOv10
- `"yolo26seg"` - YOLO26
- `"rfdetrseg"` - RF-DETR

**Classification:**
- `"torchvision-classifier"` - Torchvision models (ResNet, EfficientNet, etc.)
- `"tensorflow-classifier"` - TensorFlow/Keras models
- `"vit-classifier"` - Vision Transformers

**Video Classification:**
- `"videomae"` - VideoMAE
- `"vivit"` - ViViT
- `"timesformer"` - TimeSformer

**Optical Flow:**
- `"raft"` - RAFT optical flow

**Pose Estimation:**
- `"vitpose"` - ViTPose

**Depth Estimation:**
- `"depth_anything_v2"`, `"depth-anything-v2"` - Depth Anything V2

Canonical copy: [docs/generated/supported-model-types.md](docs/generated/supported-model-types.md).
<!-- SUPPORTED_MODEL_TYPES:END -->

- `--source=<input_source>`: Defines the input source for the object detection. It can be:
  - A live feed URL, e.g., `rtsp://cameraip:port/stream`
  - A path to a video file, e.g., `path/to/video.format`
  - A path to an image file, e.g., `path/to/image.format`

- `--labels=<path/to/labels/file>`: Optional for fixed-label models. Specifies the path to the file containing the class labels. This file should list the labels used by the model, with each label on a new line.

- `--weights=<path/to/model/weights>`: Defines the path to the file containing the model weights.

- `--text_prompts='<prompt_a;prompt_b;...>'`: Required for open-vocabulary detection with OWLv2. Prompts are semicolon-separated and passed at runtime.

- `--tokenizer_vocab=<path/to/vocab.json>`: Required for OWLv2. The app loads this tokenizer asset and passes its contents into `vision-core`.

- `--tokenizer_merges=<path/to/merges.txt>`: Required for OWLv2. The app loads this tokenizer asset and passes its contents into `vision-core`.

#### Optional Parameters

- `[--min_confidence=<confidence_value>]`: Sets the minimum confidence threshold for detections. Detections with a confidence score below this value will be discarded. The default value is `0.25`.

- `[--nms_threshold=<iou_value>]`: IoU threshold used for Non-Maximum Suppression in YOLO-based detectors and segmenters. Higher values keep more overlapping boxes. The default value is `0.45`.

- `[--mask_threshold=<value>]`: Binarization threshold applied to predicted masks in instance segmentation models. Pixels above this value are considered foreground. The default value is `0.50`.

- `[--batch | -b=<batch_size>]`: Specifies the batch size for inference. Default value is `1`, inference with batch size bigger than 1 is not currently supported.

- `[--input_sizes | -is=<input_sizes>]`: Input sizes for each model input when models have dynamic axes or the backend can't retrieve input layer information (like the OpenCV DNN module). Format: `CHW;CHW;...`. For example:
  - `'3,224,224'` for a single input
  - `'3,224,224;3,224,224'` for two inputs
  - `'3,640,640;2'` for RT-DETR/RT-DETRv2/D-FINE/DEIM/DEIMv2 models

- `[--use-gpu]`: Activates GPU support for inference. This can significantly speed up the inference process if a compatible GPU is available. Default is `false`.

- `[--warmup]`: Enables GPU warmup. Warming up the GPU before performing actual inference can help achieve more consistent and optimized performance. This parameter is relevant only if the inference is being performed on an image source. Default is `false`.

- `[--benchmark]`: Enables benchmarking mode. In this mode, the application will run multiple iterations of inference to measure and report the average inference time. This is useful for evaluating the performance of the model and the inference setup. This parameter is relevant only if the inference is being performed on an image source. Default is `false`.

- `[--iterations=<number>]`: Specifies the number of iterations for benchmarking. The default value is `10`.

### To check all available options:

```bash
./vision-inference --help
```

### Common Use Case Examples

```bash
# Object Detection - YOLOv8 ONNX Runtime image processing
./vision-inference \
  --type=yolo \
  --source=image.png \
  --weights=models/yolov8s.onnx \
  --labels=data/coco.names

# Object Detection - RT-DETR video processing
./vision-inference \
  --type=rtdetr \
  --source=video.mp4 \
  --weights=models/rtdetr-l.onnx \
  --labels=data/coco.names \
  --min_confidence=0.4

# Classification - Image classification
./vision-inference \
  --type=torchvisionclassifier \
  --source=image.png \
  --weights=models/resnet50.onnx \
  --labels=data/imagenet_labels.txt

# Instance Segmentation - YOLO segmentation
./vision-inference \
  --type=yoloseg \
  --source=video.mp4 \
  --weights=models/yolov8s-seg.onnx \
  --labels=data/coco.names \
  --min_confidence=0.4 \
  --nms_threshold=0.5 \
  --mask_threshold=0.5 \
  --use-gpu

# Optical Flow - RAFT model
./vision-inference \
  --type=raft \
  --source=video.mp4 \
  --weights=models/raft-small.onnx

# Open-vocabulary detection - OWLv2 image processing
./vision-inference \
  --type=owlv2 \
  --source=image.png \
  --weights=models/owlv2.onnx \
  --text_prompts='cat;dog;bus' \
  --tokenizer_vocab=models/owlv2/vocab.json \
  --tokenizer_merges=models/owlv2/merges.txt \
  --min_confidence=0.2
```

*Check the [`.vscode folder`](.vscode/launch.json) for other examples.*

## Docker Deployment

### Building Images
Inside the project, in the [Dockerfiles folder](docker), there will be a dockerfile for each inference backend (currently onnxruntime, libtorch, tensorrt, openvino)
```bash
# Build for specific backend
docker build --rm -t vision-inference:<backend_tag> \
    -f docker/Dockerfile.<backend_tag> .
```

### Running Containers
Replace the wildcards with your desired options and paths:
```bash
docker run --rm \
    -v<path_host_data_folder>:/app/data \
    -v<path_host_weights_folder>:/weights \
    -v<path_host_labels_folder>:/labels \
    vision-inference:<backend_tag> \
    --type=<model_type> \
    --weights=<weight_according_your_backend> \
    --source=/app/data/<image_or_video> \
    --labels=/labels/<labels_file>
```


For GPU support, add `--gpus all` to the docker run command.

### Generic End-to-End Example Script

Use the generic Docker end-to-end helper at [`docker_run_inference_e2e_example.sh`](docker_run_inference_e2e_example.sh). It replaces the old task-specific RT-DETRv4 script and provides preset-driven export and inference workflows.

Inspect the available presets:

```bash
bash docker_run_inference_e2e_example.sh --list-presets
```

Preview a workflow without executing it:

```bash
bash docker_run_inference_e2e_example.sh --preset owlv2 --dry-run
```

### Full OWLv2 End-to-End Run

OWLv2 uses the `onnxruntime` backend by default in the generic e2e script.

Build the container:

```bash
docker build --rm -t vision-inference:onnxruntime \
    -f docker/Dockerfile.onnxruntime .
```

Run the full export and inference flow:

```bash
mkdir -p /tmp/vision-inference-e2e

bash docker_run_inference_e2e_example.sh \
    --preset owlv2 \
    --text-prompts 'person;dog;bicycle' \
    --weights-dir /tmp/vision-inference-e2e
```

This flow expects:

- a sibling `vision-core` checkout at `../vision-core`
- tokenizer assets at `../vision-core/vocab.json` and `../vision-core/merges.txt`
- sample input image at `data/dog.jpg`
- a working `python3` or `python` on the host for export-time virtualenv creation

The script-level OWLv2 dry-run test is also exposed through CTest:

```bash
ctest --output-on-failure -R docker_run_inference_e2e_owlv2_dry_run
```


## Additional Resources

- [Detector Architectures Guide](docs/DetectorArchitectures.md)
- [Supported Models](docs/TablePage.md)
- [Model Export Guide](docs/ExportInstructions.md)
- [Vision-Core Export Tools](https://github.com/olibartfast/vision-core/tree/main/export) - Comprehensive export utilities for all supported models

## ⚠️ Known Limitations
- Windows builds not currently supported
- Some model/backend combinations may require specific export configurations

## 🙏 Acknowledgments
- [OpenCV YOLO detection with DNN module](https://github.com/opencv/opencv/blob/4.x/samples/dnn/yolo_detector.cpp)
- [TensorRTx](https://github.com/wang-xinyu/tensorrtx)
- [RT-DETR Deploy](https://github.com/CVHub520/rtdetr-onnxruntime-deploy)

 ## References
 - https://paperswithcode.com/sota/real-time-object-detection-on-coco (No more available)
 - https://leaderboard.roboflow.com/

## Support

- Open an [issue](https://github.com/olibartfast/vision-inference/issues) for bug reports or feature requests: contributions, corrections, and suggestions are welcome to keep this repository relevant and useful.
- Check existing issues for solutions to common problems

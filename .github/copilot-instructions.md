# Copilot Instructions

## Build Commands

```bash
# Install system dependencies
apt install libopencv-dev libgoogle-glog-dev

# Setup inference backend (choose one)
./scripts/setup_dependencies.sh --backend onnx_runtime
./scripts/setup_dependencies.sh --backend tensorrt
./scripts/setup_dependencies.sh --backend libtorch --compute-platform cpu
./scripts/setup_dependencies.sh --backend openvino
./scripts/setup_dependencies.sh --backend tensorflow

# Configure and build (DEFAULT_BACKEND options: OPENCV_DNN, ONNX_RUNTIME, LIBTORCH, TENSORRT, OPENVINO, LIBTENSORFLOW)
mkdir build && cd build
cmake -DDEFAULT_BACKEND=OPENCV_DNN -DCMAKE_BUILD_TYPE=Release ..
cmake --build . --parallel $(nproc)

# Build with tests (only OPENCV_DNN backend supports tests)
cmake -DDEFAULT_BACKEND=OPENCV_DNN -DENABLE_APP_TESTS=ON -DCMAKE_BUILD_TYPE=Release ..
cmake --build . --parallel $(nproc)

# Run all tests (requires GTest)
cd build && ctest --output-on-failure

# Run a single test by name (GTest filter)
cd build && ./app/test/runUnitTests --gtest_filter='ParseCommandLineArguments.ThresholdFlags'

# Build with video backend support
cmake -DDEFAULT_BACKEND=OPENCV_DNN -DUSE_GSTREAMER=ON ..   # or USE_FFMPEG=ON
```

## Architecture

This project is a **thin application wrapper** — it contains only CLI parsing, task dispatch, integration glue, and output handling. All vision logic lives in external libraries fetched at build time via CMake `FetchContent`:

- **[vision-core](https://github.com/olibartfast/vision-core)** — preprocessing, postprocessing, model implementations, `TaskInterface`/`TaskFactory`
- **[neuriplo](https://github.com/olibartfast/neuriplo)** — inference backend abstractions (ONNX Runtime, TensorRT, LibTorch, OpenVINO, OpenCV DNN, TensorFlow)
- **[videocapture](https://github.com/olibartfast/videocapture)** — video I/O abstraction (OpenCV / GStreamer / FFmpeg)

The configuration flow is: user selects backend → CMake sets compile definition (e.g. `USE_ONNX_RUNTIME`) → neuriplo handles versioning and linking → vision-core tasks use neuriplo API → `VisionApp` dispatches results.

**Threshold parameters** flow through a dedicated bridge struct: CLI flags → `AppConfig` fields (`confidenceThreshold`, `nmsThreshold`, `maskThreshold`) → `vision_core::TaskConfig` (built in `VisionApp.cpp`) → `TaskFactory::createTaskInstance(type, model_info, task_config)` → task constructor. `TaskConfig` also carries `top_k`, `apply_softmax`, and an `extra_params` map for future extensibility (e.g. VLM text prompts).

**Result dispatch**: `vision_core::Result` is a `std::variant<Detection, Classification, VideoClassification, SegmentationMask, OpticalFlow, PoseEstimation, DepthEstimation>`. `VisionAppRendering.cpp` switches on it with `std::holds_alternative` / `std::get`.

### This project owns only:
```
app/
├── main.cpp
├── src/
│   ├── VisionApp.cpp              # Main app class, inference engine setup
│   ├── VisionAppProcessing.cpp    # Frame processing logic
│   ├── VisionAppRendering.cpp     # Result visualization
│   ├── VisionAppTaskRouting.cpp   # Routes image/video/optical-flow/video-classification
│   ├── CommandLineParser.cpp
│   └── utils.cpp
├── inc/
│   ├── AppConfig.hpp              # Config struct (all CLI params)
│   └── ...
└── test/                          # GTest-based unit tests
```

## Key Conventions

### Branch workflow
`develop` is the integration branch for normal work. `master` is release-only. Review and implementation suggestions should assume short-lived topic branches merge into `develop`, while PRs into `master` are release PRs.

### Version management
All fetched library versions are declared in `versions.env` (read by `cmake/versions.cmake`). Inference backend versions (ONNX Runtime, TensorRT, etc.) are **not** managed here — they belong to the neuriplo library. Override neuriplo or VideoCapture versions locally by creating `versions.neuriplo.env` or `versions.videocapture.env` in the project root.

### Switching backends
Clean the build directory when switching inference backends — CMake cache will have stale backend paths.

### Model types (`--type` flag)
The `--type` string is passed to `TaskFactory` (in vision-core). Valid values include: `yolo`, `yolov10`, `yolov4`, `yolo26`, `rtdetr`, `rtdetrul`, `rfdetr`, `yoloseg`, `yolov10seg`, `yolo26seg`, `rfdetrseg`, `torchvision-classifier`, `tensorflow-classifier`, `vit-classifier`, `videomae`, `vivit`, `timesformer`, `raft`, `vitpose`, `depth_anything_v2`. Adding a new model type requires changes in vision-core, not in this repo.

Both `TaskFactory` (vision-core) and `VisionApp::getTaskType()` **normalise** the type string identically: strip spaces, `-`, and `_`, then lowercase. So `"rt-detr"`, `"rtdetr"`, and `"RT_DETR"` all resolve to the same task. Keep normalisation logic in sync if either function is updated.

### Input sizes (`--input_sizes`)
Required for models with dynamic axes or when using OpenCV DNN (which cannot introspect input shapes). Format: `C,H,W` for single input; `C,H,W;N` for multiple inputs (e.g. `'3,640,640;2'` for RT-DETR).

### Docker
Each inference backend has its own `docker/Dockerfile.<backend>`. CI builds TensorRT, OpenVINO, and TensorFlow backends via Docker; OPENCV_DNN and ONNX_RUNTIME are built natively in CI.

### Adding tests
When adding source files to `app/test/CMakeLists.txt`:
- Include **all** VisionApp translation units (`VisionApp.cpp`, `VisionAppProcessing.cpp`, `VisionAppRendering.cpp`, `VisionAppTaskRouting.cpp`) — the linker will fail if any are missing.
- Add `${vision-core_SOURCE_DIR}/include` to `target_include_directories` to expose `vision-core/core/task_config.hpp` etc.
- Link against `vision-core` (not the removed `detectors` target).

### CI
Defined in `.github/workflows/ci.yml`. Normal CI runs on push/PR to `develop`. Release validation for `master` lives in `.github/workflows/release-check.yml`. Only the `OPENCV_DNN` build runs tests in normal CI; other backends are build-only checks.

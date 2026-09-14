# neuriplo-infer

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![C++20](https://img.shields.io/badge/C++-20-blue.svg)](https://isocpp.org/std/the-standard)

Command-line inference for computer vision models: pick a task, a model file, and
an image or video, and get annotated results — on the backend of your choice.

> 🚧 Status: Under Development — expect frequent updates.

## Key Features

- **Multiple Computer Vision Tasks**: Supported via [neuriplo-tasks library](https://github.com/olibartfast/neuriplo-tasks/) (Object Detection, Open-Vocabulary Detection, Classification, Instance Segmentation, Video Classification, Optical Flow, Pose Estimation, Depth Estimation, Gaussian Splatting, Image Understanding / VLM)
- **Switchable Inference Backends**: OpenCV DNN, ONNX Runtime, TensorRT, LibTorch, OpenVINO, TensorFlow, LiteRT, ExecuTorch, llama.cpp (via [neuriplo library](https://github.com/olibartfast/neuriplo/))
- **Images, Video Files, and Streams**: via [VideoCapture library](https://github.com/olibartfast/videocapture/) (OpenCV, GStreamer, FFmpeg)
- **A Docker Image for Most Backends**: no local SDK installs needed (the default `OPENCV_DNN` backend has no image; build it from source)
- **Remote KServe Mode**: keep pre/postprocessing here and send tensors to Triton, OpenVINO Model Server, KServe, or `neuriplo-kserve-runtime`

## Quickstart (Docker)

Needs Linux, Docker, and Python 3 with `venv` (`sudo apt install python3-venv` on
Ubuntu). Runs on CPU; no GPU required.

```bash
git clone https://github.com/olibartfast/neuriplo-infer.git
cd neuriplo-infer

# 1. Get a model: export YOLO11n to ONNX
python3 -m venv environments/export && . environments/export/bin/activate
pip install ultralytics
mkdir -p models && (cd models && yolo export model=yolo11n.pt format=onnx)

# 2. Build the ONNX Runtime image
docker build -t neuriplo-infer:onnxruntime -f docker/Dockerfile.onnxruntime .

# 3. Detect objects in the sample image
docker run --rm \
  -v "$PWD/data:/app/data" -v "$PWD/models:/weights" -v "$PWD/labels:/labels" \
  neuriplo-infer:onnxruntime \
  --type=yolo --weights=/weights/yolo11n.onnx \
  --source=/app/data/dog.jpg --labels=/labels/coco.names
```

The annotated image is written to `data/output/processed_yolo_local.png`, next to
a `run_report.json` with per-stage timings. The container writes them as root.

For a video source, add `--no_display`: a container has no screen, and the preview
window would abort the run. Other backends, GPU images, and ready-made end-to-end
presets are in [docs/Deployment.md](docs/Deployment.md).

## Build from Source

Ubuntu 24.04, ONNX Runtime backend, with the model from step 1 of the quickstart:

```bash
sudo apt install -y cmake build-essential git wget curl unzip libopencv-dev libgoogle-glog-dev
./scripts/setup_dependencies.sh --backend onnx_runtime   # installs under ~/dependencies

cmake -S . -B build -DDEFAULT_BACKEND=ONNX_RUNTIME -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"

./build/app/neuriplo-infer --type=yolo --weights=models/yolo11n.onnx \
  --source=data/dog.jpg --labels=labels/coco.names
```

The default backend, `OPENCV_DNN`, needs no setup script, but Ubuntu's OpenCV 4.6
cannot load current Ultralytics ONNX exports — prefer ONNX Runtime for a first run.
Other backends, CMake options, and video backends: [docs/Deployment.md](docs/Deployment.md).

## Platform Support

| Platform | Status |
|----------|--------|
| Ubuntu 24.04 (x86_64) | Supported — CI and the Docker images run on it |
| Other Linux distributions | Expected to work; untested |
| Windows (native) | Best-effort, `OPENCV_DNN` only, untested |
| macOS | Untested |

GPU inference needs an NVIDIA driver and, for containers, the NVIDIA Container
Toolkit. Details per backend: [docs/Deployment.md § Platform support](docs/Deployment.md#platform-support).

## Supported Models

Pass the model family with `--type` — for example `yolo`, `rtdetr`, `yoloseg`,
`owlv2`, `raft`, `vitpose`, `depth_anything_v2`, or `gemma4`. The complete list,
with aliases and input contracts, is in
[docs/generated/supported-model-types.md](docs/generated/supported-model-types.md).
To produce model files, see [docs/ExportInstructions.md](docs/ExportInstructions.md)
and the [neuriplo-tasks export tools](https://github.com/olibartfast/neuriplo-tasks/tree/master/export).

## Documentation

| Guide | Read it for |
|-------|-------------|
| [Usage](docs/Usage.md) | Every CLI flag, examples, `--capabilities`, the run report |
| [Deployment](docs/Deployment.md) | Platforms, all backends, Docker and GPU, end-to-end presets, build options |
| [KServe runtime](docs/KserveRuntime.md) | Remote inference against Triton, OVMS, KServe, `neuriplo-kserve-runtime` |
| [KServe compatibility](docs/KserveCompatibility.md) | Tested server / transport / datatype matrix |
| [Supported model types](docs/generated/supported-model-types.md) | Every `--type` string and its task |
| [Export instructions](docs/ExportInstructions.md) | Producing model files for each backend |
| [VLM image understanding](docs/VLMImageUnderstanding.md) | Vision-language models via llama.cpp |
| [Detector architectures](docs/DetectorArchitectures.md) | Object-detection model families |
| [Dependency management](docs/DependencyManagement.md), [Versioning](docs/Versioning.md) | Dependency ownership, releases |

## Known Limitations

- Windows is best-effort only (see [Platform Support](#platform-support)).
- Batch sizes greater than 1 are not supported.
- Some model/backend combinations need specific export settings; see the export guides.
- KServe model management (index / load / unload) is not exposed through the CLI; see [docs/KserveRuntime.md](docs/KserveRuntime.md).

## Contributing

`develop` is the integration branch and `master` is release-only: branch
`feature/*` from `develop` and open pull requests into `develop`. Run the tests with:

```bash
cmake -S . -B build-test -DDEFAULT_BACKEND=OPENCV_DNN -DENABLE_APP_TESTS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-test
ctest --test-dir build-test --output-on-failure
```

[`AGENTS.md`](AGENTS.md) holds the full workflow, and [`specs/`](specs/README.md)
the specifications used to change the project.

## Support

- Open an [issue](https://github.com/olibartfast/neuriplo-infer/issues) for bug reports or feature requests: contributions, corrections, and suggestions are welcome.
- Check existing issues for solutions to common problems.

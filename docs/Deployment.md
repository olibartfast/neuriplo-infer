# Deployment

How to run `neuriplo-infer` on each inference backend — in Docker or built
natively — and which build options exist. For the shortest path to a first
result, follow the [README quickstart](../README.md#quickstart-docker); for the
CLI itself, see [Usage.md](Usage.md).

## Platform support

| Platform | Status | Notes |
|----------|--------|-------|
| Ubuntu 24.04, x86_64 | Supported | Every CI job runs on it, and 7 of the 8 Dockerfiles build on it. |
| Ubuntu 22.04, x86_64 | Images only | Base of `Dockerfile.libtorch`; native builds are untested. |
| Other Linux, x86_64 | Expected to work | Untested. The setup scripts assume an apt-style system. |
| Linux, ARM64 | Untested | The setup scripts and Dockerfiles download x86_64 SDK packages. |
| Windows (native) | Best-effort | See [Windows](#windows). Windows support is not a project goal. |
| macOS | Untested | |

Build requirements: CMake ≥ 3.24, a C++20 compiler, OpenCV ≥ 4.6, and glog.

### GPUs

- **Containers** need an NVIDIA driver on the host and the
  [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html);
  run with `--gpus all`, and pass `--use-gpu` to `neuriplo-infer`.
- **TensorRT** images build on `nvcr.io/nvidia/cuda:13.0.0-devel-ubuntu24.04`, so
  the host driver must support CUDA 13.0.
- **LibTorch** images are CPU unless built with
  `--build-arg COMPUTE_PLATFORM=<cu118|cu121|rocm6.0>`; natively, pass
  `--compute-platform cuda` to the setup script.
- **ONNX Runtime**: the image has no CUDA runtime and runs on CPU. Native builds
  use the ONNX Runtime GPU package, which needs a CUDA installation compatible
  with the version `neuriplo` pins.

Backend package versions are owned by `neuriplo`; see its
[`versions.env`](https://github.com/olibartfast/neuriplo/blob/master/versions.env).

### Windows

Native Windows builds are best-effort: use a Visual Studio generator and start
with `-DDEFAULT_BACKEND=OPENCV_DNN`. The shell scripts in `scripts/` are
Linux-only, so install OpenCV, glog, and any backend SDK through a package manager
or local SDK installs. There are no Windows-specific code paths, and Windows is
not tested.

## Backends

| Backend | `-DDEFAULT_BACKEND=` | `setup_dependencies.sh --backend` | Dockerfile | Model file |
|---------|----------------------|-----------------------------------|------------|------------|
| OpenCV DNN | `OPENCV_DNN` (default) | — (none needed) | — | `.onnx` with static shapes; Darknet `.weights` + `.cfg` |
| ONNX Runtime | `ONNX_RUNTIME` | `onnx_runtime` | `docker/Dockerfile.onnxruntime` | `.onnx` |
| TensorRT | `TENSORRT` | `tensorrt` | `docker/Dockerfile.tensorrt` | serialized `.engine` |
| LibTorch | `LIBTORCH` | `libtorch` | `docker/Dockerfile.libtorch` | TorchScript `.pt` |
| OpenVINO | `OPENVINO` | `openvino` | `docker/Dockerfile.openvino` | IR `.xml` + `.bin`, or `.onnx` |
| TensorFlow | `LIBTENSORFLOW` | `tensorflow` | `docker/Dockerfile.libtensorflow` | SavedModel directory |
| LiteRT | `LITERT` | `litert` | `docker/Dockerfile.litert` | `.tflite` |
| ExecuTorch | `EXECUTORCH` | `executorch` | `docker/Dockerfile.executorch` | `.pte` |
| llama.cpp | `LLAMACPP` | — (built inside the image) | `docker/Dockerfile.llamacpp` | `.gguf` model + `.gguf` projector |

One binary carries one local backend; build again with another
`DEFAULT_BACKEND` to switch. Guides for producing model files:
[ExportInstructions.md](ExportInstructions.md) and the
[neuriplo-tasks export tools](https://github.com/olibartfast/neuriplo-tasks/tree/master/export).

OpenCV DNN is the default because it needs nothing beyond OpenCV, but Ubuntu's
apt OpenCV 4.6 rejects current Ultralytics exports (YOLO11, and YOLOv8 even at
opset 12). Use ONNX Runtime for those models.

## Docker

### Images

| Dockerfile | Base image | Runs on |
|------------|------------|---------|
| `Dockerfile.onnxruntime` | `ubuntu:24.04` | CPU |
| `Dockerfile.tensorrt` | `nvcr.io/nvidia/cuda:13.0.0-devel-ubuntu24.04` | NVIDIA GPU |
| `Dockerfile.libtorch` | `ubuntu:22.04` | CPU; CUDA with `COMPUTE_PLATFORM` |
| `Dockerfile.openvino` | `openvino/ubuntu24_runtime:2025.2.0` | CPU |
| `Dockerfile.libtensorflow` | `ubuntu:24.04` | CPU |
| `Dockerfile.litert` | `ubuntu:24.04` | CPU |
| `Dockerfile.executorch` | `ubuntu:24.04` | CPU |
| `Dockerfile.llamacpp` | `ubuntu:24.04` | CPU; downloads the Gemma 4 E2B GGUF while building |

Build from the repository root. Tag images `neuriplo-infer:<backend>` — the name
the [end-to-end presets](#end-to-end-presets) look for:

```bash
docker build -t neuriplo-infer:onnxruntime -f docker/Dockerfile.onnxruntime .
docker build -t neuriplo-infer:tensorrt    -f docker/Dockerfile.tensorrt .
docker build -t neuriplo-infer:libtorch    -f docker/Dockerfile.libtorch \
  --build-arg COMPUTE_PLATFORM=cu121 .     # omit the build-arg for CPU
```

### Running

Each image's entrypoint is `neuriplo-infer`, run from `/app`. Mount inputs under
`/app/data` — results are written to `/app/data/output` — plus model and label
directories:

```bash
docker run --rm \
  -v "$PWD/data:/app/data" -v "$PWD/models:/weights" -v "$PWD/labels:/labels" \
  neuriplo-infer:<backend> \
  --type=<model_type> --weights=/weights/<model_file> \
  --source=/app/data/<image_or_video> --labels=/labels/<labels_file>
```

- **GPU:** add `--gpus all` to `docker run` and `--use-gpu` to the arguments.
- **Video:** add `--no_display`; a container has no screen, and the preview
  window would abort the run.
- **File ownership:** the container writes `data/output` as root. Running with
  `--user` does not work — the backend libraries are installed under `/root` —
  so reclaim the files afterwards with `sudo chown -R "$USER" data/output`.

### TensorRT engines

TensorRT runs serialized engines, and an engine only loads in the TensorRT
version that built it. Build it from ONNX with an NGC TensorRT container whose
TensorRT version matches the image's (`TENSORRT_VERSION` in neuriplo's
`versions.env`). The presets default to NGC release `25.12`; set `NGC_TAG` to
change it:

```bash
docker run --rm --gpus all -v "$PWD/models:/weights" \
  nvcr.io/nvidia/tensorrt:25.12-py3 \
  trtexec --onnx=/weights/yolo11n.onnx --saveEngine=/weights/yolo11n.engine

docker run --rm --gpus all \
  -v "$PWD/data:/app/data" -v "$PWD/models:/weights" -v "$PWD/labels:/labels" \
  neuriplo-infer:tensorrt \
  --type=yolo --weights=/weights/yolo11n.engine \
  --source=/app/data/dog.jpg --labels=/labels/coco.names --use-gpu
```

## End-to-end presets

[`docker_run_inference_e2e_example.sh`](../docker_run_inference_e2e_example.sh)
chains the whole flow for one model: export on the host (a Python virtual
environment under `environments/`), convert when the backend needs it
(TensorRT, LiteRT), and run inference in `neuriplo-infer:<backend>`. It does not
build the image — build that first.

```bash
bash docker_run_inference_e2e_example.sh --list-presets
bash docker_run_inference_e2e_example.sh --preset yolo26s_tflite --dry-run   # print the commands only

docker build -t neuriplo-infer:litert -f docker/Dockerfile.litert .
bash docker_run_inference_e2e_example.sh --preset yolo26s_tflite
```

| Preset | `--type` | Backend | Needs a neuriplo-tasks checkout |
|--------|----------|---------|:-------------------------------:|
| `yolo26s_tflite` | `yolo26` | LiteRT | no |
| `yolov8_executorch` | `yolov8` | ExecuTorch | no |
| `edgecrafter_det` / `_seg` / `_pose` | `ecdet` / `ecseg` / `ecpose` | ONNX Runtime | no |
| `gemma4` | `gemma4` | llama.cpp | no (downloads the GGUF) |
| `rtdetrv4` | `rtdetr` | TensorRT | yes |
| `owlv2` | `owlv2` | ONNX Runtime | yes (also its `vocab.json` / `merges.txt`) |
| `torchvision_classifier` | `torchvisionclassifier` | ONNX Runtime | yes (and `labels/imagenet_labels.txt`) |
| `yoloseg` | `yoloseg` | ONNX Runtime | yes |
| `raft` | `raft` | ONNX Runtime | yes |
| `vitpose` | `vitpose` | ONNX Runtime | yes |
| `rfdetr_keypoint` | `rfdetr_keypoint` | ONNX Runtime | yes |
| `depth_anything_v2` | `depth_anything_v2` | ONNX Runtime | yes |
| `videomae` | `videomae` | ONNX Runtime | yes (and `data/input.mp4`) |

Presets that need the checkout take `--neuriplo-tasks-dir <path>` or the
`NEURIPLO_TASKS_DIR` environment variable:

```bash
git clone https://github.com/olibartfast/neuriplo-tasks.git ../neuriplo-tasks
bash docker_run_inference_e2e_example.sh --preset owlv2 --neuriplo-tasks-dir ../neuriplo-tasks
```

`--backend` overrides a preset's backend; `--skip-export`, `--skip-convert`, and
`--skip-infer` rerun single steps. EdgeCrafter models convert to LiteRT but
produce wrong detections there — run them on ONNX Runtime or TensorRT.

## Native builds

Ubuntu 24.04:

```bash
sudo apt install -y cmake build-essential git wget curl libopencv-dev libgoogle-glog-dev
./scripts/setup_dependencies.sh --backend <backend>          # see the Backends table

cmake -S . -B build -DDEFAULT_BACKEND=<BACKEND> -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"
./build/app/neuriplo-infer --help
```

The setup script installs SDKs under `~/dependencies`, where CMake looks by
default. `--backend all` installs every supported SDK; LibTorch also takes
`--compute-platform <cpu|cuda|cu118|cu121|rocm6.0>` (default `cpu`). llama.cpp has
no setup-script entry: follow `docker/Dockerfile.llamacpp` or
[VLMImageUnderstanding.md](VLMImageUnderstanding.md).

CMake fetches `neuriplo`, `neuriplo-tasks`, `videocapture`, and
`neuriplo-kserve-client` at the versions pinned in `versions.env`. If
`../neuriplo-tasks`, `../neuriplo`, or `../neuriplo-kserve-client` exists next to
this repository, CMake builds that checkout instead of the pinned version —
convenient for cross-repo work, surprising otherwise. See
[DependencyManagement.md](DependencyManagement.md).

## Build options

| Option | Default | Effect |
|--------|---------|--------|
| `DEFAULT_BACKEND` | `OPENCV_DNN` | Local inference backend compiled into the binary. |
| `USE_FFMPEG` | `OFF` | FFmpeg video backend — widest codec support. |
| `USE_GSTREAMER` | `OFF` | GStreamer video backend. With both on, FFmpeg wins; with neither, OpenCV reads video. |
| `NEURIPLO_INFER_WITH_VIDEOWRITER` | `OFF` | Builds the writer behind `--output_video`. |
| `NEURIPLO_INFER_ENABLE_KSERVE` | `ON` | Remote KServe client. |
| `NEURIPLO_INFER_ENABLE_LOCAL_BACKENDS` | `ON` | Local backends; `OFF` gives a KServe-only build that does not fetch `neuriplo`. |
| `NEURIPLO_INFER_ENABLE_GRPC` | `ON` | gRPC transport, when Protobuf and gRPC are found. |
| `NEURIPLO_INFER_ENABLE_KSERVE_TLS` | `ON` when OpenSSL is found | HTTPS for the HTTP KServe client. |
| `ENABLE_APP_TESTS` | `OFF` | Unit tests (`ctest --test-dir <build> --output-on-failure`). |
| `WERROR` | `OFF` | Treat compiler warnings as errors. |

The KServe build modes and how they combine are described in
[KserveRuntime.md § Build modes](KserveRuntime.md#build-modes); video backend
details belong to [videocapture](https://github.com/olibartfast/videocapture).

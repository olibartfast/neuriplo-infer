# Object Detection Inference

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![C++20](https://img.shields.io/badge/C++-20-blue.svg)](https://isocpp.org/std/the-standard)

C++ framework for [real-time object detection](https://leaderboard.roboflow.com/), supporting multiple deep learning backends and input sources. Run state-of-the-art object detection models on video streams, video files, or images with configurable hardware acceleration.

> 🚧 Status: Under Development — expect frequent updates.

## Key Features

- **Multiple Object Detection Models**: Supported via [vision-core library](https://github.com/olibartfast/vision-core/) (YOLOv4-v12, RT-DETR v1/v2/v4, D-FINE, DEIM v1/v2, RF-DETR)
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

### Test Builds
```bash
# App tests
cmake -DENABLE_APP_TESTS=ON ..

# Library tests
cmake -DENABLE_DETECTORS_TESTS=ON ..
```

## 💻 App Usage

### Command Line Options

```bash
./object-detection-inference \
  [--help | -h] \
  --type=<model_type> \
  --source=<input_source> \
  --labels=<labels_file> \
  --weights=<model_weights> \
  [--min_confidence=<threshold>] \
  [--batch|-b=<batch_size>] \
  [--input_sizes|-is='<input_sizes>'] \
  [--use-gpu] \
  [--warmup] \
  [--benchmark] \
  [--iterations=<number>]
```

#### Required Parameters

- `--type=<model_type>`: Specifies the type of object detection model to use. Possible values:
  - `yolov4`: YOLOv4/YOLOv4-tiny models
  - `yolo`: YOLOv5, YOLOv6, YOLOv7, YOLOv8, YOLOv9, YOLO11, YOLOv12 models
  - `yolov10`: YOLOv10 models (different postprocessing)
  - `yolonas`: YOLO-NAS models
  - `rtdetr`: RT-DETR, RT-DETRv2, RT-DETRv4, D-FINE, DEIM, DEIMv2 models
  - `rtdetrul`: RT-DETR Ultralytics implementation
  - `rfdetr`: RF-DETR models

- `--source=<input_source>`: Defines the input source for the object detection. It can be:
  - A live feed URL, e.g., `rtsp://cameraip:port/stream`
  - A path to a video file, e.g., `path/to/video.format`
  - A path to an image file, e.g., `path/to/image.format`

- `--labels=<path/to/labels/file>`: Specifies the path to the file containing the class labels. This file should list the labels used by the model, with each label on a new line.

- `--weights=<path/to/model/weights>`: Defines the path to the file containing the model weights.

#### Optional Parameters

- `[--min_confidence=<confidence_value>]`: Sets the minimum confidence threshold for detections. Detections with a confidence score below this value will be discarded. The default value is `0.25`.

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
./object-detection-inference --help
```

### Common Use Case Examples

```bash
# YOLOv8 Onnx Runtime image processing
./object-detection-inference \
  --type=yolo \
  --source=image.png \
  --weights=models/yolov8s.onnx \
  --labels=data/coco.names

# YOLOv8 TensorRT video processing
./object-detection-inference \
  --type=yolo \
  --source=video.mp4 \
  --weights=models/yolov8s.engine \
  --labels=data/coco.names \
  --min_confidence=0.4

# RTSP stream processing using RT-DETR Ultralytics implementation
    --type=rtdetrul \
    --source="rtsp://camera:554/stream" \
    --weights=models/rtdetr-l.onnx \
    --labels=data/coco.names \
    --use-gpu
```

*Check the [`.vscode folder`](.vscode/launch.json) for other examples.*

## Docker Deployment

### Building Images
Inside the project, in the [Dockerfiles folder](docker), there will be a dockerfile for each inference backend (currently onnxruntime, libtorch, tensorrt, openvino)
```bash
# Build for specific backend
docker build --rm -t object-detection-inference:<backend_tag>  \
    -f docker/Dockerfile.backend .
```

### Running Containers
Replace the wildcards with your desired options and paths:
```bash
docker run --rm \
    -v<path_host_data_folder>:/app/data \
    -v<path_host_weights_folder>:/weights \
    -v<path_host_labels_folder>:/labels \
    object-detection-inference:<backend_tag> \
    --type=<model_type> \
    --weights=<weight_according_your_backend> \
    --source=/app/data/<image_or_video> \
    --labels=/labels/<labels_file>
```


For GPU support, add `--gpus all` to the docker run command.


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

- Open an [issue](https://github.com/olibartfast/object-detection-inference/issues) for bug reports or feature requests: contributions, corrections, and suggestions are welcome to keep this repository relevant and useful.
- Check existing issues for solutions to common problems

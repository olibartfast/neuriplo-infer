# Object Detection Inference

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![C++20](https://img.shields.io/badge/C++-20-blue.svg)](https://isocpp.org/std/the-standard)

C++ framework for [real-time object detection](https://paperswithcode.com/sota/real-time-object-detection-on-coco), supporting multiple deep learning backends and input sources. Run state-of-the-art object detection models on video streams, video files, or images with configurable hardware acceleration.


## 🚀 Key Features

- Multiple model support (YOLO series from YOLOv4 to YOLOv12, RT-DETR, D-FINE, DEIM)
- Switchable inference backends (OpenCV DNN, ONNX Runtime, TensorRT, Libtorch, OpenVINO, Libtensorflow)
- Real-time video processing with GStreamer integration
- GPU acceleration support
- Docker deployment ready
- Benchmarking tools included

## 🔧 Requirements

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


### Fetched Dependencies
The project automatically fetches and builds the following dependencies using CMake's FetchContent:

#### [VideoCapture Library](https://github.com/olibartfast/videocapture) (Only for the App module, not the library)
```cmake
FetchContent_Declare(
    VideoCapture
    GIT_REPOSITORY https://github.com/olibartfast/videocapture
    GIT_TAG main
)
```
- Handles video input processing
- Provides unified interface for various video sources
- Optional GStreamer integration


#### [Inference Engines Library](https://github.com/olibartfast/inference-engines)
```cmake
FetchContent_Declare(
    InferenceEngines
    GIT_REPOSITORY https://github.com/olibartfast/inference-engines
    GIT_TAG main
)
```
- Provides abstraction layer for multiple inference backends
- Supported backends:
  - OpenCV DNN Module 
  - ONNX Runtime (default)
  - LibTorch
  - TensorRT(10.0.7.23)
  - OpenVINO
  - LibTensorflow(2.13)
 
⚠️ **Note**: **After the CMake configuration step, fetched dependencies are cloned into the ``build/_deps`` folder.**

## 🏗 Building

### Complete Build (Shared Library + Application)
```bash
mkdir build && cd build
cmake -DDEFAULT_BACKEND=<backend> -DBUILD_ONLY_LIB=OFF -DCMAKE_BUILD_TYPE=Release ..
cmake --build .
```

#### Enabling GStreamer Support
```bash
cmake -DDEFAULT_BACKEND=<backend> -DBUILD_ONLY_LIB=OFF -DUSE_GSTREAMER=ON -DCMAKE_BUILD_TYPE=Release ..
cmake --build .
```

---

### Library-Only Build
```bash
mkdir build && cd build
cmake -DBUILD_ONLY_LIB=ON -DDEFAULT_BACKEND=<backend> -DCMAKE_BUILD_TYPE=Release ..
cmake --build .
```

---

### Backend Options
Replace `<backend>` with one of the following options:  
- **`OPENCV_DNN`**   
- **`ONNX_RUNTIME`**  
- **`LIBTORCH`**  
- **`TENSORRT`**  
- **`OPENVINO`**  
- **`LIBTENSORFLOW`**  

---

### Notes  

1. **Custom Backend Paths**  
   If the required backend package is not installed system-wide, you can manually specify its path:  

   - **Libtorch**  
     Modify [`LibTorch.cmake`](https://github.com/olibartfast/inference-engines/blob/master/cmake/LibTorch.cmake) or pass the `Torch_DIR` argument.  

   - **ONNX Runtime**  
     Modify [`ONNXRuntime.cmake`](https://github.com/olibartfast/inference-engines/blob/master/cmake/ONNXRuntime.cmake) or pass the `ONNX_RUNTIME_DIR` and `ORT_VERSION` arguments.  

   - **TensorRT**  
     Modify [`TensorRT.cmake`](https://github.com/olibartfast/inference-engines/blob/master/cmake/TensorRT.cmake) or pass the `TENSORRT_DIR` and `TRT_VERSION` arguments.  

   - ⚠️ **Important:**  
     - These CMake files belong to the [`InferenceEngines`](https://github.com/olibartfast/inference-engines) project and are cloned into the `build/_deps` folder after the configuration step.  
     - Ensure your backend version is set correctly in [cmake/AddCompileDefinitions.cmake](cmake/AddCompileDefinitions.cmake).  

2. **Cleaning the Build Folder**  
   When switching backends or changing configuration options, clean the `build` directory before reconfiguring and compiling.  

   **Full Clean (Major Changes)**  
   - **Command:**  
     ```sh
     rm -rf build && mkdir build
     ```  
   - **Use Case:**  
     - Backend switches or major config updates.  
     - Ensures a fresh, conflict-free build.  
   - **Trade-off:**  
     - Slower full rebuild.  

   **Partial Clean (Minor Changes)**  
   - **Command:**  
     ```sh
     rm build/CMakeCache.txt
     ```  
   - **Use Case:**  
     - Small tweaks without fully rebuilding.  
     - Faster than a full clean.  
   - **Trade-off:**  
     - May not catch all conflicts.  

---
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

- `--type=<model_type>`: Specifies the type of object detection model to use. Possible values include `yolov4`, `yolov5`, `yolov6`, `yolov7`, `yolov8`, `yolov9`, `yolov10`, `yolo11`, `yolov12`, `rtdetr`,`rtdetrv2`, `rtdetrul`, `dfine`, `deim`.

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
  - `'3,640,640;2'` for RT-DETR/D-FINE/DEIM models

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
  --type=yolov8 \
  --source=image.png \
  --weights=models/yolov8s.onnx \
  --labels=data/coco.names

# YOLOv8 TensorRT video processing
./object-detection-inference \
  --type=yolov8 \
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

*Check the [`.vscode` folder](.vscode/launch.json) for other examples.*

## 🐳 Docker Deployment

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


## 🗺 Project Structure

```
.
├── app/            # Main application
├── detectors/      # Detection library
├── cmake/          # CMake modules
└── docker/         # Dockerfiles
└── build/_deps/    # Fetched dependencies after CMake configuration
```

## 📚 Additional Resources

- [Supported Models](docs/TablePage.md)
- [Model Export Guide](docs/ExportInstructions.md)
- Backend-specific export documentation:
  - [YOLOv5](docs/yolov5-export.md)
  - [YOLOv8](docs/yolov8-export.md)
  - [YOLOv6](docs/yolov6-export.md)
  - [YOLOv7](docs/yolov7-export.md)
  - [YOLOv8](docs/yolov8-export.md)
  - [YOLOv9](docs/yolov9-export.md)
  - [YOLOv10](docs/yolov10-export.md)
  - [YOLO11](docs/yolo11-export.md)
  - [YOLOv12](docs/yolov12-export.md)
  - [YOLO-NAS](docs/yolo-nas-export.md)
  - [RT-DETR (lyuwenyu implementation)](docs/rtdetr-lyuwenyu-export.md)
  - [RT-DETRV2](docs/rtdetrv2-lyuwenyu-export.md)
  - [RT-DETR (Ultralytics implementation)](docs/rtdetr-ultralytics-export.md)
  - [D-FINE](docs/d-fine-export.md)
  - [DEIM](docs/deim-export.md)

## ⚠️ Known Limitations
- Windows builds not currently supported
- Some model/backend combinations may require specific export configurations

## 🙏 Acknowledgments
- [OpenCV YOLO detection with DNN module](https://github.com/opencv/opencv/blob/4.x/samples/dnn/yolo_detector.cpp)
- [TensorRTx](https://github.com/wang-xinyu/tensorrtx)
- [RT-DETR Deploy](https://github.com/CVHub520/rtdetr-onnxruntime-deploy)

 ## References
 - https://paperswithcode.com/sota/real-time-object-detection-on-coco
 - https://leaderboard.roboflow.com/

## 📫 Support

- Open an [issue](https://github.com/olibartfast/object-detection-inference/issues) for bug reports or feature requests
- Check existing issues for solutions to common problems

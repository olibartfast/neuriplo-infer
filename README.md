# Object Detection Inference

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![C++20](https://img.shields.io/badge/C++-20-blue.svg)](https://isocpp.org/std/the-standard)

C++ framework for [real-time object detection](https://paperswithcode.com/sota/real-time-object-detection-on-coco), supporting multiple deep learning backends and input sources. Run state-of-the-art object detection models (YOLOv4-11, RT-DETR, D-FINE) on video streams, video files, or images with configurable hardware acceleration.


## 🚀 Key Features

- Multiple model support (YOLO series from YOLOv4 to YOLO11, RT-DETR, D-FINE)
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
  - TensorRT
  - OpenVINO
  - LibTensorflow
 

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
   - For **Libtorch**, modify [`LibTorch.cmake`](https://github.com/olibartfast/inference-engines/blob/master/cmake/LibTorch.cmake) or pass the `Torch_DIR` argument.  
   - For **ONNX Runtime**, modify [`ONNXRuntime.cmake`](https://github.com/olibartfast/inference-engines/blob/master/cmake/ONNXRuntime.cmake) or pass the `ORT_VERSION` argument.  
   - The same approach applies to other backends—adjust their respective CMake modules or pass the relevant arguments during configuration.

2. **Cleaning the Build Folder**  
   When switching between backends or modifying configuration options, always clean the `build` directory before reconfiguring and compiling:  
   ```bash
   rm -rf build && mkdir build
   ```

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
    --type=<model_type> \
    --source=<input_source> \
    --labels=<labels_file> \
    --weights=<model_weights> \
    [--config=<model_config>] \
    [--min_confidence=<threshold>] \
    [--use-gpu] \
    [--warmup] \
    [--benchmark]
```

#### Required Parameters
- `--type=<model type>`: Specifies the type of object detection model to use. Possible values include `yolov4`, `yolov5`, `yolov6`, `yolov7`, `yolov8`, `yolov9`,  `yolov10`, `yolo11`,`rtdetr`, `rtdetrul`, `dfine`. Choose the appropriate model based on your requirements.

- `--source=<source>`: Defines the input source for the object detection. It can be:
  - A live feed URL, e.g., `rtsp://cameraip:port/somelivefeed`
  - A path to a video file, e.g., `path/to/video.format`
  - A path to an image file, e.g., `path/to/image.format`

- `--labels=<path/to/labels/file>`: Specifies the path to the file containing the class labels. This file should list the labels used by the model, each label on a new line.

- `--weights=<path/to/model/weights>`: Defines the path to the file containing the model weights. T


#### Optional Parameters

- `[--config=<path/to/model/config>]`: (Optional) Specifies the path to the model configuration file. This file contains the model architecture and other configurations necessary for setting up the inference. This parameter is primarily needed if the model is from the OpenVINO backend.

- `[--min_confidence=<confidence value>]`: (Optional) Sets the minimum confidence threshold for detections. Detections with a confidence score below this value will be discarded. The default value is `0.25`.

- `[--use-gpu]`: (Optional) Activates GPU support for inference. This can significantly speed up the inference process if a compatible GPU is available.

- `[--warmup]`: (Optional) Enables GPU warmup. Warming up the GPU before performing actual inference can help achieve more consistent and optimized performance. This parameter is relevant only if the inference is being performed on an image source.

- `[--benchmark]`: (Optional) Enables benchmarking mode. In this mode, the application will run multiple iterations of inference to measure and report the average inference time. This is useful for evaluating the performance of the model and the inference setup. This parameter is relevant only if the inference is being performed on an image source.

### To check all available options:
```
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

# YOLOv8s TensorRT video processing
./object-detection-inference \
    --type=yolov8 \
    --source=video.mp4 \
    --weights=models/yolov8s.engine \
    --labels=data/coco.names \
    --min_confidence=0.4

# RTSP stream processing using rtdetr ultralytics implementation
./object-detection-inference \
    --type=rtdetrul \
    --source="rtsp://camera:554/stream" \
    --weights=models/rtdetr-l.onnx \
    --labels=data/coco.names \
    --use-gpu
```
* check [.vscode folder](.vscode/launch.json) for other examples

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
  - [YOLO-NAS](docs/yolo-nas-export.md)
  - [RT-DETR (lyuwenyu implementation)](docs/rtdetr-lyuwenyu-export.md)
  - [RT-DETR (Ultralytics implementation)](docs/rtdetr-ultralytics-export.md)
  - [D-FINE](docs/d-fine-export.md)

## ⚠️ Known Limitations

- Models with dynamic axes not fully supported
- Windows builds not currently supported
- Some model/backend combinations may require specific export configurations

## 🙏 Acknowledgments

- [OpenCV DNN Module](https://github.com/opencv/opencv)
- [TensorRTx](https://github.com/wang-xinyu/tensorrtx)
- [RT-DETR Deploy](https://github.com/CVHub520/rtdetr-onnxruntime-deploy)

## 📫 Support

- Open an [issue](https://github.com/olibartfast/object-detection-inference/issues) for bug reports or feature requests
- Check existing issues for solutions to common problems

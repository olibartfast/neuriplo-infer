# Project Architecture

This document explains the architecture and separation of concerns in the vision-inference project.

## Overall Architecture

The project follows a **modular architecture** with clear separation of concerns:

```
│                        vision-inference                         │
│                         (This Project)                         │
├─────────────────────────────────────────────────────────────────┤
│  vision-core            │  VideoCapture       │  neuriplo       │
│                         │                     │                 │
│  • Preprocessing        │  • Video processing │  • Backend abstractions
│  • Postprocessing       │  • RTSP streams     │  • ONNX Runtime │
│  • Model implementations│  • OpenCV backend   │  • TensorRT     │
│  • Task Interface       │  • GStreamer backend │  • LibTorch     │
│  • Multi-task support   │  • FFmpeg backend   │  • OpenVINO     │
│                         │  • Unified API      │  • OpenCV DNN   │
│                         │                     │  • TensorFlow   │
└─────────────────────────────────────────────────────────────────┘
```

## **This Project: Vision Inference Application**

### What We Implement
- **Application Logic**: CLI parsing, configuration management, logging, main loop
- **Task Dispatch**: Routing inputs to the correct processing path (image, video, optical flow, video classification)
- **Integration**: Glue code connecting `vision-core`, `neuriplo`, and `VideoCapture`
- **Output Handling**: Visualization, benchmarking, result reporting

### What We Manage
- **System Dependencies**: OpenCV, glog, CMake version requirements
- **Fetched Library Versions**: `vision-core`, `neuriplo`, and `VideoCapture` library versions
- **Build Configuration**: Compile definitions for selected inference backend

### Files We Own
```
app/
├── src/
│   ├── VisionApp.cpp
│   ├── CommandLineParser.cpp
│   └── utils.cpp
├── inc/
│   ├── VisionApp.hpp
│   ├── AppConfig.hpp
│   ├── CommandLineParser.hpp
│   └── utils.hpp
├── main.cpp
└── test/
    └── ...
```

## **vision-core Library: Vision Task Logic**

### What It Provides
<!-- MODEL_TYPES:ARCHITECTURE:START -->
- **Vision Task Algorithms**:
  - **Object Detection**: YOLO-based variants, YOLO-NAS, RT-DETR family (RT-DETR v1, v2, and v4; excludes v3; includes D-FINE and DEIM v1/v2), RT-DETR (Ultralytics implementation), RF-DETR
  - **Open-Vocabulary Detection**: OWLv2 open-vocabulary detection, OWL-ViT compatible open-vocabulary detection, Generic Open Vocabulary OWL alias
  - **Instance Segmentation**: YOLOv5/YOLOv8/YOLO11, YOLO26, RF-DETR
  - **Classification**: Torchvision models (ResNet, EfficientNet, etc.), TensorFlow/Keras models, Vision Transformers
  - **Video Classification**: VideoMAE, ViViT, TimeSformer
  - **Optical Flow**: RAFT optical flow
  - **Pose Estimation**: YOLOv8 pose (single-stage, returns bbox + keypoints), YOLO11 pose, YOLO26 pose, YOLOv5 pose, ViTPose (top-down, heatmap-based)
  - **Depth Estimation**: Depth Anything V2
<!-- MODEL_TYPES:ARCHITECTURE:END -->
- **Preprocessing Implementation**: Letterbox resizing, normalization, color space conversion (using `neuriplo` compatible blobs)
- **Postprocessing Implementation**: Decoding bounding boxes, NMS (if needed), class score filtering, mask decoding
- **Unified Task Interface**: `TaskInterface` and `TaskFactory` for task-agnostic dispatch
- **Result Types**: Shared `Result` variant covering all task outputs

### What It Should Manage
- **Task-specific logic and parameters**
- **Input/Output tensor shapes and formats**
- **Model-specific pre/postprocessing**

### Files It Should Own
```
vision-core/
├── src/
│   ├── object_detection/
│   │   ├── object_detection_task.cpp
│   │   ├── detection_preprocessor.cpp
│   │   └── ...
│   ├── classification/
│   │   └── ...
│   ├── instance_segmentation/
│   │   └── ...
│   ├── optical_flow/
│   │   └── ...
│   └── core/
│       ├── task_interface.cpp
│       └── task_factory.cpp
└── include/
    └── vision-core/
        └── ...
```

## **neuriplo Library: Inference Backends**

### What It Provides
- **Inference Backend Abstractions**: Unified interface to different inference engines
- **Backend Implementations**: ONNX Runtime, TensorRT, LibTorch, OpenVINO, OpenCV DNN, TensorFlow
- **Performance Optimization**: Backend-specific optimizations
- **Version Management**: Inference backend version management

### What It Should Manage
- **ONNX Runtime versions and paths**
- **TensorRT versions and paths**
- **LibTorch versions and paths**
- **OpenVINO versions and paths**
- **TensorFlow versions and paths**
- **CUDA version compatibility**

### Files It Should Own
```
neuriplo/
├── cmake/
│   ├── versions.cmake           # Inference backend versions
│   ├── ONNXRuntime.cmake
│   ├── TensorRT.cmake
│   ├── LibTorch.cmake
│   └── ...
├── backends/
│   ├── onnx-runtime/
│   ├── tensorrt/
│   ├── libtorch/
│   └── ...
└── ...
```

## **VideoCapture Library: Video Processing**

### What It Provides
- **Video Input Processing**: RTSP streams, video files, images, camera devices
- **Multiple Video Backends**: OpenCV (default), GStreamer, FFmpeg
- **Backend Selection Priority**: FFmpeg > GStreamer > OpenCV
- **Unified Video Interface**: Consistent API for different video sources
- **Advanced Features**: Hardware acceleration, streaming protocols, complex pipelines

### What It Should Manage
- **Video processing dependencies**: OpenCV, GStreamer, FFmpeg
- **Backend version management**: Centralized in `cmake/versions.cmake`
- **Platform-specific video handling**: Linux, macOS, Windows support
- **Dependency validation**: Automatic validation of video processing libraries

## **Dependency Management Responsibilities**

### This Project Should Manage:
```cmake
# cmake/versions.cmake
set(VISION_CORE_VERSION "v1.0.0")        # Fetched library version
set(NEURIPLO_VERSION "v1.0.0")           # Fetched library version
set(VIDEOCAPTURE_VERSION "v1.0.0")       # Fetched library version
set(OPENCV_MIN_VERSION "4.6.0")          # System dependency
set(GLOG_MIN_VERSION "0.6.0")            # System dependency
set(CMAKE_MIN_VERSION "3.20")            # Build system
```

### This Project Should NOT Manage:
```cmake
# These should be in neuriplo library
set(ONNX_RUNTIME_VERSION "1.19.2")       # Inference backend
set(TENSORRT_VERSION "10.7.0.23")        # Inference backend
set(LIBTORCH_VERSION "2.0.0")            # Inference backend
set(OPENVINO_VERSION "2023.1.0")         # Inference backend
set(CUDA_VERSION "12.6")                 # Inference backend dependency
```

## **Setup Scripts Purpose**

The setup scripts in this project (`scripts/setup_dependencies.sh`) are **convenience scripts** that:

1. **Install inference backend dependencies** that will be used by the neuriplo library
2. **Provide a unified interface** for users to setup their environment
3. **Maintain backward compatibility** with existing workflows

### What They Do:
- Download and install ONNX Runtime, TensorRT, LibTorch, OpenVINO
- Set up proper directory structure
- Validate installations

### What They Don't Do:
- Manage inference backend versions (should be done by neuriplo)
- Link inference backend libraries (done by neuriplo)
- Handle inference backend configuration (done by neuriplo)

## **Correct Workflow**

### For Users:
```bash
# 1. Setup inference backend dependencies (convenience)
./scripts/setup_dependencies.sh --backend onnx_runtime

# 2. Build project (fetches neuriplo with proper versions)
mkdir build && cd build
cmake -DDEFAULT_BACKEND=ONNX_RUNTIME ..
cmake --build .
```

### For Developers:
```bash
# 1. Update neuriplo version in this project
# cmake/versions.cmake
set(NEURIPLO_VERSION "v1.1.0")

# 2. Update inference backend versions in neuriplo library
# neuriplo/cmake/versions.cmake
set(ONNX_RUNTIME_VERSION "1.20.0")
set(TENSORRT_VERSION "10.8.0.0")
```

## **Configuration Flow**

```
User selects backend
        ↓
This project sets compile definition (USE_ONNX_RUNTIME)
        ↓
neuriplo library handles:
  - Version management
  - Path configuration
  - Library linking
  - Backend-specific setup
        ↓
vision-core tasks use neuriplo API for inference
        ↓
VisionApp dispatches results to the appropriate output handler
```

## **Benefits of This Architecture**

### Separation of Concerns
- **This project**: Application wrapper, task dispatch, and integration point
- **vision-core**: Encapsulates vision task algorithms and logic for all supported tasks
- **neuriplo**: Handles inference backend complexity
- **VideoCapture**: Manages video input processing

### Maintainability
- **Version updates**: Each library manages its own versions
- **Bug fixes**: Issues are isolated to specific components
- **Feature additions**: New backends don't affect vision tasks; new tasks don't affect backends

### Reusability
- **neuriplo**: Can be used by other inference projects
- **VideoCapture**: Can be used by other projects
- **vision-core**: Can be used with different inference backends and applications

### User Experience
- **Simple setup**: One command to setup inference backends
- **Flexible configuration**: Easy to switch between backends and task types
- **Clear documentation**: Each component has its own docs

## **Future Improvements**

### For This Project:
1. **Expand task coverage**: Add support for new vision tasks as vision-core gains them
2. **Improve result rendering**: Task-specific visualization (e.g., flow maps, pose skeletons)
3. **Batch inference**: Support batch size > 1 across all task types

### For vision-core Library:
1. **Add new task types**: Depth estimation, pose estimation, panoptic segmentation
2. **Improve preprocessing**: Shared, reusable preprocessing primitives across tasks
3. **Unified result format**: Consistent output structure across all task types

### For neuriplo Library:
1. **Centralized version management**
2. **Better backend validation**
3. **Automatic backend setup**
4. **Performance benchmarking**

### For VideoCapture Library:
1. **Enhanced FFmpeg integration**: Additional codec and format support
2. **Hardware acceleration**: Improved GPU-accelerated video processing
3. **Cross-platform compatibility**: Enhanced Windows and macOS support
4. **Advanced streaming features**: WebRTC, SRT protocol support

This architecture ensures that each component has a clear responsibility and can evolve independently while providing a seamless user experience.

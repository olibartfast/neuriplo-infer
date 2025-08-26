# Project Architecture

This document explains the architecture and separation of concerns in the object-detection-inference project.

## 🏗️ Overall Architecture

The project follows a **modular architecture** with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────────┐
│                    object-detection-inference                   │
│                         (This Project)                         │
├─────────────────────────────────────────────────────────────────┤
│  🎯 Object Detectors    │  📚 VideoCapture    │  🔧 neuriplo │
│                         │                     │                     │
│  • YOLO variants        │  • Video processing │  • Backend abstractions│
│  • RT-DETR variants     │  • RTSP streams     │  • ONNX Runtime      │
│  • D-FINE, DEIM, RF-DETR│  • GStreamer        │  • TensorRT          │
│                         │  • Unified API      │  • LibTorch          │
│                         │                     │  • OpenVINO          │
│                         │                     │  • OpenCV DNN        │
│                         │                     │  • TensorFlow        │
└─────────────────────────────────────────────────────────────────┘
```

## 🎯 **This Project: Object Detectors**

### What We Implement
- **Object Detection Algorithms**: YOLO variants (v4-v12), RT-DETR variants, D-FINE, DEIM, RF-DETR
- **Detection Logic**: Preprocessing, postprocessing, bounding box handling
- **Model-Specific Implementations**: Each detector type has its own class

### What We Manage
- **System Dependencies**: OpenCV, glog, CMake version requirements
- **Fetched Library Versions**: neuriplo and VideoCapture library versions
- **Build Configuration**: Compile definitions for selected inference backend

### Files We Own
```
detectors/
├── src/
│   ├── models/
│   │   ├── YoloV4.cpp/hpp
│   │   ├── YoloVn.cpp/hpp      # YOLOv5-v12
│   │   ├── YOLOv10.cpp/hpp
│   │   ├── YoloNas.cpp/hpp
│   │   ├── RtDetr.cpp/hpp
│   │   ├── RtDetrUltralytics.cpp/hpp
│   │   └── RfDetr.cpp/hpp
│   └── DetectorSetup.cpp
└── inc/
    ├── Detector.hpp
    └── DetectorSetup.hpp
```

## 🔧 **neuriplo Library: Inference Backends**

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

## 📚 **VideoCapture Library: Video Processing**

### What It Provides
- **Video Input Processing**: RTSP streams, video files, images
- **GStreamer Integration**: Real-time video processing
- **Unified Video Interface**: Consistent API for different video sources

### What It Should Manage
- **Video processing dependencies**
- **GStreamer integration**
- **Platform-specific video handling**

## 🔄 **Dependency Management Responsibilities**

### ✅ **This Project Should Manage:**
```cmake
# cmake/versions.cmake
set(INFERENCE_ENGINES_VERSION "v1.0.0")  # Fetched library version
set(VIDEOCAPTURE_VERSION "v1.0.0")       # Fetched library version
set(OPENCV_MIN_VERSION "4.6.0")          # System dependency
set(GLOG_MIN_VERSION "0.6.0")            # System dependency
set(CMAKE_MIN_VERSION "3.20")            # Build system
```

### ❌ **This Project Should NOT Manage:**
```cmake
# These should be in neuriplo library
set(ONNX_RUNTIME_VERSION "1.19.2")       # Inference backend
set(TENSORRT_VERSION "10.7.0.23")        # Inference backend
set(LIBTORCH_VERSION "2.0.0")            # Inference backend
set(OPENVINO_VERSION "2023.1.0")         # Inference backend
set(CUDA_VERSION "12.6")                 # Inference backend dependency
```

## 🛠️ **Setup Scripts Purpose**

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

## 📋 **Correct Workflow**

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
set(INFERENCE_ENGINES_VERSION "v1.1.0")

# 2. Update inference backend versions in neuriplo library
# neuriplo/cmake/versions.cmake
set(ONNX_RUNTIME_VERSION "1.20.0")
set(TENSORRT_VERSION "10.8.0.0")
```

## 🔧 **Configuration Flow**

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
Object detectors use neuriplo API
```

## 🎯 **Benefits of This Architecture**

### **Separation of Concerns**
- **This project**: Focuses on object detection algorithms
- **neuriplo**: Handles inference backend complexity
- **VideoCapture**: Manages video input processing

### **Maintainability**
- **Version updates**: Each library manages its own versions
- **Bug fixes**: Issues are isolated to specific components
- **Feature additions**: New backends don't affect object detectors

### **Reusability**
- **neuriplobe used by other projects
- **VideoCapture**: Can be used by other projects
- **Object detectors**: Can be used with different inference backends

### **User Experience**
- **Simple setup**: One command to setup inference backends
- **Flexible configuration**: Easy to switch between backends
- **Clear documentation**: Each component has its own docs

## 🚨 **Common Misconceptions**

### ❌ **Wrong: This project manages inference backend versions**
```cmake
# This should NOT be in this project
set(ONNX_RUNTIME_VERSION "1.19.2")
```

### ✅ **Correct: This project manages fetched library versions**
```cmake
# This should be in this project
set(INFERENCE_ENGINES_VERSION "v1.0.0")
```

### ❌ **Wrong: This project links inference backend libraries**
```cmake
# This should NOT be in this project
target_link_libraries(${PROJECT_NAME} PRIVATE libonnxruntime.so)
```

### ✅ **Correct: neuriplo handles linking**
```cmake
# This should be in neuriplo library
target_link_libraries(${PROJECT_NAME} PRIVATE neuriplo)
```

## 🔮 **Future Improvements**

### **For This Project:**
1. **Focus on object detection algorithms**
2. **Improve detector implementations**
3. **Add new detector types**
4. **Enhance preprocessing/postprocessing**

### **For neuriplo Library:**
1. **Centralized version management**
2. **Better backend validation**
3. **Automatic backend setup**
4. **Performance benchmarking**

### **For VideoCapture Library:**
1. **More video source support**
2. **Better GStreamer integration**
3. **Cross-platform compatibility**

This architecture ensures that each component has a clear responsibility and can evolve independently while providing a seamless user experience. 
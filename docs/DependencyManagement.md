# Dependency Management Guide

This document describes the improved dependency management system for the object-detection-inference project.

## Overview

The project now uses a **hybrid dependency management approach** that combines:

1. **Centralized Version Management** - All versions in one place
2. **Dependency Validation** - Automatic checks for required files
3. **Unified Setup Scripts** - Single script for all dependencies
4. **CMake ExternalProject** - Alternative automated approach
5. **Docker Integration** - Containerized dependency management

## Project Architecture

### 🎯 **This Project: Object Detectors**
The object-detection-inference project implements **object detection algorithms**:

- **YOLO Series**: YOLOv4, YOLOv5, YOLOv6, YOLOv7, YOLOv8, YOLOv9, YOLOv10, YOLO11, YOLOv12
- **YOLO-NAS**: Neural Architecture Search variant
- **RT-DETR Variants**: RT-DETR, RT-DETR v2, RT-DETR Ultralytics
- **Other Detectors**: D-FINE, DEIM, RF-DETR

### 🔧 **InferenceEngines Library: Inference Backends**
The fetched `InferenceEngines` library provides **inference engine abstractions**:

- **ONNX Runtime**: Microsoft's cross-platform inference engine
- **TensorRT**: NVIDIA's GPU-optimized inference engine
- **LibTorch**: PyTorch's C++ inference engine
- **OpenVINO**: Intel's OpenVINO inference engine
- **OpenCV DNN**: OpenCV's deep learning module
- **TensorFlow**: Google's TensorFlow inference engine

### 📚 **VideoCapture Library: Video Processing**
The fetched `VideoCapture` library handles **video input processing**:

- Unified interface for various video sources
- RTSP stream support
- Optional GStreamer integration

## Quick Start

### Option 1: Unified Setup Script

```bash
# Setup inference backend dependencies for a specific backend
./scripts/setup_dependencies.sh --backend onnx_runtime

# Setup with specific compute platform for LibTorch
./scripts/setup_dependencies.sh --backend libtorch --compute-platform cu118

# Setup all inference backends
./scripts/setup_dependencies.sh --backend all
```

### Option 2: Individual Setup Scripts (Backward Compatible)

```bash
# Individual scripts still work
./scripts/setup_onnx_runtime.sh
./scripts/setup_tensorrt.sh
./scripts/setup_libtorch.sh
./scripts/setup_openvino.sh
```

### Option 3: CMake ExternalProject (Advanced)

```bash
# Build with automatic dependency download
cmake -DDEFAULT_BACKEND=ONNX_RUNTIME -DUSE_EXTERNAL_PROJECT=ON ..
cmake --build .
```

## Centralized Version Management

All dependency versions are now managed in `cmake/versions.cmake`:

```cmake
# External C++ Libraries (fetched via CMake)
set(INFERENCE_ENGINES_VERSION "v1.0.0")  # Inference backend abstractions
set(VIDEOCAPTURE_VERSION "v1.0.0")       # Video processing library

# ML Framework Versions (inference backends)
set(ONNX_RUNTIME_VERSION "1.19.2")       # Microsoft ONNX Runtime
set(TENSORRT_VERSION "10.7.0.23")        # NVIDIA TensorRT
set(LIBTORCH_VERSION "2.0.0")            # PyTorch LibTorch
set(OPENVINO_VERSION "2023.1.0")         # Intel OpenVINO
set(TENSORFLOW_VERSION "2.13.0")         # Google TensorFlow

# Platform-specific paths
set(DEFAULT_DEPENDENCY_ROOT "$ENV{HOME}/dependencies")
```

## Dependency Validation

The system automatically validates dependencies before building:

### What Gets Validated

- **System Dependencies**: OpenCV, glog, CMake version
- **Inference Backends**: Required files and directories for ML frameworks
- **CUDA Support**: GPU acceleration availability
- **Version Compatibility**: Minimum version requirements

### Validation Output

```
=== Validating Dependencies ===
✓ OpenCV 4.8.0 found
✓ glog found
✓ CMake 3.20 found
✓ ONNX Runtime validation passed (inference backend)
✓ CUDA found: 12.6
=== All Dependencies Validated Successfully ===
```

## Supported Components

| Component | Type | Setup Method | Validation | Notes |
|-----------|------|-------------|------------|-------|
| **Object Detectors** | This Project | Built-in | ✓ | YOLO, RT-DETR variants |
| **ONNX Runtime** | Inference Backend | Script/ExternalProject | ✓ | GPU support available |
| **TensorRT** | Inference Backend | Script/ExternalProject | ✓ | Requires NVIDIA account |
| **LibTorch** | Inference Backend | Script/ExternalProject | ✓ | Multiple compute platforms |
| **OpenVINO** | Inference Backend | Manual | ✓ | Complex installation |
| **OpenCV DNN** | Inference Backend | System Package | ✓ | No additional setup needed |
| **TensorFlow** | Inference Backend | System Package | ✓ | Limited support |
| **VideoCapture** | Video Processing | CMake FetchContent | ✓ | Automatic setup |

## Platform Support

### Linux (Primary)
- Full support for all object detectors
- Full support for all inference backends
- Automated setup scripts
- Docker containers available

### macOS (Experimental)
- Limited inference backend support
- Manual installation required for some dependencies

### Windows (Not Supported)
- Currently not supported
- Consider using Docker or WSL

## Advanced Configuration

### Custom Dependency Paths

You can override default paths for inference backends:

```bash
# Set custom dependency root
export DEFAULT_DEPENDENCY_ROOT="/opt/dependencies"

# Or specify individual paths
cmake -DONNX_RUNTIME_DIR="/custom/path" ..
```

### Version Overrides

Override inference backend versions at build time:

```bash
cmake -DONNX_RUNTIME_VERSION="1.18.0" ..
cmake -DLIBTORCH_VERSION="1.13.0" ..
```

### Compute Platform Selection

For LibTorch inference backend, specify the compute platform:

```bash
# CPU only
./scripts/setup_dependencies.sh --backend libtorch --compute-platform cpu

# CUDA 11.8
./scripts/setup_dependencies.sh --backend libtorch --compute-platform cu118

# CUDA 12.1
./scripts/setup_dependencies.sh --backend libtorch --compute-platform cu121

# ROCm 6.0
./scripts/setup_dependencies.sh --backend libtorch --compute-platform rocm6.0
```

## Troubleshooting

### Common Issues

#### 1. Missing System Dependencies

```bash
# Error: Missing system dependencies: cmake wget
# Solution: Install required packages
sudo apt update && sudo apt install -y cmake wget tar unzip libopencv-dev libgoogle-glog-dev
```

#### 2. Permission Denied

```bash
# Error: Permission denied when creating directories
# Solution: Check write permissions
ls -la ~/dependencies
chmod 755 ~/dependencies
```

#### 3. Download Failures

```bash
# Error: Failed to download after 3 attempts
# Solution: Check network connection and retry
./scripts/setup_dependencies.sh --backend onnx_runtime
```

#### 4. CUDA Not Found

```bash
# Warning: CUDA not found. GPU support will be disabled.
# Solution: Install CUDA toolkit
sudo apt install nvidia-cuda-toolkit
```

### Validation Failures

If validation fails, the system provides helpful error messages:

```
[ERROR] ONNX Runtime not found at /home/user/dependencies/onnxruntime-linux-x64-gpu-1.19.2
Please ensure the inference backend is properly installed or run the setup script.

=== Setup Instructions ===
If dependencies are missing, run the following commands:

  ./scripts/setup_onnx_runtime.sh

Or run the unified setup script:
  ./scripts/setup_dependencies.sh --backend ONNX_RUNTIME
```

## Docker Integration

Docker containers handle dependencies automatically:

```bash
# Build with specific inference backend
docker build --rm -t object-detection-inference:onnxruntime \
    -f docker/Dockerfile.onnxruntime .

# Run with GPU support
docker run --gpus all object-detection-inference:onnxruntime \
    --type=yolov8 --weights=model.onnx --source=image.jpg
```

## Migration from Old System

### For Existing Users

1. **Backward Compatibility**: Old scripts still work
2. **Gradual Migration**: Update one inference backend at a time
3. **Version Locking**: Use specific versions instead of `master`

### Breaking Changes

- Git tags now use specific versions instead of `master`
- Dependency paths moved to centralized location
- Validation is now mandatory

### Migration Steps

```bash
# 1. Clean old build
rm -rf build

# 2. Setup inference backend dependencies with new system
./scripts/setup_dependencies.sh --backend your_backend

# 3. Build with new system
mkdir build && cd build
cmake -DDEFAULT_BACKEND=YOUR_BACKEND ..
cmake --build .
```

## Best Practices

### For Developers

1. **Version Pinning**: Always use specific version tags
2. **Validation**: Run validation before committing
3. **Documentation**: Update versions.cmake when adding new dependencies

### For Users

1. **Unified Script**: Use `setup_dependencies.sh` for new installations
2. **Validation**: Check validation output for issues
3. **Clean Builds**: Clean build directory when switching inference backends

### For CI/CD

1. **Docker**: Use Docker containers for consistent environments
2. **Caching**: Cache dependencies between builds
3. **Validation**: Include dependency validation in CI pipeline

## Future Improvements

### Planned Features

1. **Conan Integration**: Package manager support for inference backends
2. **vcpkg Integration**: System package manager
3. **Cross-Platform**: Windows and macOS support
4. **Version Management**: Automated version updates

### Contributing

To improve dependency management:

1. Update `cmake/versions.cmake` for new inference backend versions
2. Add validation in `cmake/DependencyValidation.cmake`
3. Update setup scripts in `scripts/`
4. Test on multiple platforms
5. Update documentation 

# Dependency Management Guide

This document describes the improved dependency management system for the object-detection-inference project.

## Overview

The project now uses a **hybrid dependency management approach** that combines:

1. **Local Version Override System** - Local files override fetched repository versions
2. **Automatic Version Fetching** - Versions sourced from repositories or GitHub
3. **Selective Backend Setup** - Only setup the backend you need
4. **Auto CUDA Detection** - Automatic CUDA version detection for LibTorch
5. **CMake ExternalProject** - Alternative automated approach
6. **Docker Integration** - Containerized dependency management

## Project Architecture

### 🎯 **This Project: Object Detectors**
The object-detection-inference project implements **object detection algorithms**:

- **YOLO Series**: YOLOv4, YOLOv5, YOLOv6, YOLOv7, YOLOv8, YOLOv9, YOLOv10, YOLO11, YOLOv12
- **YOLO-NAS**: Neural Architecture Search variant
- **RT-DETR Variants**: RT-DETR, RT-DETR v2, RT-DETR Ultralytics
- **Other Detectors**: D-FINE, DEIM, RF-DETR

### 🔧 **InferenceEngines Library: Inference Backends**
The `InferenceEngines` library is **automatically fetched** and provides **inference engine abstractions**:

- **OpenCV DNN**: OpenCV's deep learning module (default - no setup required)
- **ONNX Runtime**: Microsoft's cross-platform inference engine
- **TensorRT**: NVIDIA's GPU-optimized inference engine
- **LibTorch**: PyTorch's C++ inference engine
- **OpenVINO**: Intel's OpenVINO inference engine
- **TensorFlow**: Google's TensorFlow inference engine

### 📚 **VideoCapture Library: Video Processing**
The fetched `VideoCapture` library handles **video input processing**:

- Unified interface for various video sources
- RTSP stream support
- Optional GStreamer integration

## Quick Start

### 🚀 **Default Setup (OpenCV DNN - No Additional Dependencies)**

```bash
# Setup default backend (automatically ensures version files exist)
./scripts/setup_dependencies.sh

# Build project
mkdir build && cd build
cmake ..
cmake --build .
```

### 🔧 **Alternative Backends**

```bash
# Setup ONNX Runtime dependencies (automatically ensures version files exist)
./scripts/setup_dependencies.sh --backend onnx_runtime

# Setup LibTorch with GPU support (auto-detects CUDA version)
./scripts/setup_dependencies.sh --backend libtorch --compute-platform gpu
# or equivalently:
./scripts/setup_dependencies.sh --backend libtorch --compute-platform cuda

# Setup all inference backends
./scripts/setup_dependencies.sh --backend all
```

### 📚 **Advanced Setup**

```bash
# Update backend versions from repositories
./scripts/update_backend_versions.sh --show-versions

# CMake ExternalProject (automatic download)
cmake -DDEFAULT_BACKEND=ONNX_RUNTIME -DUSE_EXTERNAL_PROJECT=ON ..
```

## Version Management System

The project uses a sophisticated version management system with local override capabilities:

### 📁 **Version File Structure**

```
object-detection-inference/
├── versions.inference-engines.env    # Overrides InferenceEngines versions (if present)
├── versions.videocapture.env         # Overrides VideoCapture versions (if present)
├── scripts/
│   ├── setup_dependencies.sh         # Main setup script
│   ├── update_backend_versions.sh    # Version management script
│   └── setup_*.sh                   # Individual backend scripts
└── build/_deps/
    ├── inferenceengines-src/versions.env  # Source InferenceEngines versions
    └── videocapture-src/versions.env     # Source VideoCapture versions
```

**Behavior**: Local version files **override** fetched repository versions **if present**, otherwise they are **created by copying** from the original repositories.

### 🔄 **Version Priority System**

1. **Local Override Files** (highest priority)
   - `versions.inference-engines.env` - **Overrides** InferenceEngines versions **if present**
   - `versions.videocapture.env` - **Overrides** VideoCapture versions **if present**

2. **Auto-Created Local Files** (medium priority)
   - If local files don't exist, they are **automatically created** by copying from:
     - `build/_deps/inferenceengines-src/versions.env` (if available)
     - `build/_deps/videocapture-src/versions.env` (if available)

3. **GitHub Fallback** (lowest priority)
   - If fetched repositories are not available, direct download from repository GitHub URLs

### 📋 **Version Management Commands**

```bash
# Auto-update versions (copied from repositories on first run)
./scripts/update_backend_versions.sh

# Force update from repositories
./scripts/update_backend_versions.sh --force

# View current versions
./scripts/update_backend_versions.sh --show-versions

# Update only InferenceEngines versions
./scripts/update_backend_versions.sh --inference-engines --show-versions

# Update only VideoCapture versions
./scripts/update_backend_versions.sh --videocapture --show-versions
```

**Note**: The `setup_dependencies.sh` script automatically calls `update_backend_versions.sh` to ensure version files exist before proceeding with dependency setup. Local version files **override** fetched repository versions **if present**, otherwise they are **created by copying** from the original repositories.

## Backend Setup Process

### 🎯 **Selective Setup**

The setup script now only installs and validates the **selected backend**:

```bash
# Only setup ONNX Runtime (not all backends)
./scripts/setup_dependencies.sh --backend onnx_runtime

# Only setup LibTorch with GPU
./scripts/setup_dependencies.sh --backend libtorch --compute-platform gpu

# Default: OpenCV DNN (no setup required)
./scripts/setup_dependencies.sh
```

### 🔍 **Auto CUDA Detection for LibTorch**

When using `--compute-platform gpu` or `--compute-platform cuda` (equivalent), the script automatically:

1. **Reads CUDA version** from `versions.inference-engines.env`
2. **Maps CUDA version** to PyTorch compute platform:
   - CUDA 12.6-12.8 → `cu121`
   - CUDA 12.0-12.5 → `cu118`
   - CUDA 11.8 → `cu118`
   - Unknown → `cu118` (fallback)

```bash
# Automatically detects CUDA 12.6 and uses cu121
./scripts/setup_dependencies.sh --backend libtorch --compute-platform gpu

# Manual override still works
./scripts/setup_dependencies.sh --backend libtorch --compute-platform cu118
```

## Dependency Validation

The system automatically validates dependencies before building:

### What Gets Validated

- **System Dependencies**: OpenCV, glog, CMake version
- **Selected Backend**: Only the backend you're using
- **CUDA Support**: GPU acceleration availability (if applicable)
- **Version Compatibility**: Minimum version requirements

### Validation Output

```
=== Validating Dependencies ===
✓ OpenCV 4.8.0 found
✓ glog found
✓ CMake 3.20 found
✓ ONNX Runtime validation passed (selected backend)
✓ CUDA found: 12.6
=== All Dependencies Validated Successfully ===
```

## Supported Components

| Component | Type | Setup Method | Validation | Notes |
|-----------|------|-------------|------------|-------|
| **Object Detectors** | This Project | Built-in | ✓ | YOLO, RT-DETR variants |
| **OpenCV DNN** | Inference Backend | System Package | ✓ | Default - no setup needed |
| **ONNX Runtime** | Inference Backend | Script/ExternalProject | ✓ | GPU support available |
| **TensorRT** | Inference Backend | Script/ExternalProject | ✓ | Requires NVIDIA account |
| **LibTorch** | Inference Backend | Script/ExternalProject | ✓ | Auto CUDA detection |
| **OpenVINO** | Inference Backend | Manual | ✓ | Complex installation |
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

Override inference backend versions using local files:

```bash
# Edit local version file
nano versions.inference-engines.env

# Or override at build time
cmake -DONNX_RUNTIME_VERSION="1.18.0" ..
cmake -DLIBTORCH_VERSION="1.13.0" ..
```

### Compute Platform Selection

For LibTorch inference backend, specify the compute platform:

```bash
# CPU only
./scripts/setup_dependencies.sh --backend libtorch --compute-platform cpu

# GPU with auto CUDA detection (equivalent commands)
./scripts/setup_dependencies.sh --backend libtorch --compute-platform gpu
./scripts/setup_dependencies.sh --backend libtorch --compute-platform cuda

# Manual CUDA version
./scripts/setup_dependencies.sh --backend libtorch --compute-platform cu118

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

#### 2. Version File Issues

```bash
# Error: CUDA version not found in versions.inference-engines.env
# Solution: Update version files
./scripts/update_backend_versions.sh --force

# Or manually set CUDA version
echo "CUDA_VERSION=12.6" >> versions.inference-engines.env
```

#### 3. Backend Not Found

```bash
# Error: LibTensorFlow not found
# Solution: Use a different backend or setup dependencies
./scripts/setup_dependencies.sh --backend opencv_dnn  # Use default backend
```

#### 4. Permission Denied

```bash
# Error: Permission denied when creating directories
# Solution: Check write permissions
ls -la ~/dependencies
chmod 755 ~/dependencies
```

### Validation Failures

If validation fails, the system provides helpful error messages:

```
[ERROR] ONNX Runtime not found at /home/user/dependencies/onnxruntime-linux-x64-gpu-1.19.2
Please ensure the inference backend is properly installed or run the setup script.

=== Setup Instructions ===
If dependencies are missing, run the following commands:

  ./scripts/setup_dependencies.sh --backend onnx_runtime
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

## Best Practices

### For Developers

1. **Version Pinning**: Always use specific version tags
2. **Local Overrides**: Use local version files for custom requirements
3. **Validation**: Run validation before committing
4. **Documentation**: Update version files when adding new dependencies

### For Users

1. **Default Backend**: Start with OpenCV DNN (no setup required)
2. **Selective Setup**: Only setup the backend you need
3. **Version Management**: Use `update_backend_versions.sh` to manage versions
4. **Clean Builds**: Clean build directory when switching inference backends

### For CI/CD

1. **Docker**: Use Docker containers for consistent environments
2. **Caching**: Cache dependencies between builds
3. **Validation**: Include dependency validation in CI pipeline
4. **Version Files**: Include local version files in CI

## Future Improvements

### Planned Features

1. **Conan Integration**: Package manager support for inference backends
2. **vcpkg Integration**: System package manager
3. **Cross-Platform**: Windows and macOS support
4. **Automated Updates**: Automated version updates from repositories

### Contributing

To improve dependency management:

1. Update version files for new inference backend versions
2. Add validation in `cmake/DependencyValidation.cmake`
3. Update setup scripts in `scripts/`
4. Test on multiple platforms
5. Update documentation 

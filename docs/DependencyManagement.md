# Dependency Management Guide

This document describes the improved dependency management system for the object-detection-inference project.

## Overview

The project now uses a **dependency management approach** that combines:

**Local Version Override System** - Local files override fetched repository versions
**Automatic Version Fetching** - Versions sourced from repositories or GitHub
**Selective Backend Setup** - Only setup the backend you need
**Docker Integration** - Containerized dependency management


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

See [README.md](../README.md) for quick setup examples for all backends.

## Version Management System

The project uses a version management system with local override capabilities:

### 📁 **Version File Structure**

```
object-detection-inference/
├── versions.env # Dependencies needed by this project
├── versions.inference-engines.env    # Overrides InferenceEngines versions (if present), otherwise will be automatically created and fetched from InferenceEngines repository
├── versions.videocapture.env         # Overrides VideoCapture versions (if present), otherwise will be automatically created and fetched from VideoCapture repository
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
   - If versions.env above fetched repositories are not available, direct download from repository GitHub URLs [inference-engines](https://github.com/olibartfast/inference-engines) and [videocapture](https://github.com/olibartfast/videocapture)


## Backend Setup Process

### 🎯 **Selective Setup**

The setup script now only installs and validates the **selected backend**. See [README.md](../README.md) for quick setup examples.

### 🔍 **LibTorch with CUDA support**

When using `--compute-platform gpu` or `--compute-platform cuda`, the script automatically detects your CUDA version and downloads the appropriate LibTorch build:

#### **How it works:**
1. **Reads CUDA version** from `versions.inference-engines.env` (e.g., `CUDA_VERSION=12.6`)
2. **Downloads the correct LibTorch version** based on your CUDA version:
   - CUDA 12.8 → Downloads LibTorch with CUDA 12.8 support (`cu128`)
   - CUDA 12.6 → Downloads LibTorch with CUDA 12.6 support (`cu126`)
   - CUDA 12.0-12.5 → Downloads LibTorch with CUDA 11.8 support (`cu118`)
   - CUDA 11.8 → Downloads LibTorch with CUDA 11.8 support (`cu118`)
   - Unknown CUDA → Downloads LibTorch with CUDA 11.8 support (`cu118`) as fallback

#### **Examples:**
```bash
# If CUDA_VERSION=12.6 in versions.inference-engines.env:
./scripts/setup_dependencies.sh --backend libtorch --compute-platform gpu
# Downloads: libtorch-cxx11-abi-shared-with-deps-2.3.0+cu126.zip

# If CUDA_VERSION=12.8 in versions.inference-engines.env:
./scripts/setup_dependencies.sh --backend libtorch --compute-platform gpu
# Downloads: libtorch-cxx11-abi-shared-with-deps-2.3.0+cu128.zip

# If CUDA_VERSION=11.8 in versions.inference-engines.env:
./scripts/setup_dependencies.sh --backend libtorch --compute-platform gpu
# Downloads: libtorch-cxx11-abi-shared-with-deps-2.3.0+cu118.zip

# Manual override (ignores auto-detection):
./scripts/setup_dependencies.sh --backend libtorch --compute-platform cu118
# Downloads: libtorch-cxx11-abi-shared-with-deps-2.3.0+cu118.zip
```

#### **Note:**
- The script downloads pre-built LibTorch binaries from PyTorch's official repository
- The `cu121` and `cu118` refer to the PyTorch build version, not your system's CUDA version
- Your system CUDA version must be compatible with the downloaded LibTorch build

## Dependency Validation

The system automatically validates dependencies before building:

### What Gets Validated

- **System Dependencies**: OpenCV, glog, CMake version
- **Selected Backend**: Only the backend you're using
- **CUDA Support**: GPU acceleration availability (if applicable)
- **Version Compatibility**: Minimum version requirements

### Validation Output Example

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
| **VideoCapture** | Video Processing | CMake FetchContent | ✓ | Automatic setup |
| **InferenceEngines** | Inference Backend Manager | CMake FetchContent | ✓ | Automatic setup |
| **OpenCV DNN** | Inference Backend | System Package | ✓ | Default - it comes with OpenCV installation, no setup needed for CPU inference, to support multiple inference backends you must customize the building process |
| **ONNX Runtime** | Inference Backend | Script| ✓ | CPU/GPU support available based on download binaries and local hardware available|
| **TensorRT** | Inference Backend | Script | ✓ | Requires NVIDIA account to download the binaries |
| **LibTorch** | Inference Backend | Script | ✓ | CPU/GPU support available based on download binaries and local hardware available |
| **OpenVINO** | Inference Backend | Script | ✓ | Complex installation |
| **TensorFlow** | Inference Backend | Script | ✓ | Complex installation |

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

For LibTorch inference backend, specify the compute platform. See the [Auto CUDA Detection](#-auto-cuda-detection-for-libtorch) section above for detailed examples.

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
# Solution: The setup script should handle this automatically, but you can manually set CUDA version
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
3. **Automatic Version Management**: Version files are managed automatically by setup scripts
4. **Clean Builds**: Clean build directory when switching inference backends

### For CI/CD

1. **Docker**: Use Docker containers for consistent environments
2. **Caching**: Cache dependencies between builds
3. **Validation**: Include dependency validation in CI pipeline
4. **Automatic Setup**: Version files are managed automatically by setup scripts

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

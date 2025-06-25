#!/bin/bash

# Unified dependency setup script for inference backend dependencies
# This script handles installation of inference backend dependencies that will be used
# by the InferenceEngines library, which is fetched by this project.

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to detect OS
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macos"
    elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
        echo "windows"
    else
        echo "unknown"
    fi
}

# Function to get dependency root directory
get_dependency_root() {
    local os=$(detect_os)
    if [[ "$os" == "windows" ]]; then
        echo "$USERPROFILE/dependencies"
    else
        echo "$HOME/dependencies"
    fi
}

# Function to create directory if it doesn't exist
ensure_directory() {
    if [[ ! -d "$1" ]]; then
        print_status "Creating directory: $1"
        mkdir -p "$1"
    fi
}

# Function to download file with retry
download_with_retry() {
    local url="$1"
    local output="$2"
    local max_retries=3
    local retry_count=0
    
    while [[ $retry_count -lt $max_retries ]]; do
        if wget --tries=3 --retry-connrefused -O "$output" "$url"; then
            return 0
        fi
        retry_count=$((retry_count + 1))
        print_warning "Download failed, retrying... ($retry_count/$max_retries)"
        sleep 2
    done
    
    print_error "Failed to download $url after $max_retries attempts"
    return 1
}

# Function to setup ONNX Runtime
setup_onnx_runtime() {
    local version="1.19.2"
    local dependency_root=$(get_dependency_root)
    local install_dir="$dependency_root/onnxruntime-linux-x64-gpu-$version"
    
    print_status "Setting up ONNX Runtime $version (for InferenceEngines library)..."
    
    if [[ -d "$install_dir" ]]; then
        print_warning "ONNX Runtime already exists at $install_dir"
        return 0
    fi
    
    ensure_directory "$dependency_root"
    
    local temp_dir=$(mktemp -d)
    cd "$temp_dir"
    
    local filename="onnxruntime-linux-x64-gpu-$version.tgz"
    local url="https://github.com/microsoft/onnxruntime/releases/download/v$version/$filename"
    
    print_status "Downloading ONNX Runtime..."
    if download_with_retry "$url" "$filename"; then
        print_status "Extracting ONNX Runtime..."
        tar -xzf "$filename"
        mv "onnxruntime-linux-x64-gpu-$version" "$install_dir"
        print_success "ONNX Runtime installed to $install_dir"
    else
        print_error "Failed to setup ONNX Runtime"
        return 1
    fi
    
    cd - > /dev/null
    rm -rf "$temp_dir"
}

# Function to setup TensorRT
setup_tensorrt() {
    local version="10.7.0.23"
    local cuda_version="12.6"
    local dependency_root=$(get_dependency_root)
    local install_dir="$dependency_root/TensorRT-$version"
    
    print_status "Setting up TensorRT $version (for InferenceEngines library)..."
    
    if [[ -d "$install_dir" ]]; then
        print_warning "TensorRT already exists at $install_dir"
        return 0
    fi
    
    ensure_directory "$dependency_root"
    
    local temp_dir=$(mktemp -d)
    cd "$temp_dir"
    
    local filename="TensorRT-$version.Linux.x86_64-gnu.cuda-$cuda_version.tar.gz"
    local url="https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.7.0/tars/$filename"
    
    print_status "Downloading TensorRT..."
    if download_with_retry "$url" "$filename"; then
        print_status "Extracting TensorRT..."
        tar -xzf "$filename"
        mv "TensorRT-$version" "$install_dir"
        print_success "TensorRT installed to $install_dir"
    else
        print_error "Failed to setup TensorRT"
        return 1
    fi
    
    cd - > /dev/null
    rm -rf "$temp_dir"
}

# Function to setup LibTorch
setup_libtorch() {
    local version="2.0.0"
    local dependency_root=$(get_dependency_root)
    local install_dir="$dependency_root/libtorch"
    
    print_status "Setting up LibTorch $version (for InferenceEngines library)..."
    
    if [[ -d "$install_dir" ]]; then
        print_warning "LibTorch already exists at $install_dir"
        return 0
    fi
    
    ensure_directory "$dependency_root"
    
    local temp_dir=$(mktemp -d)
    cd "$temp_dir"
    
    # Default to CPU version, can be overridden
    local compute_platform=${COMPUTE_PLATFORM:-cpu}
    local filename="libtorch-cxx11-abi-shared-with-deps-$version+$compute_platform.zip"
    local url="https://download.pytorch.org/libtorch/$compute_platform/$filename"
    
    print_status "Downloading LibTorch ($compute_platform)..."
    if download_with_retry "$url" "$filename"; then
        print_status "Extracting LibTorch..."
        unzip -q "$filename"
        mv "libtorch" "$install_dir"
        print_success "LibTorch installed to $install_dir"
    else
        print_error "Failed to setup LibTorch"
        return 1
    fi
    
    cd - > /dev/null
    rm -rf "$temp_dir"
}

# Function to setup OpenVINO
setup_openvino() {
    local version="2023.1.0"
    local dependency_root=$(get_dependency_root)
    local install_dir="$dependency_root/openvino-$version"
    
    print_status "Setting up OpenVINO $version (for InferenceEngines library)..."
    
    if [[ -d "$install_dir" ]]; then
        print_warning "OpenVINO already exists at $install_dir"
        return 0
    fi
    
    print_warning "OpenVINO setup requires manual installation. Please visit:"
    print_warning "https://docs.openvino.ai/latest/openvino_docs_install_guides_installing_openvino_linux.html"
    print_warning "After installation, update the InferenceEngines library configuration"
}

# Function to check system dependencies
check_system_dependencies() {
    print_status "Checking system dependencies..."
    
    local missing_deps=()
    
    # Check for required commands
    local required_commands=("cmake" "wget" "tar" "unzip")
    for cmd in "${required_commands[@]}"; do
        if ! command_exists "$cmd"; then
            missing_deps+=("$cmd")
        fi
    done
    
    # Check for required packages (Linux)
    if [[ "$(detect_os)" == "linux" ]]; then
        if ! dpkg -l | grep -q "libopencv-dev"; then
            missing_deps+=("libopencv-dev")
        fi
        if ! dpkg -l | grep -q "libgoogle-glog-dev"; then
            missing_deps+=("libgoogle-glog-dev")
        fi
    fi
    
    if [[ ${#missing_deps[@]} -gt 0 ]]; then
        print_error "Missing system dependencies: ${missing_deps[*]}"
        print_status "Please install them using your package manager:"
        if [[ "$(detect_os)" == "linux" ]]; then
            print_status "sudo apt update && sudo apt install -y ${missing_deps[*]}"
        fi
        return 1
    fi
    
    print_success "All system dependencies are available"
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "This script sets up inference backend dependencies for the InferenceEngines library."
    echo "The InferenceEngines library is fetched by this project and provides inference backend abstractions."
    echo ""
    echo "Options:"
    echo "  --backend BACKEND    Specify inference backend to setup (onnx_runtime, tensorrt, libtorch, openvino, all)"
    echo "  --compute-platform   Specify compute platform for LibTorch (cpu, cu118, cu121, rocm6.0)"
    echo "  --help              Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --backend onnx_runtime"
    echo "  $0 --backend libtorch --compute-platform cu118"
    echo "  $0 --backend all"
    echo ""
    echo "Note: These dependencies are used by the InferenceEngines library, not this project directly."
}

# Main script
main() {
    local backend=""
    local compute_platform="cpu"
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --backend)
                backend="$2"
                shift 2
                ;;
            --compute-platform)
                compute_platform="$2"
                shift 2
                ;;
            --help)
                show_usage
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
    
    # Set compute platform for LibTorch
    export COMPUTE_PLATFORM="$compute_platform"
    
    print_status "Starting inference backend dependency setup..."
    print_status "OS detected: $(detect_os)"
    print_status "Dependency root: $(get_dependency_root)"
    print_status "Note: These dependencies are for the InferenceEngines library"
    
    # Check system dependencies first
    if ! check_system_dependencies; then
        exit 1
    fi
    
    # Setup based on backend
    case "$backend" in
        onnx_runtime)
            setup_onnx_runtime
            ;;
        tensorrt)
            setup_tensorrt
            ;;
        libtorch)
            setup_libtorch
            ;;
        openvino)
            setup_openvino
            ;;
        all)
            setup_onnx_runtime
            setup_tensorrt
            setup_libtorch
            setup_openvino
            ;;
        "")
            print_error "No backend specified. Use --backend option."
            show_usage
            exit 1
            ;;
        *)
            print_error "Unknown backend: $backend"
            print_status "Supported backends: onnx_runtime, tensorrt, libtorch, openvino, all"
            exit 1
            ;;
    esac
    
    print_success "Inference backend dependency setup completed successfully!"
    print_status "These dependencies will be used by the InferenceEngines library."
    print_status "You can now build the project with:"
    print_status "mkdir build && cd build"
    print_status "cmake -DDEFAULT_BACKEND=${backend^^} .."
    print_status "cmake --build ."
}

# Run main function with all arguments
main "$@" 
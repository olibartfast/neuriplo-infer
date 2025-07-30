#!/bin/bash

# Unified dependency setup script for inference backend dependencies
# This script fetches backend versions from InferenceEngines repo and sets up dependencies

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

# Function to fetch versions from InferenceEngines repo
fetch_backend_versions() {
    local local_versions_file="versions.inference-engines.env"
    
    # Check if local versions file exists (should be created by update_backend_versions.sh)
    if [[ -f "$local_versions_file" ]]; then
        print_status "Using local InferenceEngines versions from $local_versions_file"
        source "$local_versions_file"
    else
        print_error "Local InferenceEngines versions file not found: $local_versions_file"
        print_status "This should have been created by update_backend_versions.sh"
        print_status "Please check if the script ran successfully or try manually:"
        print_status "./scripts/update_backend_versions.sh"
        return 1
    fi
    
    print_status "Fetched InferenceEngines backend versions:"
    print_status "ONNX Runtime: $ONNX_RUNTIME_VERSION"
    print_status "TensorRT: $TENSORRT_VERSION"
    print_status "LibTorch: $PYTORCH_VERSION"
    print_status "OpenVINO: $OPENVINO_VERSION"
    print_status "TensorFlow: $TENSORFLOW_VERSION"
    print_status "CUDA: $CUDA_VERSION"
}

# Function to setup ONNX Runtime
setup_onnx_runtime() {
    local version="$ONNX_RUNTIME_VERSION"
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
    local version="$TENSORRT_VERSION"
    local cuda_version="$CUDA_VERSION"
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
    local version="$PYTORCH_VERSION"
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
    
    # Determine compute platform based on CUDA version and user preference
    local compute_platform=${COMPUTE_PLATFORM:-cpu}
    
    # If user wants GPU, automatically determine CUDA version from versions file
    if [[ "$compute_platform" == "gpu" || "$compute_platform" == "cuda" ]]; then
        if [[ -n "$CUDA_VERSION" ]]; then
            # Map CUDA version to PyTorch compute platform
            case "$CUDA_VERSION" in
                "12.6"|"12.7"|"12.8")
                    compute_platform="cu121"
                    print_status "Using CUDA $CUDA_VERSION -> compute platform: $compute_platform"
                    ;;
                "12.4"|"12.5")
                    compute_platform="cu118"
                    print_status "Using CUDA $CUDA_VERSION -> compute platform: $compute_platform"
                    ;;
                "12.0"|"12.1"|"12.2"|"12.3")
                    compute_platform="cu118"
                    print_status "Using CUDA $CUDA_VERSION -> compute platform: $compute_platform"
                    ;;
                "11.8")
                    compute_platform="cu118"
                    print_status "Using CUDA $CUDA_VERSION -> compute platform: $compute_platform"
                    ;;
                *)
                    print_warning "Unknown CUDA version $CUDA_VERSION, using cu118 as fallback"
                    compute_platform="cu118"
                    ;;
            esac
        else
            print_error "CUDA version not found in versions.inference-engines.env"
            print_status "Please set CUDA_VERSION in versions.inference-engines.env or use --compute-platform explicitly"
            return 1
        fi
    fi
    
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
    local version="$OPENVINO_VERSION"
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
    echo "It automatically ensures version files exist and installs the selected backend dependencies."
    echo ""
    echo "Options:"
    echo "  --backend <backend>        Specify the inference backend to setup"
    echo "                             Supported: opencv_dnn, onnx_runtime, tensorrt, libtorch, openvino, tensorflow, all"
    echo "                             Default: opencv_dnn (no setup required)"
    echo "  --compute-platform <platform>  For LibTorch: cpu, gpu/cuda, cu118, cu121, rocm6.0"
    echo "                             Default: cpu"
    echo "                             Note: 'gpu' and 'cuda' are equivalent and auto-detect CUDA version"
    echo "  --help                     Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Setup default backend (OPENCV_DNN)"
    echo "  $0 --backend onnx_runtime            # Setup ONNX Runtime"
    echo "  $0 --backend libtorch --compute-platform gpu  # Setup LibTorch with auto CUDA detection"
    echo "  $0 --backend libtorch --compute-platform cuda # Same as above"
    echo "  $0 --backend all                     # Setup all inference backends"
    echo ""
    echo "Note: This script automatically calls update_backend_versions.sh to ensure version files exist."
    echo "Note: These dependencies are used by the InferenceEngines library, not this project directly."
}

# Main script
main() {
    local backend="opencv_dnn"  # Default to OPENCV_DNN
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
    print_status "Selected backend: $backend"
    print_status "Note: These dependencies are for the InferenceEngines library"
    
    # Automatically ensure version files exist
    print_status "Ensuring version files are available..."
    if [[ -f "scripts/update_backend_versions.sh" ]]; then
        if ! ./scripts/update_backend_versions.sh; then
            print_warning "Failed to update version files, continuing with available versions..."
        fi
    else
        print_warning "update_backend_versions.sh not found, continuing with available versions..."
    fi
    
    # Check system dependencies first
    if ! check_system_dependencies; then
        exit 1
    fi
    
    # Fetch backend versions from InferenceEngines repo
    if ! fetch_backend_versions; then
        exit 1
    fi
    
    # Setup based on backend
    case "$backend" in
        opencv_dnn)
            print_status "OPENCV_DNN is the default backend - no additional setup required"
            print_success "System dependencies (OpenCV) are already validated"
            ;;
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
        tensorflow)
            setup_tensorflow
            ;;
        all)
            setup_onnx_runtime
            setup_tensorrt
            setup_libtorch
            setup_openvino
            setup_tensorflow
            ;;
        *)
            print_error "Unknown backend: $backend"
            print_status "Supported backends: opencv_dnn, onnx_runtime, tensorrt, libtorch, openvino, tensorflow, all"
            exit 1
            ;;
    esac
    
    print_success "Inference backend dependency setup completed successfully!"
    print_status "These dependencies will be used by the InferenceEngines library."
    print_status "You can now build the project with:"
    print_status "mkdir build && cd build"
    if [[ "$backend" == "opencv_dnn" ]]; then
        print_status "cmake ..  # Uses OPENCV_DNN by default"
    else
        print_status "cmake -DDEFAULT_BACKEND=${backend^^} .."
    fi
    print_status "cmake --build ."
}

# Run main function with all arguments
main "$@" 
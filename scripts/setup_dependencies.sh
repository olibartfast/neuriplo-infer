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
        # Use curl for better URL handling (especially for URLs with + characters)
        if curl -L -o "$output" "$url"; then
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
    local local_versions_file="versions.neuriplo.env"
    
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
                "12.8")
                    compute_platform="cu128"
                    print_status "Using CUDA $CUDA_VERSION -> compute platform: $compute_platform"
                    ;;
                "12.6"|"12.7")
                    compute_platform="cu126"
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
            print_error "CUDA version not found in versions.neuriplo.env"
            print_status "Please set CUDA_VERSION in versions.neuriplo.env or use --compute-platform explicitly"
            return 1
        fi
    fi
    
    local filename="libtorch-cxx11-abi-shared-with-deps-$version+$compute_platform.zip"
    # URL encode the filename (replace + with %2B)
    local encoded_filename=$(echo "$filename" | sed 's/+/%2B/g')
    local url="https://download.pytorch.org/libtorch/$compute_platform/$encoded_filename"
    
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
    local dir="$dependency_root/openvino_$version"
    
    print_status "Setting up OpenVINO $version (for InferenceEngines library)..."
    
    if [[ -d "$dir" && "$FORCE" != "true" ]]; then
        print_warning "OpenVINO already exists at $dir"
        return 0
    fi
    
    print_status "Installing OpenVINO $version to $dir..."
    mkdir -p "$dependency_root" && cd "$dependency_root"
    
    # Download OpenVINO toolkit
    local tarball="openvino_2025.2.0.tgz"
    if [[ ! -f "$tarball" ]]; then
        print_status "Downloading OpenVINO toolkit..."
        curl -L "https://storage.openvinotoolkit.org/repositories/openvino/packages/2025.2/linux/openvino_toolkit_ubuntu24_2025.2.0.19140.c01cd93e24d_x86_64.tgz" --output "$tarball"
    fi
    
    # Extract and move to final location
    print_status "Extracting OpenVINO..."
    tar -xf "$tarball"
    if [[ -d "$dir" ]]; then
        rm -rf "$dir"
    fi
    mv openvino_toolkit_ubuntu24_2025.2.0.19140.c01cd93e24d_x86_64 "$dir"
    rm -f "$tarball"
    
    # Create a local Python virtual environment for OpenVINO tools
    print_status "Setting up OpenVINO Python tools..."
    local venv_dir="$dir/python_env"
    python3 -m venv "$venv_dir"
    source "$venv_dir/bin/activate"
    pip install openvino-dev
    deactivate
    
    # Create wrapper script for ovc
    mkdir -p "$dir/bin"
    cat > "$dir/bin/ovc" << 'EOF'
#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/../python_env"
source "$VENV_DIR/bin/activate"
ovc "$@"
deactivate
EOF
    chmod +x "$dir/bin/ovc"
    
    print_success "OpenVINO $version installed successfully to $dir"
}

# Function to setup TensorFlow
setup_tensorflow() {
    local version="$TENSORFLOW_VERSION"
    local dependency_root=$(get_dependency_root)
    local tensorflow_dir="$dependency_root/tensorflow"
    local venv_dir="$dependency_root/tensorflow_env"
    
    print_status "Setting up TensorFlow $version (for InferenceEngines library)..."
    
    # Check Python
    if ! command -v python3 &>/dev/null; then
        print_error "Python 3 required for TensorFlow installation"
        return 1
    fi
    
    # Check if already installed
    if [[ -d "$tensorflow_dir" ]]; then
        print_warning "TensorFlow C++ libraries already installed at $tensorflow_dir"
        return 0
    fi
    
    print_status "Creating Python virtual environment..."
    mkdir -p "$dependency_root"
    if [[ ! -d "$venv_dir" ]]; then
        python3 -m venv "$venv_dir" || {
            print_error "Failed to create Python virtual environment"
            return 1
        }
    fi
    
    print_status "Activating virtual environment and installing TensorFlow..."
    source "$venv_dir/bin/activate" || {
        print_error "Failed to activate virtual environment"
        return 1
    }
    
    # Install TensorFlow
    pip install --upgrade pip || {
        print_error "Failed to upgrade pip"
        return 1
    }
    
    pip install "tensorflow==$version" || {
        print_error "TensorFlow installation failed"
        return 1
    }
    
    # Verify installation
    print_status "Verifying TensorFlow installation..."
    python -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}')" || {
        print_error "TensorFlow verification failed"
        return 1
    }
    
    print_status "Setting up TensorFlow C++ libraries..."
    
    # Find TensorFlow site-packages in the virtual environment
    local python_version=$(source "$venv_dir/bin/activate" && python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    local tf_site_packages="$venv_dir/lib/python$python_version/site-packages/tensorflow"
    
    if [[ ! -d "$tf_site_packages" ]]; then
        print_error "TensorFlow site-packages not found at $tf_site_packages"
        print_status "Available directories in site-packages:"
        ls -la "$venv_dir/lib/python$python_version/site-packages/" || true
        return 1
    fi
    
    # Create TensorFlow C++ directory
    rm -rf "$tensorflow_dir"
    mkdir -p "$tensorflow_dir/lib" "$tensorflow_dir/include"
    
    # Copy libraries (activate venv for each operation)
    print_status "Copying TensorFlow libraries..."
    if [[ -f "$tf_site_packages/libtensorflow_cc.so.2" ]]; then
        cp "$tf_site_packages/libtensorflow_cc.so.2" "$tensorflow_dir/lib/"
        ln -sf "libtensorflow_cc.so.2" "$tensorflow_dir/lib/libtensorflow_cc.so"
    else
        print_warning "libtensorflow_cc.so.2 not found, searching for alternatives..."
        find "$tf_site_packages" -name "libtensorflow_cc.so*" -exec cp {} "$tensorflow_dir/lib/" \; || {
            print_warning "No libtensorflow_cc.so found in $tf_site_packages"
            print_status "Available files in TensorFlow site-packages:"
            find "$tf_site_packages" -name "*.so*" | head -10 || true
        }
    fi
    
    if [[ -f "$tf_site_packages/libtensorflow_framework.so.2" ]]; then
        cp "$tf_site_packages/libtensorflow_framework.so.2" "$tensorflow_dir/lib/"
        ln -sf "libtensorflow_framework.so.2" "$tensorflow_dir/lib/libtensorflow_framework.so"
    else
        print_warning "libtensorflow_framework.so.2 not found, searching for alternatives..."
        find "$tf_site_packages" -name "libtensorflow_framework.so*" -exec cp {} "$tensorflow_dir/lib/" \; || {
            print_warning "No libtensorflow_framework.so found in $tf_site_packages"
        }
    fi
    
    # Copy headers
    if [[ -d "$tf_site_packages/include" ]]; then
        cp -r "$tf_site_packages/include/"* "$tensorflow_dir/include/"
    fi
    
    # Copy additional headers if available
    if [[ -d "$tf_site_packages/core" ]]; then
        mkdir -p "$tensorflow_dir/include/tensorflow/"
        cp -r "$tf_site_packages/core" "$tensorflow_dir/include/tensorflow/"
    fi
    
    if [[ -d "$tf_site_packages/cc" ]]; then
        mkdir -p "$tensorflow_dir/include/tensorflow/"
        cp -r "$tf_site_packages/cc" "$tensorflow_dir/include/tensorflow/"
    fi
    
    # Create pkg-config file
    mkdir -p "$tensorflow_dir/lib/pkgconfig"
    cat > "$tensorflow_dir/lib/pkgconfig/tensorflow.pc" << EOF
prefix=$tensorflow_dir
libdir=\${prefix}/lib
includedir=\${prefix}/include
Name: TensorFlow
Version: $version
Libs: -L\${libdir} -ltensorflow_cc -ltensorflow_framework
Cflags: -I\${includedir}
EOF
    
    # Setup environment
    local env_file="$dependency_root/setup_env.sh"
    if [[ -f "$env_file" ]]; then
        grep -v "TensorFlow\|TENSORFLOW" "$env_file" > "${env_file}.tmp" 2>/dev/null || true
        mv "${env_file}.tmp" "$env_file" 2>/dev/null || true
    fi
    
    cat >> "$env_file" << EOF
export TENSORFLOW_DIR="$tensorflow_dir"
export LD_LIBRARY_PATH="\$TENSORFLOW_DIR/lib:\${LD_LIBRARY_PATH:-}"
export PKG_CONFIG_PATH="\$TENSORFLOW_DIR/lib/pkgconfig:\${PKG_CONFIG_PATH:-}"
EOF
    
    print_success "TensorFlow C++ library installed successfully at $tensorflow_dir"
    print_status "You can now build the project with:"
    print_status "cmake -DDEFAULT_BACKEND=LIBTENSORFLOW .."
    print_status "Source $env_file to use TensorFlow environment variables"
}

# Function to check system dependencies
check_system_dependencies() {
    print_status "Checking system dependencies..."
    
    local missing_deps=()
    
    # Check for required commands
    local required_commands=("cmake" "curl" "tar" "unzip")
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
            print_status "sudo apt update && sudo apt install -y curl ${missing_deps[*]}"
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
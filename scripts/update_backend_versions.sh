#!/bin/bash

# Script to update backend versions from InferenceEngines and VideoCapture repositories
# This script copies the versions.env files from the fetched repos, but local files override them

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "This script updates backend versions from the InferenceEngines and VideoCapture repositories."
    echo "Local version files override the fetched repository versions:"
    echo "  - versions.neuriplo.env overrides InferenceEngines versions"
    echo "  - versions.videocapture.env overrides VideoCapture versions"
    echo ""
    echo "Options:"
    echo "  --force              Force update even if local files exist"
    echo "  --show-versions      Show current versions after update"
    echo "  --neuriplo  Only update InferenceEngines versions"
    echo "  --videocapture       Only update VideoCapture versions"
    echo "  --help              Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                    # Update if local files don't exist"
    echo "  $0 --force           # Force update from repositories"
    echo "  $0 --show-versions   # Update and show versions"
    echo "  $0 --neuriplo --show-versions  # Only InferenceEngines"
}

# Function to update InferenceEngines versions
update_inference_engines_versions() {
    local local_versions_file="versions.neuriplo.env"
    local inference_versions_file="build/_deps/inferenceengines-src/versions.env"
    local github_url="https://raw.githubusercontent.com/olibartfast/neuriplo/master/versions.env"
    
    print_status "Updating InferenceEngines versions..."
    
    # Check if local versions file exists
    if [[ -f "$local_versions_file" && "$force" != "true" ]]; then
        print_warning "Local InferenceEngines versions file $local_versions_file already exists."
        print_status "Use --force to overwrite with versions from InferenceEngines repo."
        print_status "Current local versions:"
        source "$local_versions_file"
        echo "ONNX Runtime: $ONNX_RUNTIME_VERSION"
        echo "TensorRT: $TENSORRT_VERSION"
        echo "LibTorch: $PYTORCH_VERSION"
        echo "OpenVINO: $OPENVINO_VERSION"
        echo "TensorFlow: $TENSORFLOW_VERSION"
        echo "CUDA: $CUDA_VERSION"
        return 0
    fi
    
    # Try to get versions from local build folder first, then GitHub
    if [[ -f "$inference_versions_file" ]]; then
        print_status "Using versions from local build folder..."
        cp "$inference_versions_file" "$local_versions_file"
    else
        print_status "Local build folder not found, fetching from GitHub..."
        print_status "URL: $github_url"
        
        # Download versions from GitHub
        if curl -s -o "$local_versions_file" "$github_url"; then
            print_success "Successfully downloaded InferenceEngines versions from GitHub"
        else
            print_error "Failed to download InferenceEngines versions from GitHub"
            return 1
        fi
    fi
    
    if [[ "$show_versions" == "true" ]]; then
        print_success "InferenceEngines versions updated successfully:"
        source "$local_versions_file"
        echo "ONNX Runtime: $ONNX_RUNTIME_VERSION"
        echo "TensorRT: $TENSORRT_VERSION"
        echo "LibTorch: $PYTORCH_VERSION"
        echo "OpenVINO: $OPENVINO_VERSION"
        echo "TensorFlow: $TENSORFLOW_VERSION"
        echo "CUDA: $CUDA_VERSION"
    else
        print_success "InferenceEngines versions updated successfully to $local_versions_file"
    fi
}

# Function to update VideoCapture versions
update_videocapture_versions() {
    local local_versions_file="versions.videocapture.env"
    local videocapture_versions_file="build/_deps/videocapture-src/versions.env"
    local github_url="https://raw.githubusercontent.com/olibartfast/videocapture/master/versions.env"

    print_status "Updating VideoCapture versions..."
    
    # Check if local versions file exists
    if [[ -f "$local_versions_file" && "$force" != "true" ]]; then
        print_warning "Local VideoCapture versions file $local_versions_file already exists."
        print_status "Use --force to overwrite with versions from VideoCapture repo."
        print_status "Current local versions:"
        source "$local_versions_file"
        echo "GStreamer: $GSTREAMER_VERSION"
        echo "OpenCV: $OPENCV_VERSION"
        echo "CMake: $CMAKE_VERSION"
        return 0
    fi
    
    # Try to get versions from local build folder first, then GitHub
    if [[ -f "$videocapture_versions_file" ]]; then
        print_status "Using versions from local build folder..."
        cp "$videocapture_versions_file" "$local_versions_file"
    else
        print_status "Local build folder not found, fetching from GitHub..."
        print_status "URL: $github_url"
        
        # Download versions from GitHub
        if curl -s -o "$local_versions_file" "$github_url"; then
            print_success "Successfully downloaded VideoCapture versions from GitHub"
        else
            print_error "Failed to download VideoCapture versions from GitHub"
            return 1
        fi
    fi
    
    if [[ "$show_versions" == "true" ]]; then
        print_success "VideoCapture versions updated successfully:"
        source "$local_versions_file"
        echo "GStreamer: $GSTREAMER_VERSION"
        echo "OpenCV: $OPENCV_VERSION"
        echo "CMake: $CMAKE_VERSION"
    else
        print_success "VideoCapture versions updated successfully to $local_versions_file"
    fi
}

# Main function
main() {
    local force=false
    local show_versions=false
    local update_inference_engines=true
    local update_videocapture=true
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --force)
                force=true
                shift
                ;;
            --show-versions)
                show_versions=true
                shift
                ;;
            --neuriplo)
                update_inference_engines=true
                update_videocapture=false
                shift
                ;;
            --videocapture)
                update_inference_engines=false
                update_videocapture=true
                shift
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
    
    print_status "Starting version update process..."
    
    # Update InferenceEngines versions
    if [[ "$update_inference_engines" == "true" ]]; then
        if ! update_inference_engines_versions; then
            print_error "Failed to update InferenceEngines versions"
            exit 1
        fi
    fi
    
    # Update VideoCapture versions
    if [[ "$update_videocapture" == "true" ]]; then
        if ! update_videocapture_versions; then
            print_error "Failed to update VideoCapture versions"
            exit 1
        fi
    fi
    
    print_success "Version update process completed successfully!"
}

# Run main function with all arguments
main "$@" 
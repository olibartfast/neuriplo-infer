#!/bin/bash
# RTDETRv4 end-to-end workflow: export -> ONNX -> TensorRT -> inference
#
# Usage: bash docker_run_inference_example_rtdetrv4.sh
#
# Prerequisites:
#   - vision-core repo fetched under build/_deps/vision-core-src 
#   - vision-inference Docker image built:
#       docker build --rm -t vision-inference:trt -f docker/Dockerfile.tensorrt .
#   - NVIDIA GPU with drivers and nvidia-container-toolkit installed
#   - A test image at $(pwd)/data/dog.jpg
#   - COCO labels at $(pwd)/labels/coco.names

# Exit immediately if a command exits with a non-zero status, 
# treat unset variables as an error, and catch failures in pipes.
set -euo pipefail

# ─────────────────────────────────────────────
# Set Up Absolute Paths FIRST
# ─────────────────────────────────────────────
# Determine the absolute path of the directory this script is inside
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$SCRIPT_DIR"

# Core directories (using absolute paths to prevent Docker/cd errors)
VISION_CORE_DIR="$ROOT_DIR/build/_deps/vision-core-src"
WEIGHTS_DIR="$VISION_CORE_DIR/weights"
VENV_NAME="rtdetr-pytorch"
VENV_DIR="$VISION_CORE_DIR/environments/$VENV_NAME"

# RT-DETR repository and config paths
MODEL_NAME="rtv4_hgnetv2_s_model"
REPO_DIR="$ROOT_DIR/3rdparty/repositories/pytorch/RT-DETRv4"
CONFIG="$REPO_DIR/configs/rtv4/rtv4_hgnetv2_s_coco.yml"

# TensorRT Docker Tag
NGC_TAG="25.12"  # Update to match your desired CUDA/TensorRT container version

# ─────────────────────────────────────────────
# Step 0: Set up and activate Python venv
# ─────────────────────────────────────────────
echo "=== Step 0: Setting up Python environment ==="

# Move to the vision-core directory where the setup scripts live
cd "$VISION_CORE_DIR"

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment: $VENV_NAME..."
    bash export/detection/rtdetr/setup_env.sh \
        --env-name "$VENV_NAME" \
        --output-dir "$VISION_CORE_DIR/environments"
fi

# Activate the virtual environment
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"
echo "Virtual environment activated: $VENV_DIR"

# Ensure the RT-DETRv4 repository is cloned BEFORE installing dependencies
if [ ! -d "$REPO_DIR" ]; then
    echo "=== RT-DETRv4 repository not found. Cloning now... ==="
    bash export/detection/rtdetr/clone_repo.sh \
        --version v4 \
        --output-dir "$ROOT_DIR/3rdparty/repositories/pytorch"
fi

# Bulletproof Dependency Installation
echo "=== Installing required Python packages ==="

# 1. Install the main repository requirements explicitly
if [ -f "$REPO_DIR/requirements.txt" ]; then
    echo "Installing RT-DETR requirements.txt..."
    pip install -r "$REPO_DIR/requirements.txt"
fi

# 2. Install ONNX specific dependencies that are often missing from PyTorch repos
echo "Installing ONNX export dependencies..."
pip install onnx onnxscript onnxruntime

# ─────────────────────────────────────────────
# Step 1: Export RTDETRv4 to ONNX
# ─────────────────────────────────────────────
echo "=== Step 1: Exporting RTDETRv4 to ONNX ==="

bash export/detection/rtdetr/export.sh \
    --config "$CONFIG" \
    --checkpoint "$WEIGHTS_DIR/${MODEL_NAME}.pth" \
    --repo-dir "$REPO_DIR" \
    --install-deps \
    --download-weights \
    --weights-dir "$WEIGHTS_DIR" \
    --format onnx \
    --output-dir "$WEIGHTS_DIR"

# Exit the Python virtual environment since we don't need it for Docker
deactivate
echo "ONNX export done: $WEIGHTS_DIR/${MODEL_NAME}.onnx"

# ─────────────────────────────────────────────
# Step 2: Convert ONNX to TensorRT engine
# ─────────────────────────────────────────────
echo "=== Step 2: Converting ONNX to TensorRT engine ==="

docker run --rm --gpus=all \
    -v "$WEIGHTS_DIR":/weights \
    nvcr.io/nvidia/tensorrt:${NGC_TAG}-py3 \
    trtexec \
        --onnx=/weights/${MODEL_NAME}.onnx \
        --saveEngine=/weights/${MODEL_NAME}.engine \
        --fp16

echo "TensorRT engine saved: $WEIGHTS_DIR/${MODEL_NAME}.engine"

# ─────────────────────────────────────────────
# Step 3: Run inference with vision-inference
# ─────────────────────────────────────────────
echo "=== Step 3: Running inference ==="

# Move back to the original script directory just to be safe
cd "$SCRIPT_DIR"

docker run --gpus=all --rm \
    -v "$SCRIPT_DIR/data":/app/data \
    -v "$WEIGHTS_DIR":/weights \
    -v "$SCRIPT_DIR/labels":/labels \
    vision-inference:trt \
    --type=rtdetr \
    --weights=/weights/${MODEL_NAME}.engine \
    --source=/app/data/dog.jpg \
    --labels=/labels/coco.names \
    --input_sizes='3,640,640;2'

echo "=== Inference workflow completed successfully! ==="
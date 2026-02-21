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

# Exit on error, unset variables, and pipe failures
set -euo pipefail

# Define absolute paths FIRST, before any directory changes
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Assuming the script runs from the root of your project
ROOT_DIR="$SCRIPT_DIR"

# Use absolute paths for everything so Docker and cd commands don't break
VISION_CORE_DIR="$ROOT_DIR/build/_deps/vision-core-src"
WEIGHTS_DIR="$VISION_CORE_DIR/weights"
VENV_NAME="rtdetr-pytorch"
VENV_DIR="$VISION_CORE_DIR/environments/$VENV_NAME"

MODEL_NAME="rtv4_hgnetv2_s_model"
CONFIG="3rdparty/repositories/pytorch/RT-DETRv4/configs/rtv4/rtv4_hgnetv2_s_coco.yml"
NGC_TAG="24.02"  # Ensure this tag exists (changed from 25.12 as an example)

# ─────────────────────────────────────────────
# Step 0: Set up and activate Python venv
# ─────────────────────────────────────────────
echo "=== Step 0: Setting up Python environment ==="
cd "$VISION_CORE_DIR"

if [ ! -d "$VENV_DIR" ]; then
    bash export/detection/rtdetr/setup_env.sh \
        --env-name "$VENV_NAME" \
        --output-dir "$VISION_CORE_DIR/environments"
fi

# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"
echo "Virtual environment activated: $VENV_DIR"

# ─────────────────────────────────────────────
# Step 1: Export RTDETRv4 to ONNX
# ─────────────────────────────────────────────
echo "=== Step 1: Exporting RTDETRv4 to ONNX ==="

bash export/detection/rtdetr/export.sh \
    --config "$CONFIG" \
    --checkpoint "$WEIGHTS_DIR/${MODEL_NAME}.pth" \
    --repo-dir 3rdparty/repositories/pytorch/RT-DETRv4 \
    --install-deps \
    --download-weights \
    --weights-dir "$WEIGHTS_DIR" \
    --format onnx \
    --output-dir "$WEIGHTS_DIR"

deactivate
echo "ONNX export done: $WEIGHTS_DIR/${MODEL_NAME}.onnx"

# ─────────────────────────────────────────────
# Step 2: Convert ONNX to TensorRT engine
# ─────────────────────────────────────────────
echo "=== Step 2: Converting ONNX to TensorRT engine ==="

# Note: WEIGHTS_DIR is now an absolute path, which Docker requires
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

# Go back to the script directory just to be safe, though absolute paths make this optional
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

echo "=== Inference completed successfully ==="
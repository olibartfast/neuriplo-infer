#!/bin/bash
# Generic vision-inference end-to-end workflow: export -> (convert) -> inference
#
# Supports any model family (rtdetr, yolo, rfdetr) and any backend
# (tensorrt, onnxruntime, openvino).
#
# Usage:
#   bash docker_run_inference_e2e.sh --model-family rtdetr --model-name rtv4_hgnetv2_s_model --backend tensorrt
#   bash docker_run_inference_e2e.sh --model-family yolo   --model-name yolov10n            --backend onnxruntime
#   bash docker_run_inference_e2e.sh --model-family rfdetr --model-name rfdetr_medium        --backend openvino
#   bash docker_run_inference_e2e.sh --help
#
# Prerequisites:
#   - vision-core repo fetched under build/_deps/vision-core-src
#   - Relevant vision-inference Docker image built (see docker/ folder)
#   - For tensorrt backend: NVIDIA GPU with drivers and nvidia-container-toolkit installed
#   - A test image at $(pwd)/data/dog.jpg
#   - COCO labels at $(pwd)/labels/coco.names

set -euo pipefail

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$SCRIPT_DIR"
VISION_CORE_DIR="$ROOT_DIR/build/_deps/vision-core-src"
NEURIPLO_VERSIONS_ENV="$ROOT_DIR/build/_deps/neuriplo-src/versions.env"

# ─────────────────────────────────────────────
# Defaults
# ─────────────────────────────────────────────
MODEL_FAMILY="rtdetr"
MODEL_NAME="rtv4_hgnetv2_s_model"
BACKEND="tensorrt"
SOURCE="$ROOT_DIR/data/dog.jpg"
LABELS="$ROOT_DIR/labels/coco.names"
WEIGHTS_DIR="$VISION_CORE_DIR/weights"
NGC_TAG="25.12"          # nvcr.io/nvidia/tensorrt tag used by trtexec
TRT_PRECISION="fp16"     # fp16 (default, faster) or fp32 (higher precision)
SKIP_EXPORT=false
SKIP_CONVERT=false
# Override inference --type (auto-set per model family if empty)
INFERENCE_TYPE_OVERRIDE=""
# Override --input_sizes (auto-set per model family if empty)
INPUT_SIZES_OVERRIDE=""

# ─────────────────────────────────────────────
# Help
# ─────────────────────────────────────────────
usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Generic end-to-end workflow: ONNX export -> backend conversion -> inference

MODEL OPTIONS:
  --model-family  FAMILY   Model family: rtdetr, yolo, rfdetr  (default: rtdetr)
  --model-name    NAME     Model weights base name             (default: rtv4_hgnetv2_s_model)
  --inference-type TYPE    Override --type for vision-inference  (auto-set by model family)
  --input-sizes   SIZES    Override --input_sizes               (auto-set by model family)

BACKEND OPTIONS:
  --backend       BACKEND  Inference backend: tensorrt, onnxruntime, openvino (default: tensorrt)

INPUT OPTIONS:
  --source        PATH     Image / video source   (default: data/dog.jpg)
  --labels        PATH     Class labels file      (default: labels/coco.names)
  --weights-dir   PATH     Directory for weights  (default: vision-core/weights)

TENSORRT OPTIONS:
  --ngc-tag       TAG      NVIDIA TensorRT container tag for trtexec (default: 25.11)
  --precision     PREC     TensorRT precision: fp16 or fp32            (default: fp16)
SKIP OPTIONS:
  --skip-export            Skip ONNX export step
  --skip-convert           Skip backend-format conversion step

  -h, --help               Show this help

EXAMPLES:
  # RT-DETRv4 with TensorRT
  $0 --model-family rtdetr --model-name rtv4_hgnetv2_s_model --backend tensorrt

  # YOLOv10n with ONNX Runtime
  $0 --model-family yolo --model-name yolov10n --backend onnxruntime

  # RF-DETR medium with OpenVINO
  $0 --model-family rfdetr --model-name rfdetr_medium --backend openvino

  # Run only inference (weights already prepared)
  $0 --model-family yolo --model-name yolov10n --backend tensorrt --skip-export --skip-convert
EOF
    exit 0
}

# ─────────────────────────────────────────────
# Parse arguments
# ─────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-family)     MODEL_FAMILY="$2";          shift 2 ;;
        --model-name)       MODEL_NAME="$2";            shift 2 ;;
        --inference-type)   INFERENCE_TYPE_OVERRIDE="$2"; shift 2 ;;
        --input-sizes)      INPUT_SIZES_OVERRIDE="$2";  shift 2 ;;
        --backend)          BACKEND="$2";               shift 2 ;;
        --source)           SOURCE="$2";                shift 2 ;;
        --labels)           LABELS="$2";                shift 2 ;;
        --weights-dir)      WEIGHTS_DIR="$2";           shift 2 ;;
        --ngc-tag)          NGC_TAG="$2";               shift 2 ;;
        --precision)        TRT_PRECISION="$2";          shift 2 ;;
        --skip-export)      SKIP_EXPORT=true;           shift   ;;
        --skip-convert)     SKIP_CONVERT=true;          shift   ;;
        -h|--help)          usage ;;
        *) echo "ERROR: Unknown option: $1"; echo ""; usage ;;
    esac
done

# ─────────────────────────────────────────────
# Load neuriplo versions (e.g. OPENVINO_VERSION)
# ─────────────────────────────────────────────
if [[ -f "$NEURIPLO_VERSIONS_ENV" ]]; then
    # shellcheck source=/dev/null
    . "$NEURIPLO_VERSIONS_ENV"
fi

# ─────────────────────────────────────────────
# Model family configuration
# ─────────────────────────────────────────────
# Defines per-family defaults that can be overridden via --inference-type / --input-sizes.
#
# VENV_NAME      : virtualenv name under vision-core/environments/
# HAS_SETUP_ENV  : whether export/detection/<family>/setup_env.sh exists
# HAS_CLONE_REPO : whether export/detection/<family>/clone_repo.sh exists
# EXPORT_RELPATH : export script path relative to VISION_CORE_DIR (empty = Python export.py)
# DEF_INFERENCE_TYPE : default --type value for vision-inference
# DEF_INPUT_SIZES    : default --input_sizes value

case "$MODEL_FAMILY" in
    rtdetr)
        VENV_NAME="rtdetr-pytorch"
        HAS_SETUP_ENV=true
        HAS_CLONE_REPO=true
        EXPORT_RELPATH="export/detection/rtdetr/export.sh"
        DEF_INFERENCE_TYPE="rtdetr"
        DEF_INPUT_SIZES="3,640,640;2"
        # RT-DETR needs a config file and a repo dir; set sane defaults.
        # Pass --rtdetr-config / --rtdetr-repo to override via extra args if needed.
        RTDETR_REPO_DIR="$ROOT_DIR/3rdparty/repositories/pytorch/RT-DETRv4"
        RTDETR_CONFIG="$RTDETR_REPO_DIR/configs/rtv4/rtv4_hgnetv2_s_coco.yml"
        ;;
    yolo)
        VENV_NAME="yolo-export"
        HAS_SETUP_ENV=true
        HAS_CLONE_REPO=true
        EXPORT_RELPATH="export/detection/yolo/export.sh"
        DEF_INFERENCE_TYPE="yolov10"
        DEF_INPUT_SIZES="3,640,640"
        RTDETR_REPO_DIR=""
        RTDETR_CONFIG=""
        ;;
    rfdetr)
        VENV_NAME="rfdetr-export"
        HAS_SETUP_ENV=false   # rfdetr uses pip install, no setup_env.sh
        HAS_CLONE_REPO=false
        EXPORT_RELPATH=""     # uses export.py directly
        DEF_INFERENCE_TYPE="rfdetr"
        DEF_INPUT_SIZES="3,640,640;2"
        RTDETR_REPO_DIR=""
        RTDETR_CONFIG=""
        ;;
    *)
        echo "ERROR: Unknown model family '$MODEL_FAMILY'."
        echo "       Supported: rtdetr, yolo, rfdetr"
        exit 1
        ;;
esac

INFERENCE_TYPE="${INFERENCE_TYPE_OVERRIDE:-$DEF_INFERENCE_TYPE}"
INPUT_SIZES="${INPUT_SIZES_OVERRIDE:-$DEF_INPUT_SIZES}"
VENV_DIR="$VISION_CORE_DIR/environments/$VENV_NAME"

# ─────────────────────────────────────────────
# Backend configuration
# ─────────────────────────────────────────────
# DOCKER_IMAGE   : vision-inference Docker image to run
# WEIGHT_EXT     : file extension of the final weights file
# NEEDS_TRT_CONV : whether to run trtexec ONNX → engine conversion
# NEEDS_OV_CONV  : whether to run ovc ONNX → OpenVINO IR conversion
# DOCKER_FLAGS   : extra docker run flags

case "$BACKEND" in
    tensorrt)
        DOCKER_IMAGE="vision-inference:trt"
        WEIGHT_EXT="engine"
        NEEDS_TRT_CONV=true
        NEEDS_OV_CONV=false
        DOCKER_FLAGS="--gpus=all"
        ;;
    onnxruntime)
        DOCKER_IMAGE="vision-inference:ort"
        WEIGHT_EXT="onnx"
        NEEDS_TRT_CONV=false
        NEEDS_OV_CONV=false
        DOCKER_FLAGS=""
        ;;
    openvino)
        DOCKER_IMAGE="vision-inference:openvino"
        WEIGHT_EXT="xml"
        NEEDS_TRT_CONV=false
        NEEDS_OV_CONV=true
        DOCKER_FLAGS=""
        ;;
    *)
        echo "ERROR: Unknown backend '$BACKEND'."
        echo "       Supported: tensorrt, onnxruntime, openvino"
        exit 1
        ;;
esac

WEIGHT_FILE="$WEIGHTS_DIR/${MODEL_NAME}.${WEIGHT_EXT}"

# ─────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────
echo "════════════════════════════════════════════════"
echo " vision-inference end-to-end workflow"
echo "════════════════════════════════════════════════"
echo " Model family  : $MODEL_FAMILY"
echo " Model name    : $MODEL_NAME"
echo " Inference type: $INFERENCE_TYPE"
echo " Input sizes   : $INPUT_SIZES"
echo " Backend       : $BACKEND"
echo " Docker image  : $DOCKER_IMAGE"
echo " Weights dir   : $WEIGHTS_DIR"
echo " Source        : $SOURCE"
echo " Labels        : $LABELS"
echo "════════════════════════════════════════════════"

mkdir -p "$WEIGHTS_DIR"

# ─────────────────────────────────────────────
# Step 0 + 1: Setup Python env & Export to ONNX
# ─────────────────────────────────────────────
if [[ "$SKIP_EXPORT" == false ]]; then
    echo ""
    echo "=== Step 0: Setting up Python environment ($VENV_NAME) ==="
    cd "$VISION_CORE_DIR"

    if [[ "$HAS_SETUP_ENV" == true ]]; then
        if [[ ! -d "$VENV_DIR" ]]; then
            echo "Creating virtual environment: $VENV_NAME..."
            bash "export/detection/${MODEL_FAMILY}/setup_env.sh" \
                --env-name "$VENV_NAME" \
                --output-dir "$VISION_CORE_DIR/environments"
        fi
        # shellcheck source=/dev/null
        source "$VENV_DIR/bin/activate"
    else
        # rfdetr: use a simple venv with pip install
        if [[ ! -d "$VENV_DIR" ]]; then
            python3 -m venv "$VENV_DIR"
        fi
        # shellcheck source=/dev/null
        source "$VENV_DIR/bin/activate"
        pip install --quiet rfdetr onnx onnxruntime onnx-simplifier
    fi
    echo "Virtual environment activated: $VENV_DIR"

    echo ""
    echo "=== Step 1: Exporting $MODEL_FAMILY/$MODEL_NAME to ONNX ==="

    case "$MODEL_FAMILY" in
        rtdetr)
            if [[ "$HAS_CLONE_REPO" == true && ! -d "$RTDETR_REPO_DIR" ]]; then
                echo "Cloning RT-DETRv4 repository..."
                bash "export/detection/rtdetr/clone_repo.sh" \
                    --version v4 \
                    --output-dir "$ROOT_DIR/3rdparty/repositories/pytorch"
            fi
            pip install --quiet onnx onnxscript onnxruntime
            bash "$EXPORT_RELPATH" \
                --config "$RTDETR_CONFIG" \
                --checkpoint "$WEIGHTS_DIR/${MODEL_NAME}.pth" \
                --repo-dir "$RTDETR_REPO_DIR" \
                --install-deps \
                --download-weights \
                --weights-dir "$WEIGHTS_DIR" \
                --format onnx \
                --output-dir "$WEIGHTS_DIR"
            ;;
        yolo)
            if [[ "$HAS_CLONE_REPO" == true ]]; then
                # Clone underlying repo only for legacy versions (v5/v6/v7) if needed.
                # For v8+ ultralytics handles it automatically.
                :
            fi
            pip install --quiet onnx onnxscript onnxruntime
            bash "$EXPORT_RELPATH" \
                --model "$WEIGHTS_DIR/${MODEL_NAME}.pt" \
                --format onnx \
                --output-dir "$WEIGHTS_DIR"
            ;;
        rfdetr)
            python3 "export/detection/rfdetr/export.py" \
                --output_dir "$WEIGHTS_DIR" \
                --simplify
            # Rename the exported file to the expected MODEL_NAME if needed
            EXPORTED_ONNX=$(find "$WEIGHTS_DIR" -maxdepth 1 -name "*.onnx" | head -n1)
            if [[ -n "$EXPORTED_ONNX" && "$EXPORTED_ONNX" != "$WEIGHTS_DIR/${MODEL_NAME}.onnx" ]]; then
                mv "$EXPORTED_ONNX" "$WEIGHTS_DIR/${MODEL_NAME}.onnx"
            fi
            ;;
    esac

    deactivate
    echo "ONNX export done: $WEIGHTS_DIR/${MODEL_NAME}.onnx"
else
    echo ""
    echo "=== Step 0+1: Skipping export (--skip-export) ==="
fi

# ─────────────────────────────────────────────
# Step 2: Convert ONNX to backend format
# ─────────────────────────────────────────────
if [[ "$SKIP_CONVERT" == false ]]; then
    if [[ "$NEEDS_TRT_CONV" == true ]]; then
        echo ""
        echo "=== Step 2: Converting ONNX → TensorRT engine ==="
        TRT_PRECISION_FLAG=""
        [[ "$TRT_PRECISION" == "fp16" ]] && TRT_PRECISION_FLAG="--fp16"
        docker run --rm --gpus=all \
            -v "$WEIGHTS_DIR":/weights \
            nvcr.io/nvidia/tensorrt:${NGC_TAG}-py3 \
            trtexec \
                --onnx=/weights/${MODEL_NAME}.onnx \
                --saveEngine=/weights/${MODEL_NAME}.engine \
                ${TRT_PRECISION_FLAG}
        echo "TensorRT engine saved: $WEIGHT_FILE"

    elif [[ "$NEEDS_OV_CONV" == true ]]; then
        echo ""
        echo "=== Step 2: Converting ONNX → OpenVINO IR ==="
        # ovc is the OpenVINO Model Converter available in openvino/ubuntu24_runtime images
        docker run --rm \
            -v "$WEIGHTS_DIR":/weights \
            openvino/ubuntu24_runtime:${OPENVINO_VERSION:-2025.4.1} \
            ovc /weights/${MODEL_NAME}.onnx \
                --output_model /weights/${MODEL_NAME}
        echo "OpenVINO IR saved: $WEIGHTS_DIR/${MODEL_NAME}.xml"

    else
        echo ""
        echo "=== Step 2: No conversion needed for '$BACKEND' — using .onnx directly ==="
    fi
else
    echo ""
    echo "=== Step 2: Skipping conversion (--skip-convert) ==="
fi

# ─────────────────────────────────────────────
# Step 3: Run inference
# ─────────────────────────────────────────────
echo ""
echo "=== Step 3: Running inference ==="
cd "$SCRIPT_DIR"

# shellcheck disable=SC2086
docker run $DOCKER_FLAGS --rm \
    -e GLOG_minloglevel=1 \
    -v "$SCRIPT_DIR/data":/app/data \
    -v "$WEIGHTS_DIR":/weights \
    -v "$(dirname "$LABELS")":/labels \
    "$DOCKER_IMAGE" \
    --type="$INFERENCE_TYPE" \
    --weights="/weights/${MODEL_NAME}.${WEIGHT_EXT}" \
    --source="/app/data/$(basename "$SOURCE")" \
    --labels="/labels/$(basename "$LABELS")" \
    --input_sizes="$INPUT_SIZES"

echo ""
echo "════════════════════════════════════════════════"
echo " Workflow completed: $MODEL_FAMILY/$MODEL_NAME on $BACKEND"
echo "════════════════════════════════════════════════"

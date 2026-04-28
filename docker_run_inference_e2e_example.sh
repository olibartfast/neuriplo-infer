#!/bin/bash
# Generic end-to-end workflow: export -> optional TensorRT conversion -> inference.
#
# Presets cover the task families currently supported by vision-inference:
#   - rtdetrv4              Object detection
#   - owlv2                 Open-vocabulary detection
#   - torchvision_classifier Classification
#   - yoloseg               Instance segmentation
#   - raft                  Optical flow
#   - vitpose               Pose estimation
#   - depth_anything_v2     Depth estimation
#   - videomae              Video classification
#
# Typical usage:
#   bash docker_run_inference_e2e_example.sh --preset owlv2 --dry-run
#   bash docker_run_inference_e2e_example.sh --preset rtdetrv4

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$SCRIPT_DIR"

PRESET="rtdetrv4"
BACKEND=""
BACKEND_SET=false
DRY_RUN=false
SKIP_EXPORT=false
SKIP_CONVERT=false
SKIP_INFER=false
VISION_CORE_DIR="${VISION_CORE_DIR:-}"
WEIGHTS_DIR="${ROOT_DIR}/models/e2e"
DATA_DIR="${ROOT_DIR}/data"
LABELS_DIR="${ROOT_DIR}/labels"
DOCKER_IMAGE=""
NGC_TAG="${NGC_TAG:-25.12}"
TEXT_PROMPTS="${TEXT_PROMPTS:-cat;dog;bus}"

if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
else
    echo "Neither 'python' nor 'python3' was found on PATH." >&2
    exit 1
fi

show_help() {
    cat <<'EOF'
Generic docker end-to-end workflow for vision-inference examples.

Usage:
  bash docker_run_inference_e2e_example.sh [options]

Options:
  --preset <name>            Task preset to run. Default: rtdetrv4
  --backend <name>           Runtime backend: onnxruntime or tensorrt
  --vision-core-dir <path>   Path to a vision-core checkout with export tooling
  --weights-dir <path>       Export/model output directory. Default: ./models/e2e
  --data-dir <path>          Host data directory to mount. Default: ./data
  --labels-dir <path>        Host labels directory to mount. Default: ./labels
  --docker-image <name>      vision-inference image override
  --text-prompts <value>     Open-vocab prompts for owlv2. Default: cat;dog;bus
  --dry-run                  Print commands without executing them
  --skip-export              Skip export step
  --skip-convert             Skip TensorRT conversion step
  --skip-infer               Skip inference step
  --list-presets             Print available presets
  --help                     Show this help
EOF
}

list_presets() {
    cat <<'EOF'
rtdetrv4
owlv2
torchvision_classifier
yoloseg
raft
vitpose
depth_anything_v2
videomae
EOF
}

print_cmd() {
    printf '+'
    for arg in "$@"; do
        printf ' %q' "$arg"
    done
    printf '\n'
}

run_cmd() {
    print_cmd "$@"
    if [[ "$DRY_RUN" == false ]]; then
        "$@"
    fi
}

run_shell_cmd() {
    local cmd="$1"
    printf '+ %s\n' "$cmd"
    if [[ "$DRY_RUN" == false ]]; then
        bash -lc "$cmd"
    fi
}

ensure_dir() {
    if [[ "$DRY_RUN" == false ]]; then
        mkdir -p "$1"
    fi
}

ensure_file() {
    if [[ "$DRY_RUN" == false && ! -f "$1" ]]; then
        echo "Required file not found: $1" >&2
        exit 1
    fi
}

ensure_files() {
    local path_list="$1"
    local path=""
    IFS=',' read -r -a paths <<< "$path_list"
    for path in "${paths[@]}"; do
        ensure_file "$path"
    done
}

ensure_vision_core() {
    if [[ -z "$VISION_CORE_DIR" ]]; then
        echo "vision-core checkout path is required. Pass --vision-core-dir or set VISION_CORE_DIR." >&2
        exit 1
    fi
    if [[ "$DRY_RUN" == false && ! -d "$VISION_CORE_DIR" ]]; then
        echo "vision-core checkout not found: $VISION_CORE_DIR" >&2
        exit 1
    fi
}

normalize_preset() {
    echo "$1" | tr '[:upper:]' '[:lower:]' | tr '-' '_'
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --preset)
            PRESET="$2"
            shift 2
            ;;
        --backend)
            BACKEND="$2"
            BACKEND_SET=true
            shift 2
            ;;
        --vision-core-dir)
            VISION_CORE_DIR="$2"
            shift 2
            ;;
        --weights-dir)
            WEIGHTS_DIR="$2"
            shift 2
            ;;
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --labels-dir)
            LABELS_DIR="$2"
            shift 2
            ;;
        --docker-image)
            DOCKER_IMAGE="$2"
            shift 2
            ;;
        --text-prompts)
            TEXT_PROMPTS="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --skip-export)
            SKIP_EXPORT=true
            shift
            ;;
        --skip-convert)
            SKIP_CONVERT=true
            shift
            ;;
        --skip-infer)
            SKIP_INFER=true
            shift
            ;;
        --list-presets)
            list_presets
            exit 0
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            show_help >&2
            exit 1
            ;;
    esac
done

PRESET="$(normalize_preset "$PRESET")"

case "$PRESET" in
    rtdetrv4)
        if [[ "$BACKEND_SET" == false ]]; then
            BACKEND="tensorrt"
        fi
        MODEL_TYPE="rtdetr"
        MODEL_BASENAME="rtv4_hgnetv2_s_model"
        SOURCE_IN_CONTAINER="/app/data/dog.jpg"
        HOST_SOURCE_PATH="${DATA_DIR}/dog.jpg"
        LABELS_IN_CONTAINER="/labels/coco.names"
        HOST_LABELS_PATH="${LABELS_DIR}/coco.names"
        INPUT_SIZES="3,640,640;2"
        EXPORT_VENV_NAME="rtdetr-pytorch"
        EXTRA_REQUIREMENTS=()
        RTDETR_REPO_DIR="${ROOT_DIR}/3rdparty/repositories/pytorch/RT-DETRv4"
        RTDETR_CONFIG="${RTDETR_REPO_DIR}/configs/rtv4/rtv4_hgnetv2_s_coco.yml"
        EXPORT_COMMANDS=(
            "if [[ ! -d \"${VISION_CORE_DIR}/environments/${EXPORT_VENV_NAME}\" ]]; then bash \"${VISION_CORE_DIR}/export/detection/rtdetr/setup_env.sh\" --env-name \"${EXPORT_VENV_NAME}\" --output-dir \"${VISION_CORE_DIR}/environments\"; fi"
            "source \"${VISION_CORE_DIR}/environments/${EXPORT_VENV_NAME}/bin/activate\" && if [[ ! -d \"${RTDETR_REPO_DIR}\" ]]; then bash \"${VISION_CORE_DIR}/export/detection/rtdetr/clone_repo.sh\" --version v4 --output-dir \"${ROOT_DIR}/3rdparty/repositories/pytorch\"; fi && if [[ -f \"${RTDETR_REPO_DIR}/requirements.txt\" ]]; then pip install -r \"${RTDETR_REPO_DIR}/requirements.txt\"; fi && pip install onnx onnxscript onnxruntime && bash \"${VISION_CORE_DIR}/export/detection/rtdetr/export.sh\" --config \"${RTDETR_CONFIG}\" --checkpoint \"${WEIGHTS_DIR}/${MODEL_BASENAME}.pth\" --repo-dir \"${RTDETR_REPO_DIR}\" --install-deps --download-weights --weights-dir \"${WEIGHTS_DIR}\" --format onnx --output-dir \"${WEIGHTS_DIR}\""
        )
        RUNTIME_EXTRA_ARGS=("--labels=${LABELS_IN_CONTAINER}" "--input_sizes=${INPUT_SIZES}")
        ;;
    owlv2)
        if [[ "$BACKEND_SET" == false ]]; then
            BACKEND="onnxruntime"
        fi
        MODEL_TYPE="owlv2"
        MODEL_BASENAME="owlv2"
        SOURCE_IN_CONTAINER="/app/data/dog.jpg"
        HOST_SOURCE_PATH="${DATA_DIR}/dog.jpg"
        TOKENIZER_VOCAB_HOST="${VISION_CORE_DIR}/vocab.json"
        TOKENIZER_MERGES_HOST="${VISION_CORE_DIR}/merges.txt"
        TOKENIZER_VOCAB_IN_CONTAINER="/weights/vocab.json"
        TOKENIZER_MERGES_IN_CONTAINER="/weights/merges.txt"
        EXTRA_REQUIREMENTS=("${VISION_CORE_DIR}/export/open_vocab_detection/owlv2/requirements.txt")
        EXPORT_COMMANDS=(
            "${PYTHON_BIN} -m venv \"${VISION_CORE_DIR}/environments/open-vocab-export\""
            "source \"${VISION_CORE_DIR}/environments/open-vocab-export/bin/activate\" && python -m pip install --upgrade pip setuptools wheel && python -m pip install -r \"${VISION_CORE_DIR}/export/open_vocab_detection/owlv2/requirements.txt\" && python -m pip install numpy && ${PYTHON_BIN} \"${VISION_CORE_DIR}/export/open_vocab_detection/owlv2/export_owlv2_to_onnx.py\" --model google/owlv2-base-patch16-ensemble --output \"${WEIGHTS_DIR}/${MODEL_BASENAME}.onnx\" --image-height 960 --image-width 960 --max-queries 16 --sequence-length 16 --test"
        )
        RUNTIME_EXTRA_ARGS=("--text_prompts=${TEXT_PROMPTS}" "--tokenizer_vocab=${TOKENIZER_VOCAB_IN_CONTAINER}" "--tokenizer_merges=${TOKENIZER_MERGES_IN_CONTAINER}" "--min_confidence=0.2")
        ;;
    torchvision_classifier)
        if [[ "$BACKEND_SET" == false ]]; then
            BACKEND="onnxruntime"
        fi
        MODEL_TYPE="torchvisionclassifier"
        MODEL_BASENAME="resnet50"
        SOURCE_IN_CONTAINER="/app/data/dog.jpg"
        HOST_SOURCE_PATH="${DATA_DIR}/dog.jpg"
        LABELS_IN_CONTAINER="/labels/imagenet_labels.txt"
        HOST_LABELS_PATH="${LABELS_DIR}/imagenet_labels.txt"
        EXTRA_REQUIREMENTS=("${VISION_CORE_DIR}/export/classification/torchvision/requirements.txt")
        EXPORT_COMMANDS=(
            "${PYTHON_BIN} -m venv \"${VISION_CORE_DIR}/environments/classification-export\""
            "source \"${VISION_CORE_DIR}/environments/classification-export/bin/activate\" && pip install -r \"${VISION_CORE_DIR}/export/requirements.txt\" -r \"${VISION_CORE_DIR}/export/classification/torchvision/requirements.txt\" onnx onnxruntime && ${PYTHON_BIN} \"${VISION_CORE_DIR}/export/classification/torchvision/export_torchvision_classifier.py\" --library torchvision --model resnet50 --export_format onnx --output_onnx \"${WEIGHTS_DIR}/${MODEL_BASENAME}.onnx\""
        )
        RUNTIME_EXTRA_ARGS=("--labels=${LABELS_IN_CONTAINER}")
        ;;
    yoloseg)
        if [[ "$BACKEND_SET" == false ]]; then
            BACKEND="onnxruntime"
        fi
        MODEL_TYPE="yoloseg"
        MODEL_BASENAME="yolov8n-seg"
        SOURCE_IN_CONTAINER="/app/data/dog.jpg"
        HOST_SOURCE_PATH="${DATA_DIR}/dog.jpg"
        LABELS_IN_CONTAINER="/labels/coco.names"
        HOST_LABELS_PATH="${LABELS_DIR}/coco.names"
        EXTRA_REQUIREMENTS=()
        EXPORT_COMMANDS=(
            "source \"${VISION_CORE_DIR}/environments/yolo-export/bin/activate\" 2>/dev/null || true; bash \"${VISION_CORE_DIR}/export/detection/yolo/export.sh\" --model \"${MODEL_BASENAME}.pt\" --format onnx --download-weights --weights-dir \"${WEIGHTS_DIR}\" --output-dir \"${WEIGHTS_DIR}\""
        )
        RUNTIME_EXTRA_ARGS=("--labels=${LABELS_IN_CONTAINER}" "--min_confidence=0.4" "--nms_threshold=0.5" "--mask_threshold=0.5")
        ;;
    raft)
        if [[ "$BACKEND_SET" == false ]]; then
            BACKEND="onnxruntime"
        fi
        MODEL_TYPE="raft"
        MODEL_BASENAME="raft_large"
        SOURCE_IN_CONTAINER="/app/data/frame_001.png,/app/data/frame_002.png"
        HOST_SOURCE_PATH="${DATA_DIR}/frame_001.png,${DATA_DIR}/frame_002.png"
        EXTRA_REQUIREMENTS=()
        EXPORT_COMMANDS=(
            "${PYTHON_BIN} -m venv \"${VISION_CORE_DIR}/environments/raft-export\""
            "source \"${VISION_CORE_DIR}/environments/raft-export/bin/activate\" && pip install -r \"${VISION_CORE_DIR}/export/requirements.txt\" onnx onnxruntime && ${PYTHON_BIN} \"${VISION_CORE_DIR}/export/optical_flow/raft/raft_exporter.py\" --model-type large --output-dir \"${WEIGHTS_DIR}\" --format onnx"
        )
        RUNTIME_EXTRA_ARGS=("--input_sizes=3,520,960;3,520,960")
        ;;
    vitpose)
        if [[ "$BACKEND_SET" == false ]]; then
            BACKEND="onnxruntime"
        fi
        MODEL_TYPE="vitpose"
        MODEL_BASENAME="vitpose"
        SOURCE_IN_CONTAINER="/app/data/person.jpg"
        HOST_SOURCE_PATH="${DATA_DIR}/person.jpg"
        EXTRA_REQUIREMENTS=()
        EXPORT_COMMANDS=(
            "${PYTHON_BIN} -m venv \"${VISION_CORE_DIR}/environments/vitpose-export\""
            "source \"${VISION_CORE_DIR}/environments/vitpose-export/bin/activate\" && pip install -r \"${VISION_CORE_DIR}/export/requirements.txt\" onnx onnxruntime && ${PYTHON_BIN} \"${VISION_CORE_DIR}/export/pose_estimation/vitpose/export_vitpose_to_onnx.py\" --output \"${WEIGHTS_DIR}/${MODEL_BASENAME}.onnx\""
        )
        RUNTIME_EXTRA_ARGS=()
        ;;
    depth_anything_v2)
        if [[ "$BACKEND_SET" == false ]]; then
            BACKEND="onnxruntime"
        fi
        MODEL_TYPE="depth_anything_v2"
        MODEL_BASENAME="depth_anything_v2"
        SOURCE_IN_CONTAINER="/app/data/dog.jpg"
        HOST_SOURCE_PATH="${DATA_DIR}/dog.jpg"
        EXTRA_REQUIREMENTS=()
        EXPORT_COMMANDS=(
            "${PYTHON_BIN} -m venv \"${VISION_CORE_DIR}/environments/depth-export\""
            "source \"${VISION_CORE_DIR}/environments/depth-export/bin/activate\" && pip install -r \"${VISION_CORE_DIR}/export/requirements.txt\" onnx onnxruntime && ${PYTHON_BIN} \"${VISION_CORE_DIR}/export/depth_estimation/depth_anything_v2/export_depth_anything_v2_to_onnx.py\" --output \"${WEIGHTS_DIR}/${MODEL_BASENAME}.onnx\""
        )
        RUNTIME_EXTRA_ARGS=()
        ;;
    videomae)
        if [[ "$BACKEND_SET" == false ]]; then
            BACKEND="onnxruntime"
        fi
        MODEL_TYPE="videomae"
        MODEL_BASENAME="videomae"
        SOURCE_IN_CONTAINER="/app/data/input.mp4"
        HOST_SOURCE_PATH="${DATA_DIR}/input.mp4"
        EXTRA_REQUIREMENTS=()
        EXPORT_COMMANDS=(
            "${PYTHON_BIN} -m venv \"${VISION_CORE_DIR}/environments/video-export\""
            "source \"${VISION_CORE_DIR}/environments/video-export/bin/activate\" && pip install -r \"${VISION_CORE_DIR}/export/requirements.txt\" onnx onnxruntime && ${PYTHON_BIN} \"${VISION_CORE_DIR}/export/video_classification/videomae/export_videomae_to_onnx.py\" --output \"${WEIGHTS_DIR}/${MODEL_BASENAME}.onnx\""
        )
        RUNTIME_EXTRA_ARGS=("--num_frames=16")
        ;;
    *)
        echo "Unsupported preset: $PRESET" >&2
        echo "Use --list-presets to inspect available values." >&2
        exit 1
        ;;
esac

if [[ -z "$DOCKER_IMAGE" ]]; then
    if [[ "$BACKEND" == "tensorrt" ]]; then
        DOCKER_IMAGE="vision-inference:tensorrt"
    else
        DOCKER_IMAGE="vision-inference:${BACKEND}"
    fi
fi

case "$BACKEND" in
    onnxruntime)
        MODEL_ARTIFACT="${WEIGHTS_DIR}/${MODEL_BASENAME}.onnx"
        MODEL_ARTIFACT_IN_CONTAINER="/weights/${MODEL_BASENAME}.onnx"
        ;;
    tensorrt)
        MODEL_ARTIFACT="${WEIGHTS_DIR}/${MODEL_BASENAME}.engine"
        MODEL_ARTIFACT_IN_CONTAINER="/weights/${MODEL_BASENAME}.engine"
        ;;
    *)
        echo "Unsupported backend: $BACKEND" >&2
        exit 1
        ;;
esac

ensure_dir "$WEIGHTS_DIR"
ensure_vision_core

if [[ "$PRESET" == "owlv2" ]]; then
    ensure_file "$TOKENIZER_VOCAB_HOST"
    ensure_file "$TOKENIZER_MERGES_HOST"
fi

if [[ "${HOST_LABELS_PATH:-}" != "" ]]; then
    ensure_file "$HOST_LABELS_PATH"
fi

ensure_files "$HOST_SOURCE_PATH"

if [[ "$SKIP_EXPORT" == false ]]; then
    echo "=== Step 1: Exporting preset '${PRESET}' ==="
    for export_cmd in "${EXPORT_COMMANDS[@]}"; do
        run_shell_cmd "$export_cmd"
    done
fi

if [[ "$BACKEND" == "tensorrt" && "$SKIP_CONVERT" == false ]]; then
    ONNX_ARTIFACT="${WEIGHTS_DIR}/${MODEL_BASENAME}.onnx"
    echo "=== Step 2: Converting ONNX to TensorRT ==="
    if [[ "$DRY_RUN" == false ]]; then
        ensure_file "$ONNX_ARTIFACT"
    fi
    run_cmd docker run --rm --gpus=all \
        -v "${WEIGHTS_DIR}:/weights" \
        "nvcr.io/nvidia/tensorrt:${NGC_TAG}-py3" \
        trtexec \
        "--onnx=/weights/${MODEL_BASENAME}.onnx" \
        "--saveEngine=/weights/${MODEL_BASENAME}.engine"
fi

if [[ "$SKIP_INFER" == false ]]; then
    echo "=== Step 3: Running inference ==="
    if [[ "$DRY_RUN" == false ]]; then
        ensure_file "$MODEL_ARTIFACT"
    fi

    DOCKER_ARGS=(
        docker run --rm
        -e GLOG_minloglevel=1
        -v "${DATA_DIR}:/app/data"
        -v "${WEIGHTS_DIR}:/weights"
    )

    if [[ "${HOST_LABELS_PATH:-}" != "" ]]; then
        DOCKER_ARGS+=(-v "${LABELS_DIR}:/labels")
    fi

    if [[ "$PRESET" == "owlv2" ]]; then
        DOCKER_ARGS+=(-v "${TOKENIZER_VOCAB_HOST}:${TOKENIZER_VOCAB_IN_CONTAINER}")
        DOCKER_ARGS+=(-v "${TOKENIZER_MERGES_HOST}:${TOKENIZER_MERGES_IN_CONTAINER}")
    fi

    if [[ "$BACKEND" == "tensorrt" ]]; then
        DOCKER_ARGS+=(--gpus=all)
    fi

    DOCKER_ARGS+=(
        "$DOCKER_IMAGE"
        "--type=${MODEL_TYPE}"
        "--weights=${MODEL_ARTIFACT_IN_CONTAINER}"
        "--source=${SOURCE_IN_CONTAINER}"
    )

    DOCKER_ARGS+=("${RUNTIME_EXTRA_ARGS[@]}")

    run_cmd "${DOCKER_ARGS[@]}"
fi

echo "=== Workflow completed for preset '${PRESET}' ==="

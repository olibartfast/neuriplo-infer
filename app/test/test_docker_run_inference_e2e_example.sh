#!/bin/bash
set -euo pipefail

ROOT_DIR="${1:?repo root required}"
NEURIPLO_TASKS_DIR="${2:?neuriplo-tasks dir required}"
SCRIPT_PATH="${ROOT_DIR}/docker_run_inference_e2e_example.sh"

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT

mkdir -p "${TMP_DIR}/data" "${TMP_DIR}/labels" "${TMP_DIR}/weights"
touch "${TMP_DIR}/data/dog.jpg"
touch "${TMP_DIR}/data/person.jpg"
touch "${TMP_DIR}/labels/coco.names"

# ── owlv2 dry-run ─────────────────────────────────────────────────────────────
OUTPUT_FILE="${TMP_DIR}/owlv2_dry_run.txt"

bash "${SCRIPT_PATH}" \
    --preset owlv2 \
    --neuriplo-tasks-dir "${NEURIPLO_TASKS_DIR}" \
    --weights-dir "${TMP_DIR}/weights" \
    --data-dir "${TMP_DIR}/data" \
    --labels-dir "${TMP_DIR}/labels" \
    --docker-image neuriplo-infer:test \
    --text-prompts 'cat;dog;bus' \
    --dry-run > "${OUTPUT_FILE}"

grep -F -- "export/open_vocab_detection/owlv2/export_owlv2_to_onnx.py" "${OUTPUT_FILE}"
grep -F -- "--type=owlv2" "${OUTPUT_FILE}"
grep -F -- "--text_prompts=cat\\;dog\\;bus" "${OUTPUT_FILE}"
grep -F -- "--tokenizer_vocab=/weights/vocab.json" "${OUTPUT_FILE}"
grep -F -- "--tokenizer_merges=/weights/merges.txt" "${OUTPUT_FILE}"
grep -F -- "neuriplo-infer:test" "${OUTPUT_FILE}"

# -- yolo26s_tflite dry-run ---------------------------------------------------
OUTPUT_FILE="${TMP_DIR}/yolo26s_tflite_dry_run.txt"

bash "${SCRIPT_PATH}" \
    --preset yolo26s_tflite \
    --weights-dir "${TMP_DIR}/weights" \
    --data-dir "${TMP_DIR}/data" \
    --labels-dir "${TMP_DIR}/labels" \
    --docker-image neuriplo-infer:litert \
    --dry-run > "${OUTPUT_FILE}"

grep -F -- "YOLO('yolo26s.pt')" "${OUTPUT_FILE}"
grep -F -- "format='tflite'" "${OUTPUT_FILE}"
grep -F -- "--type=yolo26" "${OUTPUT_FILE}"
grep -F -- "--weights=/weights/yolo26s.tflite" "${OUTPUT_FILE}"
grep -F -- "neuriplo-infer:litert" "${OUTPUT_FILE}"

# ── edgecrafter_det dry-run ───────────────────────────────────────────────────
OUTPUT_FILE="${TMP_DIR}/edgecrafter_det_dry_run.txt"

bash "${SCRIPT_PATH}" \
    --preset edgecrafter_det \
    --weights-dir "${TMP_DIR}/weights" \
    --data-dir "${TMP_DIR}/data" \
    --labels-dir "${TMP_DIR}/labels" \
    --docker-image neuriplo-infer:onnxruntime \
    --dry-run > "${OUTPUT_FILE}"

grep -F -- "Intellindust-AI-Lab/EdgeCrafter" "${OUTPUT_FILE}"
grep -F -- "tools/deployment/export_onnx.py" "${OUTPUT_FILE}"
grep -F -- "configs/ecdet/ecdet_s.yml" "${OUTPUT_FILE}"
grep -F -- "--type=ecdet" "${OUTPUT_FILE}"
grep -F -- "--weights=/weights/ecdet_s.onnx" "${OUTPUT_FILE}"
grep -F -- "--input_sizes=3\\,640\\,640\\;2" "${OUTPUT_FILE}"
grep -F -- "neuriplo-infer:onnxruntime" "${OUTPUT_FILE}"

# ── edgecrafter_det (litert) dry-run ──────────────────────────────────────────
OUTPUT_FILE="${TMP_DIR}/edgecrafter_det_litert_dry_run.txt"

bash "${SCRIPT_PATH}" \
    --preset edgecrafter_det \
    --backend litert \
    --weights-dir "${TMP_DIR}/weights" \
    --data-dir "${TMP_DIR}/data" \
    --labels-dir "${TMP_DIR}/labels" \
    --docker-image neuriplo-infer:litert \
    --dry-run > "${OUTPUT_FILE}"

grep -F -- "Intellindust-AI-Lab/EdgeCrafter" "${OUTPUT_FILE}"
grep -F -- "tools/deployment/export_onnx.py" "${OUTPUT_FILE}"
grep -F -- "configs/ecdet/ecdet_s.yml" "${OUTPUT_FILE}"
grep -F -- "--type=ecdet" "${OUTPUT_FILE}"
# litert lowers the exported ONNX to TFLite via onnx2tf before inference
grep -F -- "onnx2tf -i" "${OUTPUT_FILE}"
grep -F -- "ecdet_s_float32.tflite" "${OUTPUT_FILE}"
grep -F -- "--weights=/weights/ecdet_s.tflite" "${OUTPUT_FILE}"
grep -F -- "--input_sizes=3\\,640\\,640\\;2" "${OUTPUT_FILE}"
grep -F -- "neuriplo-infer:litert" "${OUTPUT_FILE}"

# ── edgecrafter_seg dry-run ───────────────────────────────────────────────────
OUTPUT_FILE="${TMP_DIR}/edgecrafter_seg_dry_run.txt"

bash "${SCRIPT_PATH}" \
    --preset edgecrafter_seg \
    --weights-dir "${TMP_DIR}/weights" \
    --data-dir "${TMP_DIR}/data" \
    --labels-dir "${TMP_DIR}/labels" \
    --docker-image neuriplo-infer:onnxruntime \
    --dry-run > "${OUTPUT_FILE}"

grep -F -- "configs/ecseg/ecseg_s.yml" "${OUTPUT_FILE}"
grep -F -- "--type=ecseg" "${OUTPUT_FILE}"
grep -F -- "--weights=/weights/ecseg_s.onnx" "${OUTPUT_FILE}"
grep -F -- "--mask_threshold=0.5" "${OUTPUT_FILE}"
grep -F -- "--input_sizes=3\\,640\\,640\\;2" "${OUTPUT_FILE}"

# ── edgecrafter_pose dry-run ──────────────────────────────────────────────────
OUTPUT_FILE="${TMP_DIR}/edgecrafter_pose_dry_run.txt"

bash "${SCRIPT_PATH}" \
    --preset edgecrafter_pose \
    --weights-dir "${TMP_DIR}/weights" \
    --data-dir "${TMP_DIR}/data" \
    --docker-image neuriplo-infer:onnxruntime \
    --dry-run > "${OUTPUT_FILE}"

grep -F -- "configs/ecpose/ecpose_s_coco.yml" "${OUTPUT_FILE}"
grep -F -- "--type=ecpose" "${OUTPUT_FILE}"
grep -F -- "--weights=/weights/ecpose_s.onnx" "${OUTPUT_FILE}"
grep -F -- "--source=/app/data/person.jpg" "${OUTPUT_FILE}"
grep -F -- "--input_sizes=3\\,640\\,640\\;2" "${OUTPUT_FILE}"

# ── gemma4 dry-run ────────────────────────────────────────────────────────────
OUTPUT_FILE="${TMP_DIR}/gemma4_dry_run.txt"

bash "${SCRIPT_PATH}" \
    --preset gemma4 \
    --weights-dir "${TMP_DIR}/weights" \
    --data-dir "${TMP_DIR}/data" \
    --docker-image neuriplo-infer:llamacpp \
    --prompt "Describe what you see in this image." \
    --dry-run > "${OUTPUT_FILE}"

grep -F -- "gemma-4-E2B-it-Q4_K_M.gguf" "${OUTPUT_FILE}"
grep -F -- "--type=gemma4" "${OUTPUT_FILE}"
grep -F -- "--weights=/weights/gemma-4-E2B-it-Q4_K_M.gguf" "${OUTPUT_FILE}"
grep -F -- "--prompt=Describe" "${OUTPUT_FILE}"
grep -F -- "neuriplo-infer:llamacpp" "${OUTPUT_FILE}"

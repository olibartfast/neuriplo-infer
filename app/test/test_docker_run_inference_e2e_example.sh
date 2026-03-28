#!/bin/bash
set -euo pipefail

ROOT_DIR="${1:?repo root required}"
SCRIPT_PATH="${ROOT_DIR}/docker_run_inference_e2e_example.sh"
VISION_CORE_DIR="${ROOT_DIR}/../vision-core"

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT

mkdir -p "${TMP_DIR}/data" "${TMP_DIR}/labels" "${TMP_DIR}/weights"
touch "${TMP_DIR}/data/dog.jpg"
touch "${TMP_DIR}/labels/coco.names"

OUTPUT_FILE="${TMP_DIR}/owlv2_dry_run.txt"

bash "${SCRIPT_PATH}" \
    --preset owlv2 \
    --vision-core-dir "${VISION_CORE_DIR}" \
    --weights-dir "${TMP_DIR}/weights" \
    --data-dir "${TMP_DIR}/data" \
    --labels-dir "${TMP_DIR}/labels" \
    --docker-image vision-inference:test \
    --text-prompts 'cat;dog;bus' \
    --dry-run > "${OUTPUT_FILE}"

grep -F -- "export/open_vocab_detection/owlv2/export_owlv2_to_onnx.py" "${OUTPUT_FILE}"
grep -F -- "--type=owlv2" "${OUTPUT_FILE}"
grep -F -- "--text_prompts=cat\\;dog\\;bus" "${OUTPUT_FILE}"
grep -F -- "--tokenizer_vocab=/weights/vocab.json" "${OUTPUT_FILE}"
grep -F -- "--tokenizer_merges=/weights/merges.txt" "${OUTPUT_FILE}"
grep -F -- "vision-inference:test" "${OUTPUT_FILE}"

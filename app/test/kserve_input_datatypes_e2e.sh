#!/bin/bash
# End-to-end check that neuriplo-infer honors a model's advertised image-input
# datatype over KServe (issue #44), against a real neuriplo-kserve-runtime.
#
# It builds four tiny ONNX classifiers whose 1x3x32x32 image input is typed
# UINT8, INT8, BOOL, and FP32, serves them from one neuriplo-kserve-runtime in
# repository mode, and runs neuriplo-infer against each:
#   UINT8, FP32  must succeed, and the runtime must accept exactly one request.
#   INT8, BOOL   must fail at pipeline setup naming the datatype, and the
#                runtime must accept no request.
#
# Modes:
#   --dry-run       Print every command without executing it; exits 0. This is
#                   what ctest runs.
#   --live          Run it. Skips (exit 0) when a prerequisite is missing,
#                   unless --require-live is set.
#
# Inputs (flag or environment variable):
#   --runtime-bin PATH   neuriplo-kserve-runtime built with ONNX Runtime
#                        (NEURIPLO_KSERVE_RUNTIME_BIN)
#   --infer-bin PATH     neuriplo-infer built with KServe (NEURIPLO_INFER_BIN)
#   --port N             runtime HTTP port (PORT, default 18090)
#
# Usage:
#   bash app/test/kserve_input_datatypes_e2e.sh --dry-run
#   bash app/test/kserve_input_datatypes_e2e.sh --live \
#     --runtime-bin ../neuriplo-kserve-runtime/build/real-onnx/neuriplo-kserve-runtime \
#     --infer-bin build/app/neuriplo-infer
#
# Exit codes: 0 ok/skipped, non-zero on a real failure.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

DRY_RUN=false
REQUIRE_LIVE=false
RUNTIME_BIN="${NEURIPLO_KSERVE_RUNTIME_BIN:-}"
INFER_BIN="${NEURIPLO_INFER_BIN:-}"
PORT="${PORT:-18090}"
GRPC_PORT="${GRPC_PORT:-18091}"
SOURCE_IMAGE="${REPO_ROOT}/data/dog.jpg"

usage() {
  sed -n '2,31p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=true ;;
    --live) DRY_RUN=false ;;
    --require-live) REQUIRE_LIVE=true; DRY_RUN=false ;;
    --runtime-bin) RUNTIME_BIN="${2:?}"; shift ;;
    --infer-bin) INFER_BIN="${2:?}"; shift ;;
    --port) PORT="${2:?}"; shift ;;
    -h | --help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
  shift
done

log() { printf '%s\n' "$*"; }
section() { printf '\n=== %s ===\n' "$*"; }

skip_or_fail() {
  if [[ "${REQUIRE_LIVE}" == true ]]; then
    echo "ERROR: $1 (and --require-live was set)" >&2
    exit 1
  fi
  log "SKIP (live): $1"
  exit 0
}

WORK_DIR=""
RUNTIME_PID=""
FAILURES=0

# Invoked indirectly via `trap ... EXIT`.
# shellcheck disable=SC2317
cleanup() {
  if [[ -n "${RUNTIME_PID}" ]]; then
    kill "${RUNTIME_PID}" 2>/dev/null || true
    wait "${RUNTIME_PID}" 2>/dev/null || true
  fi
  if [[ -n "${WORK_DIR}" && -d "${WORK_DIR}" ]]; then
    rm -rf "${WORK_DIR}"
  fi
  return 0
}
trap cleanup EXIT

write_models() {
  python3 - "$1" <<'PY'
import os
import sys

import onnx
from onnx import TensorProto, helper

root = sys.argv[1]
for name, elem in [("cls_uint8", TensorProto.UINT8), ("cls_int8", TensorProto.INT8),
                   ("cls_bool", TensorProto.BOOL), ("cls_fp32", TensorProto.FLOAT)]:
    images = helper.make_tensor_value_info("images", elem, [1, 3, 32, 32])
    logits = helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 3])
    nodes, source = [], "images"
    if elem != TensorProto.FLOAT:
        nodes.append(helper.make_node("Cast", ["images"], ["as_float"], to=TensorProto.FLOAT))
        source = "as_float"
    nodes.append(helper.make_node("ReduceMean", [source], ["logits"], axes=[2, 3], keepdims=0))
    model = helper.make_model(helper.make_graph(nodes, name, [images], [logits]),
                              opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    out = os.path.join(root, name, "1")
    os.makedirs(out, exist_ok=True)
    onnx.save(model, os.path.join(out, "model.onnx"))
PY
}

# Requests the runtime has accepted for a model, from its Prometheus metrics.
accepted() {
  curl -s "http://127.0.0.1:${PORT}/metrics" |
    awk -v tag="model=\"$1\"" '
      /^neuriplo_scheduler_requests_accepted_total\{/ && index($0, tag) { print $NF; found = 1 }
      END { if (!found) { print "no accepted-requests metric for " tag > "/dev/stderr"; print 0 } }'
}

# run_case MODEL run|refuse DATATYPE
run_case() {
  local model="$1" expect="$2" datatype="$3"
  local bin="${INFER_BIN:-<infer-bin>}" work="${WORK_DIR:-<workdir>}"
  local cmd=("${bin}" --type=torchvision-classifier --source="${SOURCE_IMAGE}"
    --labels="${work}/labels.txt" --kserve_endpoint="http://127.0.0.1:${PORT}"
    --kserve_model_name="${model}" --kserve_transport=http)
  printf '+ (cd %s && %s)\n' "${work}" "${cmd[*]}"
  if [[ "${DRY_RUN}" == true ]]; then
    return 0
  fi

  local before after status=0 output
  before="$(accepted "${model}")"
  output="$(cd "${WORK_DIR}" && "${cmd[@]}" 2>&1)" || status=$?
  after="$(accepted "${model}")"
  local sent=$((after - before))

  if [[ "${expect}" == run ]]; then
    if [[ ${status} -eq 0 && ${sent} -eq 1 ]]; then
      log "PASS ${model}: ran; the runtime accepted 1 request"
      return 0
    fi
    log "FAIL ${model}: expected success with 1 request, got exit ${status} and ${sent} requests"
  else
    if [[ ${status} -ne 0 && ${sent} -eq 0 &&
      "${output}" == *"is ${datatype}, but image preprocessing"* ]]; then
      log "PASS ${model}: refused at setup naming ${datatype}; no request sent"
      return 0
    fi
    log "FAIL ${model}: expected a setup refusal naming ${datatype} with no request, got exit ${status} and ${sent} requests"
  fi
  printf '%s\n' "${output}" | tail -5
  FAILURES=$((FAILURES + 1))
}

section "KServe input datatypes e2e ($([[ ${DRY_RUN} == true ]] && echo dry-run || echo live))"

if [[ "${DRY_RUN}" == false ]]; then
  [[ -n "${RUNTIME_BIN}" && -x "${RUNTIME_BIN}" ]] ||
    skip_or_fail "neuriplo-kserve-runtime binary not set or not executable (--runtime-bin / NEURIPLO_KSERVE_RUNTIME_BIN)"
  [[ -n "${INFER_BIN}" && -x "${INFER_BIN}" ]] ||
    skip_or_fail "neuriplo-infer binary not set or not executable (--infer-bin / NEURIPLO_INFER_BIN)"
  # run_case changes into WORK_DIR, so both binaries must be absolute.
  INFER_BIN="$(realpath "${INFER_BIN}")"
  RUNTIME_BIN="$(realpath "${RUNTIME_BIN}")"
  command -v curl >/dev/null 2>&1 || skip_or_fail "curl not available"
  python3 -c 'import onnx' >/dev/null 2>&1 ||
    skip_or_fail "python3 'onnx' package not available to build the test models"
  [[ -f "${SOURCE_IMAGE}" ]] || skip_or_fail "sample image not found: ${SOURCE_IMAGE}"

  WORK_DIR="$(mktemp -d)"
  write_models "${WORK_DIR}/repo"
  printf 'red\ngreen\nblue\n' >"${WORK_DIR}/labels.txt"
fi

section "Serve"
printf '+ %s --host 127.0.0.1 --port %s --grpc-port %s --models %s/repo\n' \
  "${RUNTIME_BIN:-<runtime-bin>}" "${PORT}" "${GRPC_PORT}" "${WORK_DIR:-<workdir>}"
if [[ "${DRY_RUN}" == false ]]; then
  "${RUNTIME_BIN}" --host 127.0.0.1 --port "${PORT}" --grpc-port "${GRPC_PORT}" \
    --models "${WORK_DIR}/repo" >"${WORK_DIR}/runtime.log" 2>&1 &
  RUNTIME_PID=$!
  if ! curl -sf -o /dev/null --retry 30 --retry-connrefused --retry-delay 1 \
    "http://127.0.0.1:${PORT}/v2/health/ready"; then
    tail -20 "${WORK_DIR}/runtime.log" >&2
    echo "ERROR: neuriplo-kserve-runtime did not become ready on port ${PORT}" >&2
    exit 1
  fi
fi

section "Infer"
run_case cls_uint8 run UINT8
run_case cls_fp32 run FP32
run_case cls_int8 refuse INT8
run_case cls_bool refuse BOOL

if [[ ${FAILURES} -ne 0 ]]; then
  section "RESULT: FAIL (${FAILURES} case(s))"
  exit 1
fi
section "RESULT: PASS ($([[ ${DRY_RUN} == true ]] && echo 'dry-run: commands well-formed' || echo 'datatypes honored end to end'))"

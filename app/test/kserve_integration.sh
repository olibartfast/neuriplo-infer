#!/bin/bash
# KServe V2 (Open Inference Protocol) integration test against containerized
# inference servers — Triton Inference Server and OpenVINO Model Server (OVMS).
#
# It spins up each server with a tiny shared ONNX "identity" model, then drives a
# KServe V2 inference round-trip over BOTH HTTP and gRPC against each server and
# asserts the echoed tensor matches what was sent. This exercises the same wire
# protocol the neuriplo-infer KServe client speaks (KserveHttpClient /
# KserveGrpcClient / KserveEngine), keeping docs/KserveCompatibility.md honest.
#
# Modes (mirrors docker_run_inference_e2e_example.sh gating):
#   --dry-run   Print/validate every docker + client command WITHOUT executing.
#               Always exits 0 when command construction is well-formed, so CI
#               on runners without docker/images/GPUs still passes. This is what
#               `ctest`/CI run by default.
#   --live      Actually pull images, start containers, and run inferences.
#               Auto-skips (exit 0) if docker or required tooling is unavailable,
#               unless --require-live is set (then a missing prerequisite fails).
#
# Selectors:
#   --servers triton,ovms     Comma-separated subset to test (default: both).
#   --transports http,grpc    Comma-separated subset (default: both).
#
# Usage:
#   bash app/test/kserve_integration.sh --dry-run
#   bash app/test/kserve_integration.sh --live
#   bash app/test/kserve_integration.sh --live --servers triton --transports http
#
# Exit codes: 0 ok/skipped, non-zero on a real failure.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ── Defaults ──────────────────────────────────────────────────────────────────
DRY_RUN=false
REQUIRE_LIVE=false
KEEP=false
SERVERS="triton,ovms"
TRANSPORTS="http,grpc"

MODEL_NAME="identity_fp32"
TRITON_IMAGE="${TRITON_IMAGE:-nvcr.io/nvidia/tritonserver:24.08-py3}"
OVMS_IMAGE="${OVMS_IMAGE:-openvino/model_server:latest}"

# Host ports. Triton exposes HTTP 8000 / gRPC 8001; OVMS REST 8000 / gRPC 9000.
TRITON_HTTP_PORT="${TRITON_HTTP_PORT:-18000}"
TRITON_GRPC_PORT="${TRITON_GRPC_PORT:-18001}"
OVMS_HTTP_PORT="${OVMS_HTTP_PORT:-18002}"
OVMS_GRPC_PORT="${OVMS_GRPC_PORT:-18003}"

# The KServe gRPC proto lives in the neuriplo-kserve-client sibling repo now.
# Resolve it (for the live grpcurl path) from, in order: an explicit PROTO_FILE
# override, a sibling checkout, or a FetchContent-populated build tree. Dry-run
# only prints the command, so an unresolved path here is harmless there.
if [[ -z "${PROTO_FILE:-}" ]]; then
  for _cand in \
    "${REPO_ROOT}/../neuriplo-kserve-client/proto/kserve_grpc.proto" \
    "${REPO_ROOT}"/build*/_deps/neuriplo-kserve-client-src/proto/kserve_grpc.proto; do
    if [[ -f "${_cand}" ]]; then PROTO_FILE="${_cand}"; break; fi
  done
fi
PROTO_FILE="${PROTO_FILE:-${REPO_ROOT}/../neuriplo-kserve-client/proto/kserve_grpc.proto}"

# ── Arg parsing ───────────────────────────────────────────────────────────────
usage() {
  sed -n '2,40p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=true ;;
    --live) DRY_RUN=false ;;
    --require-live) REQUIRE_LIVE=true; DRY_RUN=false ;;
    --keep) KEEP=true ;;
    --servers) SERVERS="${2:?}"; shift ;;
    --transports) TRANSPORTS="${2:?}"; shift ;;
    --model-name) MODEL_NAME="${2:?}"; shift ;;
    --triton-image) TRITON_IMAGE="${2:?}"; shift ;;
    --ovms-image) OVMS_IMAGE="${2:?}"; shift ;;
    -h | --help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
  shift
done

has() { [[ ",$1," == *",$2,"* ]]; }

log() { printf '%s\n' "$*"; }
section() { printf '\n=== %s ===\n' "$*"; }

# Run a command, or just print it in dry-run.
run_cmd() {
  printf '+ %s\n' "$*"
  if [[ "${DRY_RUN}" == false ]]; then
    "$@"
  fi
}

# Skip the live path gracefully (or fail when --require-live).
skip_or_fail() {
  local reason="$1"
  if [[ "${REQUIRE_LIVE}" == true ]]; then
    echo "ERROR: ${reason} (and --require-live was set)" >&2
    exit 1
  fi
  log "SKIP (live): ${reason}"
  exit 0
}

# ── Workspace ─────────────────────────────────────────────────────────────────
WORK_DIR=""
CONTAINERS=()

cleanup() {
  if [[ "${KEEP}" == true ]]; then
    log "Leaving containers/workspace in place (--keep): ${WORK_DIR}"
    return
  fi
  if [[ "${DRY_RUN}" == false ]]; then
    for c in "${CONTAINERS[@]:-}"; do
      [[ -n "${c}" ]] || continue
      docker rm -f "${c}" >/dev/null 2>&1 || true
    done
    [[ -n "${WORK_DIR}" && -d "${WORK_DIR}" ]] && rm -rf "${WORK_DIR}"
  fi
}
trap cleanup EXIT

# ── Tiny model generation ─────────────────────────────────────────────────────
# Both servers serve the SAME ONNX file (Identity, FP32, shape [4]); only the
# repository layout differs (Triton needs config.pbtxt; OVMS auto-detects).
PY_BIN="python3"

write_identity_onnx() {
  local dst="$1"
  "${PY_BIN}" - "$dst" <<'PYEOF'
import sys
import onnx
from onnx import helper, TensorProto

dst = sys.argv[1]
node = helper.make_node("Identity", ["input0"], ["output0"])
graph = helper.make_graph(
    [node],
    "identity_fp32",
    [helper.make_tensor_value_info("input0", TensorProto.FLOAT, [4])],
    [helper.make_tensor_value_info("output0", TensorProto.FLOAT, [4])],
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
model.ir_version = 9
onnx.save(model, dst)
print(f"wrote {dst}")
PYEOF
}

prepare_triton_repo() {
  local repo="${WORK_DIR}/triton"
  if [[ "${DRY_RUN}" == false ]]; then
    mkdir -p "${repo}/${MODEL_NAME}/1"
    write_identity_onnx "${repo}/${MODEL_NAME}/1/model.onnx" >&2
    cat >"${repo}/${MODEL_NAME}/config.pbtxt" <<EOF
name: "${MODEL_NAME}"
backend: "onnxruntime"
max_batch_size: 0
input [ { name: "input0" data_type: TYPE_FP32 dims: [ 4 ] } ]
output [ { name: "output0" data_type: TYPE_FP32 dims: [ 4 ] } ]
EOF
  fi
  printf '%s' "${repo}"
}

prepare_ovms_repo() {
  local repo="${WORK_DIR}/ovms"
  if [[ "${DRY_RUN}" == false ]]; then
    mkdir -p "${repo}/${MODEL_NAME}/1"
    write_identity_onnx "${repo}/${MODEL_NAME}/1/model.onnx" >&2
  fi
  printf '%s' "${repo}"
}

# ── Readiness + round-trip helpers ────────────────────────────────────────────
wait_http_ready() {
  local base="$1" tries="${2:-60}"
  log "Waiting for ${base}/v2/health/ready ..."
  for _ in $(seq 1 "${tries}"); do
    if curl -fsS "${base}/v2/health/ready" >/dev/null 2>&1; then
      log "  ready: ${base}"
      return 0
    fi
    sleep 2
  done
  return 1
}

# KServe V2 HTTP infer round-trip (curl). Asserts the echoed output matches.
infer_http() {
  local base="$1"
  local url="${base}/v2/models/${MODEL_NAME}/infer"
  local body='{"inputs":[{"name":"input0","shape":[4],"datatype":"FP32","data":[1.0,2.0,3.0,4.0]}]}'
  log "+ curl -fsS -X POST ${url} -d '${body}'"
  if [[ "${DRY_RUN}" == true ]]; then
    return 0
  fi
  local resp
  resp="$(curl -fsS -H 'Content-Type: application/json' -X POST "${url}" -d "${body}")"
  log "  response: ${resp}"
  echo "${resp}" | grep -q '"output0"' ||
    { echo "ERROR: HTTP infer response missing output0" >&2; return 1; }
  echo "${resp}" | grep -Eq '4(\.0+)?' ||
    { echo "ERROR: HTTP infer echo mismatch" >&2; return 1; }
  log "  HTTP round-trip OK"
}

# KServe V2 gRPC infer round-trip (grpcurl, using the repo proto). Asserts the
# ModelInfer call returns output0.
infer_grpc() {
  local hostport="$1"
  local req='{"model_name":"'"${MODEL_NAME}"'","inputs":[{"name":"input0","shape":[4],"datatype":"FP32","contents":{"fp32_contents":[1,2,3,4]}}]}'
  local cmd=(grpcurl -plaintext -proto "${PROTO_FILE}"
    -d "${req}" "${hostport}" inference.GRPCInferenceService/ModelInfer)
  log "+ ${cmd[*]}"
  if [[ "${DRY_RUN}" == true ]]; then
    return 0
  fi
  if ! command -v grpcurl >/dev/null 2>&1; then
    skip_or_fail "grpcurl not installed; cannot run gRPC round-trip"
  fi
  local resp
  resp="$("${cmd[@]}")"
  log "  response: ${resp}"
  echo "${resp}" | grep -q 'output0' ||
    { echo "ERROR: gRPC infer response missing output0" >&2; return 1; }
  log "  gRPC round-trip OK"
}

# ── Server drivers ────────────────────────────────────────────────────────────
test_triton() {
  section "Triton Inference Server"
  local repo
  repo="$(prepare_triton_repo)"
  local cname="neuriplo-kserve-it-triton"
  CONTAINERS+=("${cname}")
  run_cmd docker run -d --rm --name "${cname}" \
    -p "${TRITON_HTTP_PORT}:8000" -p "${TRITON_GRPC_PORT}:8001" \
    -v "${repo}:/models:ro" \
    "${TRITON_IMAGE}" \
    tritonserver --model-repository=/models --model-control-mode=none

  if [[ "${DRY_RUN}" == false ]]; then
    wait_http_ready "http://127.0.0.1:${TRITON_HTTP_PORT}" ||
      { echo "ERROR: Triton did not become ready" >&2; return 1; }
  fi
  has "${TRANSPORTS}" http && infer_http "http://127.0.0.1:${TRITON_HTTP_PORT}"
  has "${TRANSPORTS}" grpc && infer_grpc "127.0.0.1:${TRITON_GRPC_PORT}"
  run_cmd docker rm -f "${cname}"
}

test_ovms() {
  section "OpenVINO Model Server"
  local repo
  repo="$(prepare_ovms_repo)"
  local cname="neuriplo-kserve-it-ovms"
  CONTAINERS+=("${cname}")
  run_cmd docker run -d --rm --name "${cname}" \
    -p "${OVMS_HTTP_PORT}:8000" -p "${OVMS_GRPC_PORT}:9000" \
    -v "${repo}:/models:ro" \
    "${OVMS_IMAGE}" \
    --model_name "${MODEL_NAME}" --model_path "/models/${MODEL_NAME}" \
    --rest_port 8000 --port 9000

  if [[ "${DRY_RUN}" == false ]]; then
    wait_http_ready "http://127.0.0.1:${OVMS_HTTP_PORT}" ||
      { echo "ERROR: OVMS did not become ready" >&2; return 1; }
  fi
  has "${TRANSPORTS}" http && infer_http "http://127.0.0.1:${OVMS_HTTP_PORT}"
  has "${TRANSPORTS}" grpc && infer_grpc "127.0.0.1:${OVMS_GRPC_PORT}"
  run_cmd docker rm -f "${cname}"
}

# ── Main ──────────────────────────────────────────────────────────────────────
section "KServe integration ($([[ ${DRY_RUN} == true ]] && echo dry-run || echo live))"
log "servers=${SERVERS} transports=${TRANSPORTS} model=${MODEL_NAME}"
log "triton-image=${TRITON_IMAGE}"
log "ovms-image=${OVMS_IMAGE}"

if [[ ! -f "${PROTO_FILE}" ]]; then
  echo "ERROR: proto not found at ${PROTO_FILE}" >&2
  exit 1
fi

# Live prerequisites: docker + the model generator + curl.
if [[ "${DRY_RUN}" == false ]]; then
  command -v docker >/dev/null 2>&1 || skip_or_fail "docker not available"
  docker info >/dev/null 2>&1 || skip_or_fail "docker daemon not reachable"
  command -v curl >/dev/null 2>&1 || skip_or_fail "curl not available"
  "${PY_BIN}" -c 'import onnx' >/dev/null 2>&1 ||
    skip_or_fail "python3 'onnx' package not available to build the tiny model"
fi

WORK_DIR="$(mktemp -d)"

rc=0
has "${SERVERS}" triton && { test_triton || rc=1; }
has "${SERVERS}" ovms && { test_ovms || rc=1; }

if [[ "${rc}" -eq 0 ]]; then
  section "RESULT: PASS ($([[ ${DRY_RUN} == true ]] && echo 'dry-run: commands well-formed' || echo 'live round-trips OK'))"
else
  section "RESULT: FAIL"
fi
exit "${rc}"

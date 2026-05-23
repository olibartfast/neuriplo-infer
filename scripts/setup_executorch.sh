#!/bin/bash
# ExecuTorch C++ runtime setup for vision-inference.
# Delegates to neuriplo's setup_executorch.sh which builds ExecuTorch from source.
#
# Usage: bash scripts/setup_executorch.sh [--install-dir <path>]
# Default install dir: ~/dependencies/executorch
#
# Prerequisite: run cmake configure first so FetchContent populates build/_deps/,
# or point directly to a local neuriplo checkout via NEURIPLO_DIR env var.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

find_neuriplo_setup() {
    # 1. Explicit override
    if [[ -n "${NEURIPLO_DIR:-}" ]]; then
        echo "${NEURIPLO_DIR}/scripts/setup_executorch.sh"
        return
    fi

    # 2. Common build/_deps locations (cmake configure populates these)
    for build_dir in build build-release build-test build-followup build_ort build-llamacpp; do
        local candidate="${ROOT_DIR}/${build_dir}/_deps/neuriplo-src/scripts/setup_executorch.sh"
        if [[ -f "${candidate}" ]]; then
            echo "${candidate}"
            return
        fi
    done

    echo ""
}

NEURIPLO_SETUP="$(find_neuriplo_setup)"

if [[ -z "${NEURIPLO_SETUP}" || ! -f "${NEURIPLO_SETUP}" ]]; then
    echo "Error: neuriplo setup_executorch.sh not found." >&2
    echo "" >&2
    echo "Run cmake configure first to populate build/_deps/:" >&2
    echo "  cmake -S . -B build -DDEFAULT_BACKEND=EXECUTORCH" >&2
    echo "" >&2
    echo "Or set NEURIPLO_DIR to a local neuriplo checkout:" >&2
    echo "  NEURIPLO_DIR=/path/to/neuriplo bash scripts/setup_executorch.sh" >&2
    exit 1
fi

echo "Using neuriplo setup script: ${NEURIPLO_SETUP}"
exec bash "${NEURIPLO_SETUP}" "$@"

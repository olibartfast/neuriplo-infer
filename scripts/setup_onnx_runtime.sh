#!/bin/bash

# Backward-compatible ONNX Runtime setup script
# This script calls the unified setup script for ONNX Runtime

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"$SCRIPT_DIR/setup_dependencies.sh" --backend onnx_runtime "$@" 
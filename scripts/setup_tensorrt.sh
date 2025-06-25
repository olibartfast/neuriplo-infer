#!/bin/bash

# Backward-compatible TensorRT setup script
# This script calls the unified setup script for TensorRT

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"$SCRIPT_DIR/setup_dependencies.sh" --backend tensorrt "$@" 
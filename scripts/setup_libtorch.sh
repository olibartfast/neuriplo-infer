#!/bin/bash

# Backward-compatible LibTorch setup script
# This script calls the unified setup script for LibTorch

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"$SCRIPT_DIR/setup_dependencies.sh" --backend libtorch "$@" 
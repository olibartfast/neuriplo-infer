#!/bin/bash

# Backward-compatible OpenVINO setup script
# This script calls the unified setup script for OpenVINO

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"$SCRIPT_DIR/setup_dependencies.sh" --backend openvino "$@" 
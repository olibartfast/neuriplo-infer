#!/bin/bash

# Backward-compatible TensorFlow setup script
# This script calls the unified setup script for TensorFlow

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"$SCRIPT_DIR/setup_dependencies.sh" --backend tensorflow "$@"
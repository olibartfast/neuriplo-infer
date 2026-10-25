#!/bin/bash
# Build OpenCV 5 from source into $HOME/dependencies (non-apt, like the
# other backend SDKs). apt OpenCV 4.6 cannot parse current Ultralytics
# exports (Split/Unsqueeze); the OPENCV_DNN backend needs 5.x.
# Usage: scripts/setup_opencv.sh [--version 5.0.0] [--jobs N]
set -euo pipefail

VERSION="5.0.0"
JOBS="$(nproc)"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --version) VERSION="$2"; shift 2 ;;
        --jobs) JOBS="$2"; shift 2 ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

ROOT="$HOME/dependencies/opencv-${VERSION}"
SRC="$ROOT"
BUILD="$ROOT/build"
INSTALL="$ROOT/install"

if [[ ! -d "$SRC" ]]; then
    git clone --depth 1 --branch "$VERSION" \
        https://github.com/opencv/opencv.git "$SRC"
fi
cmake -S "$SRC" -B "$BUILD" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$INSTALL" \
    -DBUILD_TESTS=OFF \
    -DBUILD_PERF_TESTS=OFF \
    -DBUILD_EXAMPLES=OFF \
    -DBUILD_opencv_apps=OFF \
    -DBUILD_JAVA=OFF \
    -DBUILD_opencv_python3=OFF
cmake --build "$BUILD" -j"$JOBS"
cmake --install "$BUILD"

echo "OpenCV $VERSION installed to $INSTALL"
echo "Point CMake at it: -DOpenCV_DIR=$INSTALL/lib/cmake/opencv5"
echo "(picked automatically when the directory exists; explicit -DOpenCV_DIR wins)"

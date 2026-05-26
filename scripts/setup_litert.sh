#!/bin/bash

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -f "$PROJECT_ROOT/versions.neuriplo.env" ]]; then
    source "$PROJECT_ROOT/versions.neuriplo.env"
elif [[ -f "$PROJECT_ROOT/versions.env" ]]; then
    source "$PROJECT_ROOT/versions.env"
else
    echo "Error: versions.neuriplo.env or versions.env file not found" >&2
    exit 1
fi

DEPENDENCY_ROOT="${DEPENDENCY_ROOT:-$HOME/dependencies}"
INSTALL_DIR="${LITERT_DIR:-$DEPENDENCY_ROOT/litert}"
SOURCE_DIR="$DEPENDENCY_ROOT/tensorflow-src-$LITERT_VERSION"
BUILD_DIR="$SOURCE_DIR/litert-build"
FORCE="${FORCE:-false}"

if [[ -d "$INSTALL_DIR" && "$FORCE" != "true" ]]; then
    echo "LiteRT already exists at $INSTALL_DIR"
    echo "Set FORCE=true to rebuild."
    exit 0
fi

mkdir -p "$DEPENDENCY_ROOT"

if [[ ! -d "$SOURCE_DIR" ]]; then
    git clone --depth 1 --branch "v$LITERT_VERSION" https://github.com/tensorflow/tensorflow.git "$SOURCE_DIR"
fi

cmake -S "$SOURCE_DIR/tensorflow/lite" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=ON \
    -DTFLITE_ENABLE_INSTALL=OFF

cmake --build "$BUILD_DIR" --target tensorflow-lite --parallel "$(nproc)"
cmake --build "$BUILD_DIR" --target \
    absl_log_internal_message \
    absl_log_internal_check_op \
    absl_log_internal_conditions \
    absl_log_internal_format \
    absl_log_internal_globals \
    absl_log_internal_log_sink_set \
    absl_log_internal_nullguard \
    absl_log_entry \
    absl_log_globals \
    absl_log_initialize \
    absl_log_sink \
    --parallel "$(nproc)"

rm -rf "$INSTALL_DIR"
mkdir -p "$INSTALL_DIR/include" "$INSTALL_DIR/lib"
cp -a "$SOURCE_DIR/tensorflow" "$INSTALL_DIR/include/"

while IFS= read -r include_dir; do
    cp -a "$include_dir"/. "$INSTALL_DIR/include/"
done < <(find "$BUILD_DIR" -type d -name include)

while IFS= read -r header_dir; do
    cp -a "$header_dir" "$INSTALL_DIR/include/"
done < <(find "$BUILD_DIR" "$SOURCE_DIR/third_party" -type d \( -name absl -o -name Eigen -o -name flatbuffers -o -name ruy \) 2>/dev/null)

find "$BUILD_DIR" \( -name '*.so*' -o -name '*.a' \) -exec cp -a {} "$INSTALL_DIR/lib/" \;

if [[ ! -f "$INSTALL_DIR/lib/libtensorflow-lite.so" ]]; then
    echo "Error: libtensorflow-lite.so was not produced by the LiteRT build" >&2
    exit 1
fi

ln -sf libtensorflow-lite.so "$INSTALL_DIR/lib/libtensorflowlite.so"

echo "LiteRT installed to $INSTALL_DIR"

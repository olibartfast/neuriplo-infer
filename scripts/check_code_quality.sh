#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_ASAN="${ROOT_DIR}/build-quality-asan"
BUILD_TSAN="${ROOT_DIR}/build-quality-tsan"

CPP_FILES=(
  "${ROOT_DIR}/app/main.cpp"
  "${ROOT_DIR}/app/inc/AppConfig.hpp"
  "${ROOT_DIR}/app/inc/CLICommands.hpp"
  "${ROOT_DIR}/app/inc/InferencePipeline.hpp"
  "${ROOT_DIR}/app/inc/ResultRenderer.hpp"
  "${ROOT_DIR}/app/inc/TaskRouting.hpp"
  "${ROOT_DIR}/app/inc/VisionApp.hpp"
  "${ROOT_DIR}/app/src/CLICommands.cpp"
  "${ROOT_DIR}/app/src/CommandLineParser.cpp"
  "${ROOT_DIR}/app/src/InferencePipeline.cpp"
  "${ROOT_DIR}/app/src/ResultRenderer.cpp"
  "${ROOT_DIR}/app/src/VisionApp.cpp"
  "${ROOT_DIR}/app/src/VisionAppTaskRouting.cpp"
)

if command -v clang-format >/dev/null 2>&1; then
  clang-format --dry-run --Werror "${CPP_FILES[@]}"
else
  echo "clang-format not found; skipping format check" >&2
fi

if command -v cppcheck >/dev/null 2>&1; then
  cppcheck --enable=warning,performance,portability \
    --std=c++20 \
    --error-exitcode=1 \
    --inline-suppr \
    --suppress=missingIncludeSystem \
    "${ROOT_DIR}/app"
else
  echo "cppcheck not found; skipping static analysis" >&2
fi

cmake -S "${ROOT_DIR}" -B "${BUILD_ASAN}" \
  -DDEFAULT_BACKEND=OPENCV_DNN \
  -DENABLE_APP_TESTS=ON \
  -DSANITIZER=address-undefined \
  -DCMAKE_BUILD_TYPE=Debug
cmake --build "${BUILD_ASAN}" --parallel "$(nproc)"
ctest --test-dir "${BUILD_ASAN}" --output-on-failure

cmake -S "${ROOT_DIR}" -B "${BUILD_TSAN}" \
  -DDEFAULT_BACKEND=OPENCV_DNN \
  -DENABLE_APP_TESTS=ON \
  -DSANITIZER=thread \
  -DCMAKE_BUILD_TYPE=Debug
cmake --build "${BUILD_TSAN}" --parallel "$(nproc)"
ctest --test-dir "${BUILD_TSAN}" --output-on-failure

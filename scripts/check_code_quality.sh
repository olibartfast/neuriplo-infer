#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_ASAN="${ROOT_DIR}/build-quality-asan"
BUILD_TSAN="${ROOT_DIR}/build-quality-tsan"

mapfile -t CPP_FILES < <(
  find "${ROOT_DIR}/app/src" "${ROOT_DIR}/app/inc" "${ROOT_DIR}/app/test" \
    \( -name '*.cpp' -o -name '*.hpp' \) | sort
)

CLANG_FORMAT_BIN="${CLANG_FORMAT_BIN:-clang-format-18}"
if ! command -v "${CLANG_FORMAT_BIN}" >/dev/null 2>&1; then
  CLANG_FORMAT_BIN="clang-format"
fi

if command -v "${CLANG_FORMAT_BIN}" >/dev/null 2>&1; then
  "${CLANG_FORMAT_BIN}" --dry-run --Werror "${CPP_FILES[@]}"
else
  echo "clang-format not found; skipping format check" >&2
fi

if command -v cppcheck >/dev/null 2>&1; then
  cppcheck --enable=warning,performance,portability \
    --std=c++20 \
    --error-exitcode=1 \
    --inline-suppr \
    --suppress=missingIncludeSystem \
    --suppress=unmatchedSuppression \
    -I "${ROOT_DIR}/app/inc" \
    "${ROOT_DIR}/app/src" \
    "${ROOT_DIR}/app/inc"
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

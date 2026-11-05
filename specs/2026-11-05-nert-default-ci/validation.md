# Validation — NERT default and CI

Written before implementation.

- [x] Fresh default configures and builds with app tests in build-nert.
- [x] NERT app suite passes; capabilities validates and advertises only `nert`
      in the local workflow. Configure omits DEFAULT_BACKEND.
- [x] Explicit OPENCV_DNN build and app suite remain green.
- [x] KServe-only build still configures without fetching neuriplo/NERT.
- [x] Setup-script syntax and help state NERT as default; OpenCV remains explicit.
- [x] CI workflow parses and the fail-fast matrix includes distinct NERT and
      OpenCV jobs with separate cache keys; NERT verifies its advertised backend.
- [x] Generated model documentation matches the pinned tasks README.
- [x] Relative and GitHub documentation links resolve; diff has no whitespace errors.
- [ ] Hosted feature/develop CI and master release check are green before release.

## Limits

This change certifies application wiring and tests, not new model numerical
parity. Locally gRPC is disabled because grpc_cpp_plugin is missing and sudo
requires an interactive password; hosted CI covers the gRPC configuration.

## Local results

- Fresh default configured with `cmake -S . -B build-nert -DENABLE_APP_TESTS=ON
  -DCMAKE_BUILD_TYPE=Release -DNEURIPLO_INFER_ENABLE_GRPC=OFF`; built with
  `cmake --build build-nert --parallel 4`. App suite: 192/192 passed.
- Explicit OPENCV_DNN build-test retains its backend; rebuilt and 192/192 passed.
- `DEFAULT_BACKEND:STRING=NERT` and local capabilities `backends: ["nert"]`
  confirmed; capabilities schema validation passes.
- KServe-only configures with local backends off and no neuriplo/NERT sources.
- YAML parses; the fail-fast matrix has independent NERT/OpenCV entries and cache
  keys; cache checker and SDK-version tests pass. Setup syntax/help passes.
- Generated docs check passes against the fetched tasks v0.8.5 README.
- Relative links resolve and all 15 existing GitHub documentation links return 200.
- No C++ source changes; pre-commit was not run. Hosted lint remains a release gate.
- No real model numerical gate was added; coverage statuses are unchanged.

## Hosted installation diagnosis

The synchronized run passed NERT, OpenCV, ONNX Runtime, TensorRT, TensorFlow,
OpenVINO, E2E, and lint. KServe gRPC and LibTorch spent their install budget
downloading Ubuntu packages from azure.archive.ubuntu.com at roughly 40 KB/s.
Master Release Check was similarly delayed. Change only that mirror URL in
the shared action; hosted results after the correction remain pending.
A supplemental parallel local CTest run collided on shared fixture directories;
the canonical serial rerun passed 192/192. CI uses serial CTest.

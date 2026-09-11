# Feature Plan — Model input datatypes reach preprocessing

## Group 1 — neuriplo-tasks: honor image input types

1. `Preprocessor` gains an image-input-type hook: `Float32` leaves the config
   untouched; `UInt8` switches to raw pixels (`data_type = UINT8`, no `/255`, no
   ImageNet statistics); anything else throws. Preprocessors whose `preprocess`
   bypasses the shared config path (RAFT pairs, VideoMAE / ViViT / TimeSformer)
   report that raw pixels are unsupported, so `UInt8` throws there too instead of
   silently emitting floats.
2. A core helper applies `ModelInfo::input_types` of the image inputs (rank ≥ 3)
   to a task's preprocessor, naming the offending input in every error, and
   exposes the image-input rule so consumers do not re-derive it.
3. Call the helper wherever a task builds its image preprocessor: object detection,
   instance segmentation, pose, classification, depth, optical flow, Gaussian
   splatting, video classification, open-vocabulary detection.
4. Tests: `UInt8` yields `W*H*C` raw bytes equal to the resized input (NCHW and
   NHWC, letterboxed YOLO); `Float32` output is byte-identical to today;
   `Int32` and `UInt8`-on-RAFT are rejected with the input name; the TensorFlow
   classifier keeps emitting `UINT8` under the default tag.
5. `CHANGELOG.md` `[Unreleased]`; lint (format, clang-tidy, cppcheck, WERROR,
   tests, valgrind job's config) before commit; PR into `develop`.

## Group 2 — neuriplo-infer: carry and check datatypes

6. App-local contract: `TensorDataType` + `LayerInfo::datatype` + defaulted
   `addInput`/`addOutput` parameter, mirroring neuriplo v0.9.1.
7. `KserveEngine::ensureMetadata()` records representable datatypes on inputs and
   outputs.
8. `buildModelInfo()` drops the forced `Float32`. In `preprocessed` mode, for each
   image input: `FP32` → `Float32`, `UINT8` → `UInt8`, anything else → error naming
   input, datatype, and the way out (an FP32/UINT8 model, or `--input_mode=
   encoded-image`). For KServe the datatype comes from the raw server tag, so
   `FP16` and friends are named precisely. `encoded-image` is exempt.
9. `KserveEngine::get_infer_results()` byte-count guard: bytes must be a whole
   multiple of the datatype width (times the static dims when one axis is
   dynamic) and must equal the element count when the shape is fully static.
10. Unit tests with the fake client: datatype propagation; guard accepts FP32,
    UINT8, INT64 size inputs and a dynamic `[-1]` encoded input; guard rejects
    Float32 bytes under `INT8`, `BOOL`, `UINT8`, `FP16` before `infer()` is called.
    Setup-time rejection and the `UINT8` mapping are covered through a fake engine.
11. `versions.env`: `NEURIPLO_VERSION=v0.9.1`, `NEURIPLO_TASKS_VERSION` to the new
    neuriplo-tasks tag. `CHANGELOG.md` including the neuriplo-tasks v0.8.1
    preprocessing change the pin pulls in; `docs/KserveRuntime.md` constraints.

## Group 3 — End to end

12. Build `neuriplo-kserve-runtime` (`real-onnx`) against neuriplo v0.9.1. Serve
    three tiny ONNX classifiers with `[1,3,32,32]` image inputs typed `UINT8`,
    `INT8`, `BOOL`. Run `neuriplo-infer --type=torchvision-classifier
    --kserve_endpoint=...` against each: `UINT8` succeeds (server accepts the
    element count and datatype), `INT8` and `BOOL` exit non-zero at setup with the
    datatype named, and the runtime logs no inference request for them.
13. Record commands and output in `validation.md`.

## Group 4 — Integration

14. Lint and test neuriplo-infer (format, cppcheck, clang-tidy, WERROR build,
    tests) before commit. Merge the neuriplo-tasks PR, cut its release, pin it,
    then open the neuriplo-infer PR.

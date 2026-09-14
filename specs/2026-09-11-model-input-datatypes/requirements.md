# Feature Requirements — Model input datatypes reach preprocessing

Roadmap phase: none — fixes issue
[#44](https://github.com/olibartfast/neuriplo-infer/issues/44).
Branch: `feature/kserve-input-datatypes` (neuriplo-infer),
`feature/model-input-datatypes` (neuriplo-tasks).

## Goal

A model's advertised input datatypes decide what preprocessing sends it. An image
input advertised as `UINT8` receives raw pixels and runs; an image input whose
datatype preprocessing cannot produce (`INT8`, `BOOL`, `INT32`, `INT64`, `FP16`,
…) fails at pipeline setup with an error naming the input and its datatype; and no
KServe request is ever sent with bytes that do not fit the datatype it is labelled
with. Today `buildModelInfo()` forces the first input to `Float32` and
`KserveEngine` labels Float32 bytes with whatever the server advertised.

## In Scope

- **neuriplo-infer**
  - `KserveEngine` carries each advertised datatype into `InferenceMetadata`
    (`LayerInfo::datatype`) for the types neuriplo can represent.
  - The app-local KServe-only contract (`app/inc/contract/`) gains the same
    `TensorDataType` / `LayerInfo::datatype` shape as neuriplo v0.9.1.
  - `buildModelInfo()` stops forcing `Float32`. In `preprocessed` mode it maps
    each image input's datatype into `ModelInfo::input_types` and rejects image
    inputs whose datatype preprocessing cannot produce. `encoded-image` mode is
    exempt: the server preprocesses there.
  - `KserveEngine::get_infer_results()` rejects, before calling the client, an
    input whose byte count cannot hold whole elements of its advertised datatype,
    or does not match a fully static shape.
  - Unit tests for the datatype propagation, the setup-time rejection, and the
    byte-count guard; an end-to-end check through `neuriplo-kserve-runtime` with
    `UINT8` (runs), `INT8` and `BOOL` (rejected) image inputs.
  - `versions.env`: `NEURIPLO_VERSION=v0.9.1` (requested); `NEURIPLO_TASKS_VERSION`
    moves to the neuriplo-tasks release carrying the change below.
- **neuriplo-tasks**
  - Preprocessing honors `ModelInfo::input_types` for image inputs (rank ≥ 3):
    `Float32` keeps each task's current preprocessing unchanged; `UInt8` emits raw
    0–255 pixels — same resize, letterbox, color order and layout, no `/255`, no
    ImageNet statistics; any other type is rejected with an error naming the input.
  - Applied in every task that preprocesses images through `Preprocessor`.
  - Released as a tagged version (maintainer asked me to cut it after merge).

## Out of Scope

- `INT8` / `BOOL` image preprocessing (maintainer decision: reject).
- Image understanding (llama.cpp raw-RGB contract) — no `Preprocessor`, no
  model-advertised image datatype.
- Non-image inputs (size tensors, token ids): their types stay task-owned; only the
  pre-request byte-count guard applies to them.
- Output datatype handling (already decoded from the wire datatype).
- tritonic changes; it already fills `input_types` from Triton metadata and gains
  the `UInt8` behavior through the neuriplo-tasks release.

## Decisions

- **Maintainer-approved exception** to neuriplo-tasks `REPO_META.yaml`, which
  forbids agent `model-io-change` / `inference-logic-change` (2026-09-11).
- **`UINT8` image inputs get raw pixels; `INT8` / `BOOL` are rejected** (maintainer,
  2026-09-11). INT8 would need quantization parameters the metadata does not carry.
- **`Float32` means "unchanged", not "force float".** The TensorFlow classifier
  already emits `UINT8` under the default `Float32` tag; honoring `Float32` as a
  command would silently change it. Only an explicit non-default type changes
  output.
- **Rejection lives where the type is visible.** neuriplo-tasks `PixelType` cannot
  represent `INT8`/`BOOL`/`INT64`/`FP16`, so neuriplo-infer rejects those; the
  neuriplo-tasks helper rejects `Int32`.
- **Setup-time over request-time** for image inputs, so `--export_metadata` and a
  misconfigured model fail before any frame; the byte-count guard covers the rest.

## Constraints and Context

- `tech-stack.md`: dtype semantics belong to neuriplo-tasks; this repo wires them.
- neuriplo v0.9.1 `LayerInfo::datatype` = `{Float32, Int32, Int64, UInt8, Int8, Bool}`;
  ONNX Runtime and TensorRT populate it; KServe did not.
- The pinned neuriplo-tasks moves from v0.8.0 past v0.8.1, which also changed
  RT-DETR / D-FINE / DEIM normalization — call it out in `CHANGELOG.md`.
- Contract surfaces: app-local `InferenceMetadata`, error messages, `versions.env`.

## Open Questions

- None blocking.

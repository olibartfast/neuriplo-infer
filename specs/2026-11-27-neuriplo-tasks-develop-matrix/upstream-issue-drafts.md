# Upstream issue drafts — neuriplo-tasks@develop compatibility

Status: DRAFT, not filed. Target repo: `olibartfast/neuriplo-tasks`.
File with `gh issue create -R olibartfast/neuriplo-tasks` after maintainer
approval. Producer SHA for all below:
`fd94cabca822201341f6bd3d710ac265eb55957c`.

Context (same for all): downstream `neuriplo-infer` mirrors
`TaskFactory` matching rules in
`app/src/NeuriploInferTaskRouting.cpp` (AGENTS.md routing rule). Audit at
the above SHA found five families with no downstream target; their gates
are halted per `specs/2026-11-27-neuriplo-tasks-develop-matrix/plan.md`
Group 2. This is downstream lag on a moving target, not necessarily an
upstream defect — each draft asks for the contract confirmation needed to
mirror it.

---

## 1. New video-classification families unroutable downstream

Title: `develop: pevideo / vjepa / videoprism have no downstream routing target`

Body:
`TaskFactory` at `fd94cab` routes `pevideo*`, `vjepa*`, `videoprism*` to
VideoClassification (`src/core/task_factory.cpp`, VideoClassification
entry). Downstream `getTaskTypeForModel` only knows
`timesformer`/`videomae`/`vivit`, so these fall through to the Detection
default — a silent misroute, not a clean failure.

Ask: confirm the prefix rules and input contracts (frames, size, export
source) are stable enough to mirror downstream, or flag them experimental
in the README so downstream holds them out.

Impact: video-classification gates for these three families halted
downstream until confirmed.

---

## 2. FlexGlue two-model contract needs downstream guidance

Title: `develop: flexglue descriptor+matcher contract vs image-in task path`

Body: `flexglue*` routes to KeypointMatching, but unlike the end-to-end
families it is two models (`flexglue_descriptor.onnx` +
`flexglue_matcher.onnx`) needing externally detected keypoints
(`[1, N, 2]` alongside the matcher). Downstream `processKeypointMatching`
takes exactly two still images and returns one pair result — there is no
slot for external keypoints or a second weights file.

Ask: is the intended downstream integration (a) detector-choice left to
the app with a two-weights invocation, or (b) a fixed reference detector
we should wire? Point to the canonical detector + weights if (b).

Impact: flexglue gate halted downstream.

---

## 3. Depth Anything 3 contract vs downstream depth path

Title: `develop: depth_anything_3 / da3* routing and relative-depth contract`

Body: `TaskFactory` adds `depthanything3`/`da3*` (relative depth map
`[B,H,W]`, optional `conf` ignored) and generalizes `yolo*depth*`.
Downstream only routes `depthanythingv2` and YOLO-depth metric maps
(`[B,1,H,W]` unbounded). `da3*` strings fall to Detection.

Ask: confirm the v3 input normalization (504x504, /14 multiples,
ImageNet mean/std) and the relative-vs-metric distinction as the
downstream branching rule (v2 path must not accept v3 weights).

Impact: DA3 gate halted downstream.

---

## 4. New task types have no downstream consumer

Title: `develop: GazeEstimation / Monocular3DDetection need app-facing task specs`

Body: `TaskFactory` adds `TaskFamily::GazeEstimation`
(`gaze`/`ethxgaze`/`gazetr`) and `TaskFamily::Monocular3DDetection`
(`urbanomni*`, `monocular3d`, `mono3d`) with new `TaskType` values and
result structs (`GazeEstimation`, `Monocular3DDetection`). Downstream has
no such task types, no preprocessing/postprocessing, no renderer, and no
`--type` routing — support is structural work, not a routing tweak.

Ask: publish the app-facing contract per family (inputs, outputs,
confidence semantics, render suggestion), or state explicitly that app
integration is not expected yet so downstream records them as known
unsupported rather than silently misrouting to Detection.

Impact: both families recorded as structurally unsupported downstream.

---

## 5. `yolo*depth*` generalization vs downstream's narrower rule

Title: `develop: confirm yolo-depth matching rule for downstream mirror`

Body: downstream deliberately routes only YOLO-prefixed types containing
`depth` (comment in `NeuriploInferTaskRouting.cpp`: a broader match
routed names `TaskFactory` would not build). Develop generalizes the
YOLO-depth rule — please confirm the exact predicate (prefix list,
`depth` vs `depthanything` overlap with the DA3 rule at 135) so the
mirror stays exact, especially the `da3*` vs `yolo*depth*` precedence
(our seg/pose branches run first; confirm factory order).

Impact: none yet (current rule still matches); blocks exact mirroring.

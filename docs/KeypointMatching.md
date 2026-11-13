# Keypoint matching with SuperPoint + LightGlue

Build the application feature with a neuriplo-tasks checkout containing the
LightGlue pipeline decoder and empty-match fix
(`develop` after integration of commit `f867d5a`).
TRT also needs the neuriplo allocator correction in commit `8c95d5a`
(or its successor on `develop`).
The existing release pins are not yet an application validation claim for this task.

```bash
cmake -S . -B build-lightglue-ort -DDEFAULT_BACKEND=ONNX_RUNTIME \
  -DNEURIPLO_INFER_ENABLE_KSERVE=OFF \
  -DFETCHCONTENT_SOURCE_DIR_NEURIPLO-TASKS=/path/to/neuriplo-tasks
cmake --build build-lightglue-ort -j4

build-lightglue-ort/app/neuriplo-infer \
  --type=lightglue --weights=/path/to/superpoint_lightglue_pipeline.onnx \
  --source=data/keypoint-matching-phototourism-capitol-98169888-3347710852.jpg,data/keypoint-matching-phototourism-capitol-26757027-6717084061.jpg \
  --batch=2 --input_sizes=1,512,512 --min_confidence=0.2 --use-gpu=false
```

The source flag takes exactly two still-image paths, separated by a comma.
For Sacré-Cœur use
`data/keypoint-matching-lightglue-sacre-coeur1.jpg,data/keypoint-matching-lightglue-sacre-coeur2.jpg`.

The application preprocesses both images in one task call, runs the selected
backend, decodes the tensors, and writes a side-by-side PNG to
`data/output/processed_lightglue_local.png`. Each image keeps its original
coordinates and size; the shorter image has black padding below it. Copy the
output before running the next pair. Logs report the match count at the
strict `--min_confidence` threshold.

On Jetson, build with `-DDEFAULT_BACKEND=TENSORRT` and supply the engine built
for that device and installed TensorRT version via `--weights`, keeping the
same source, shape and threshold arguments. Use `--use-gpu=true`.
Both the packed input and variable-size output tensors are handled by the
shared task/backend libraries. Warmup and benchmark flags currently apply to
single-image workflows and are not supported for this pair workflow.

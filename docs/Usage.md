# Usage

The full command-line reference for `neuriplo-infer`. For a first run, see the
[README quickstart](../README.md#quickstart-docker); for backends, images, and
build options, see [Deployment.md](Deployment.md).

`neuriplo-infer --help` prints every flag with its default for the binary you
built; it is the authority when this page and a build disagree.

## Synopsis

```bash
# Local inference
neuriplo-infer --type=<model_type> --weights=<model_file> --source=<input> [options]

# Remote inference against a KServe V2 endpoint (no --weights)
neuriplo-infer --type=<model_type> --kserve_endpoint=<url> --source=<input> [options]

# Inspect a model without running it
neuriplo-infer --type=<model_type> --weights=<model_file> --export_metadata

# Machine-readable description of this build
neuriplo-infer --capabilities
```

## Outputs

- **Image source (detection, segmentation, pose, classification, depth):** the
  annotated image is written to `data/output/processed_<type>_<mode>.png` (for
  example `processed_yolo_local.png`), relative to the working directory.
- **Optical flow:** the flow visualization is written next to the source image at
  `<source-dir>/output/processed_frame_optical_flow.jpg`, not under `data/output/`.
- **Image understanding (`--type=gemma4` and other vision-language models):** the
  model's response is written to standard output; no annotated image is produced.
  The run report is still written.
- **Video or stream source:** frames are shown in a preview window. Pass
  `--no_display` wherever there is no screen — a container, SSH, CI — because
  the window aborts the process without one. `--output_video` writes the
  annotated video to a file in builds that include the writer.
- **Every run:** a [run report](#run-diagnostics) at
  `data/output/run_report.json`.

## Parameters

### Model and input

| Flag | Default | Description |
|------|---------|-------------|
| `--type=<model_type>` | `yolov10` | Model type string; routes to the task. Set it to match your model — see [supported-model-types.md](generated/supported-model-types.md). |
| `--weights`, `-w=<path>` | — | Local model file. Required for local inference and `--export_metadata`; not used with `--kserve_endpoint`. |
| `--source`, `-s=<input>` | — | Image, video file, or stream URL (e.g. `rtsp://...`). Optical flow takes two comma-separated images. Omit for text-only image understanding and for `--export_metadata`. |
| `--labels`, `--lb=<path>` | — | Class labels, one per line, for fixed-label models (`labels/` ships COCO and VOC lists). |
| `--use-gpu` | `false` | Run inference on the GPU when the backend supports it. |
| `--input_sizes`, `--is='<CHW;...>'` | — | Input sizes for models with dynamic axes, or for backends that cannot report input shapes (OpenCV DNN). E.g. `'3,224,224'`, or `'3,640,640;2'` for RT-DETR / D-FINE / DEIM. |
| `--batch`, `-b=<n>` | `1` | Batch size. Values above 1 are not currently supported. |

### Detection and segmentation

| Flag | Default | Description |
|------|---------|-------------|
| `--min_confidence=<v>` | `0.25` | Minimum detection confidence. |
| `--nms_threshold=<v>` | `0.45` | IoU threshold for NMS in YOLO detectors and segmenters. |
| `--mask_threshold=<v>` | `0.50` | Mask binarization threshold for instance segmentation. |
| `--segmentation_output`, `--so=<mask\|polygon>` | `mask` | Instance-segmentation representation. `polygon` returns polygon exteriors and holes instead of a dense mask and renders them as outlines, never filled. The polygons are convex hulls of each instance, not tight contours. |

### Task-specific

| Flag | Default | Used by |
|------|---------|---------|
| `--text_prompts`, `--tp='<a;b;...>'` | — | Open-vocabulary detection: semicolon-separated prompts. Required for OWLv2. |
| `--tokenizer_vocab=<vocab.json>`, `--tokenizer_merges=<merges.txt>` | — | OWLv2 / OWL-ViT tokenizer assets. |
| `--bert_tokenizer_vocab=<vocab.txt>` | — | Grounding DINO BERT vocabulary. |
| `--prompt='<text>'` | — | Image understanding / VLM: freeform prompt. |
| `--mmproj=<path>` | — | Image understanding / VLM: multimodal projector GGUF (llama.cpp). |
| `--output_format=<text\|json>` | — | Multimodal output hint; `json` for parseable responses. |
| `--sample_stride=<n>`, `--max_frames=<n>` | `0` | Frame-sampling stride and cap for multimodal video tasks. |
| `--num_frames`, `--nf=<n>` | `0` | Frames per clip for video classification; `0` uses the model default (16 for VideoMAE). |

### Output and measurement

| Flag | Default | Description |
|------|---------|-------------|
| `--no_display` | `false` | Do not open the preview window. Needed for video without a screen. |
| `--output_video=<path>` | — | Write the annotated video (fixed 30 fps, codec auto-selected, container from the extension). Only in builds configured with `-DNEURIPLO_INFER_WITH_VIDEOWRITER=ON`; image sources are rejected. |
| `--timings_csv=<path>` | — | Write one row per inference (`frame,latency_us`) for a video run. Parent directories are created; the file is opened before the first frame. |
| `--warmup` | `false` | GPU warmup before inference; image sources only. |
| `--benchmark`, `--iterations=<n>` | `false`, `10` | Repeat inference and report the average time; image sources only. |
| `--export_metadata` | `false` | Print model type, routed task, and input/output layers, then exit. Requires `--weights`, not `--source`. |
| `--no_gif` | `false` | Reserved; current paths emit no GIFs. |

### Remote inference (KServe)

`--kserve_endpoint`, `--kserve_model_name`, `--kserve_model_version`,
`--kserve_transport`, `--kserve_timeout_ms`, `--input_mode`, `--task_model`,
`--task_model_version`, and `--postprocess_mode` are documented with their
defaults and examples in
[KserveRuntime.md § CLI flags](KserveRuntime.md#cli-flags).

## Examples

```bash
# Object detection on an image
./neuriplo-infer --type=yolo --weights=models/yolo11n.onnx \
  --source=data/dog.jpg --labels=labels/coco.names

# RT-DETR on a video, headless, with per-frame timings
./neuriplo-infer --type=rtdetr --weights=models/rtdetr-l.onnx \
  --source=video.mp4 --labels=labels/coco.names --input_sizes='3,640,640;2' \
  --min_confidence=0.4 --no_display --timings_csv=data/output/timings.csv

# Classification
./neuriplo-infer --type=torchvision-classifier --weights=models/resnet50.onnx \
  --source=data/dog.jpg --labels=imagenet_labels.txt

# Instance segmentation on the GPU
./neuriplo-infer --type=yoloseg --weights=models/yolov8n-seg.onnx \
  --source=data/dog.jpg --labels=labels/coco.names \
  --min_confidence=0.4 --nms_threshold=0.5 --mask_threshold=0.5 --use-gpu

# Optical flow between two frames
./neuriplo-infer --type=raft --weights=models/raft_large.onnx \
  --source=data/frame_001.png,data/frame_002.png --input_sizes='3,520,960;3,520,960'

# Open-vocabulary detection
./neuriplo-infer --type=owlv2 --weights=models/owlv2.onnx --source=data/dog.jpg \
  --text_prompts='cat;dog;bus' --tokenizer_vocab=models/owlv2/vocab.json \
  --tokenizer_merges=models/owlv2/merges.txt --min_confidence=0.2

# Inspect a model without a source
./neuriplo-infer --type=yolo --weights=models/yolo11n.onnx --export_metadata
```

Remote-inference examples are in [KserveRuntime.md](KserveRuntime.md#cli-flags);
more invocations are in [`.vscode/launch.json`](../.vscode/launch.json), and
runnable export-plus-inference flows in the
[end-to-end presets](Deployment.md#end-to-end-presets).

## Capabilities

Use `--capabilities` without run arguments to print the versioned, build-specific
JSON contract consumed by tools such as Neuriplo UI:

```bash
./neuriplo-infer --capabilities
```

The response reports task/model selectors, source and parameter requirements,
the local backend compiled into the executable, and the client-server protocols
and transports available in the current build. Local and client-server
execution are separate workflows. Enum defaults are build-specific when an
optional transport is compiled. The schema is documented in
[`capabilities.schema.json`](capabilities.schema.json), currently at
`schema_version` 2. Version 2 added the required `diagnostics` section; because
the schema forbids unknown properties, that is a breaking change in both
directions, so version 1 stays published as
[`capabilities.schema.v1.json`](capabilities.schema.v1.json) for
documents produced by an older binary.

## Run diagnostics

Every run writes a versioned JSON report next to its output, at the path the
capabilities document advertises under `diagnostics.run_report` (currently
`data/output/run_report.json`, relative to the working directory):

```json
{
  "schema_version": 1,
  "status": "failed",
  "stage": "model_load",
  "metrics": {
    "wall_time_ms": 546.9,
    "samples": 0,
    "frames": null,
    "throughput_per_second": null,
    "stages_ms": {
      "model_load": 546.7,
      "preprocess": null,
      "inference": null,
      "postprocess": null,
      "render": null
    }
  },
  "error": { "stage": "model_load", "message": "..." }
}
```

It exists so a caller can tell *where* a run failed and *how long each stage
took* without parsing log text. Five rules make it safe to consume:

- **Absent is not zero.** A stage nobody measured is `null`, never `0`, and
  `throughput_per_second` appears only when both a processed count and the
  inference time it belongs to were measured. It is also `null` on a failed
  run: counts are added after work succeeds while the stage timer still records
  the attempt that threw, so a rate computed from both would describe work that
  did not happen. The counts and the stage sums stay; only the ratio is
  withheld.
- **`stages_ms` values are sums** over the whole run, in milliseconds;
  `wall_time_ms` is measured inside `main`, so it is always smaller than the
  caller's own process wall time. Warmup and benchmark iterations are excluded:
  they repeat inference without producing a sample, and counting them would
  inflate every stage total and collapse `throughput_per_second`.
- **`samples` counts completed sources** — one still image, one video read to
  its end, one optical-flow pair, one image-understanding request — while
  `frames` counts video frames and stays `null` for a run with no video. A
  video the operator stopped with `q` or Escape is not a completed source and
  is not counted; its frames still are, because they were processed.
- **A source that cannot be read is a failure, not a skip.** An optical-flow
  pair with an unreadable half ends the run at `source` instead of returning
  success with no samples and no artifact.
- **The stage is recorded, not inferred.** A failure is attributed to the stage
  the run had reached — `configuration`, `model_load`, `source`, `preprocess`,
  `inference`, `postprocess`, `render`, or `unknown` — and the message stays
  the producer's own. A backend that throws a non-`std::exception` (some ONNX
  graphs abort inside OpenCV DNN) still leaves an attributed report before the
  process dies.

Writing the report never changes the outcome it describes: a report that cannot
be written is logged and dropped.

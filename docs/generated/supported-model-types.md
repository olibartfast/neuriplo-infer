# Supported Model Types

Auto-generated from `neuriplo-tasks` TaskFactory documentation.
Do not edit manually; run `python scripts/sync_supported_model_types.py`.

Source: [https://github.com/olibartfast/neuriplo-tasks](https://github.com/olibartfast/neuriplo-tasks)

`TaskFactory` routes model type strings through a **built-in, compile-time**
registration table in `src/core/task_factory.cpp`. New built-in tasks require
editing that table and the README list below. **Third-party or runtime task
plugins are not supported**; if plugin extension becomes a product goal, add a
separate explicit extension registry rather than growing the internal table
indefinitely.

<!-- TASKFACTORY_MODEL_LIST:START -->
The TaskFactory supports the following model type strings. Matching normalizes strings by lowercasing and stripping whitespace, hyphens, and underscores, so `YOLO-V8`, `yolo_v8`, and ` yolo v8 ` route identically. Specific segmentation, pose, and depth aliases are checked before generic detection aliases.

**Object Detection:**

- `"yolo"`, `"yolov7e2e"`, `"yolov10"`, `"yolo26"`, `"yolov4"` - YOLO-based variants
- `"yolonas"` - YOLO-NAS
- `"yoloworld"` - Ultralytics YOLO-World exported with a fixed class list (same output as YOLOv8; see the [YOLO-World export guide](https://github.com/olibartfast/neuriplo-tasks/blob/master/export/detection/yolo_world/README.md))
- `"rtdetr"`, `"rtdetrv2"` - RT-DETR family (RT-DETR v1, v2, and v4; excludes v3)
- `"dfine"`, `"deim"`, `"deimv2"` - D-FINE and DEIM; same export signature and postprocessing as RT-DETR, matched by prefix so later revisions resolve too
- `"rtdetrul"`, `"rtdetrultralytics"` - RT-DETR (Ultralytics implementation)
- `"rfdetr"` - RF-DETR
- `"ecdet"` - EdgeCrafter detection (any string starting with `ecdet`)
- `"edgecrafter"`, `"edgecrafter-det"` - EdgeCrafter detection unless the normalized string contains `seg` or `pose`

For EdgeCrafter detection export details, see [export/detection/edgecrafter/README.md](https://github.com/olibartfast/neuriplo-tasks/blob/master/export/detection/edgecrafter/README.md).

**Instance Segmentation:**
- `"ecseg"` - EdgeCrafter segmentation (any string starting with `ecseg` or `edgecrafter` and containing `seg`)
- `"yoloseg"`, `"yolo-seg"`, `"yolov8-seg"` - YOLOv5/YOLOv8/YOLO11-style segmentation
- `"yolov10seg"`- YOLOv10
- `"yolo26seg"` - YOLO26
- `"rfdetrseg"` - RF-DETR

For EdgeCrafter segmentation export details, see [export/segmentation/edgecrafter/README.md](https://github.com/olibartfast/neuriplo-tasks/blob/master/export/segmentation/edgecrafter/README.md).

Instance-segmentation tasks return masks by default. Set `TaskConfig::segmentation_output` to `SegmentationOutput::Polygon` to return full-image exterior rings and holes instead; see [instance-segmentation output representations](https://github.com/olibartfast/neuriplo-tasks/blob/master/docs/segmentation_outputs.md).

**Classification:**
- `"torchvision-classifier"` - Torchvision models (ResNet, EfficientNet, etc.)
- `"tensorflow-classifier"` - TensorFlow/Keras models
- `"vit-classifier"` - Vision Transformers

Any model type starting with `resnet` (e.g. `resnet50`) or containing `tensorflow` also routes to classification.

**Video Classification:**
- `"videomae"` - VideoMAE
- `"vivit"` - ViViT
- `"timesformer"` - TimeSformer
- `"pevideo"` - PE Video (any string starting with `pevideo`; classification needs a fixed-vocabulary export, see the export guide)
- `"vjepa"` - V-JEPA 2 (any string starting with `vjepa`, e.g. `vjepa2-vitl-fpc16-256-ssv2`)
- `"videoprism"` - VideoPrism (any string starting with `videoprism`)

**Optical Flow:**
- `"raft"` - RAFT optical flow

**Keypoint Matching:**
- `"superglue"` - SuperGlue-style end-to-end matching pipelines (any string starting with `superglue`)
- `"lightglue"` - LightGlue matching pipelines, including the LightGlue-ONNX exports (any string starting with `lightglue`)
- `"matchanything"` - MatchAnything detector-free matchers (EfficientLoFTR / RoMa backbones; any string starting with `matchanything`)

One image pair per call — `getRequiredFrames()` returns 2. Results are `KeypointMatching` entries containing source/target keypoints and matched index pairs.

`"superglue"` covers end-to-end `SuperGlueForKeypointMatching` exports (transformers/Optimum, e.g. `magic-leap-community/superglue_outdoor`). Those take the pair as **one** rank-5 input `pixel_values [1, 2, 3, 480, 640]` — grayscale replicated across three channels in `[0, 1]` — so `preprocess()` returns **one** packed buffer, source image first, and every buffer maps 1:1 onto a model input. Keypoints come back normalized (`x / width`, `y / height`) and are rescaled to each original image; matches at or below `TaskConfig::confidence_threshold` (default 0.25) are dropped, as are masked-out padding keypoints. The export carries no SuperPoint detector score, so `Keypoint::confidence` is 0 — use `KeypointMatch::score` for match quality.

`"lightglue"` covers end-to-end LightGlue pipelines (extractor and matcher in one graph, e.g. `thomasonzhou/superpoint-lightglue`). Their single input is the pair interleaved on the batch axis, `images [2, C, H, W]`, so `preprocess()` again returns one packed buffer, source first, with the channel count taken from the model input (1 for the SuperPoint pipeline, 3 for ALIKED/DISK variants). Keypoints come back as input-space pixels (int64) and are rescaled to each original image; `matches [M, 3]` rows are `[pair_index, source_index, target_index]`, and rows for other pairs plus scores at or below `TaskConfig::confidence_threshold` are dropped. The decoder needs the model input size (from `ModelInfo`) to rescale correctly.

**MatchAnything decoding is not implemented yet**: its postprocessor throws `std::logic_error` until its output layout has been verified against a real ONNX export. Matcher-only exports (keypoints and descriptors in — the original SuperGlue matcher) do not fit the image-in task contract and are not supported.

For export and verification details, see [export/keypoint_matching/README.md](https://github.com/olibartfast/neuriplo-tasks/blob/master/export/keypoint_matching/README.md).

**Pose Estimation (2D Human Pose):**
- `"yolov8pose"`, `"yolov8-pose"` - YOLOv8 pose (single-stage, returns bbox + keypoints)
- `"yolo11pose"`, `"yolo11-pose"` - YOLO11 pose
- `"yolo26pose"`, `"yolo26-pose"` - YOLO26 pose
- `"yolov5pose"`, `"yolov5-pose"` - YOLOv5 pose
- `"rfdetrpose"`, `"rfdetr-pose"`, `"rfdetrkeypoint"`, `"rfdetr-keypoint"`, `"rfdetrkpt"`, `"rfdetr-kpt"` - RF-DETR keypoint pose (single-stage, returns bbox + 17 coco keypoints with visibility and per-keypoint covariance)
- `"vitpose"` - ViTPose (top-down, heatmap-based)
- `"ecpose"` - EdgeCrafter pose estimation (any string starting with `ecpose`, or `edgecrafter` and containing `pose`)

For EdgeCrafter pose-estimation export details, see [export/pose_estimation/edgecrafter/README.md](https://github.com/olibartfast/neuriplo-tasks/blob/master/export/pose_estimation/edgecrafter/README.md).

RF-DETR keypoint models output per-keypoint visibility and 2×2 pixel covariance (decoded from Cholesky L via the ONNX `log_l11`, `l21`, `log_l22` channels). Keypoints are filtered by an uncertainty-weighted score fusion that discounts high-covariance predictions.

**Depth Estimation:**
- `"depth_anything_v2"`, `"depth-anything-v2"` - Depth Anything V2
- `"yolo-depth"`, `"yolo26n-depth"` - Ultralytics YOLO26 depth models; any YOLO-prefixed model type containing `depth` routes here

YOLO26 depth expects RGB NCHW float input normalized to `[0,1]` with YOLO letterboxing. Its exported output is an unbounded metric-depth map shaped `[B,1,H,W]`; postprocessing removes letterbox padding and resizes the map to the original frame. See the [YOLO depth export guide](https://github.com/olibartfast/neuriplo-tasks/blob/master/export/depth_estimation/yolo_depth/README.md).

**Open-Vocabulary Detection:**
- `"owlv2"` - OWLv2 open-vocabulary detection
- `"owlvit"` - OWL-ViT compatible open-vocabulary detection
- `"groundingdino"` - Grounding DINO text-conditioned detection
- `"yoloworld"` - YOLO-World with the class list baked in at export time; no runtime prompts, so it routes to the YOLO detection task and returns `Detection` results (see the [YOLO-World export guide](https://github.com/olibartfast/neuriplo-tasks/blob/master/export/detection/yolo_world/README.md))

OWLv2, OWL-ViT, and Grounding DINO use text prompts supplied at runtime through `TaskConfig::text_prompts`. Tokenizer assets can be passed either as file paths (`tokenizer_vocab_path`, `tokenizer_merges_path`) or preloaded text blobs (`tokenizer_vocab_json`, `tokenizer_merges_text`).

Their expected ONNX contract is:
- Inputs: `pixel_values`, `input_ids`, `attention_mask`
- Outputs: `logits`, `pred_boxes`, and optional `objectness_logits`

Results are returned as `OpenVocabDetection` entries containing `bbox`, `score`, `prompt_index`, and resolved `label`.

YOLO-World exports instead use the YOLOv8 contract — `images` in, `output0` `[1, 4 + num_classes, 8400]` out — with class IDs in the exported class-list order; changing the vocabulary means re-exporting.

For export details, see [export/open_vocab_detection/OWLv2.md](https://github.com/olibartfast/neuriplo-tasks/blob/master/export/open_vocab_detection/OWLv2.md).

**Image Understanding (VLM):**
- `"gemma4"`, `"gemma"`, `"llama"`, `"llamacpp"`, `"imageunderstanding"` - Vision-language model image captioning / Q&A via llama.cpp backend

Input contract: `preprocess()` returns two tensors — `[0]` UTF-8 prompt bytes, `[1]` raw RGB pixels with an 8-byte header `[uint32 width LE][uint32 height LE][H×W×3 bytes]`. When no image is provided only tensor `[0]` is returned (text-only mode). Output is a UTF-8 string returned as float-encoded bytes (one `float` per byte value).

Requires the llama.cpp `LLAMACPP` backend with an mmproj (vision projector) GGUF.

For model download and setup details, see [export/image_understanding/ImageUnderstanding.md](https://github.com/olibartfast/neuriplo-tasks/blob/master/export/image_understanding/ImageUnderstanding.md).

**Gaussian Splatting:**
- `"lgm"`, `"lgm-mini"` - LGM (Large Gaussian Model)
- `"grm"` - GRM
- `"gaussiansplatting"`, any string containing `"splat"` - generic alias

<!-- TASKFACTORY_MODEL_LIST:END -->

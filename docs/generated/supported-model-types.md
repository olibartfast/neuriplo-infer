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
The TaskFactory supports the following model type strings. Matching normalizes strings by lowercasing and stripping whitespace, hyphens, and underscores, so `YOLO-V8`, `yolo_v8`, and ` yolo v8 ` route identically. Specific segmentation and pose aliases are checked before generic detection aliases.

**Object Detection:**

- `"yolo"`, `"yolov7e2e"`, `"yolov10"`, `"yolo26"`, `"yolov4"` - YOLO-based variants
- `"yolonas"` - YOLO-NAS
- `"rtdetr"` - RT-DETR family (RT-DETR v1, v2, and v4; excludes v3; includes D-FINE and DEIM v1/v2)
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

**Classification:**
- `"torchvision-classifier"` - Torchvision models (ResNet, EfficientNet, etc.)
- `"tensorflow-classifier"` - TensorFlow/Keras models
- `"vit-classifier"` - Vision Transformers

Any model type starting with `resnet` (e.g. `resnet50`) or containing `tensorflow` also routes to classification.

**Video Classification:**
- `"videomae"` - VideoMAE
- `"vivit"` - ViViT
- `"timesformer"` - TimeSformer

**Optical Flow:**
- `"raft"` - RAFT optical flow

**Pose Estimation:**
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

**Open-Vocabulary Detection:**
- `"owlv2"` - OWLv2 open-vocabulary detection
- `"owlvit"` - OWL-ViT compatible open-vocabulary detection
- `"groundingdino"` - Grounding DINO text-conditioned detection

Open-vocabulary models use text prompts supplied at runtime through `TaskConfig::text_prompts`. Tokenizer assets can be passed either as file paths (`tokenizer_vocab_path`, `tokenizer_merges_path`) or preloaded text blobs (`tokenizer_vocab_json`, `tokenizer_merges_text`).

The expected ONNX contract is:
- Inputs: `pixel_values`, `input_ids`, `attention_mask`
- Outputs: `logits`, `pred_boxes`, and optional `objectness_logits`

Results are returned as `OpenVocabDetection` entries containing `bbox`, `score`, `prompt_index`, and resolved `label`.

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

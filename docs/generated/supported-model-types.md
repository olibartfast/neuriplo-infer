# Supported Model Types

Auto-generated from `vision-core` TaskFactory documentation.
Do not edit manually; run `python scripts/sync_supported_model_types.py`.

Source: [https://github.com/olibartfast/vision-core](https://github.com/olibartfast/vision-core)

<!-- TASKFACTORY_MODEL_LIST:START -->
The TaskFactory supports the following model type strings:

**Object Detection:**

- `"yolo"`, `"yolov7e2e"`, `"yolov10"`, `"yolo26"`, `"yolov4"` - YOLO-based variants
- `"yolonas"` - YOLO-NAS
- `"rtdetr"` - RT-DETR family (RT-DETR v1, v2, and v4; excludes v3; includes D-FINE and DEIM v1/v2)
- `"rtdetrul"`, `"rtdetrultralytics"` - RT-DETR (Ultralytics implementation)
- `"rfdetr"` - RF-DETR
- `"ecdet"` - EdgeCrafter detection (any string starting with `ecdet` or `edgecrafter`)
- `"edgecrafter"` - EdgeCrafter generic (defaults to detection)

**Instance Segmentation:**
- `"ecseg"` - EdgeCrafter segmentation (any string starting with `ecseg` or `edgecrafter` and containing `seg`)
- `"yoloseg"` - YOLOv5/YOLOv8/YOLO11
- `"yolov10seg"`- YOLOv10
- `"yolo26seg"` - YOLO26
- `"rfdetrseg"` - RF-DETR

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
- `"vitpose"` - ViTPose (top-down, heatmap-based)
- `"ecpose"` - EdgeCrafter pose estimation (any string starting with `ecpose` or `edgecrafter` and containing `pose`)

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

For export details, see [export/open_vocab_detection/OWLv2.md](https://github.com/olibartfast/vision-core/blob/master/export/open_vocab_detection/OWLv2.md).

**Image Understanding (VLM):**
- `"gemma4"`, `"gemma"`, `"llama"`, `"llamacpp"`, `"imageunderstanding"` - Vision-language model image captioning / Q&A via llama.cpp backend

Input contract: `preprocess()` returns two tensors — `[0]` UTF-8 prompt bytes, `[1]` raw RGB pixels with an 8-byte header `[uint32 width LE][uint32 height LE][H×W×3 bytes]`. When no image is provided only tensor `[0]` is returned (text-only mode). Output is a UTF-8 string returned as float-encoded bytes (one `float` per byte value).

Requires the llama.cpp `LLAMACPP` backend with an mmproj (vision projector) GGUF.

For model download and setup details, see [export/image_understanding/ImageUnderstanding.md](https://github.com/olibartfast/vision-core/blob/master/export/image_understanding/ImageUnderstanding.md).

**Gaussian Splatting:**
- `"lgm"`, `"lgm-mini"` - LGM (Large Gaussian Model)
- `"grm"` - GRM
- `"gaussiansplatting"`, any string containing `"splat"` - generic alias

**EdgeCrafter:**
- `"ecdet"` / `"ecseg"` / `"ecpose"` — detection, segmentation, and pose estimation via the [EdgeCrafter](https://github.com/Intellindust-AI-Lab/EdgeCrafter) model family. All variants share a common ONNX contract:

| Role   | Name                | Dtype  | Shape              | Description                       |
|--------|---------------------|--------|--------------------|-----------------------------------|
| Input  | `images`            | float  | `[1, 3, H, W]`     | NCHW preprocessed image           |
| Input  | `orig_target_sizes` | int64  | `[1, 2]`           | Original `[width, height]`        |
| Output | `labels`            | int64  | `[1, N]`           | Class IDs (0-indexed COCO)        |
| Output | `boxes`             | float  | `[1, N, 4]`        | `[x1,y1,x2,y2]` in orig coords   |
| Output | `scores`            | float  | `[1, N]`           | Confidence scores                 |

Detection models output `labels` + `boxes` + `scores`. Segmentation adds a `masks` `[1, N, MH, MW]` float tensor. Pose estimation replaces `boxes` with `keypoints` `[1, N, 17, 2|3]` and applies a label offset of –1 (person `1` → `0`); bounding boxes are derived from visible keypoints.

Preprocessing: direct resize to `[H, W]` (no letterbox) → BGR to RGB → scale to `[0,1]` → ImageNet normalization (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`). The ONNX graph performs top-k selection and coordinate scaling internally.

Export instructions: see [export/detection/edgecrafter/README.md](https://github.com/olibartfast/vision-core/blob/master/export/detection/edgecrafter/README.md), [export/segmentation/edgecrafter/README.md](https://github.com/olibartfast/vision-core/blob/master/export/segmentation/edgecrafter/README.md), [export/pose_estimation/edgecrafter/README.md](https://github.com/olibartfast/vision-core/blob/master/export/pose_estimation/edgecrafter/README.md).
<!-- TASKFACTORY_MODEL_LIST:END -->

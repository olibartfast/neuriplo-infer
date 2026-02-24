# Supported Model Types

Auto-generated from `vision-core` TaskFactory documentation.
Do not edit manually; run `python scripts/sync_supported_model_types.py`.

Source: `vision-inference/build/_deps/vision-core-src/README.md`

The TaskFactory supports the following model type strings:

**Object Detection:**

- `"yolo"`, `"yolov7e2e"`, `"yolov10"`, `"yolo26"`, `"yolov4"` - YOLO-based variants
- `"yolonas"` - YOLO-NAS
- `"rtdetr"` - RT-DETR family (RT-DETR v1, v2, and v4; excludes v3; includes D-FINE and DEIM v1/v2)
- `"rtdetrul"` - RT-DETR (Ultralytics implementation)
- `"rfdetr"` - RF-DETR

**Instance Segmentation:**
- `"yoloseg"` - YOLOv5/YOLOv8/YOLO11
- `"yolov10seg"`- YOLOv10
- `"yolo26seg"` - YOLO26
- `"rfdetrseg"` - RF-DETR

**Classification:**
- `"torchvision-classifier"` - Torchvision models (ResNet, EfficientNet, etc.)
- `"tensorflow-classifier"` - TensorFlow/Keras models
- `"vit-classifier"` - Vision Transformers

**Video Classification:**
- `"videomae"` - VideoMAE
- `"vivit"` - ViViT
- `"timesformer"` - TimeSformer

**Optical Flow:**
- `"raft"` - RAFT optical flow

**Pose Estimation:**
- `"vitpose"` - ViTPose

**Depth Estimation:**
- `"depth_anything_v2"`, `"depth-anything-v2"` - Depth Anything V2

# YOLOv5 Export Instructions

## OnnxRuntime
Run from the [YOLOv5 repository](https://github.com/ultralytics/yolov5/issues/251) export script:

```bash
python export.py --weights <yolov5_version>.pt --include onnx
```

## OpenVINO
From the YOLOv5 repository:

```bash
python export.py --weights <yolov5_version>.pt --include openvino
```

## Libtorch
From the YOLOv5 repository:

```bash
python export.py --weights <yolov5_version>.pt --include torchscript
```
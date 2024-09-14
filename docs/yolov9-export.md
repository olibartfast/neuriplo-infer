# YOLOv9 Export Instructions

## OnnxRuntime
Run from the [YOLOv9 repository](https://github.com/WongKinYiu/yolov9):

```bash
python export.py --weights yolov9-c/e-converted.pt --include onnx
```

## Torchscript
```bash
python export.py --weights yolov9-c/e-converted.pt --include torchscript
```

## TensorRT
```bash
trtexec --onnx=yolov9-c/e-converted.onnx --saveEngine=yolov9-c/e.engine --fp16
```
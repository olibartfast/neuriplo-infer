# YOLO11 Export Instructions

* The process is the same for YOLOv8 and other models managed by Ultralytics in their library.

Install YOLO11 [following Ultralytics official documentation](https://docs.ultralytics.com/quickstart/) and export the model in different formats using the following commands:

## Torchscript
```bash
yolo export model=best.pt format=torchscript
```
Note: `best.pt` corresponds to your trained YOLOv8n/s/m/x model.

## OnnxRuntime
```bash
yolo export model=best.pt format=onnx
```

## OpenVINO
```bash
yolo export model=best.pt format=openvino
```

## TensorFlow
```bash
yolo export model=best.pt format=saved_model
```
After running, check that SignatureDefs `serving_default` is not empty:
```bash
saved_model_cli show --dir <your_yolov8_saved_model> --all
```

## TensorRT
```bash
yolo export model=best.pt format=engine
```

Alternatively, use `trtexec`:
```bash
trtexec --onnx=best.onnx --saveEngine=best.engine
```

Export with dynamic axis example:
```bash
trtexec --onnx=yourmodel.onnx --minShapes=images:1x3x640x640 --optShapes=images:1x3x640x640 --maxShapes=images:32x3x640x640 --saveEngine=yourmodel.engine --fp16
```

Note: When using TensorRT, ensure that the version installed under Ultralytics python environment matches the C++ version you plan to use for inference.
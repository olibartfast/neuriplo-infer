# YOLOv10 Export Instructions

## OnnxRuntime 
Using the [Ultralytics pip package](https://docs.ultralytics.com/quickstart/):

```bash
yolo export format=onnx model=yolov10model.pt
```

## Torchscript
Same method as OnnxRuntime:

```bash
yolo export format=torchscript model=yolov10model.pt
```

## OpenVINO
```bash
yolo export format=openvino model=yolov10model.pt
```

## TensorRT
```bash
trtexec --onnx=yolov10model.onnx --saveEngine=yolov10model.engine --fp16
```

## TensorFlow
```bash
onnx2tf -i yolov10x.onnx -o <your_yolov10x_saved_model> --disable_group_convolution -osd
```

After running the following command, check that SignatureDefs `serving_default` is not empty:

```bash
saved_model_cli show --dir yolov10_saved_model --all
```
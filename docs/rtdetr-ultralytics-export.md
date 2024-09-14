# RT-DETR (Ultralytics) Export Instructions

Always use the [Ultralytics pip package](https://docs.ultralytics.com/quickstart/) to export the model.

## OnnxRuntime
Export the model to ONNX using the following command:

```bash
yolo export model=best.pt format=onnx
```
Note: In this case, `best.pt` is a trained RTDETR-L or RTDETR-X model.

## Libtorch
Similar to the ONNX case, change format to torchscript:

```bash
yolo export model=best.pt format=torchscript 
```

## TensorRT
Same as explained for YOLOv8:

```bash
trtexec --onnx=yourmodel.onnx --saveEngine=yourmodel.engine
```

Or:

```bash
yolo export model=yourmodel.pt format=engine
```

For more information, visit: https://docs.ultralytics.com/models/rtdetr/
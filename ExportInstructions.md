## Instructions

### Export the model for the inference
### YoloV8

Install YoloV8 [following official documentation](https://docs.ultralytics.com/quickstart/) and export the model in different formats, you can use the following commands:

#### Torchscript

To export the model in the TorchScript format:

```
yolo export model=best.pt format=torchscript device=0
```

#### OnnxRuntime

To export the model in the ONNXRuntime format:

```
yolo export model=best.pt format=onnx
```

#### TensorRT

To export the model in the TensorRT format:

```
yolo export model=best.pt format=engine device=0
```

Please note that when using TensorRT, ensure that the version installed under Ultralytics python environment matches the C++ version you plan to use for inference. Another way to export the model is to use `trtexec` with the following command:

```
trtexec --onnx=best.onnx --saveEngine=best.engine
```

By following these steps, you can successfully export the model in the desired formats for further inference.

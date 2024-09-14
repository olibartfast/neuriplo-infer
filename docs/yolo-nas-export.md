# YOLO-NAS Export Instructions

## OnnxRuntime
Weights can be exported in ONNX format as described in the [YOLO-NAS Quickstart](https://github.com/Deci-AI/super-gradients/blob/master/documentation/source/YoloNASQuickstart.md#export-to-onnx).

Here's an example of exporting the model, specifying input and output layer names for the YOLO-NAS-S version:

```python
from super_gradients.training import models

net = models.get("yolo_nas_s", pretrained_weights="coco")
models.convert_to_onnx(model=net, input_shape=(3,640,640), out_path="yolo_nas_s.onnx", torch_onnx_export_kwargs={"input_names": ['input'], "output_names": ['output0', 'output1']})
```
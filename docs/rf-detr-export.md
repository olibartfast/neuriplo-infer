# **RF-DETR Export Instructions**  
* Follow the procedure listed https://github.com/roboflow/rf-detr?tab=readme-ov-file#onnx-export

### ONNX Export

> [!IMPORTANT]
> Starting with RF-DETR 1.2.0, you'll have to run `pip install rfdetr[onnxexport]` before exporting model weights to ONNX format.  

RF-DETR supports exporting models to the ONNX format, which enables interoperability with various inference frameworks and can improve deployment efficiency. To export your model, simply initialize it and call the `.export()` method.

```python
from rfdetr import RFDETRBase

model = RFDETRBase(pretrain_weights=<CHECKPOINT_PATH>)

model.export()
```

This command saves the ONNX model to the `output` directory.

# **RF-DETR Export Instructions**  
* Follow the procedure listed https://github.com/roboflow/rf-detr?tab=readme-ov-file#onnx-export

### ONNX Export for onnxruntime

> [!IMPORTANT]
> Starting with RF-DETR 1.2.0, you'll have to run `pip install rfdetr[onnxexport]` before exporting model weights to ONNX format.  

RF-DETR supports exporting models to the ONNX format, which enables interoperability with various inference frameworks and can improve deployment efficiency. To export your model, simply initialize it and call the `.export()` method.

```python
from rfdetr import RFDETRBase # or RFDETRBase/Nano/Small/Medium/Large

model = RFDETRBase(pretrain_weights=<OPTIONAL_CHECKPOINT_PATH>)

model.export()
```

This command saves the ONNX model to the `output` directory.

### TensorRT Export
```bash
trtexec --onnx=/path/to/model.onnx --saveEngine=/path/to/model.engine --memPoolSize=workspace:4096 --fp16 --useCudaGraph --useSpinWait --warmUp=500 --avgRuns=1000 --duration=10
```

* using a tensorrt docker container:
```bash
export NGC_TAG_VERSION=24.12
docker run --rm -it --gpus=all -v $(pwd)/exports:/exports --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v $(pwd)/model.onnx:/workspace/model.onnx -w /workspace nvcr.io/nvidia/tensorrt$NGC_TAG_VERSION:-py3 /bin/bash -cx "trtexec --onnx=model.onnx --saveEngine=/exports/model.engine --memPoolSize=workspace:4096 --fp16 --useCudaGraph --useSpinWait --warmUp=500 --avgRuns=1000 --duration=10"
```
 

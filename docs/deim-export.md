# **DEIM Export Instructions**  
* Export procedure is similar to [D-FINE](d-fine-export.md)

## **Exporting ONNX Models with ONNXRuntime**  
To export DEIM models to ONNX format, follow the steps below:  

### **Repository**  
[ShihuaHuang95 DEIM Repository](https://github.com/ShihuaHuang95/DEIM)  

### **Steps:**  
1. Navigate to the DEIM repository directory:  
   ```bash
   cd DEIM  
   ```  
2. then select and download a model from the model zoo (in example below `deim_dfine_hgnetv2_s_coco_120e.pth`)
3. Run the export script:  
   ```bash
   python tools/deployment/export_onnx.py --check -c configs/deim_dfine/deim_hgnetv2_s_coco.yml -r deim_dfine_hgnetv2_s_coco_120e.pth
   ```  

### **Notes:**  
- Ensure the batch size hardcoded in the `export_onnx.py` script is appropriate for your system's available RAM. If not, modify the batch size in the script to avoid out-of-memory errors.  
- Verify that `pth` model selected corresponds to the correct pre-trained model checkpoint for the configuration file you're using.  
- The `--check` flag ensures that the exported ONNX model is validated after the export process.  


## **Convert ONNX Model to TensorRT**
```bash
mkdir exports
docker run --rm -it --gpus=all -v $(pwd)/exports:/exports --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v $(pwd)/model.onnx:/workspace/model.onnx -w /workspace nvcr.io/nvidia/tensorrt:24.12-py3 /bin/bash -cx "trtexec --onnx="model.onnx" --saveEngine="/exports/model.engine" --fp16"
```


# **DEIMv2 Export Instructions**  
* Export procedure is similar to [D-FINE](d-fine-export.md) and [DEIM](deim-export.md)

## **Exporting ONNX Models with ONNXRuntime**  
To export DEIMv2 models to ONNX format, follow the steps below:  

### **Repository**  
[Intellidust AI Lab DEIMv2 Repository](https://github.com/Intellindust-AI-Lab/DEIMv2)  

### **Steps:**  
1. Navigate to the DEIMv2 repository directory:  
   ```bash
   cd DEIMv2  
   ```  
2. Download the pretrained model (e.g., `vitt_distill.pt`) to the `ckpts` folder  
3. Select a model from the model zoo. For example: `deimv2_dinov3_s_coco` (check the model zoo for other available models)
4. Run the export script:  
   ```bash
   export MODEL=deimv2_dinov3_s_coco
   python tools/deployment/export_onnx.py --check -c configs/deimv2/$MODEL.yml -r /path/to/$MODEL.pth
   ```  

### **Notes:**  
- Same considerations as [D-FINE](d-fine-export.md) regarding batch size and model verification
- Ensure the batch size hardcoded in the `export_onnx.py` script is appropriate for your system's available RAM. If not, modify the batch size in the script to avoid out-of-memory errors.  
- Verify that the `.pth` model selected corresponds to the correct pre-trained model checkpoint for the configuration file you're using.  
- The `--check` flag ensures that the exported ONNX model is validated after the export process.  


## **Convert ONNX Model to TensorRT**
For D-FINE, DEIM, and DEIMv2 models, follow the same procedure as [lyuwenyu RT-DETR models](rtdetr-lyuwenyu-export.md#convert-onnx-model-to-tensorrt):

```bash
mkdir exports
docker run --rm -it --gpus=all -v $(pwd)/exports:/exports --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v $(pwd)/model.onnx:/workspace/model.onnx -w /workspace nvcr.io/nvidia/tensorrt:24.12-py3 /bin/bash -cx "trtexec --onnx="model.onnx" --saveEngine="/exports/model.engine" --fp16"
```

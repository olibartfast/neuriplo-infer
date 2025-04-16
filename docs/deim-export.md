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

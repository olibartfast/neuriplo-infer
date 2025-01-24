# **D-FINE Export Instructions**  

## **Exporting ONNX Models with ONNXRuntime**  
To export D-FINE models to ONNX format, follow the steps below:  

### **Repository**  
[Peterande D-FINE Repository](https://github.com/Peterande/D-FINE)  

### **Steps:**  
1. Navigate to the D-FINE repository directory:  
   ```bash
   cd D-FINE
   ```  

2. Define the model size you want to export (`n`, `s`, `m`, `l`, or `x`). For example:  
   ```bash
   export model=l
   ```  

3. Run the export script:  
   ```bash
   python tools/deployment/export_onnx.py --check -c configs/dfine/dfine_hgnetv2_${model}_coco.yml -r model.pth
   ```  

### **Notes:**  
- Ensure the batch size hardcoded in the `export_onnx.py` script is appropriate for your system's available RAM. If not, modify the batch size in the script to avoid out-of-memory errors.  
- Verify that `model.pth` corresponds to the correct pre-trained model checkpoint for the configuration file you're using.  
- The `--check` flag ensures that the exported ONNX model is validated after the export process.  

### **Example:**  
To export the large model (`l`) with the corresponding configuration:  
```bash
cd D-FINE
export model=l
python tools/deployment/export_onnx.py --check -c configs/dfine/dfine_hgnetv2_l_coco.yml -r model.pth
```

## **Convert ONNX Model to TensorRT**
```bash
mkdir exports
docker run --rm -it --gpus=all -v $(pwd)/exports:/exports -u $(id -u):$(id -g) --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v $(pwd)/model.onnx:/workspace/model.onnx -w /workspace nvcr.io/nvidia/tensorrt:24.12-py3 /bin/bash -cx "trtexec --onnx="model.onnx" --saveEngine="model.engine" --fp16"
```

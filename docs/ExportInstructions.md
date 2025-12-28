### Export the model for the inference

For comprehensive model export instructions, see the [vision-core export section](https://github.com/olibartfast/vision-core/tree/main/export) which provides generic export utilities for:

- **YOLO models**: Universal export for YOLOv5-v12, YOLO11, YOLO-NAS
- **Detection models**: RT-DETR (all versions), D-FINE, DEIM, DEIMv2, RF-DETR
- **Multiple formats**: ONNX, TorchScript, TensorRT, SavedModel

#### Quick Export Examples

```bash
# Export any YOLO model (v5-v12, YOLO11, NAS)
python export/detection/yolo/export.py --model your_model.pt --format onnx

# Export RT-DETR models
python export/detection/rtdetr/export.py --model your_rtdetr.pt --format onnx

# Export RF-DETR models  
python export/detection/rfdetr/export.py --model your_rfdetr.pt --format onnx
```

## Note
The opencv-dnn module is configured to load ONNX models(not dynamic axis) and .weights(i.e. darknet format) for YOLOv4.

For detailed instructions on exporting each model, please refer to the linked documents above.

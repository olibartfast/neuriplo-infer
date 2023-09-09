
The table represents a specific model and includes the following details:

- **Model**: The name or identifier of the model.
- **Model Type**: The type of the model, indicating the underlying architecture or framework used.
- **o**: Supported backend.
- **x**: Not supported backend.


| Model                                              | Model Type | OpenCV-DNN | TensorRT | LibTorch | Onnx-runtime | LibTensorflow |
|----------------------------------------------------|------------|------------|----------|----------|--------------|---------------|
| yolov4/yolov4-tiny                                 | yolov4     | o          | x        | x        | x            | x             |
| yolov5 models                              | yolov5     | o          | x        | x        | x            | x             |
| yolov6 models                              | yolov6     | o          | x        | x        | x            | x             |
| yolov7 models                              | yolov7     | o          | x        | x        | x            | x             |
| yolov8 models                           | yolov8     | o          | o        | o        | o            | x             |
| yolo-nas models                            | yolonas    | o          | x        | x        | o            | x             |
| rt-detr models                            | rtdetr    |x          | x        | o        | o            | x             |
| models from TF2 Object Detection API model zoo | tensorflow | x          | x        | x        | x            | o             |

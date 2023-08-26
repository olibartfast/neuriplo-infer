
The table represents a specific model and includes the following details:

- **Model**: The name or identifier of the model.
- **Model Type**: The type of the model, indicating the underlying architecture or framework used.


| Model                                              | Model Type | OpenCV-DNN | TensorRT | LibTorch | Onnx-runtime | LibTensorflow |
|----------------------------------------------------|------------|------------|----------|----------|--------------|---------------|
| yolov4/yolov4-tiny                                 | yolov4     | o          | x        | x        | x            | x             |
| Any yolov5 model                              | yolov5     | o          | x        | x        | x            | x             |
| Any yolov6 model                              | yolov6     | o          | x        | x        | x            | x             |
| Any yolov7 model                              | yolov7     | o          | x        | x        | x            | x             |
| Any yolov8 model                              | yolov8     | o          | x        | o        | o            | x             |
| Any yolo-nas model                            | yolonas    | o          | x        | x        | x            | x             |
| Any models from TF2 Object Detection API model zoo | tensorflow | x          | x        | x        | x            | o             |


The table represents a specific model and includes the following details:

- **Model**: The name or identifier of the model.
- **Model Type**: The type of the model, indicating the underlying architecture or framework used.


| Model                                              | Model Type | OpenCV-DNN | TensorRT | LibTorch | Onnx-runtime | LibTensorflow |
|----------------------------------------------------|------------|------------|----------|----------|--------------|---------------|
| yolov4/yolov4-tiny                                 | yolov4     | o          | x        | x        | x            | x             |
| Any yolov5 640p model                              | yolov5     | o          | x        | x        | x            | x             |
| Any yolov6 640p model                              | yolov6     | o          | x        | x        | x            | x             |
| Any yolov7 640p model                              | yolov7     | o          | x        | x        | x            | x             |
| Any yolov8 640p model                              | yolov8     | o          | x        | x        | x            | x             |
| Any yolo-nas 640p model                            | yolonas    | o          | x        | x        | x            | x             |
| Any models from TF2 Object Detection API model zoo | tensorflow | x          | x        | x        | x            | o             |

- **Demo**: The command-line demo to execute the object detection inference using the model. It includes the necessary parameters, such as the model type and input video stream.
- **Notes**: Additional notes or instructions related to the model, such as where to download the model files, required file formats, or specific considerations.
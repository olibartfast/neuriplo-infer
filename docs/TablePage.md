
The table represents a specific model and includes the following details:

- **Model**: The name or identifier of the model.
- **Model Type**: The type of the model, indicating the underlying architecture or framework used.
- **o**: Supported backend.
- **x**: Not supported backend.


| Model                                              | Model Type | OpenCV-DNN | TensorRT | LibTorch | Onnx-runtime | LibTensorflow | OpenVino |
|----------------------------------------------------|------------|------------|----------|----------|--------------|---------------|----------|
| yolov4/yolov4-tiny                                 | yolov4     | o          | x        | x        | x            | x             | x |
| yolov5 models                                      | yolov5     | o          | x        | o        | o            | x             | o |
| yolov6 models                                      | yolov6     | o          | x        | x        | o            | x             | o |
| yolov7 models                                      | yolov7     | o          | x        | o        | o            | x             | o |
| yolov8 models                                      | yolov8     | o          | o        | o        | o            | x             | x |
| yolo-nas models                                    | yolonas    | o          | x        | x        | o            | x             | x |
| rt-detr models                                     | rtdetr     | x          | o        | o        | o            | x             | x |
| models from TF2 Object Detection API model zoo     | tensorflow | x          | x        | x        | x            | o             | x |

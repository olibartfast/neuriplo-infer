
The table represents a specific model and includes the following details:

- **Model**: The name or identifier of the model.
- **Model Type**: The type of model, indicating the underlying architecture or framework used (also the string corresponding to the param to be used with ``--type`` during the inference step of this project).
- **o**: Supported backend.
- **x**: Not supported backend (or at least not tested yet/not tried to export).


| Model                                              | Model Type | OpenCV-DNN | TensorRT | LibTorch | Onnx-runtime  | OpenVino |
|----------------------------------------------------|------------|------------|----------|----------|--------------|-----------|
| yolov4/yolov4-tiny                                 | yolov4     | o          | x        | x        | x            | x         |
| yolov5 models                                      | yolov5     | o          | x        | o        | o            | o         |
| yolov6 models                                      | yolov6     | o          | x        | x        | o            | x         |
| yolov7 models                                      | yolov7     | o          | x        | o        | o            | x         |
| yolov8 models                                      | yolov8     | o          | o        | o        | o            | o         |
| yolov9 models                                      | yolov9     | o          | o        | o        | o            | x         |
| yolov10 models                                     | yolov10    | o          | o        | o        | o            | x         |
| yolo-nas models                                    | yolonas    | o          | x        | x        | o            | x         |
| rt-detr models                                     | rtdetr     | x          | o        | x        | o            | x         |
| rt-detr ultralytics models                         | rtdetrul   | x          | o        | o        | o            | x         |

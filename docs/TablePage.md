
The table represents a specific model and includes the following details:

- **Model**: The name or identifier of the model.
- **Model Type**: The type of model, indicating the underlying architecture or framework used (also the string corresponding to the param to be used with ``--type`` during the inference step of this project).
- **o**: Supported backend.
- **x**: Not supported backend (or at least not tested yet/not tried to export).
- **Note**:  OpenCV-DNN --> tested only onnx weights on cpu, opencv-dnn currently supports only static shapes, dynamic input (or layer) shape does not work with opencv 


| Model                                              | Model Type | OpenCV-DNN | TensorRT | LibTorch | Onnx-runtime | OpenVino  | Libtensorflow |
|----------------------------------------------------|------------|------------|----------|----------|--------------|-----------|-----------|
| yolov4/yolov4-tiny                                 | yolov4     | o          | x        | x        | x            | x         | x         | 
| yolov5 models                                      | yolov5     | o          | x        | o        | o            | o         | x         |
| yolov6 models                                      | yolov6     | o          | x        | x        | o            | x         | x         |
| yolov7 models                                      | yolov7     | o          | x        | o        | o            | x         | x         |
| yolov8 models                                      | yolov8     | o          | o        | o        | o            | o         | o         |
| yolov9 models                                      | yolov9     | o          | o        | o        | o            | x         | x         |
| yolov10 models                                     | yolov10    | x          | o        | o        | o            | o         | o         |
| yolo11 models                                      | yolo11     | o          | o        | o        | o            | x         | x         |
| yolov12 models                                     | yolov12    | o          | o        | o        | o            | o         | o         |
| yolo-nas models                                    | yolonas    | o          | x        | x        | o            | x         | x         |
| rt-detr models                                     | rtdetr     | x          | o        | x        | o            | x         | x         |
| rt-detr v2 models                                  | rtdetrv2   | x          | o        | x        | o            | x         | x         |
| rt-detr ultralytics models                         | rtdetrul   | x          | o        | o        | o            | x         | x         |
| d-fine models                                      | dfine      | x          | o        | x        | o            | x         | x         |
| deim models                                        | deim       | x          | o        | x        | o            | x         | x         |
| deimv2 models                                      | deim     | x          | o        | x        | o            | x         | x         |
| rf-detr models                                     | rfdetr     | x          | o        | x        | o            | x         | x         |

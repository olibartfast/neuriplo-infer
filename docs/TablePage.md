
The table represents a specific model and includes the following details:

- **Model**: The name or identifier of the model.
- **Model Type**: The type of model, indicating the underlying architecture or framework used (also the string corresponding to the param to be used with ``--type`` during the inference step of this project).
- **o**: Supported backend.
- **x**: Not supported backend (or at least not tested yet/not tried to export).
- **Note**:  OpenCV-DNN --> tested only onnx weights on cpu, opencv-dnn currently supports only static shapes, dynamic input (or layer) shape does not work with opencv 


| Model                                              | Model Type | OpenCV-DNN | TensorRT | LibTorch | Onnx-runtime | OpenVino  | Libtensorflow |
|----------------------------------------------------|------------|------------|----------|----------|--------------|-----------|-----------|
| yolov4/yolov4-tiny                                 | yolov4     | o          | x        | x        | x            | x         | x         | 
| yolov5 models                                      | yolo     | o          | x        | o        | o            | o         | x         |
| yolov6 models                                      | yolo     | o          | x        | x        | o            | x         | x         |
| yolov7 models                                      | yolo     | o          | x        | o        | o            | x         | x         |
| yolov8 models                                      | yolo     | o          | o        | o        | o            | o         | o         |
| yolov9 models                                      | yolo     | o          | o        | o        | o            | x         | x         |
| yolov10 models                                     | yolo    | x          | o        | o        | o            | o         | o         |
| yolo11 models                                      | yolo     | o          | o        | o        | o            | x         | x         |
| yolov12 models                                     | yolo    | o          | o        | o        | o            | o         | o         |
| yolo-nas models                                    | yolonas    | o          | x        | x        | o            | x         | x         |
| rt-detr models                                     | rtdetr     | x          | o        | x        | o            | x         | x         |
| rt-detr v2 models                                  | rtdetr   | x          | o        | x        | o            | x         | x         |
| rt-detr ultralytics models                         | rtdetrul   | x          | o        | o        | o            | x         | x         |
| d-fine models                                      | rtdetr      | x          | o        | x        | o            | x         | x         |
| deim models                                        | rtdetr       | x          | o        | x        | o            | x         | x         |
| deimv2 models                                      | rtdetr     | x          | o        | x        | o            | x         | x         |
| rf-detr models                                     | rfdetr     | x          | o        | o        | o            | x         | x         |

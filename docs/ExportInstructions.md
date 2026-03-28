### Export the model for the inference

Refer to:
* [Object Detection](https://github.com/olibartfast/vision-core/blob/master/export/detection/ObjectDetection.md) 
* [Classification](https://github.com/olibartfast/vision-core/blob/master/export/classification/Classification.md)
* [Instance Segmentation](https://github.com/olibartfast/vision-core/blob/master/export/segmentation/InstanceSegmentation.md)
* [Optical Flow](https://github.com/olibartfast/vision-core/blob/master/export/optical_flow/OpticalFlow.md)

For local export-plus-inference examples inside this repo, use [`docker_run_inference_e2e_example.sh`](../docker_run_inference_e2e_example.sh). It provides preset-driven end-to-end flows, including OWLv2 with ONNX Runtime.

## Note
The opencv-dnn module is configured to load ONNX models(not dynamic axis) and .weights(i.e. darknet format) for YOLOv4.

For detailed instructions on exporting each model, please refer to the linked documents above.

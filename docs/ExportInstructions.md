# Model Export

Model export and tensor contracts are owned by `neuriplo-tasks`. Start with its
[export index](https://github.com/olibartfast/neuriplo-tasks/blob/master/export/README.md)
and the [supported model inventory](generated/supported-model-types.md).

Task-specific guides (use the export index for additional model-specific recipes):

* [Object Detection](https://github.com/olibartfast/neuriplo-tasks/blob/master/export/detection/ObjectDetection.md) 
* [Classification](https://github.com/olibartfast/neuriplo-tasks/blob/master/export/classification/Classification.md)
* [Instance Segmentation](https://github.com/olibartfast/neuriplo-tasks/blob/master/export/segmentation/InstanceSegmentation.md)
* [Optical Flow](https://github.com/olibartfast/neuriplo-tasks/blob/master/export/optical_flow/OpticalFlow.md)
* [Keypoint Matching](https://github.com/olibartfast/neuriplo-tasks/blob/master/export/keypoint_matching/README.md)
* [Pose Estimation](https://github.com/olibartfast/neuriplo-tasks/blob/master/export/pose_estimation/PoseEstimation.md)
* [Depth Estimation](https://github.com/olibartfast/neuriplo-tasks/blob/master/export/depth_estimation/DepthEstimation.md)
* [Open-Vocabulary Detection](https://github.com/olibartfast/neuriplo-tasks/blob/master/export/open_vocab_detection/OWLv2.md)
* [Video Classification](https://github.com/olibartfast/neuriplo-tasks/blob/master/export/video_classification/VideoClassification.md)
* [Gaussian Splatting export tools](https://github.com/olibartfast/neuriplo-tasks/tree/master/export/gaussian_splatting)
* [Image Understanding (VLM)](https://github.com/olibartfast/neuriplo-tasks/blob/master/export/image_understanding/ImageUnderstanding.md) — download the model and vision projector GGUF files for the llama.cpp backend; no ONNX export is needed.

For local export-plus-inference examples, use
[`docker_run_inference_e2e_example.sh`](../docker_run_inference_e2e_example.sh).
From the repository root, list presets and preview the OWLv2 workflow (replace
the checkout path with your local `neuriplo-tasks` directory):

```bash
bash docker_run_inference_e2e_example.sh --list-presets
bash docker_run_inference_e2e_example.sh --preset owlv2 \
  --neuriplo-tasks-dir /path/to/neuriplo-tasks --dry-run
```

Follow [Deployment](Deployment.md) for image preparation, required assets, and
running the selected preset. Use `--neuriplo-tasks-dir /path/to/neuriplo-tasks`
to select a checkout containing the upstream export tools.

Choose the export format for the backend you built; see the
[backend format table](Deployment.md#backends) and
[backend coverage](BackendCoverage.md) for tested combinations and limitations.
OpenCV DNN uses ONNX models with fixed input dimensions, or Darknet `.weights`
with a `.cfg` file for YOLOv4. Exporting a model to ONNX alone does not guarantee
that its operators or inputs are supported by every backend.

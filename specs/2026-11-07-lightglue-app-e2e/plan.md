# Plan

1. Consume the existing grayscale pair batch and typed decoder from neuriplo-tasks develop snapshot 1331639. Correct empty-match output classification with a regression test. Add bounded allocation and notified shapes for data-dependent TensorRT outputs in neuriplo on feature/trt-data-dependent-outputs.
2. Route LightGlue through the app, require exactly two still images, run task preprocessing/backend/postprocessing and render both original views. Preserve stage reporting and output naming.
3. Validate owning tests before ORT application runs and Slurm-dispatched Jetson TensorRT application runs. Update user docs and generated model inventory; promote coverage only with app evidence.

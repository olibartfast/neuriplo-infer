# LightGlue application validation

Maintainer request: run SuperPoint + LightGlue through neuriplo-infer, ORT on this machine and TensorRT on Jetson Orin dispatched by Slurm using its Docker worker. Both Capitol and Sacré-Cœur image pairs are required. Render correspondences with score strictly greater than 0.2. Raw backend inference is insufficient for PASS. Complete the missing tasks-owned preprocessing/decoding and app routing/rendering for the verified LightGlue-ONNX v2.0 export. Preserve other families and existing CLI flags.

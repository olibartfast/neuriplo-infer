# RT-DETR V2  Export Instructions

The process is similar to RTDETR export, From the [lyuwenyu RT-DETR repository, rtdetr v2 pytorch folder](https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetrv2_pytorch):

## OnnxRuntime
```bash
cd RT-DETR/rtdetr_pytorch
python tools/export_onnx.py -c configs/rtdetrv2/rtdetrv2_r18vd_120e_coco.yml -r path/to/checkpoint --check
```
Note: You can use other versions instead of `rtdetrv2_r18vd_120e_coco.yml`.

## TensorRT
```bash
trtexec --onnx=<model>.onnx --saveEngine=<model>.engine --minShapes=images:1x3x640x640,orig_target_sizes:1x2 --optShapes=images:1x3x640x640,orig_target_sizes:1x2 --maxShapes=images:1x3x640x640,orig_target_sizes:1x2
```
Note: This assumes you exported the ONNX model in the previous step.



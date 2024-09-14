# RT-DETR [lyuwenyu] Export Instructions

From the [lyuwenyu RT-DETR repository](https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetr_pytorch):

## OnnxRuntime
```bash
cd RT-DETR/rtdetr_pytorch
python tools/export_onnx.py -c configs/rtdetr/rtdetr_r18vd_6x_coco.yml -r path/to/checkpoint --check
```
Note: You can use other versions instead of `rtdetr_r18vd_6x_coco.yml`.

## TensorRT
```bash
trtexec --onnx=<model>.onnx --saveEngine=rtdetr_r18vd_dec3_6x_coco_from_paddle.engine --minShapes=images:1x3x640x640,orig_target_sizes:1x2 --optShapes=images:1x3x640x640,orig_target_sizes:1x2 --maxShapes=images:1x3x640x640,orig_target_sizes:1x2
```
Note: This assumes you exported the ONNX model in the previous step.
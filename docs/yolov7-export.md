# YOLOv7 Export Instructions

## OnnxRuntime and/or Libtorch
Run from the [YOLOv7 repository](https://github.com/WongKinYiu/yolov7#export):

```bash
python export.py --weights <yolov7_version>.pt --grid --simplify --topk-all 100 --iou-thres 0.65 --conf-thres 0.35 --img-size 640 640 --max-wh 640
```

Note: Don't use the end-to-end parameter.
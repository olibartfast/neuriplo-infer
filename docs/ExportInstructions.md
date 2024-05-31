## Instructions


## Note
Opencv-dnn module loads onnx models except for yolov4 .weights  

### Export the model for the inference

## YOLOv10
### OnnxRuntime 
* From [yolov10 repo](https://github.com/THU-MIG/yolov10):
```
yolo export format=onnx model=yolov10model.pt

```
#### Torchscript
* Same way as above:
```
yolo export format=torchscript model=yolov10model.pt

```


## YOLOv9
#### OnnxRuntime
* Run from [yolov9 repo](https://github.com/WongKinYiu/yolov9):
```
 python export.py --weights yolov9-c/e-converted.pt --include onnx
```
#### Torchscript
```
 python export.py --weights yolov9-c/e-converted.pt --include torchscript
```
#### TensorRT
```
 trtexec --onnx=yolov9-c/e-converted.onnx --saveEngine=yolov9-c/e.engine --fp16
```

## YOLOv8

Install YOLOv8 [following Ultralytics official documentation](https://docs.ultralytics.com/quickstart/) and export the model in different formats, you can use the following commands:

#### Torchscript

```
yolo export model=best.pt(the best corrisponding to your trained yolov8n/s/m/x) format=torchscript
```

#### OnnxRuntime

```
yolo export model=best.pt format=onnx
```

#### OpenVino

```
yolo export model=best.pt format=openvino

```

#### TensorRT

```
yolo export model=best.pt format=engine
```

Please note that when using TensorRT, ensure that the version installed under Ultralytics python environment matches the C++ version you plan to use for inference. Another way to export the model is to use `trtexec` with the following command:

```
trtexec --onnx=best.onnx --saveEngine=best.engine
```

Export with dynamic axis example:
```
trtexec --onnx=yourmodel.onnx --minShapes=images:1x3x640x640 --optShapes=images:1x3x640x640 --maxShapes=images:32x3x640x640   --saveEngine=yourmodel.engine --fp16
```


## YOLOv5 
#### OnnxRuntime
* Run from [yolov5 repo](https://github.com/ultralytics/yolov5/issues/251) export script:  ```python export.py  --weights <yolov5_version>.pt  --include onnx```

#### Libtorch
* from yolov5 repo: ```python export.py  --weights <yolov5_version>.pt  --include torchscript```

## YOLOv6
#### OnnxRuntime
Weights to export in ONNX format or download from [yolov6 repo](https://github.com/meituan/YOLOv6/tree/main/deploy/ONNX). Posteprocessing code is identical to yolov5-v7.

## RT-DETR (Ultralytics)
#### OnnxRuntime
Always using [Ultralytics pip package](https://docs.ultralytics.com/quickstart/) export the model to onnx using the following command:
```
yolo export model=best.pt(in this case best.pt is a trained rtdetr-l or rtdetr-x model) format=onnx
```
#### Libtorch
As previous for onnx case, change format=torchscript, i.e.
```
yolo export model=best.pt format=torchscript 
```
More infos here: https://docs.ultralytics.com/models/rtdetr/

#### TensorRT
Same as explained for YoloV8 above:

```
trtexec --onnx=yourmodel.onnx --saveEngine=yourmodel.engine
```
or 

```
yolo export model=yourmodel.pt format=engine
```


## RT-DETR [lyuwenyu](https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetr_pytorch)
- From lyuwenyu rt-detr repo:
#### OnnxRuntime
```
cd RT-DETR/rtdetr_pytorch
python tools/export_onnx.py -c configs/rtdetr/rtdetr_r18vd_6x_coco.yml(or other version) -r path/to/checkpoint --check
```

#### TensorRT
```
trtexec  --onnx=<model>.onnx --saveEngine=rtdetr_r18vd_dec3_6x_coco_from_paddle.engine(supposing you exported onnx above) --minShapes=images:1x3x640x640,orig_target_sizes:1x2 --optShapes=images:1x3x640x640,orig_target_sizes:1x2 --maxShapes=images:1x3x640x640,orig_target_sizes:1x2
```

## YOLOv7
#### OnnxRuntime and/or Libtorch
* Run from [yolov7 repo](https://github.com/WongKinYiu/yolov7#export): ```python export.py --weights <yolov7_version>.pt --grid  --simplify --topk-all 100 --iou-thres 0.65 --conf-thres 0.35 --img-size 640 640 --max-wh 640``` (Don't use end-to-end parameter)


## YOLO-NAS export 
#### OnnxRuntime
* Weights can be export in ONNX format like in [YoloNAS Quickstart](https://github.com/Deci-AI/super-gradients/blob/master/documentation/source/YoloNASQuickstart.md#export-to-onnx).  
I export the model specifying input and output layers name, for example here below in the case of yolo_nas_s version:
```
from super_gradients.training import models

net = models.get("yolo_nas_s", pretrained_weights="coco")
models.convert_to_onnx(model=net, input_shape=(3,640,640), out_path="yolo_nas_s.onnx", torch_onnx_export_kwargs={"input_names": ['input'], "output_names": ['output0', 'output1']})
```

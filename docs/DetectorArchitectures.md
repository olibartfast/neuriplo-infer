# Object Detection Architectures Guide 

## Table of Contents

* [At-a-Glance Selection](#at-a-glance-selection)
* [YOLO Family](#yolo-family)

  * [Core Principles](#core-principles)
  * [Evolution Timeline](#evolution-timeline)
  * [Strengths](#strengths)
  * [Weaknesses](#weaknesses)
* [RT-DETR Family](#rt-detr-family)

  * [Core Concept](#core-concept)
  * [Architecture Overview](#architecture-overview)
  * [Training & Inference Notes](#training--inference-notes)
  * [Variants](#variants)
  * [Strengths](#strengths-1)
  * [Weaknesses](#weaknesses-1)
* [YOLO-NAS](#yolo-nas)

  * [Core Innovations](#core-innovations)
  * [Key Features](#key-features)
  * [Strengths](#strengths-2)
  * [Weaknesses](#weaknesses-2)
* [Specialized Architectures (Project-Specific)](#specialized-architectures-project-specific)

  * [D-FINE](#d-fine)
  * [DEIM](#deim)
  * [RF-DETR](#rf-detr)
* [Architecture Comparison Summary](#architecture-comparison-summary)
* [Deployment Notes](#deployment-notes)
* [References](#references)

---

## At-a-Glance Selection

* **Edge real-time (speed/accuracy balanced):** YOLOv8/YOLO11 (S/M), YOLO-NAS (with QAT) 
* **Highest accuracy on server GPUs:** RT-DETR or transformer-based refinements 
* **NMS-free pipeline:** YOLOv10, RT-DETR, RF-DETR 
* **Strict memory/latency budgets:** YOLO-NAS (QAT) or Tiny/Small YOLO models 

---

## YOLO Family

YOLO ("You Only Look Once") represents fast, single-stage detectors that perform dense predictions directly over feature maps. Modern YOLO versions refine this paradigm with anchor‑free heads, strong backbones, and increasingly deployment‑friendly designs.

### Core Principles

* Single-stage detection with dense predictions.
* Multi-scale heads for improved small-object performance.
* Increasingly anchor-free designs (v8+).
* Reparameterizable blocks (v6–v7) for speed at inference.
* Mostly NMS-based—except YOLOv10.

### Evolution Timeline

#### **YOLOv1 (2016)**

* Introduced the single-shot paradigm.
* Grid-based prediction; limited small-object handling.

#### **YOLOv2 / YOLO9000 (2017)**

* Introduced anchor boxes, BN everywhere, high-res pretraining.
* Backbone: Darknet‑19.

#### **YOLOv3 (2018)**

* Added multi-scale predictions (FPN), deeper backbone (Darknet‑53).
* Strong improvement in robustness.

#### **YOLOv4 (2020)**

* CSPDarknet53 backbone; SPP + PAN neck.
* Heavy use of training augmentations.

#### **YOLOv5 (2020)**

* First major PyTorch implementation; highly production‑ready.
* AutoAnchor, C2f/CSP modules, strong training recipes.

#### **YOLOv6 (2022)**

* Industrial focus; RepVGG-style reparameterizable designs.
* Decoupled heads and SimOTA label assignment.

#### **YOLOv7 (2022)**

* E-ELAN architecture; extensive reparameterization.
* Strong accuracy for its speed.

#### **YOLOv8 (2023)**

* Anchor-free design, C2f modules, improved heads.
* Unified detection/segmentation/pose framework.

#### **YOLOv9 (2024)**

* Introduced PGI (better gradient flow) and GELAN backbone.

#### **YOLOv10 (2024)**

* Fully NMS-free using dual assignments.
* End-to-end latency benefits.

#### **YOLO11 (2024)**

* Ultralytics successor to YOLOv8 with incremental accuracy and speed gains.

### Strengths

* Excellent real-time performance.
* Mature ecosystem and tooling.
* Works across detection, segmentation, classification, pose.
* Strong ONNX/TensorRT export story.

### Weaknesses

* NMS adds latency and can affect recall.
* Crowded/small-object scenes remain challenging.
* Requires careful threshold tuning (pre-v10).

---

## RT-DETR Family

RT-DETR combines CNN backbones with efficient multi-scale transformers to create real-time, NMS-free detectors.

### Core Concept

* Hybrid CNN + transformer design.
* Learnable object queries for global scene understanding.
* Hungarian-matching-based set prediction (NMS-free).

### Architecture Overview

1. **Backbone:** Efficient CNN generating multi-scale features.
2. **Encoder:** Multi-scale transformer encoder.
3. **Decoder:** Object queries producing end-to-end predictions.
4. **Heads:** Box + class predictions trained with L1/GIoU + CE/Focal.

### Training & Inference Notes

* Uses bipartite matching for direct assignment.
* Auxiliary losses on decoder layers improve convergence.
* NMS-free by design (optional in some repos).

### Variants

* **RT-DETR (2023):** Original real-time DETR.
* **RT-DETR v2:** Improved training dynamics and efficiency.
* **Ultralytics RT-DETR:** Production-friendly implementation.

### Strengths

* Excellent global context modeling.
* NMS-free inference.
* Scales well with larger backbones.

### Weaknesses

* Higher compute/memory vs YOLO.
* More complex training dynamics.
* Smaller ecosystem compared to YOLO.

---

## YOLO-NAS

YOLO-NAS applies Neural Architecture Search to discover optimized accuracy–latency architectures tailored for deployment.

### Core Innovations

* Hardware-aware NAS.
* QAT-ready from day one.
* Built-in distillation.

### Key Features

* Search-optimized backbones and heads.
* Strong INT8 performance on edge.
* Integrated into SuperGradients for training.

### Strengths

* Outstanding deployment efficiency.
* Excellent quantization support.
* Strong accuracy for cost.

### Weaknesses

* Architecture is less interpretable.
* Tight integration with specific tooling.
* Search process not reproducible by end users.

---

## Specialized Architectures (Project-Specific)

### D-FINE

Detail-oriented architecture optimized for fine-grained, small-object, or subtle-localization tasks.

**Key Features**

* Enhanced multi-scale fusion.
* Strong detail preservation.
* Precise localization.

### DEIM

Instance-focused model designed for complex, crowded scenes.

**Key Features**

* Enhanced instance representation.
* Strong relational/context modeling.
* Useful for segmentation or tracking pipelines.

### RF-DETR

A NAS‑optimized DETR variant exploring the accuracy–latency frontier using weight sharing.

**Core Innovations**

* Weight-sharing NAS enabling thousands of architecture evaluations.
* Tunable transformer/CNN components.
* Superior domain transfer vs heavy VLM detectors.

**Performance Highlights**

* Nano model: +5.3 AP over D-FINE (nano) at similar latency.
* 2×Large: First real-time detector to surpass 60 AP on COCO.
* 20× faster than GroundingDINO (tiny) on Roboflow100-VL.

**Strengths**

* NMS-free inference (DETR-based set prediction).
* Exceptional accuracy/latency balance.
* Lightweight enough for real-time.
* Strong generalization across domains.

---

## Architecture Comparison Summary

| Family     | Paradigm                  | Anchors | NMS              | Scales   | Typical Deployments | Compute/Memory       |
| ---------- | ------------------------- | ------- | ---------------- | -------- | ------------------- | -------------------- |
| YOLO v4–v7 | Single-stage CNN          | Yes     | Yes              | T–XL     | Edge → Server       | Low → Very High      |
| YOLO v8–11 | Single-stage, anchor-free | No      | Yes (except v10) | T–XL     | Mobile → Server     | Very Low → Very High |
| YOLOv10    | Single-stage              | No      | No               | T–XL     | Edge → Server       | Very Low → Very High |
| RT-DETR    | CNN + Transformer         | N/A     | No               | S–XL     | Edge GPU → Server   | Medium → Very High   |
| YOLO-NAS   | NAS CNN                   | Varies  | Yes              | S–L      | Edge → Server       | Low → High           |
| D-FINE     | Detail-focused CNN        | Varies  | Usually          | T–L      | Precision tasks     | Very Low → High      |
| DEIM       | Instance modeling         | Varies  | Usually          | S–L      | Cluttered scenes    | Low → High           |
| RF-DETR    | NAS DETR                  | N/A     | No               | Nano–2XL | Edge → Server       | Very Low → High      |

---

## Deployment Notes

### Export & Inference

* **YOLO & YOLO-NAS:** Mature ONNX/TensorRT support; many runtimes.
* **RT-DETR:** NMS-free simplifies graph export but transformer ops may need vendor kernels.

### Quantization

* **YOLO-NAS:** Best-in-class QAT support.
* **YOLOv8/11:** Well-supported by vendor toolchains.

### NMS-Free Benefits

* Predictable latency.
* Cleaner end-to-end graphs.
* No threshold tuning.

---

## References

- YOLOv1 (2016): https://arxiv.org/abs/1506.02640  
- YOLOv2/YOLO9000 (2017): https://arxiv.org/abs/1612.08242  
- YOLOv3 (2018): https://arxiv.org/abs/1804.02767  
- YOLOv4 (2020): https://arxiv.org/abs/2004.10934  
- YOLOv5 (2020, repo): https://github.com/ultralytics/yolov5  
- YOLOX (SimOTA, 2021): https://arxiv.org/abs/2107.08430  
- YOLOv6 (2022): https://arxiv.org/abs/2209.02976  
- YOLOv7 (2022): https://arxiv.org/abs/2207.02696  
- YOLOv8 (2023, repo): https://github.com/ultralytics/ultralytics  
- YOLOv9 (2024): https://arxiv.org/abs/2402.13616  
- YOLOv10 (2024): https://arxiv.org/abs/2405.14458  
- Ultralytics YOLO11 (2024, docs): https://docs.ultralytics.com/models/yolo11/  
- RT-DETR (2023): https://arxiv.org/abs/2304.08069  
- RT-DETR (Ultralytics, docs): https://docs.ultralytics.com/models/rtdetr/  
- YOLO-NAS (repo): https://github.com/Deci-AI/super-gradients
- RF-DETR (2025): https://arxiv.org/abs/2511.09554  
- RF-DETR (repo): https://github.com/roboflow/rf-detr  
- RF-DETR (project page): https://rfdetr.roboflow.com/

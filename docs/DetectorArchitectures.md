# Object Detection Architectures Guide

Overview of major detector families used in this project, with trade-offs, links, and deployment notes.

## Table of Contents

- [At-a-Glance Selection](#at-a-glance-selection)
- [YOLO Family](#yolo-family)
  - [Architecture and Core Idea](#architecture-and-core-idea)
  - [Evolution Timeline](#evolution-timeline)
  - [Strengths](#strengths)
  - [Weaknesses](#weaknesses)
- [RT-DETR Family](#rt-detr-family)
  - [Core Concept](#core-concept)
  - [Architecture Components](#architecture-components)
  - [Training and Inference Notes](#training-and-inference-notes)
  - [Variants](#variants)
  - [Strengths](#strengths-1)
  - [Weaknesses](#weaknesses-1)
- [YOLO-NAS](#yolo-nas)
  - [Core Innovation](#core-innovation)
  - [Key Features](#key-features)
  - [Strengths](#strengths-2)
  - [Weaknesses](#weaknesses-2)
- [Specialized Architectures (Project-Specific)](#specialized-architectures-project-specific)
  - [D-FINE](#d-fine)
  - [DEIM](#deim)
  - [RF-DETR](#rf-detr)
- [Architecture Comparison Summary](#architecture-comparison-summary)
- [Deployment Notes](#deployment-notes)
- [References](#references)

---

## At-a-Glance Selection

- Edge real-time (balanced speed/accuracy): YOLOv8/YOLO11 (S/M) or YOLO-NAS (with QAT) 🚀  
- Highest accuracy on server GPUs: RT-DETR variants or transformer-based refinements 🧠  
- NMS-free pipeline: YOLOv10 or RT-DETR ✅  
- Tight memory/latency budget: YOLO-NAS with quantization, or small YOLO models 📦

---

## YOLO Family

YOLO (“You Only Look Once”) is a family of single-stage detectors optimized for speed. Predictions are made densely over feature maps, producing bounding boxes and classes in one forward pass.

### Architecture and Core Idea

- Early YOLO (v1–v2)
  - Grid-based predictions (S×S cells); each cell predicts a fixed number of boxes, objectness, and class probabilities.
  - v2 introduces anchors (dimension clustering), BN, and a stronger backbone.

- Modern YOLO (v3+)
  - Dense predictions on multi-scale feature maps (FPN/PAN-like necks).
  - Anchor-based heads (v2–v7) → anchor-free heads (v8+).
  - Decoupled heads for classification vs box regression are common (v6+).
  - Inference typically uses NMS; YOLOv10 is NMS-free by design.

### Evolution Timeline

- YOLOv1 (2016)
  - Innovation: Single-shot detection paradigm.
  - Architecture: CNN with fully connected detection head.
  - Limitation: Struggles with small objects and precise localization.
  - Paper: You Only Look Once: Unified, Real-Time Object Detection — https://arxiv.org/abs/1506.02640

- YOLOv2 / YOLO9000 (2017)
  - Improvements: Anchor boxes, BN, higher-resolution pretraining.
  - Backbone: Darknet-19.
  - Paper: YOLO9000: Better, Faster, Stronger — https://arxiv.org/abs/1612.08242

- YOLOv3 (2018)
  - Key Changes: Multi-scale predictions (FPN-style), 3 detection layers.
  - Backbone: Darknet-53 with residuals.
  - Loss: BCE/logistic for multi-label classification.
  - Paper: YOLOv3: An Incremental Improvement — https://arxiv.org/abs/1804.02767

- YOLOv4 (2020)
  - Backbone: CSPDarknet53 (better gradient flow).
  - Neck: SPP + PANet.
  - Training: Mosaic, CutMix, DropBlock; Mish activations used in some configs.
  - Paper: YOLOv4: Optimal Speed and Accuracy of Object Detection — https://arxiv.org/abs/2004.10934

- YOLOv5 (2020)
  - Notes: Popular PyTorch implementation (no official paper).
  - Architecture: CSP bottlenecks; AutoAnchor; evolved training pipeline.
  - Historical: Early versions used a “Focus” layer; later removed in favor of standard conv for compatibility.
  - Repo: Ultralytics YOLOv5 — https://github.com/ultralytics/yolov5

- YOLOv6 (2022)
  - Focus: Industrial deployment.
  - Blocks: RepVGG-style reparameterizable convs (e.g., EfficientRep/RepConv).
  - Head: Decoupled classification/localization heads.
  - Label Assignment: SimOTA (from YOLOX — https://arxiv.org/abs/2107.08430).
  - Paper: YOLOv6: A Single-Stage Object Detection Framework for Industrial Applications — https://arxiv.org/abs/2209.02976

- YOLOv7 (2022)
  - E-ELAN: Extended Efficient Layer Aggregation Networks.
  - Training: Trainable bag-of-freebies; optional auxiliary heads for coarse-to-fine learning.
  - Re-parameterization strategies for inference.
  - Paper: YOLOv7 — https://arxiv.org/abs/2207.02696

- YOLOv8 (2023)
  - Anchor-free heads; C2f modules; decoupled head.
  - Unified tasks: detection, segmentation, classification, pose.
  - Commonly uses TaskAlignedAssigner for training.
  - Repo: Ultralytics YOLOv8 — https://github.com/ultralytics/ultralytics

- YOLOv9 (2024)
  - PGI: Programmable Gradient Information to improve gradient flow and training stability.
  - GELAN: Generalized Efficient Layer Aggregation Network for efficiency–accuracy balance.
  - Paper: YOLOv9 — https://arxiv.org/abs/2402.13616

- YOLOv10 (2024)
  - NMS-free: One-to-many + one-to-one dual assignments during training; end-to-end inference.
  - Efficiency: Reduced post-processing latency; real-time optimized.
  - Paper: YOLOv10 — https://arxiv.org/abs/2405.14458

- YOLO11 (2024)
  - Ultralytics release (successor line to YOLOv8; not a peer-reviewed paper).
  - Anchor-free heads, C2f-style components, and training refinements for improved accuracy/speed.
  - Docs/Repo: https://docs.ultralytics.com/models/yolo11/

### Strengths

- Real-time performance on commodity GPUs and many edge devices.
- End-to-end trainable single-stage designs.
- Strong, active ecosystem (repos, pretrained weights, tutorials).
- Flexible: detection, segmentation, classification, and pose support (especially Ultralytics line).

### Weaknesses

- Small/occluded/crowded objects remain challenging for one-stage dense predictors (mitigated by multi-scale features and modern assigners, but not fully solved).
- Pre–YOLOv10 variants rely on NMS; threshold tuning can impact recall/latency and duplicate suppression behavior.
- Class imbalance and long-tail distributions often require careful loss/assigner tuning.

---

## RT-DETR Family

RT-DETR combines efficient CNN backbones with transformer encoders/decoders to achieve real-time, end-to-end detection without NMS.

### Core Concept

- CNN backbone for efficient feature extraction.
- Transformer encoder/decoder with learnable object queries for global reasoning.
- Set-based prediction with bipartite (Hungarian) matching and no NMS at inference.

### Architecture Components

1. Backbone: ResNet-like or other efficient CNN for multi-scale features.
2. Encoder: Multi-scale transformer encoder to aggregate global context.
3. Decoder: Transformer decoder with learnable queries producing object slots.
4. Prediction Heads: Boxes (e.g., L1/GIoU) and classes (CE/Focal), trained end-to-end.

### Training and Inference Notes

- Matching: Hungarian matching aligns predictions with ground truth (set prediction).
- Auxiliary losses on intermediate decoder layers help convergence.
- NMS-free by design; some repos expose optional NMS for convenience/compatibility.

### Variants

- RT-DETR (Original, 2023)
  - Innovation: Real-time DETR with hybrid CNN–Transformer design.
  - Paper: DETRs Beat YOLOs on Real-time Object Detection — https://arxiv.org/abs/2304.08069

- RT-DETR v2
  - Description: Enhanced training strategies and architectural tweaks for better accuracy/throughput.
  - Note: Provide a citation/repo if this is a project-specific variant.

- RT-DETR (Ultralytics)
  - Implementation: Integrated into the Ultralytics framework with a simplified API.
  - Docs: https://docs.ultralytics.com/models/rtdetr/

### Strengths

- Captures long-range dependencies with global attention.
- End-to-end training and inference (NMS-free).
- Scales well across model sizes for different speed/accuracy targets.

### Weaknesses

- Typically higher memory/compute than YOLO on the same hardware budget.
- Training dynamics (matching, auxiliary losses) can be more complex to tune.
- Ecosystem and tooling are newer than classic YOLO pipelines.

---

## YOLO-NAS

YOLO-NAS leverages Neural Architecture Search (NAS) to discover architectures optimized for accuracy–latency trade-offs and deployment constraints.

### Core Innovation

- Automated architecture search to balance accuracy, latency, and memory.
- Quantization-aware design for efficient edge deployment.
- Training strategies often include knowledge distillation.

### Key Features

1. NAS-optimized backbones/necks/heads for target hardware.
2. QAT-ready (Quantization-Aware Training) and export-friendly.
3. Built on Deci’s SuperGradients training library and deployment stack.
4. Production-focused recipes and tools.

### Strengths

- Strong real-world deployment focus with competitive accuracy–latency trade-offs.
- Built-in quantization/compression support.
- Modern training techniques (distillation, QAT) out of the box.

### Weaknesses

- NAS/search itself is resource-intensive (done by provider; not replicated by users).
- Architectures can feel “black box” compared to hand-designed networks.
- Some features are tightly integrated with a specific tooling ecosystem.

Repository: YOLO-NAS by Deci — https://github.com/Deci-AI/super-gradients

---

## Specialized Architectures (Project-Specific)

These architectures are specific to this project. Add internal links/papers where available to help readers dive deeper.

### D-FINE

Fine-grained object detection focused on precise localization and subtle detail differences.

- Key Features
  - Fine-grained detection of small details.
  - High-precision localization.
  - Advanced multi-scale feature fusion.

- Typical Use Cases
  - Medical imaging analysis
  - Manufacturing quality control
  - Scientific image analysis
  - Detail-focused surveillance

### DEIM

Detection with Enhanced Instance Modeling for improved instance-level reasoning.

- Key Features
  - Enhanced instance representations.
  - Context-aware modeling of object relations.
  - Robust detection under cluttered or complex scenes.

- Typical Applications
  - Crowded scenes
  - Instance segmentation workflows
  - Complex scene understanding
  - Multi-object tracking

### RF-DETR

RF-DETR is a lightweight specialist detection transformer that uses weight-sharing Neural Architecture Search (NAS) to discover accuracy-latency Pareto curves for target datasets. Unlike heavy-weight vision-language models, RF-DETR fine-tunes a pre-trained base network and evaluates thousands of network configurations with different accuracy-latency tradeoffs without re-training.

- Core Innovation
  - Weight-sharing NAS for real-time detection transformers.
  - Fine-tunes on target datasets to discover optimal accuracy-latency tradeoffs.
  - Evaluates thousands of configurations without re-training each one.
  - Revisits "tunable knobs" for NAS to improve DETR transferability across diverse domains.

- Key Performance Highlights
  - RF-DETR (nano): 48.0 AP on COCO, beating D-FINE (nano) by 5.3 AP at similar latency.
  - RF-DETR (2x-large): First real-time detector to surpass 60 AP on COCO.
  - Outperforms GroundingDINO (tiny) by 1.2 AP on Roboflow100-VL while running 20× faster.
  - State-of-the-art on both COCO and Roboflow100-VL benchmarks.

- Strengths
  - Exceptional accuracy-latency balance through NAS optimization.
  - Strong generalization to real-world datasets with out-of-distribution classes.
  - Lightweight architecture suitable for real-time deployment.
  - Better domain transfer than heavy VLMs.

- References
  - Paper: RF-DETR: Neural Architecture Search for Real-Time Detection Transformers — https://arxiv.org/abs/2511.09554
  - Code: https://github.com/roboflow/rf-detr
  - Project Page: https://rfdetr.roboflow.com/

---

## Architecture Comparison Summary

| Family | Paradigm | Anchors | NMS at Inference | Scales | Typical Deployment by Scale | Relative Compute/Memory by Scale |
|---|---|---|---|---|---|---|
| YOLO (v4–v7) | Single-stage CNN | Yes | Yes | T/S/M/L/XL | T: mobile/SBC CPU or NPUs, low-power edge<br>S: edge CPU/GPU (e.g., Jetson Nano/Orin Nano)<br>M: edge GPU or midrange desktop GPU<br>L: desktop GPU or small server GPU<br>XL: server GPU | T: very low<br>S: low<br>M: medium<br>L: high<br>XL: very high |
| YOLO (v8–YOLO11) | Single-stage CNN | No (anchor-free) | Yes (YOLOv10 is NMS-free; see below) | T/S/M/L/XL | T: mobile/SBC or edge NPUs with INT8/QAT<br>S: edge CPU/GPU, entry desktop GPU<br>M: strong edge GPU or desktop GPU<br>L: desktop/server GPU<br>XL: server GPU | T: very low<br>S: low<br>M: medium<br>L: high<br>XL: very high |
| YOLOv10 | Single-stage CNN | No | No (end-to-end) | T/S/M/L/XL | T: mobile/edge with hardware accel<br>S: edge GPU, desktop GPU<br>M: strong edge GPU or desktop GPU<br>L/XL: desktop/server GPU with strict E2E latency needs | T: very low<br>S: low<br>M: medium<br>L: high<br>XL: very high |
| RT-DETR | CNN + Transformer | N/A | No (by design) | S/M/L/XL | S: edge GPU (e.g., Jetson Orin-class), desktop GPU<br>M: desktop/server GPU<br>L: server GPU<br>XL: high-memory server GPU | S: medium<br>M: medium–high<br>L: high<br>XL: very high |
| YOLO-NAS | NAS-optimized CNN | Varies | Typically Yes | S/M/L | S: edge CPU/GPU, mobile/edge NPUs with QAT<br>M: edge or desktop GPU<br>L: desktop/server GPU | S: low<br>M: medium<br>L: high |
| D-FINE (project) | Fine-grained CNN | Varies | Usually Yes | T/S/M/L | T/S: edge devices where detail matters<br>M: desktop/edge GPU<br>L: desktop/server GPU | T: very low<br>S: low<br>M: medium<br>L: high |
| DEIM (project) | Instance-focused | Varies | Usually Yes | S/M/L | S: edge GPU, desktop GPU<br>M: desktop/server GPU<br>L: server GPU | S: low–medium<br>M: medium–high<br>L: high |
| RF-DETR | NAS-optimized DETR | N/A | No | Nano/S/M/L/XL/2XL | Nano: mobile/edge devices with hardware accel<br>S/M: edge/desktop GPU<br>L/XL: desktop/server GPU<br>2XL: server GPU (60+ AP COCO) | Nano: very low<br>S: low<br>M: medium<br>L: medium–high<br>XL/2XL: high |

Scale key: T = Tiny, S = Small, M = Medium, L = Large, XL = Extra Large.  
Notes:
- Categories are relative and resolution-dependent. Doubling input width/height roughly quadruples FLOPs and increases memory accordingly.
- Availability of exact scale names varies by repo (e.g., n/s/m/l/x for Ultralytics); mapping here is conceptual.
---

## Deployment Notes

- Export/Inference
  - YOLO families and YOLO-NAS: strong support for ONNX, TensorRT, and various runtimes (e.g., OpenVINO, CoreML) depending on repo/tools.
  - RT-DETR: exportability improving; NMS-free simplifies deployment graphs, but attention blocks may need optimized kernels.

- Quantization
  - YOLO-NAS: QAT-ready by design; good fit for INT8 on edge.
  - YOLO (v8/YOLO11): widely used with post-training quantization or QAT via vendor toolchains.

- NMS-free benefits
  - Eliminates post-processing latency and threshold sensitivity (e.g., YOLOv10, RT-DETR).
  - Can simplify end-to-end pipelines and make latency more predictable.

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

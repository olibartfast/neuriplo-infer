# Plan

1. Run the existing exporter regression and task tests.
2. Export pretrained ResNet50 from cached weights; compute a reference on data/dog.jpg using the same resize.
3. Rebuild the TensorFlow consumer against current local siblings and compare classification.
4. Record exact commands, provenance, results, and deployment limits; update coverage and changelog.

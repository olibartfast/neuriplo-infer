# Validation defined before execution

- Existing tests/test_tensorflow_export.py must pass with Session variable restoration and BGR mean subtraction.
- neuriplo-tasks CTest suite must pass.
- Python pretrained ResNet50 and exported Session must agree on probabilities for bilinear resized dog.jpg.
- C++ tensorflow-classifier must return the same top class and score within 0.01 of the Python reference.
- Record sibling SHAs, runtime version, model hash, command and output. Keep KNOWN-BAD if correctness fails; distinguish local sibling evidence from pinned releases.
- Check documentation links and git diff --check.

## Results — 2026-11-04

All required checks passed. No new task implementation was necessary: the upstream fix already ships in v0.8.5. This branch corrects stale coverage and records evidence.

- Exporter Session regression: 1/1 passed (Python TensorFlow 2.21.0).
- Existing neuriplo-tasks build CTest: 31/31 passed, 17.34 s.
- `cmake --build build-tensorflow -j 4`: passed; local TensorFlow 2.19.0 C++ runtime and OpenCV 5.0.0.
- CLI: exit 0, 2850 ms inference, rendered `data/output/processed_tensorflow-classifier_local.png`.
- Exported Python Session versus Keras reference: maximum logit error 0.0.
- C++ pipeline versus Python reference: top class 249, score difference 0.0002264 (< 0.01). Top-five classes agree: 249, 248, 250, 160, 537.
- Python uses OpenCV bilinear resize; C++ uses the task library's native bilinear resize. Minor score differences include that resize implementation difference; this is not a bitwise parity claim.
- Relevant upstream paths have no diff versus pinned tags: tasks `src/classification`, `include/neuriplo/tasks/classification`, `export/classification/tensorflow`, `tests/test_tensorflow_export.py` versus v0.8.5; neuriplo `backends/libtensorflow` versus v0.11.3.
- The supported-recipe GitHub URL returned HTTP 200. New relative documentation links resolve.

### Provenance

- App base: eb48791ea245854e3e4e39475abae46682e0eceb.
- Local tasks: 1331639d56b6e275649dd9b4bc52df180e5025fa; v0.8.5 resolves to aac606858c28ddab94ab04af39b190e4fd7370e0.
- Local neuriplo: 06b69034750a298f3fe62fc30d145bca9e59b107.
- SavedModel protobuf SHA256: 7d911ef92a113eee4b228b0b568ad70e418ff4fa5f2605f1254c49ca95051a45. This identifies this export, not a deterministic future serialization.
- Model: Keras ResNet50 ImageNet cached weights, classifier_activation=None, corrected export_classifier wrapper. No weight download needed.
- Both sibling working trees were clean. Build cache has local source overrides; no release or remote-fetch gate is claimed.

### Commands

Run from neuriplo-infer unless a subshell changes directory. Replace the Python path with an environment providing TensorFlow 2.21 and NumPy.

```bash
(cd ../neuriplo-tasks && /tmp/opencode/tf221/bin/python tests/test_tensorflow_export.py)
ctest --test-dir ../neuriplo-tasks/build --output-on-failure
/tmp/opencode/tf221/bin/python ../neuriplo-tasks/export/classification/tensorflow/export_tf_saved_model_classifier.py --output /tmp/tf-regate-resnet50
cmake --build build-tensorflow -j 4
TF_NUM_INTRAOP_THREADS=4 TF_NUM_INTEROP_THREADS=2 build-tensorflow/app/neuriplo-infer --type=tensorflow-classifier --weights=/tmp/tf-regate-resnet50 --source=data/dog.jpg --no_display=true --no_gif=true
```

The configure cache uses DEFAULT_BACKEND=LIBTENSORFLOW, TENSORFLOW_DIR=/home/oli/dependencies/tensorflow and FETCHCONTENT_SOURCE_DIR_NEURIPLO / FETCHCONTENT_SOURCE_DIR_NEURIPLO-TASKS pointing to local siblings. Backend SDK installation and a populated build are prerequisites, not provided by this documentation change.

### Python reference

Create `/tmp/tf-regate-bgr.npy` using system Python with cv2 and NumPy:

```python
import cv2, numpy as np
np.save('/tmp/tf-regate-bgr.npy', cv2.resize(cv2.imread('data/dog.jpg'), (224,224)).astype(np.float32)[None])
```

Run the following in the TensorFlow environment:

```python
import json, numpy as np, tensorflow as tf
bgr=np.load('/tmp/tf-regate-bgr.npy')
image=bgr[...,::-1].copy()
model=tf.keras.applications.ResNet50(weights='imagenet',classifier_activation=None)
ref=model(tf.keras.applications.resnet50.preprocess_input(image.copy()),training=False).numpy()
with tf.compat.v1.Session(graph=tf.Graph()) as session:
    meta=tf.compat.v1.saved_model.loader.load(session,['serve'],'/tmp/tf-regate-resnet50')
    sig=meta.signature_def['serving_default']
    out=session.run(next(iter(sig.outputs.values())).name,{next(iter(sig.inputs.values())).name:bgr})
np.testing.assert_allclose(out,ref,rtol=1e-4,atol=1e-4)
p=tf.nn.softmax(out).numpy()[0]
print(json.dumps({'tensorflow':tf.__version__,'top5':[(int(i),float(p[i])) for i in np.argsort(p)[-5:][::-1]],'max_logit_error':float(np.max(np.abs(out-ref)))},indent=2))
```

### Numerical C++ probe

The CLI renders scores without emitting a numerical result schema. This temporary probe uses its existing InferencePipelineBuilder, preprocessing, backend and task decoder to print scores, with the same tensor conversion as CLICommands.cpp:

```cpp
#include "InferencePipeline.hpp"
#include "neuriplo/tasks/core/vision/opencv_adapter.hpp"
#include <opencv2/imgcodecs.hpp>
#include <iostream>
int main() {
 AppConfig c; c.detectorType="tensorflow-classifier"; c.weights="/tmp/tf-regate-resnet50";
 auto p=InferencePipelineBuilder(c).build();
 auto image=cv::imread("/home/oli/repos/neuriplo-infer/data/dog.jpg");
 auto input=p.task->preprocess({neuriplo_tasks::vision::opencv::copyFromCvMat(image)});
 auto raw=p.engine->get_infer_results(input);
 std::vector<neuriplo_tasks::Tensor> tensors;
 auto &[outputs,shapes]=raw;
 for(size_t i=0;i<outputs.size();++i) tensors.emplace_back(outputs[i],shapes[i]);
 auto results=p.task->postprocess({image.cols,image.rows},tensors);
 for(auto &r:results) {auto v=std::get<neuriplo_tasks::Classification>(r); std::cout<<v.class_id<<" "<<v.class_confidence<<"\n";}
}
```

Compile the probe using the `app/main.cpp` entry from `build-tensorflow/compile_commands.json`, replacing its input/output paths with the probe and a temporary object. Link with `build-tensorflow/app/CMakeFiles/neuriplo-infer-cli.dir/link.txt`, replacing the main object and executable destination. Run with TF_NUM_INTRAOP_THREADS=4 and TF_NUM_INTEROP_THREADS=2.

Observed C++ scores:

```text
249 0.532603
248 0.181920
250 0.0803334
160 0.0285540
537 0.0245811
```

## Limits

No app source changed, so unrelated app configurations and its full test suite were not rebuilt. No hosted CI, remote-only build, KServe classification, GPU correctness or latency gate was run. OpenVINO classification remains KNOWN-BAD and was not re-gated here. Older unwrapped SavedModels remain incompatible and require re-export.

### Final documentation checks

- `git diff --check`: passed.
- Generated model inventory check passes against the pinned v0.8.5 README:
  `git -C ../neuriplo-tasks show v0.8.5:README.md > /tmp/tf-regate-tasks-v085-README.md`
  then `python3 scripts/sync_supported_model_types.py --neuriplo-tasks-readme /tmp/tf-regate-tasks-v085-README.md --check`.
- Default generated-doc check cannot find its default build source directory. Checking against local tasks develop instead reports unrelated inventory drift; the pinned README is the appropriate check for this app's current dependency contract. No generated files were changed.
- Render inspected: dog image with class 249 @ 0.532603 and the expected four remaining ranked labels.

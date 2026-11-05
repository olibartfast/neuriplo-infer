# Validation before implementation

- Compare EdgeCrafter ONNX/C++ LiteRT/Python LiteRT on identical preprocessed input; inspect all outputs, not just rendering.
- Same high-confidence labels and scores within 0.01 for replacement artifacts; record box errors and mask agreement where emitted.
- RF-DETR LiteRT presets must use native exporter and dry-run without selecting generic conversion.
- Focused automated tests cover changed routing/export/backend behavior; run owning-repo checks first.
- Re-run app live gates, document runtime/model hashes, commands and limitations. Do not label an unrun transport PASS.
- Legacy incorrect artifacts remain unsupported even when replacement recipes pass.

## Results — 2026-11-04

- Neuriplo LiteRT build and all 77 CTest entries passed after the backend/link-group change. Direct LiteRT test binary: 8 tests passed, one optional `model_path.txt` smoke skipped; the fixture-backed EdgeCrafter regression ran and passed.
- Neuriplo full-tree clang-format and cppcheck scripts passed.
- App LiteRT rebuild, shell syntax and two Python regression tests passed. The regression tests cover both RF-DETR model families and both LiteRT/ONNX preset selections.
- Generated model inventory passes against the pinned tasks v0.8.5 README. No generated model types changed.
- All relative links in changed documentation resolve; seven GitHub URLs returned HTTP 200.
- EdgeCrafter ONNX and LiteRT received byte-identical NCHW images and INT64 sizes. Default Python builtin and default C++ paths reproduce deflated scores; Python XNNPACK and explicit C++ XNNPACK subgraph reshaping match ORT.
- Patched app local gate: dog 0.929990, bicycle 0.927981, car 0.780462, motorbike 0.697765, truck 0.439418, cup 0.427723. Maximum sorted score error over 300 queries: 4.4815242e-6. Six high-confidence labels and box coordinates agree (boxes within 0.01 pixel). Two low-score query ties exchange labels/boxes; no all-query ordered-box claim is made.
- YOLOv8 LiteRT regression passed (445 ms); render inspected with dog 0.93, bicycle 0.89, car 0.61.
- EdgeCrafter local CLI: 3005 ms inference, render inspected. KServe HTTP: 17139 ms end to end, render produced; a separate binary HTTP request returned all tensors exactly equal to the local LiteRT tensors. Server metrics: two successful inference requests, zero failures.
- RF-DETR native medium versus ORT: same selected labels and scores; maximum normalized box error 0.00025934, logit error 0.00164986.
- RF-DETR native seg medium versus ORT: maximum normalized box error 0.00029513, logit error 0.00686646, mask-logit maximum error 0.209843 (mean 0.00079385), mask sign agreement 0.9999991426611797. These are numerical tolerance checks, not bitwise equivalence.
- Same `rtdetrul` ONNX on current ORT and OpenVINO: outputs [1,300,6], maximum raw difference 1.0550022e-5. The current tasks decoder already handles the normalized center-size/class-ID output; no new tasks code was needed.
- Legacy generic RF-DETR LiteRT artifacts remain KNOWN-BAD. The supported replacement is the native LiteRT export, now wired into both presets.

### Provenance and exact artifacts

See `docs/evidence/known-bad-regate-2026-11-04.json` for base commits, artifact hashes and numerical comparisons. Libraries are built against local siblings. The neuriplo fix is commit `9bde6c0fd07ab88b70e1076f5fc667c20be41d7d` on `feature/litert-dynamic-xnnpack`; app work is on `feature/known-bad-backends`. No released pins moved. All validation is local and must not be described as an isolated remote-fetch release gate.

The original two untracked version override files in the app were left untouched. No task-source changes were required, and its working tree remains clean.

### Owning-repo verification commands

```bash
cd ../neuriplo
./scripts/quality/format.sh --check
./scripts/quality/cppcheck.sh
cmake --build build-litert -j 4
NEURIPLO_LITERT_PARITY_FIXTURE=/tmp/kb-ecdet-fixture ctest --test-dir build-litert --output-on-failure
NEURIPLO_LITERT_PARITY_FIXTURE=/tmp/kb-ecdet-fixture build-litert/backends/litert/test/LiteRTInferTest
```

Fixture generation uses the diagnostic probe below: capture ORT app inputs and outputs, then place the reconverted TFLite as `model.tflite`, captured inputs as `input0.bin`/`input1.bin`, and float32 reference outputs as `output0.bin` through `output2.bin`. No weights or captured tensors are committed.

### App commands

```bash
cmake --build build-litert -j 4
python3 scripts/test_export_rfdetr_litert.py
bash -n docker_run_inference_e2e_example.sh
bash docker_run_inference_e2e_example.sh --preset rfdetr --backend litert --neuriplo-tasks-dir ../neuriplo-tasks --dry-run
bash docker_run_inference_e2e_example.sh --preset rfdetrseg --backend litert --neuriplo-tasks-dir ../neuriplo-tasks --dry-run
bash docker_run_inference_e2e_example.sh --preset edgecrafter_det --backend litert --neuriplo-tasks-dir ../neuriplo-tasks --dry-run
build-litert/app/neuriplo-infer --type=ecdet --weights=/tmp/opencode/ecdet-fix/ecdet_orig_reconvert/model_float32.tflite --source=data/dog.jpg --no_display=true --no_gif=true --min_confidence=0.4
```

Runtime prerequisite: LiteRT SDK TF 2.19.0 with XNNPACK and matching headers. App build cache uses local sibling overrides and OpenCV 5.0.0. The EdgeCrafter artifact is an existing `onnx2tf 2.6.9` reconversion; that conversion itself was not repeated during this work.

### Live serving commands

The existing real-litert runtime binary was used without changing its source. Its library path deliberately loads the patched app-build libneuriplo. Stop this temporary server after verification.

```bash
LD_LIBRARY_PATH=/home/oli/repos/neuriplo-infer/build-litert/_deps/neuriplo-build:/home/oli/dependencies/litert/lib ../neuriplo-kserve-runtime/build/real-litert/neuriplo-kserve-runtime --host 127.0.0.1 --port 18088 --backend litert --model-name ecdet_fixed --model-path /tmp/opencode/ecdet-fix/ecdet_orig_reconvert/model_float32.tflite
build-litert/app/neuriplo-infer --type=ecdet --source=data/dog.jpg --kserve_endpoint=http://127.0.0.1:18088 --kserve_model_name=ecdet_fixed --kserve_transport=http --no_display=true --no_gif=true --min_confidence=0.4
curl -s http://127.0.0.1:18088/metrics
```

A first client attempt omitted the explicit HTTP transport and hit the HTTP port using its default gRPC mode; it was corrected using `--kserve_transport=http`. No gRPC correctness claim is made.

### Numerical probe source

Save as `/tmp/knownbad-probe.cpp`; compile it against each app build with the following driver. The probe uses the same task and tensor conversion contract as CLICommands.cpp.

```cpp
#include "InferencePipeline.hpp"
#include "neuriplo/tasks/core/vision/opencv_adapter.hpp"
#include <opencv2/imgcodecs.hpp>
#include <fstream>
#include <iostream>
int main(int argc,char** argv) {
 if(argc<4) return 2;
 AppConfig c; c.detectorType=argv[1]; c.weights=argv[2]; c.confidenceThreshold=0.4f;
 auto p=InferencePipelineBuilder(c).build();
 auto image=cv::imread("/home/oli/repos/neuriplo-infer/data/dog.jpg");
 auto input=p.task->preprocess({neuriplo_tasks::vision::opencv::copyFromCvMat(image)});
 for(size_t i=0;i<input.size();++i) {std::ofstream f(std::string(argv[3])+"-in"+std::to_string(i)+".bin",std::ios::binary); f.write(reinterpret_cast<const char*>(input[i].data()),input[i].size());}
 auto [outputs,shapes]=p.engine->get_infer_results(input);
 std::vector<neuriplo_tasks::Tensor> tensors;
 for(size_t i=0;i<outputs.size();++i) {
  tensors.emplace_back(outputs[i],shapes[i]);
  std::ofstream f(std::string(argv[3])+"-out"+std::to_string(i)+".bin",std::ios::binary);
  std::cout<<"OUTPUT "<<i; for(auto d:shapes[i]) std::cout<<" "<<d; std::cout<<"\n";
  for(auto &v:outputs[i]) {float x=std::visit([](auto x){return static_cast<float>(x);},v); f.write(reinterpret_cast<const char*>(&x),sizeof(x));}
 }
 auto results=p.task->postprocess({image.cols,image.rows},tensors);
 for(auto &r:results) if(auto v=std::get_if<neuriplo_tasks::Detection>(&r)) std::cout<<v->class_id<<" "<<v->class_confidence<<"\n";
}
```

Save as `/tmp/build-knownbad-probe.py`:

```python
import json,shlex,subprocess,sys
from pathlib import Path
build=Path(sys.argv[1]).resolve()
e=next(x for x in json.load(open(build/'compile_commands.json')) if x['file'].endswith('/app/main.cpp'))
a=shlex.split(e['command']);a[a.index('-o')+1]='/tmp/knownbad-'+build.name+'.o';a[-1]='/tmp/knownbad-probe.cpp';subprocess.run(a,cwd=e['directory'],check=True)
a=shlex.split((build/'app/CMakeFiles/neuriplo-infer-cli.dir/link.txt').read_text());a[a.index('CMakeFiles/neuriplo-infer-cli.dir/main.cpp.o')]='/tmp/knownbad-'+build.name+'.o';a[a.index('-o')+1]='/tmp/knownbad-'+build.name;subprocess.run(a,cwd=build/'app',check=True)
```

Example invocations:

```bash
python3 /tmp/build-knownbad-probe.py build-ort
python3 /tmp/build-knownbad-probe.py build-litert
python3 /tmp/build-knownbad-probe.py build-openvino
/tmp/knownbad-build-ort ecdet /home/oli/model_repository/ecdet_s_onnx/1/model.onnx /tmp/kb-ecdet-ort
/tmp/knownbad-build-litert ecdet /tmp/opencode/ecdet-fix/ecdet_orig_reconvert/model_float32.tflite /tmp/kb-ecdet-fixed
/tmp/knownbad-build-ort rtdetrul /home/oli/repos/neuriplo-ui/artifacts/installer/workspace/catalog/rtdetrul.onnx /tmp/kb-rtdetrul-ort
/tmp/knownbad-build-openvino rtdetrul /home/oli/repos/neuriplo-ui/artifacts/installer/workspace/catalog/rtdetrul.onnx /tmp/kb-rtdetrul-openvino
/tmp/knownbad-build-ort rfdetr /tmp/opencode/rfdetr-torch-test/onnx/rfdetr-medium.onnx /tmp/kb-rfdetr-ort
/tmp/knownbad-build-litert rfdetr /tmp/opencode/litert-native/rfdetr-medium.tflite /tmp/kb-rfdetr-native
/tmp/knownbad-build-ort rfdetrseg /tmp/opencode/rfdetr-torch-test/onnx/rfdetrseg-medium.onnx /tmp/kb-rfdetrseg-ort
/tmp/knownbad-build-litert rfdetrseg /tmp/opencode/litert-native/rfdetr-seg-medium.tflite /tmp/kb-rfdetrseg-native
```

## Delivery limits

At validation time no release, pin update, remote CI, gRPC gate or GPU gate was performed. The maintainer subsequently authorized commit and push of both feature branches. The old generic RF-DETR converter route is still numerically unsupported: this change supplies and tests the native replacement instead of patching the third-party converter. Old EdgeCrafter artifacts also require reconversion. The app's pinned released neuriplo does not contain this local backend patch yet.

## Fresh native exports and final checks

Both commands below completed successfully using rfdetr 1.11.2 and cached pretrained weights under `~/.roboflow/models` (no download needed):

```bash
/tmp/opencode/rfdetr-torch-export/bin/python scripts/export_rfdetr_litert.py --task detection --output /tmp/kb-native-export/rfdetr-medium.tflite
/tmp/opencode/rfdetr-torch-export/bin/python scripts/export_rfdetr_litert.py --task segmentation --output /tmp/kb-native-export/rfdetrseg-medium.tflite
```

Both freshly exported models were run through the patched C++ pipeline and compared to the same ONNX references. Their raw errors exactly match the native-artifact error figures above; fresh-export hashes are in the evidence JSON. Fresh segmentation CLI completed in 3564 ms and rendered successfully. These are real export/inference checks in addition to the mocked API/preset regression tests.

Workflow YAML parses. `act push --job dry-run-presets --matrix preset:rfdetr --dryrun -W .github/workflows/e2e-example-emulation.yml` passes; this is workflow dry-run resolution, not a hosted or full container CI run. Python syntax checks and both repository diffs pass whitespace checks. The temporary HTTP server was stopped after verification.

The requested matrix refresh adds a dedicated `rtdetrul` row and marks unreleased EdgeCrafter PASS cells with a backend-commit footnote. OpenVINO `rtdetrul` CLI re-gate completed in 910 ms; its rendered dog/bicycle/car/truck boxes were inspected. The archived TensorFlow-classifier OpenVINO result is explicitly not re-gated by the separate LIBTENSORFLOW check.

# Validation defined before implementation

V1/R1: Preserve baseline logs and executable/model hashes for each blocked cell.
V2/R2: Run owning-repo regression tests, build affected app backends, and compare detections/poses/masks with established ORT results on identical source images. Reject silent dtype coercion or backend substitution.
V3/R3: Record sbatch IDs, node, start/end and sacct exit states; verify at least two independent jobs overlap. Use explicit CPU/container memory/time limits because cluster nodes advertise placeholder memory.
V4/R4: Check documentation links and diff; distinguish unreleased source changes, SDK experiments, numerical passes and remaining upstream limitations.

## Discovery

Local Slurm client cannot communicate with controller 25.11.2; submit via SSH to the existing controller. No cluster configuration changes required. Existing three workers are idle at discovery.

## Results — local correctness complete; Slurm parallel rerun blocked

All five originally BLOCKED table cells have a local passing route, covering
six model/backend cases. docs/evidence/blocked-slurm-2026-11-04.json records
identical-input hashes, output errors, model/library/executable hashes and SDK
provenance. RT-DETRv4 requires the tasks export adapter and --type=rtdetr.
Original versus prepared ORT maxima: labels 0, boxes 0.000091553 pixels,
scores 9.536743e-7. The original artifact's vector MatMul is still unsupported
in OpenCV. Legacy formats and unrelated transport/ABI blockers stay visible.

Slurm baseline jobs 39/40 reproduce DNN and all three ExecuTorch failures.
They ran sequentially; no parallel overlap is claimed. Preflight jobs 36–38
are excluded (incomplete staging / incorrect CLI option); baseline v4 used
the unsupported --type=rtdetrv4 alias, corrected to --type=rtdetr in the
prepared manifest. Accounting is disabled; captured controller output is at
/tmp/blocked-slurm-20261104/baseline-control.txt.

The existing controller became unreachable before repaired submissions.
Tailnet status reports the controller offline; the OptiPlex worker remains
reachable. No cluster configuration was changed and no replacement cluster
was started. Parallel repaired tests remain PENDING, not passed.

### Validation performed

- neuriplo: OpenCV 4.6 80/80, OpenCV 5 80/80, ExecuTorch 77/77 CTest entries.
- ExecuTorch real fixture regression ran, not skipped; every nano output
  element agrees with ORT within 0.001.
- App: 191/191 CTest entries on OpenCV 5 and ExecuTorch.
- Tasks: 31/31 CTest entries; export adapter 3 Python tests.
- Slurm runner: 4 real-process tests (success, nonzero exit, timeout, missing
  artifact). It records RUNTIME_PASS only, never numerical PASS.
- Source formatting and cppcheck passed. Generated model inventory matches
  the pinned tasks v0.8.5 README. No task types or release pins changed.
- RT-DETR/OWLv2/RF-DETR renders inspected; numeric errors are in the JSON.

### Exact build commands

```bash
cmake -S /tmp/blocked-executorch-src/executorch -B /tmp/blocked-et-build -GNinja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/tmp/blocked-et-sdk \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON \
  -DEXECUTORCH_BUILD_XNNPACK=ON -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
  -DEXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP=ON -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
  -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON -DEXECUTORCH_BUILD_TESTS=OFF \
  -DEXECUTORCH_BUILD_EXAMPLES=OFF \
  -DPYTHON_EXECUTABLE=/tmp/opencode/rfdetr-torch-export/bin/python
cmake --build /tmp/blocked-et-build -j 12
cmake --install /tmp/blocked-et-build
cmake -S . -B build-executorch-optimized -DDEFAULT_BACKEND=EXECUTORCH \
  -DEXECUTORCH_DIR=/tmp/blocked-et-sdk -DEXECUTORCH_VERSION=v1.2.0 \
  -DFETCHCONTENT_SOURCE_DIR_NEURIPLO=/home/oli/repos/neuriplo \
  -DFETCHCONTENT_SOURCE_DIR_NEURIPLO-TASKS=/home/oli/repos/neuriplo-tasks \
  -DENABLE_APP_TESTS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-executorch-optimized -j 4
ctest --test-dir build-executorch-optimized --output-on-failure
cmake --build build-dnn5 -j 3
ctest --test-dir build-dnn5 --output-on-failure
```

The isolated SDK is pinned source v1.2.0; Python codegen used the existing
RF-DETR export environment (PyTorch 2.14.0+cpu). SDK logging defaults OFF in
this validation build; the setup script explicitly enables logging. Tests
inherit SDK compile definitions so either setting is consistent.

### Resume the requested concurrent Slurm gate

Local staging root: /tmp/blocked-slurm-20261104. Worker staging root:
/home/oli/slurm-blocked-20261104. Manifest cases.json selects DNN on OptiPlex
and ExecuTorch on the controller worker, alternating submissions. Scripts
and fixtures must exist at matching paths on both workers. OptiPlex has the
updated DNN capsule; the controller needs the updated ExecuTorch capsule and
runner/manifest once it returns. Copy the current slurm_backend_matrix.py,
cases.json and executorch/{neuriplo-infer,lib,data/person.jpg} from local staging.

On the restored controller:

```bash
python3 /home/oli/slurm-blocked-20261104/slurm_backend_matrix.py \
  --manifest /home/oli/slurm-blocked-20261104/cases.json \
  --output /home/oli/slurm-blocked-20261104/repaired --partition cpu --cpus 2 --minutes 20
squeue
scontrol show job JOB_ID
```

Retrieve results from each worker, validate artifact hashes and rendered task
results, and verify at least one cross-backend start/end interval overlaps
before changing the pending Slurm status. This step has not been performed.

Local Slurm client is 23.11.4; existing controller/worker Slurm is 25.11.2.
CMake is 3.28.3, clang-format 18.1.3. Person.jpg pose rendering matches ORT
(score 0.850784); primary raw keypoint max error is 0.000644684. The manifest
uses person.jpg for pose. Paired segmentation files give bicycle 0.943308,
horse 0.916267 and car 0.801956 on both runtimes; archived E13 artifact
confidence numbers are not reused as this gate's reference.

Source fixes are published as neuriplo 042bcefc8a20771d7c40d54c86f8ea588fcde1b8
and neuriplo-tasks d38e8b3984386c93a053215a1a88964e6e054e74. No merge, release
or pin move was performed. The exact pending manifest is also tracked as
docs/evidence/blocked-slurm-plan-2026-11-04.json, independent of temporary files.

# Application validation — PASS on tested development builds

Date: 2026-11-07. All learned-model executions below use the neuriplo-infer
CLI, with task preprocessing, backend inference, task decoding and rendering.
Machine-readable hashes and scope are in [evidence.json](evidence.json).

## Results

| Pair, score > 0.2 | ORT CPU | TRT FP16 |
| --- | --- | --- |
| United States Capitol | 77 | 74 |
| Sacré-Cœur | 45 | 49 |
| Blank black/white pair (edge case) | 0 | 0 |

All runs return success and save pair PNGs. Inspected the actual application
renders on both backends. The two source images retain their original
coordinates and dimensions; the shorter image is padded below. Different
ORT/TRT counts are expected numerical differences, not a parity claim.

## Checks

- neuriplo-tasks: strict-warning build, 31/31 CTest suites PASS, including
  zero-match decoder regression.
- Local application: build and 149/149 CTest tests PASS. Tests cover pair
  rejection before inference, one engine call per pair, sample reporting,
  routing, renderer target offset, strict confidence threshold, capabilities.
- Jetson application: 149/149 CTest tests PASS (final Slurm job 65).
- Jetson neuriplo backend: 76/76 CTest tests PASS (final Slurm job 65).
  TensorRT tests include actual GPU execution for NonZero output [M,1],
  counts 0/2/4/1/0, concrete runtime shapes, metadata [-1,1], typed/raw values,
  public context recreation and successful/failed reload. Static TensorRT
  tests use a synthetic ReduceMean+broadcast fixture with input [1,3,224,224]
  and output [1,1000], solely to exercise the existing static contracts.
- clang-format 18 and cppcheck; generated supported model list checked
  against the pinned tasks README. Relative and GitHub links checked.
- GPT-6.1-sol code review: all concrete findings addressed.

## Reproduction

Build and run commands: [CLI guide](../../docs/KeypointMatching.md).
Set `ONNX_RUNTIME_DIR` to the installed ORT SDK as needed. Tested ORT 1.19.2
CPU provider, app `feature/lightglue-app-e2e`, tasks snapshot `1331639` plus
empty-match fix, neuriplo snapshot `ff575af` plus data-dependent-output fix.

ORT uses `--weights=superpoint_lightglue_pipeline.onnx --batch=2
--input_sizes=1,512,512 --min_confidence=0.2 --use-gpu=false` on
zenbook-ux425uaz. Both pairs were dispatched by Slurm job 60.

TRT uses the same CLI shape, sources and threshold with `--use-gpu=true`
and `--weights=/tmp/superpoint_lightglue_pipeline-56.engine` on Jetson Orin.
Engine: TensorRT 10.16.2 FP16, built by Slurm job 56. Job 65 executes final
backend tests, both image pairs, app tests and zero-match pair. Slurm 25.11.2
runs in Docker on the Jetson; its batch command invokes the host ARM app over
SSH using the worker host's existing identity and known-hosts file. Host GPU
execution is restricted to CPUs 0,1 with taskset.

Local artifacts: `/tmp/keypoint-matching/app-e2e/ort/`,
`/tmp/keypoint-matching/app-e2e/trt/`,
`/tmp/keypoint-matching/showcase/keypoint-matching-showcase.mp4`.
The 32-second English video uses these real app PNGs, not raw-runtime outputs.

These are validated development builds. No release pin or release tag moved.

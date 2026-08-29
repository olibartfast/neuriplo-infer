# Mission

## Problem

Running a computer-vision model locally means writing the same glue every time:
decode the source, preprocess for one specific runtime, call one specific
backend API, decode the raw tensors, draw the result. That glue is rewritten per
model, per backend, and per task, and it is where most of the breakage lives.

`neuriplo-infer` removes that glue from the user's side of the problem. It is the
application layer of the `vision-stack` cluster: one CLI that takes a model type,
a weights file (or a remote endpoint), and a source, and runs the task end to end.

## Audience

- Engineers evaluating a CV model against a real image, video, or stream before
  committing it to a product.
- Engineers who need the *same* task pre/postprocessing whether the model runs
  locally or behind a KServe V2 server.
- The sibling repos in the cluster, which use this app as the reference consumer
  of their contracts.

It is a developer tool, not a product surface. There is no GUI here;
[neuriplo-ui](https://github.com/olibartfast/neuriplo-ui) consumes the
`--capabilities` contract for that.

## Promise

- **One CLI across tasks.** Detection, open-vocabulary detection, classification,
  segmentation, pose, depth, optical flow, video classification, and VLM image
  understanding are reached through the same flags and the same execution flow.
- **Switchable backends.** OpenCV DNN, ONNX Runtime, TensorRT, LibTorch, OpenVINO,
  and LibTensorFlow are a configure-time choice, not a code change.
- **Local and remote are the same pipeline.** In KServe mode the transport moves;
  task preprocessing, postprocessing, and rendering stay in this app unless the
  server is explicitly asked to take them (`--input_mode`, `--postprocess_mode`).
- **Reproducible builds.** Every sibling dependency is pinned to a concrete tag,
  so checking out a release tag rebuilds the same code.
- **Honest capability reporting.** `--capabilities` emits a machine-readable
  contract validated against [`docs/capabilities.schema.json`](../docs/capabilities.schema.json);
  what it advertises is what the binary can actually route.

## Success criteria

A change is good for the mission when, after it:

- The CLI contract still holds for existing invocations, or the break is
  explicitly reviewed and recorded in `CHANGELOG.md`.
- A new model type is reachable end to end — routed, run, rendered, tested —
  not merely present in the upstream inventory.
- `--capabilities` output still validates and still describes reality.
- The E2E presets in `docker_run_inference_e2e_example.sh` still run.
- `master`, `develop`, and the release tag can converge on one commit.

## Tone

Technical, terse, and literal. Documentation states what the code does and what
it does not; it does not sell. Error messages name the offending flag and the
way out. Generated documentation is generated, never hand-polished.

## Assumptions to confirm

This file was reconstructed from `README.md`, `AGENTS.md`, `CHANGELOG.md`, and the
build files rather than from a product brief. Treat these as drafts until the
maintainer confirms them:

- The primary audience is model-evaluation engineering, not production serving —
  production serving is expected to move to a KServe deployment with this app as
  the client.
- Windows support is a non-goal rather than deferred work (see
  [`tech-stack.md`](tech-stack.md#explicit-non-choices)).

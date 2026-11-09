# TPU via KServe Integration Evidence

> Coverage owner: [docs/BackendCoverage.md](../../docs/BackendCoverage.md) — this README documents the runner; coverage claims live there.

Validation status: dry-run scaffold only; no live GPU/TPU result is claimed.

## Purpose

Exercise the remote-TPU path: `neuriplo-infer` keeps preprocessing,
postprocessing, and rendering locally and sends tensors to an
operator-supplied TPU serving process over the existing KServe V2 client
(`--kserve_endpoint`, `--kserve_model_name`, `--kserve_transport`). The
server owns device placement; the client selects a model name and endpoint,
not a TPU device. There is no in-process TPU/PJRT backend in this repo.

## Prerequisites (live only)

- A Cloud TPU VM (or any host) serving a classifier over KServe V2 HTTP,
  e.g. `neuriplo-kserve-runtime` or a custom TPU server exposing
  `/v2/health/live`, `/v2/health/ready`, and `/v2/models/<name>/infer`.
- A `neuriplo-infer` binary built with `NEURIPLO_INFER_ENABLE_KSERVE=ON`
  (the default).
- The same input image and labels used for the CPU baseline
  (`data/dog.jpg`, `labels/imagenet_labels.txt`).

## Commands

```bash
# Dry-run (no network, no hardware): validates the scaffold files exist and
# prints the exact live command that would run.
python3 integration-tests/tpu-kserve/run.py --dry-run

# Live (needs the TPU endpoint + binaries); the numeric check compares the
# served top-5 classes/scores against baselines/cpu_top5.json when populated.
python3 integration-tests/tpu-kserve/run.py --live \
  --endpoint http://<tpu-host>:8000 \
  --model-name <served-model> \
  --infer-bin <path-to>/neuriplo-infer \
  --source data/dog.jpg \
  --labels labels/imagenet_labels.txt
```

## Baselines

- `baselines/cpu_top5.json`: CPU reference top-5 classes/scores for the
  representative input. Values are `null` until a measured run is recorded;
  the keys stay present so validation can rely on a stable shape.
- `labels/imagenet_labels.txt` is operator-supplied and not shipped in this
  repo. Provide the same labels file used for the CPU baseline when running
  `--live`; the default `--labels` path assumes the operator has placed it.
- `data/dog.jpg` is the representative source image.

## Served top-5 for numeric comparison

- `--served-top5 <json>` shape: `{"classes": [...], "scores": [...]}` with five
  entries each, in rank order.
- Produce it from the served run's top-5 output, e.g. by saving the served
  classes/scores to that JSON file.
- When both the CPU baseline (`classes`/`scores` non-null) and `--served-top5`
  are available, `--live` enforces an exact class match and a score tolerance
  (`score_tolerance`, default `1e-4`); otherwise the numeric step is skipped
  with a message.
- `evidence.yaml`: attestation record. The live check stays `skipped` until
  a TPU run is attested with its endpoint, image, command, and results.

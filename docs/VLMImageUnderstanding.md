# VLM Image Understanding

End-to-end guide for running vision-language models (VLMs) via the llama.cpp backend.
Covers Gemma4 as the reference model, but the pattern applies to any GGUF-format VLM
supported by llama.cpp's `libmtmd`.

---

## Prerequisites

### 1. llama.cpp shared libraries

Build or install llama.cpp so that the following shared libraries are available:

```
libllama.so
libggml.so
libggml-base.so
libggml-cpu.so
libmtmd.so      ← multimodal projector library (required for image input)
```

The default install prefix used by this project's Dockerfile is `/usr/local/lib`.
For local development the path is typically `$HOME/dependencies/llamacpp/lib`.

Build with:

```bash
cmake -B build -DLLAMA_BUILD_TESTS=OFF -DGGML_NATIVE=OFF -DLLAMA_BUILD_EXAMPLES=ON
cmake --build build --config Release -j$(nproc)
cmake --install build --prefix $HOME/dependencies/llamacpp
```

### 2. Model files

| File | Size | Description |
|------|------|-------------|
| `gemma-4-E4B-it-Q4_K_M.gguf` | ~5 GB | Language model weights |
| `mmproj-gemma4-E4B-F16.gguf` | ~944 MB | Vision projector (SigLIP encoder) |

Download via Hugging Face Hub or use the preset in `docker_run_inference_e2e_example.sh`.

### 3. Build vision-inference with LLAMACPP backend

```bash
cmake -B build-llamacpp \
  -DDEFAULT_BACKEND=LLAMACPP \
  -DLLAMACPP_DIR=$HOME/dependencies/llamacpp \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-llamacpp -j$(nproc)
```

---

## CLI Usage

```bash
LD_LIBRARY_PATH=/path/to/llamacpp/lib \
  build-llamacpp/app/vision-inference \
  --type=gemma4 \
  --weights=/path/to/gemma-4-E4B-it-Q4_K_M.gguf \
  --mmproj=/path/to/mmproj-gemma4-E4B-F16.gguf \
  --source=data/dog.jpg \
  --prompt="Describe what you see in this image."
```

### Key flags

| Flag | Description |
|------|-------------|
| `--type=gemma4` | Selects the `ImageUnderstanding` task in vision-core |
| `--weights` | Path to the language model GGUF |
| `--mmproj` | Path to the vision projector GGUF (enables multimodal mode) |
| `--source` | Path to the input image (`jpg`, `png`); omit for text-only mode |
| `--prompt` | Freeform text prompt sent to the model |

### Text-only mode

Omit `--source` (or provide an empty path) to run without an image:

```bash
build-llamacpp/app/vision-inference \
  --type=gemma4 \
  --weights=/path/to/model.gguf \
  --prompt="What is 2+2?"
```

### Without the projector (pure LLM)

Omit `--mmproj` entirely. The `--source` flag is then ignored and only text inference runs.

---

## Docker Usage

### Build the image

```bash
docker build \
  --build-arg LLAMACPP_VERSION=b9085 \
  -f docker/Dockerfile.llamacpp \
  -t vision-inference-llamacpp:latest \
  .
```

The Dockerfile downloads model GGUFs from Hugging Face during the build
(set `HF_TOKEN` as a build secret if the repo is gated).

### Run via preset

```bash
./docker_run_inference_e2e_example.sh --preset gemma4
```

This runs inside the container with a pre-downloaded `dog.jpg` sample image and
the default prompt `"Describe what you see in this image."`.

### Run manually inside the container

```bash
docker run --rm \
  -v /path/to/models:/models:ro \
  -v /path/to/images:/data:ro \
  vision-inference-llamacpp:latest \
  --type=gemma4 \
  --weights=/models/gemma-4-E4B-it-Q4_K_M.gguf \
  --mmproj=/models/mmproj-gemma4-E4B-F16.gguf \
  --source=/data/dog.jpg \
  --prompt="What breed is this dog?"
```

---

## How It Works

1. **CLI layer** (`CommandLineParser`) parses `--mmproj` into `AppConfig.mmprojectPath`.
2. **Pipeline wiring** (`InferencePipelineBuilder`) concatenates the projector path into the engine weights string:
   `model.gguf|mmproj=/path/to/mmproj.gguf`
3. **LlamaCppInfer** constructor splits on `|mmproj=`, loads the language model, then
   calls `mtmd_init_from_file()` with the projector path.
4. **Preprocessing** (vision-core `ImageUnderstandingTask::preprocess`):
   - Tensor 0: UTF-8 prompt bytes
   - Tensor 1 (when image provided): `[4B width LE][4B height LE][H×W×3 RGB bytes]`
5. **Inference dispatch** (`LlamaCppInfer::get_infer_results`):
   - Two tensors + projector loaded → `infer_multimodal()`
   - Otherwise → `infer_text_only()`
6. **Multimodal inference** (`infer_multimodal`):
   - Constructs prompt using Gemma4's native token strings (see Troubleshooting)
   - `mtmd_tokenize()` merges text tokens and image embeddings
   - `mtmd_helper_eval_chunks()` evaluates the combined sequence
   - `autoregressiveGenerate()` samples tokens until EOG or 512-token limit
7. **Response** is returned as a UTF-8 string and printed by `ResultRenderer`.

---

## Tested Results

| Image | Prompt | Response (truncated) |
|-------|--------|----------------------|
| `dog.jpg` | "Describe what you see in this image." | "The image shows a dog, appearing to be a golden retriever or similar breed..." |
| `horses.jpg` | "How many animals are in this photo?" | "There are two horses in the photo..." |
| `bus.jpg` | "What text can you read on this vehicle?" | "The text reads 'School District'..." |

---

## Troubleshooting

See [neuriplo/docs/TROUBLESHOOTING.md](../../../neuriplo/docs/TROUBLESHOOTING.md) for:

- **Empty/silent response from multimodal inference** — Gemma4 chat template mismatch
- **`n_ctx` too small** — image tokens require at least 8192 context length
- **`libmtmd.so` not found** — check `LD_LIBRARY_PATH` and `LinkBackend.cmake` mtmd linkage
- **`mtmd_init_from_file` returns NULL** — projector GGUF path wrong or file corrupted

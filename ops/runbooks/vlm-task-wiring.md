# Runbook: Wiring a New VLM Task Across the Vision Stack

This runbook describes how a vision-language model (VLM) task is threaded through the
three-repo vision stack: neuriplo-tasks → neuriplo → neuriplo-infer. Use it as a template
when adding support for a new VLM or multimodal task type.

Reference implementation: **Gemma4 ImageUnderstanding** (commit `c55daf3` and preceding).

---

## Stack Overview

```
neuriplo-infer   (application layer — CLI, routing, NeuriploInferApp)
      │
      ▼
  neuriplo          (backend orchestration — InferenceInterface, LlamaCppInfer)
      │
      ▼
  neuriplo-tasks       (task contracts — TaskFactory, preprocess/postprocess)
```

Each repo has a well-defined seam. Changes cross all three in a bottom-up order
(neuriplo-tasks first, neuriplo second, neuriplo-infer last).

---

## 1. neuriplo-tasks: Define the Task Contract

### 1a. Register the task type

In `neuriplo-tasks`, add the new type to `TaskType` enum and `TaskFactory`:

```cpp
// include/neuriplo_tasks/TaskType.hpp
enum class TaskType { ..., ImageUnderstanding, ... };

// src/TaskFactory.cpp
if (type == "gemma4" || type == "imageunderstanding") {
    return std::make_unique<ImageUnderstandingTask>(model_info, task_config);
}
```

### 1b. Implement preprocess() — the two-tensor encoding contract

For VLM tasks, `preprocess()` must produce **two** `std::vector<uint8_t>` tensors:

| Index | Content |
|-------|---------|
| 0 | UTF-8 prompt bytes (no null terminator) |
| 1 | `[uint32_t width LE][uint32_t height LE][H×W×3 RGB bytes]` |

When no image is available, return only tensor 0 (text-only fallback).

```cpp
std::vector<std::vector<uint8_t>> ImageUnderstandingTask::preprocess(
    const std::vector<cv::Mat>& imgs) {

    std::vector<uint8_t> prompt_bytes(prompt_.begin(), prompt_.end());

    if (imgs.empty() || imgs[0].empty()) {
        return {std::move(prompt_bytes)};
    }

    cv::Mat rgb;
    cv::cvtColor(imgs[0], rgb, cv::COLOR_BGR2RGB);
    const uint32_t nx = static_cast<uint32_t>(rgb.cols);
    const uint32_t ny = static_cast<uint32_t>(rgb.rows);

    std::vector<uint8_t> image_bytes(8 + static_cast<size_t>(nx) * ny * 3);
    std::memcpy(image_bytes.data() + 0, &nx, 4);
    std::memcpy(image_bytes.data() + 4, &ny, 4);
    // copy row-by-row to handle non-contiguous Mats
    for (uint32_t row = 0; row < ny; ++row) {
        std::memcpy(image_bytes.data() + 8 + row * nx * 3,
                    rgb.ptr(static_cast<int>(row)), nx * 3);
    }
    return {std::move(prompt_bytes), std::move(image_bytes)};
}
```

**Why raw RGB instead of JPEG?** Lossless, no decompression overhead on the backend side.
The 8-byte header is cheap to parse with `std::memcpy`.

### 1c. Implement postprocess()

For text generation tasks, the backend returns UTF-8 bytes encoded as `float` values
(one byte per element). Decode them:

```cpp
auto results = task->postprocess(cv::Size{0,0}, tensors);
// tensors[0] elements are float(uint8_t(c)) — cast back to string
```

---

## 2. neuriplo: Implement the Backend Dispatch

### 2a. Parse the mmproj path from model_path

The `InferenceInterface` signature is `(model_path, use_gpu, batch_size, input_sizes)`.
To pass a projector path without changing the signature, embed it with a separator:

```
model.gguf|mmproj=/path/to/mmproj.gguf
```

Parse in the constructor:

```cpp
const std::string sep = "|mmproj=";
const auto sep_pos = model_path.find(sep);
if (sep_pos != std::string::npos) {
    actual_model_path = model_path.substr(0, sep_pos);
    mmproj_path       = model_path.substr(sep_pos + sep.size());
}
```

### 2b. Load the multimodal projector

After the language model and context are initialized:

```cpp
if (!mmproj_path.empty()) {
    mtmd_context_params mparams = mtmd_context_params_default();
    mparams.use_gpu       = use_gpu;
    mparams.print_timings = false;
    mparams.n_threads     = 4;
    mparams.warmup        = false;
    ctx_mtmd_ = mtmd_init_from_file(mmproj_path.c_str(), model_, mparams);
    if (!ctx_mtmd_) throw std::runtime_error("Failed to load mmproj: " + mmproj_path);
}
```

### 2c. Dispatch in get_infer_results()

```cpp
if (input_tensors.size() >= 2 && !input_tensors[1].empty() && ctx_mtmd_) {
    return infer_multimodal(raw_prompt, input_tensors[1]);
}
return infer_text_only(raw_prompt);
```

### 2d. infer_multimodal() — critical: Gemma4 native token format

**Do NOT use `llama_chat_apply_template` for the multimodal path.** Gemma4's 16KB Jinja2
template is not supported by llama.cpp's minimal C template engine. Construct the prompt
using Gemma4's native special-token strings instead:

```cpp
const std::string marker = mtmd_default_marker();  // "<__media__>" or similar
const std::string formatted_prompt =
    "<bos><|turn>user\n" + marker + "\n" + raw_prompt + "<turn|>\n<|turn>model\n";

mtmd_input_text input_text{formatted_prompt.c_str(), /*add_special=*/true, /*parse_special=*/true};
```

With `parse_special=true`, llama.cpp tokenizes `<bos>` → 2, `<|turn>` → 105, `<turn|>` → 106.
Token 106 (`<turn|>`) is also the EOG token — the model stops generation at the correct point.

For other models, find the equivalent turn/BOS tokens in the model's vocabulary and
construct the prompt string accordingly.

### 2e. Evaluate and generate

```cpp
llama_pos n_past = 0;
mtmd_helper_eval_chunks(ctx_mtmd_, ctx_llama_, chunks.ptr.get(),
                        /*n_past=*/0, /*seq_id=*/0, /*n_batch=*/512,
                        /*logits_last=*/true, &n_past);
std::string response = autoregressiveGenerate(n_past);
```

### 2f. Context size

Set `n_ctx` to at least **8192**. Gemma4 produces ~266 image tokens; with the text
prompt and generation budget, 2048 is too small.

```cpp
ctx_params.n_ctx = 8192;
```

### 2g. Link libmtmd

In `cmake/LinkBackend.cmake`:

```cmake
find_library(VI_MTMD_LIB NAMES mtmd PATHS ${LLAMACPP_DIR}/lib NO_DEFAULT_PATH)
foreach(_lib VI_LLAMA_LIB VI_GGML_LIB VI_GGML_BASE_LIB VI_GGML_CPU_LIB VI_MTMD_LIB)
    if(${_lib})
        target_link_libraries(${PROJECT_NAME} PRIVATE "${${_lib}}")
    endif()
endforeach()
```

---

## 3. neuriplo-infer: Plumb CLI → NeuriploInferApp → Task

### 3a. AppConfig

Add a field for the projector path:

```cpp
// app/inc/AppConfig.hpp
std::string mmprojectPath;
```

### 3b. CommandLineParser

```cpp
// In params string:
"{ mmproj | | path to multimodal projector GGUF for VLM image inference }"

// In parseCommandLineArguments():
config.mmprojectPath = parser.get<std::string>("mmproj");

// In validateArguments():
const std::string mmproj = parser.get<std::string>("mmproj");
if (!mmproj.empty() && !isFile(mmproj)) {
    LOG(ERROR) << "Multimodal projector file " << mmproj << " doesn't exist";
    std::exit(1);
}
```

### 3c. NeuriploInferApp constructor

Embed the projector path before calling `setup_inference_engine`:

```cpp
std::string engine_weights = config.weights;
if (!config.mmprojectPath.empty()) {
    engine_weights += "|mmproj=" + config.mmprojectPath;
}
engine = setup_inference_engine(engine_weights, use_gpu, config.batch_size, config.input_sizes);
```

### 3d. NeuriploInferApp::run() routing

Add the `ImageUnderstanding` early-return before the image/video dispatch:

```cpp
if (getTaskType(config.detectorType) == neuriplo_tasks::TaskType::ImageUnderstanding) {
    processImageUnderstanding();
    return;
}
```

### 3e. processImageUnderstanding()

Load the source image (if provided) and pass it to `task->preprocess()`:

```cpp
std::vector<cv::Mat> images;
if (!config.sources.empty() && !config.sources[0].empty()) {
    cv::Mat img = cv::imread(config.sources[0]);
    if (!img.empty()) images.push_back(std::move(img));
}
const auto preprocessed = task->preprocess(images);
const auto [outputs, shapes] = engine->get_infer_results(preprocessed);
```

The source validation in `CommandLineParser::validateArguments` must allow empty sources
for text-only task types. Guard it:

```cpp
const bool is_text_task = (normalizedType == "gemma4" || normalizedType == "imageunderstanding"
                           || normalizedType == "llamacpp" /* add new types here */);
if (source.empty() && !is_text_task) {
    LOG(ERROR) << "Cannot open video stream"; std::exit(1);
}
```

---

## 4. Checklist for a New VLM Task

- [ ] neuriplo-tasks: `TaskType` enum entry
- [ ] neuriplo-tasks: `TaskFactory` registration (normalize name variants)
- [ ] neuriplo-tasks: `preprocess()` returns 2-tensor encoding when image provided
- [ ] neuriplo-tasks: `postprocess()` decodes float-encoded bytes to text
- [ ] neuriplo: `LlamaCppInfer` constructor parses `|mmproj=` separator
- [ ] neuriplo: `mtmd_init_from_file()` called with parsed path
- [ ] neuriplo: `get_infer_results()` dispatches on tensor count + `ctx_mtmd_`
- [ ] neuriplo: multimodal prompt uses model's native token strings (not `llama_chat_apply_template`)
- [ ] neuriplo: `n_ctx ≥ 8192`
- [ ] neuriplo: `libmtmd` linked in `cmake/LinkBackend.cmake`
- [ ] neuriplo-infer: `AppConfig.mmprojectPath` field
- [ ] neuriplo-infer: `--mmproj` CLI flag + validation
- [ ] neuriplo-infer: `engine_weights += "|mmproj=..."` in NeuriploInferApp constructor
- [ ] neuriplo-infer: `processImageUnderstanding()` loads image from sources
- [ ] neuriplo-infer: `validateArguments` allows empty source for text-only task types
- [ ] neuriplo-infer: `getTaskType()` / `normalizeModelType()` maps new name variants

---

## Reference Files

| File | Purpose |
|------|---------|
| `neuriplo-tasks/src/image_understanding/image_understanding_task.cpp` | Two-tensor preprocess impl |
| `neuriplo/backends/llamacpp/src/LlamaCppInfer.cpp` | mtmd dispatch and Gemma4 prompt format |
| `neuriplo/backends/llamacpp/src/LlamaCppInfer.hpp` | `ctx_mtmd_` member, method declarations |
| `neuriplo-infer/app/inc/AppConfig.hpp` | `mmprojectPath` field |
| `neuriplo-infer/app/src/CommandLineParser.cpp` | `--mmproj` flag and validation |
| `neuriplo-infer/app/src/NeuriploInferApp.cpp` | Path embedding and task setup |
| `neuriplo-infer/app/src/NeuriploInferProcessing.cpp` | `processImageUnderstanding()` |
| `neuriplo-infer/cmake/LinkBackend.cmake` | `libmtmd` linkage |
| `neuriplo-infer/docker/Dockerfile.llamacpp` | Multi-stage build with mtmd |

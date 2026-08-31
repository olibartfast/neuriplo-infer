# Feature Requirements — OpenCV-free application layer

Roadmap phase: [Candidates → *OpenCV-free application layer*](../roadmap.md)
Branch: `feature/opencv-free-app`

## Goal

`neuriplo-infer` builds, runs, renders, and writes results with no OpenCV of its
own, so OpenCV becomes an internal detail of a single neuriplo backend
(`OPENCV_DNN`) rather than a system dependency of the application. A TensorRT,
ONNX Runtime, LibTorch, or KServe-only build then configures, links, and ships
without OpenCV present at all.

## Why this is achievable now

Three of the four repos in the cluster already made this move; only the
application still opts in:

| Repo | State | Evidence |
|---|---|---|
| `neuriplo` | **Done.** OpenCV is requested only by the `OPENCV_DNN` backend, linked `PRIVATE` | the single `find_package(OpenCV REQUIRED)` sits inside the `OPENCV_DNN` branch of `cmake/LinkBackend.cmake`, with a comment stating exactly this intent |
| `neuriplo-tasks` | **Done.** OpenCV is an optional adapter, default OFF | `NEURIPLO_TASKS_WITH_OPENCV` defaults `OFF`; dependency-free `Image`/`ImageView`, stb I/O, and an `image_ops` layer whose comment reads *"Replaces cv::resize"* |
| `videocapture` | **Done in v0.4.0.** `cv::Mat` removed from the public frame API | `readFrame()` fills a `videocapture::Frame`; OpenCV is confined to the OpenCV capture backend |
| `neuriplo-infer` | **Outstanding.** `find_package(OpenCV REQUIRED)` is unconditional at `CMakeLists.txt:82` | a TensorRT build drags in OpenCV that neither sibling asked for |

The inference contract is already OpenCV-free: `InferenceInterface::get_infer_results`
takes `std::vector<std::vector<uint8_t>>`. Nothing about the inference path
requires this work — it is entirely an application-layer concern.

## Scale of the change

~140 `cv::` uses in `app/`, very unevenly distributed:

| Bucket | Sites | Replacement | Notes |
|---|---|---|---|
| Rendering | ~70 | new `vision::draw` in `neuriplo-tasks` | the substance of this feature |
| Image I/O (`imread`/`imwrite`) | 9 | `loadImage` / `saveImage` | exists in tasks |
| Pixel ops (`resize`, `normalize`, `convertTo`) | ~8 | `image_ops`, `Image::convertTo` | exists; needs `Nearest` |
| `cv::CommandLineParser` | 4 | hand-rolled parser | contract-guarded by existing tests |
| `cv::cuda` GPU probe | 3 | delete | filesystem fallbacks already exist below it |
| `cv::imencode(".jpg")` | 1 | `encodeImage` | **gap** — tasks has decode, not encode |
| Display (`imshow`/`waitKey`) | 4 | SDL2 behind `NEURIPLO_INFER_WITH_DISPLAY` | default OFF; see Decision 3 |

## In Scope

- Every `cv::` use in `app/`, including `app/test/`.
- The application's image type: `cv::Mat` → `neuriplo_tasks::Image` through
  `AppConfig`, `CLICommands`, `InferencePipeline`, `ResultRenderer`, `utils`.
- A primitive rasterizer in `neuriplo-tasks` (`vision::draw`) covering exactly
  what `ResultRenderer` and `utils::draw_label` use: `line`, `rectangle`
  (stroked and filled), `circle` (filled), closed `polyline`, masked fill,
  alpha blend, `TURBO` colormap, and text.
- Text as vendored Hershey vector glyphs with `drawText` / `measureText`.
- Two small `neuriplo-tasks` additions: `Interpolation::Nearest` and
  `encodeImage` (encode to memory).
- CMake: remove the app's unconditional `find_package(OpenCV REQUIRED)` and its
  `neuriplo-tasks::vision-opencv` link.
- Dockerfiles: drop `libopencv-dev` from every image except the `OPENCV_DNN` one.
- `specs/tech-stack.md`, `README.md`, `CHANGELOG.md`, `../roadmap.md`.

## Out of Scope

- **Rendering policy.** Colors, stroke thicknesses, the COCO skeleton topology,
  label placement, and the `std::rand()` mask colouring are reproduced exactly
  as they are. Changing how output *looks* while also changing what draws it
  would make any visual regression impossible to attribute. Improvements go in a
  later phase.
- **`neuriplo`.** Already correct; not touched.
- **`neuriplo-track`, `tritonic`, `object-detection-inference`.** They can adopt
  `vision::draw` once it exists; that is their own work.
- **GPU-accelerated drawing.** The rasterizer is scalar CPU code.
- **Removing OpenCV from the `OPENCV_DNN` image.** That backend *is* OpenCV;
  it keeps it, `PRIVATE`, via `neuriplo`. This feature makes that the only
  place it appears.

## Decisions

1. **The rasterizer lives in `neuriplo-tasks`, not in the app.** `Image` and
   `image_ops` already live there, and `image_ops` exists precisely to replace
   `cv::` functions. A rasterizer over `Image` written in the app would invert
   the layering and be unavailable to `neuriplo-track` and `tritonic`, which
   draw the same overlays. Rendering *policy* stays in the app's
   `ResultRenderer`; `vision::draw` knows nothing about `Result` types.

2. **Text uses Hershey vector fonts, not stb_truetype or a bundled TTF.**
   `FONT_HERSHEY_SIMPLEX` and `FONT_HERSHEY_DUPLEX` — the only two fonts this
   app uses — *are* the public-domain Hershey fonts, whose glyphs are polylines.
   Rendering them reuses the polyline rasterizer needed anyway for
   `cv::polylines`, reproduces the same glyph shapes and the same `getTextSize`
   advance metrics, and ships no font file, no font loader, and no runtime font
   lookup. A TTF path would add a parser, a rasterizer, a file to install, and a
   different-looking result. Glyph data is vendored under `3rdparty/`, as stb
   already is, with its provenance notice.

3. **The live preview is kept, backed by SDL2, behind
   `NEURIPLO_INFER_WITH_DISPLAY` (default OFF).** `cv::imshow` / `cv::waitKey`
   is the only OpenCV use with no in-family replacement, and the maintainer
   chose to keep the capability rather than lose it: SDL2 hands over a window,
   an event loop, and `SDL_UpdateTexture` + `SDL_RenderCopy`, and it is the
   choice `videocapture` v0.4.0 already made for the same problem.

   This is a deliberate, approved exception to *"No new third-party runtime
   dependency without maintainer approval"* in `specs/tech-stack.md`. It is
   narrow on purpose: the flag defaults OFF, so no headless or serving image
   acquires SDL2 or anything it pulls, and the exception is recorded here rather
   than left implicit in a CMake option.

   Measured before deciding, installed-size delta with `--no-install-recommends`,
   on `ubuntu:24.04` and on `nvcr.io/nvidia/cuda:13.0.0-devel-ubuntu24.04` (the
   TensorRT base). The two agree within 1 MB, so the CUDA image carries no GL of
   its own:

   | Option | Added | What dominates |
   |---|---:|---|
   | Raw X11 (`libx11-6` + `libxext6`) | +4 MB | nothing — no GL at all |
   | GLFW · sokol via GLX · `libgl1` alone | +225 MB | `libllvm20` 137 MB + `mesa-libgallium` 42 MB |
   | sokol via GLES3/EGL | +224 MB | `libegl1` + `libgles2` |
   | **SDL2 — chosen** | **+237 MB** | the same GL stack; SDL2 itself is 1.9 MB |
   | OpenCV highgui | +433 MB | Qt5 — Ubuntu builds highgui against Qt5, not GTK |
   | `libopencv-dev`, installed today | +1088 MB | the full OpenCV stack |

   SDL2 is 4.6x lighter than the highgui it replaces and 4.6x lighter again than
   what the Dockerfiles install today. It is on record that 225 of its 237 MB is
   a Mesa/LLVM GL stack a BGR blit never uses, and that raw X11 `XShmPutImage`
   would be +4 MB — the maintainer weighed that against ~150 lines of
   hand-written X11 window and event code and chose the library. Because the
   flag defaults OFF, the difference is paid only by someone who asked for a
   preview.

   **sokol was evaluated and rejected**, recorded so it is not revisited from
   scratch. It is a vendored single header under a permissive license, adding no
   package and fitting the pattern stb and the Hershey data already fit — but it
   needs a GL context, so it pays GLFW's +225 MB rather than nothing; its Linux
   window and GL code is, by its own header, *"taken from GLFW"*; and
   `sokol_app` is an application wrapper that supplies the entry point and
   drives the program through `init_cb` / `frame_cb`, inverting control away
   from a CLI that owns its own frame loop. Displaying a frame would also mean a
   texture upload and a fullscreen quad. It is GLFW's runtime cost, more code
   than SDL2, plus a header to maintain.

4. **The replacement is validated against OpenCV, while OpenCV is still there.**
   `neuriplo-tasks` already builds both ways (`build-ocv` / `build-no-ocv`), so
   each primitive gets a test that draws the same shape with OpenCV and with
   `vision::draw` and asserts the difference is within tolerance. OpenCV becomes
   a *test-only* dependency of tasks and then leaves the app entirely. This is
   what turns "hand-rolled rasterizer" from a risk into a measured claim.

5. **Phased so that every group ships green with OpenCV still linked.** The
   `find_package` removal is the last step, not the first. During the type
   migration a zero-copy `Image` → `cv::Mat` view keeps the untouched renderer
   working, so no group leaves the tree broken.

## Constraints and Context

- **`specs/tech-stack.md` states "OpenCV ≥ 4.6 and glog are system
  dependencies."** This feature changes that rule, so tech-stack.md is updated
  in this branch — the file's own instruction.
- **"No new third-party runtime dependency without maintainer approval."**
  SDL2 is that dependency, and Decision 3 records the approval and its bounds:
  optional, default OFF, display only. Nothing else is added — Hershey glyph
  data is vendored public-domain data, not a dependency.
- **"No silent CLI breakage."** Any change to the live preview is a reviewed
  contract change with a `CHANGELOG.md` entry.
- **The `--capabilities` contract is unaffected.** Neither `Capabilities.cpp`
  nor `docs/capabilities.schema.json` mentions OpenCV, so no `schema_version`
  bump. The existing `capabilities_cli_contract` and
  `capabilities_schema_contract` tests become the safety net for the CLI parser
  rewrite.
- **Deleting the `cv::cuda` probe loses nothing.** `getGPUModels()` and
  `hasNvidiaGPU()` already fall through to `/proc/driver/nvidia/gpus/*/information`,
  which reports model names. Stock Ubuntu OpenCV is built without CUDA, so
  `getCudaEnabledDeviceCount()` already returns 0 and the fallback already does
  all the work on every machine that is not running a custom OpenCV.
- **Cross-repo sequencing.** `neuriplo-tasks` must release `vision::draw`,
  `Interpolation::Nearest`, and `encodeImage` before this app change can be
  released, since `versions.env` pins concrete tags
  (`scripts/validate_release_pins.sh` blocks a branch pin). Development can
  resolve the local sibling checkout; the release cannot.
- **Depends on `feature/videocapture-0.4.0`.** That branch adds
  `FrameConversion` (`Frame` → `cv::Mat`); this feature deletes it in favour of
  `Frame` → `Image`. Landing them in the other order means writing the bridge
  twice.

## Open Questions

1. **What pixel tolerance counts as parity** for the anti-aliased primitives?
   Exact match with OpenCV is not a goal and not achievable — OpenCV's AA uses
   its own fixed-point coverage model. Proposal: assert per-pixel |Δ| ≤ 8 on
   ≥ 99% of pixels for AA strokes and exact match for un-AA fills, plus layout
   invariants for text rather than pixel equality.

2. **Does `neuriplo-tasks::vision-opencv` survive?** After this, the app is its
   only consumer. Keeping it costs nothing and helps external adopters; removing
   it deletes the last `cv::Mat` interop in the family. Not urgent either way.

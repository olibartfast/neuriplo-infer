# Feature Plan — OpenCV-free application layer

Derived from [`requirements.md`](requirements.md). Groups are in dependency
order. Every group ends with the tree building and `ctest` green — OpenCV stays
linked until Group 7, so no group leaves a broken intermediate state.

Groups 1 and 5–8 are `neuriplo-tasks` and `neuriplo-infer` respectively; Group 1
must be released and pinned before this branch can be released, though local
sibling resolution unblocks development immediately.

## Group 1 — `neuriplo-tasks`: the drawing layer

Delivered in `neuriplo-tasks` as `neuriplo_tasks::vision::draw`, beside
`vision::ops`. It operates on `Image` and knows nothing of `Result` types.

1. **Fill the two API gaps.** Add `Interpolation::Nearest` to `vision::ops`
   (the mask overlay needs `INTER_NEAREST`; the enum currently has Linear, Area,
   Cubic only). Add `encodeImage(const Image&, Format, std::vector<uint8_t>&)`
   to `stb_io`, via `stbi_write_jpg_to_func` / `stbi_write_png_to_func` —
   `stb_image_write.h` is already vendored, and only `loadImage`/`decodeImage`/
   `saveImage` are exposed today.

2. **Primitive rasterizer.** `drawLine`, `drawRectangle` (stroked and filled),
   `drawCircle` (filled), `drawPolyline` (open and closed), each taking a colour,
   a thickness, and an anti-alias flag. Thick strokes are rendered as the union
   of the segment's offset outline so joins do not gap — matching what
   `cv::polylines` with `thickness=2` produces. Anti-aliasing uses per-pixel
   coverage; the un-AA path is Bresenham.

3. **Composite operations.** `blend(dst, src, alpha)` for the mask overlay
   (`cv::addWeighted(image, 1, colorMask, 0.7, 0, image)`), `fillMasked(dst,
   mask, colour)` for `Mat::setTo(colour, mask)`, and `applyColormap(Image&,
   Colormap::Turbo)` from a vendored 256-entry LUT.

4. **Hershey text.** Vendor the SIMPLEX and DUPLEX glyph tables for ASCII 32–126
   under `3rdparty/hershey/` with their provenance notice, then implement
   `drawText(Image&, text, origin, scale, colour, thickness)` and
   `measureText(text, scale, thickness) -> {Size, baseline}` on top of the
   polyline rasterizer from step 2. `measureText` must reproduce
   `cv::getTextSize`'s contract, because `utils::draw_label` sizes its filled
   background rectangle from it — a wrong width there is a visibly clipped label,
   not a subtle shading difference.

5. **Differential tests against OpenCV.** For each primitive, draw the same
   shape into an `Image` via `vision::draw` and into a `cv::Mat` via OpenCV, and
   assert agreement within the tolerance fixed in `validation.md`. These build
   only when `NEURIPLO_TASKS_WITH_OPENCV=ON`, so they run in `build-ocv` and are
   absent from `build-no-ocv` — OpenCV becomes a test-only dependency of tasks
   and never a runtime one.

## Group 2 — `neuriplo-infer`: mechanical replacements

OpenCV stays linked throughout this group; each task is independently revertible.

6. **Image I/O.** `cv::imread` → `loadImage`, `cv::imwrite` → `saveImage`,
   `cv::imencode(".jpg", …)` → `encodeImage` (the KServe encoded-image path).
   Preserve the existing failure behaviour, including the fallback-path retry in
   `CLICommands.cpp`.

7. **Pixel ops.** `cv::resize` → `ops::resize`, `cv::normalize(NORM_MINMAX)` →
   `ops::minMax` plus `Image::convertTo`, `cv::cvtColor` → `ops::swapBgrRgb`
   where it is a channel swap.

8. **Delete the `cv::cuda` probes** in `utils.cpp`, keeping the filesystem
   fallbacks that already sit below them. Confirm `getGPUModels()` still reports
   model names on a GPU host.

## Group 3 — `neuriplo-infer`: the image type

9. **Migrate `cv::Mat` → `neuriplo_tasks::Image`** through the app's own
   signatures: `AppConfig`, `CLICommands`, `InferencePipeline::renderResults`,
   `utils`, and the `ResultRenderer` interface. To keep this group mechanical,
   `ResultRenderer` keeps its OpenCV body behind a **zero-copy** `Image` →
   `cv::Mat` view (`cv::Mat(h, w, CV_8UC3, image.data(), image.stride())`,
   valid because both are packed 8-bit BGR). Group 4 deletes that view.

10. **Drop `FrameConversion`.** With the app on `Image`, the videocapture v0.4.0
    bridge becomes `Frame` → `Image` — the same aliasing argument, one fewer
    type. `toTaskImage`/`copyFromCvMat` disappear along with it, since frames
    arrive as `Image` already.

## Group 4 — `neuriplo-infer`: the renderer

11. **Rewrite `ResultRenderer` on `vision::draw`**, one task type at a time —
    detection, open-vocab detection, classification, video classification,
    instance segmentation (mask *and* polygon paths), optical flow, pose,
    depth, image understanding. Colours, thicknesses, font scales, the skeleton
    edge list, and the `std::rand()` colour draw are reproduced exactly; see
    *Out of Scope* in `requirements.md`.

12. **Rewrite `utils::draw_label`** on `measureText` + filled rectangle + text.

13. **Delete the temporary `cv::Mat` view** from step 9. At this point `app/`
    contains no `cv::` outside the display calls.

## Group 5 — `neuriplo-infer`: the CLI parser

14. **Replace `cv::CommandLineParser`** with a small parser preserving the
    current surface: `--key=value`, `-k value`, defaults, `has()`, typed `get()`,
    and the generated help text. The four call sites are thin; the 429 lines
    around them are validation that does not change. `test_parseCommandLineArguments`,
    `capabilities_cli_contract`, and `capabilities_schema_contract` are the
    regression net — run them before and after and diff `--help` output byte for
    byte.

## Group 6 — the display decision

15. **Implement the option the maintainer picks** for `cv::imshow` /
    `cv::waitKey` (Open Question 1 in `requirements.md`). Whichever it is, it
    carries a `CHANGELOG.md` entry, because the video loops' observable
    behaviour changes. This group is the only one that cannot start before that
    answer.

## Group 7 — flip the dependency

16. **Remove `find_package(OpenCV REQUIRED)`** from `CMakeLists.txt`, drop the
    `neuriplo-tasks::vision-opencv` link and `${OpenCV_LIBS}` / `${OpenCV_INCLUDE_DIRS}`
    from `app/CMakeLists.txt` and `app/test/CMakeLists.txt`, and leave
    `NEURIPLO_TASKS_WITH_OPENCV` at its default `OFF`.

17. **Slim the images.** Drop `libopencv-dev` from every `docker/Dockerfile.*`
    except the `OPENCV_DNN` one, which keeps it because `neuriplo` requires it
    for that backend. Record the image size delta.

## Group 8 — contract and documentation

18. Update `specs/tech-stack.md` — the "OpenCV ≥ 4.6 and glog are system
    dependencies" line is what this feature falsifies, and that file requires
    amendment in the same branch as the change. Update `README.md` build
    prerequisites, `CHANGELOG.md`, and the phase status in `../roadmap.md`.

19. Execute [`validation.md`](validation.md) from the file, not from memory.

> Commit by group — this is higher-risk work touching every rendering path.

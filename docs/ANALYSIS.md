# 독립 인페인팅 아키텍처 분석

## 의존성 경계
```
PySide6 GUI (Python)       [MIT proprietary-authored application]
  -> queued Job/Mask       [Python, data validation + hashing]
  -> QThread / BatchRunner [Python, cancel/progress]
  -> SpatialTemporalInpaint [OpenCV C++ backend called by Python]
      -> mask ROI / Telea [spatial, mandatory]
      -> Farneback+warp   [temporal, optional]
  -> FFmpeg rawvideo pipe  [external command, installed separately]
      -> libx264 / NVENC  [CPU or NVIDIA encoder]
      -> audio from input [AAC transcode]
      -> atomic target publish
```
GPU **inference** is NOT implemented; NVENC only speeds video encoding. Application is
independent of JASNA and requires no JASNA executable/activation/model.

## Key correctness invariants
1. Only selected mask pixels are modified by the compute engine, verified with bitwise equality tests.
2. Existing output files are not overwritten; same-volume hard-link publication is atomic/no-clobber.
3. Each job streams frames, retains at most the current and previous frames and ROI optical flow.
4. CPU calculation uses OpenCV C++ routines instead of Python pixel loops.
5. The GUI never runs decode/restoration/encoding on the UI thread.
6. Cancelling stops between frames and deletes uncommitted staged results.
7. FFmpeg subprocess uses argv (never `shell=True`); output lives in temporary same-volume file.
8. Full-frame mask rejected; a wholly unknown frame cannot be spatially reconstructed from neighbors.
9. A single fixed rectangle/image mask per job: moving masks and automatic detection are not supported.
10. Variable frame rate inputs are currently converted to constant reported FPS. Verify timing before production use.

## Design trade-offs
- Telea fast mode favors low memory, predictable compute, and relatively small masks;
  it cannot reconstruct plausible large hidden textures or structures.
- Temporal mode aligns previous restored result with Farneback optical flow and blends only inside
  the mask. A coarse scene-change guard skips temporal blending across large visible changes.
  Optical flow over strongly occluded/mosaicked regions is unreliable and may smear content.
- Current optional GPU use: FFmpeg `h264_nvenc`, only if installed GPU/FFmpeg support it.
- Automated mask detection, transformer-based inpainting, trained LTX inference, HIP/ROCm,
  LTX diffusion conditioning, frame-accurate VFR, scene-cut detector remain unimplemented.

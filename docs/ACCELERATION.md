# v0.6.0 — TensorRT / FP8 / CUDA Graph / NVDEC–NVENC

## Scope and verification status

This implementation accelerates the **independently trained LTX_GUI compact VideoVAE + DiT model**. It does **not** make the model structurally or numerically compatible with the official LTX-2.5 22B weights. Per user request, official-22B weight compatibility was omitted. Neither Jasna nor its runtime/license/model components are imported.

| Feature | Implemented | Executed on NVIDIA GPU in this development environment |
|---|---|---|
| Full-coverage TensorRT compiler gate for VAE encoder, DiT, decoder | Yes | **No** |
| FP8 calibration, ModelOpt Q/DQ path, Hopper preflight | Yes | **No** |
| Shape-fixed CUDA Graph of VAE encoding → denoising → VAE decoding → pixel composition | Yes | **No** |
| PyNvVideoCodec GPU NVDEC DLPack → PyTorch inference → GPU NVENC | Yes (static mask, CFR assumption) | **No** |
| Real GPU median/p95/peak-VRAM correctness benchmark | Runner | **No measurements** |
| Paired Jasna and LTX_GUI reference quality comparison | Existing runner | **No comparable output supplied** |

Python/PyTorch CPU tests confirm option validation, paths, source-compatibility, error handling and deterministic CPU inference. No GPU execution, TRT conversion success, FP8 quality equivalence, end-to-end NVCodec output or measured speedups should be inferred from CPU tests.

## Runtime requirements

- Separate compatible NVIDIA driver, CUDA-enabled PyTorch and `torch-tensorrt`/TensorRT. Installation instructions vary by driver, GPU and OS. Check the current [Torch-TensorRT instructions](https://docs.pytorch.org/TensorRT/).
- For FP8: an NVIDIA GPU of **Hopper compute capability 9.0 or newer**, NVIDIA ModelOpt (`nvidia-modelopt`), and real representative input clips. FP8 is not a generic `.to(torch.float8)` operation. ModelOpt quantization modifies **copies** of the trained modules, preserving the loaded checkpoint.
- For GPU video I/O: compatible NVIDIA [PyNvVideoCodec](https://docs.nvidia.com/video-technologies/pynvvideocodec/), NVDEC/NVENC, FFmpeg and a constant-frame-rate H.264-capable source. Encoder uses GPU ABGR buffers and only H.264 is supported in this path.
- For CUDA Graph: fixed padded `[B,3,T,H,W]` shapes (T=8k+1, H/W multiples of 32), enough VRAM for fixed allocations, and supported GPU kernels. CUDA Graph captures **inference only**; NVDEC/NVENC hardware engines and muxer cannot be captured inside PyTorch CUDA Graph.

## TensorRT full conversion

`ltx_gui/acceleration.py` makes three **separate** compile requests with `require_full_compilation=True`, disabling custom RMSNorm dispatch during export. If any one cannot be compiled completely, it reports an error and the normal caller does not publish a partial output. There is no automatic unreported PyTorch fallback for an explicit TRT selection. This is a strict **attempt**, not a claim that every current model operator is supported by TRT. Model architecture and PyTorch/TensorRT versions may require additional converter development.

```powershell
ltx-restore in.mp4 --rectangle 0.2,0.2,0.7,0.7 --method neural `
  --checkpoint checkpoints/native.pt --device cuda:0 `
  --accelerator tensorrt --precision fp16 --gpu-backend torch
```

FP8 requires `.npz` samples with `frames[N,T,H,W,3] uint8` and `masks[N,T,H,W] uint8`, with 1–16 **representative** samples and the exact padded input shape (not arbitrary random calibration). The per-component ModelOpt forward calibration is performed before TensorRT compilation. Inspect the resulting output against FP32 and set deployment-specific quality limits; full-graph compilation alone is not a numerical quality guarantee.

```powershell
ltx-restore in.mp4 --rectangle 0.2,0.2,0.7,0.7 --method neural `
  --checkpoint checkpoints/native.pt --device cuda:0 `
  --accelerator tensorrt --precision fp8 --fp8-calibration calibration.npz
```

`--amp` is the existing PyTorch mixed-precision option and is separate from TRT `--precision`. Set `--amp none` for the TRT path while inspecting numerical compatibility.

## CUDA Graph

`ltx_gui/cuda_graph.py` explicitly owns static image/mask/noise buffers, warms up on a side stream, captures one graph and replays using `.copy_()` into the same buffers. Seeded noise is refreshed *outside* capture for each run; graph output is cloned so subsequent replays cannot overwrite caller-owned output. The graph is cached for one clip shape and recreated when the padded geometry changes. A full video can use multiple graph replays; the entire video stream is not captured as one graph.

```powershell
ltx-restore in.mp4 --rectangle 0.2,0.2,0.7,0.7 --method neural `
  --checkpoint checkpoints/native.pt --device cuda:0 `
  --cuda-graph --gpu-backend torch
```

## NVIDIA device-video pipeline

`ltx_gui/nvcodec_pipeline.py` decodes RGBP frames into device memory and imports **each decoded frame** using `torch.from_dlpack(frame)` with no host frame copy. Stacking, floating-point normalization, model processing and ABGR color conversion remain **on GPU**, but allocate and move GPU memory. The NVENC GPU-frame adapter exposes a CUDA Array Interface, and the encoded H.264 byte stream is written to a temporary file before FFmpeg muxes the original audio. The raw RGB pixels are not transferred back to host for this path.

**This is GPU-resident processing, not literally end-to-end zero-copy**: GPU buffers are allocated for stacking and conversion, encoded bytes are copied to host, and FFmpeg muxing runs separately. Explicit stream synchronization prioritizes correctness over maximal overlap. Real input/output verification on the target PyNvVideoCodec build is required. VFR/timestamp handling, HDR, GPU ONNX auto-detection, NV12/P010, multiprocess ring buffers, and GPU encode quality parity are not implemented here. Use the OpenCV path for automatic tracking.

```powershell
ltx-restore in.mp4 --rectangle 0.2,0.2,0.7,0.7 --method neural `
  --checkpoint checkpoints/native.pt --device cuda:0 `
  --video-io nvcodec --encoder h264_nvenc
```

## Measurement protocol

```powershell
python -m pytest -q
python -m scripts.validate_acceleration --checkpoint checkpoints/native.pt `
  --device cuda:0 --frames 9 --width 64 --height 64 `
  --iterations 30 --warmup 5 --output gpu_perf.json
```

The GPU script measures CUDA Event p50/p95 execution time, peak allocated VRAM and maximum absolute output difference against FP32 PyTorch. If no GPU is present, it exits without producing simulated results. It deliberately uses **synthetic random input only to test compute throughput**, never to claim video restoration quality.

For real restoration comparisons, run independently prepared **Jasna and LTX_GUI** outputs on the **same GPU and same input**, with logged exact masks, frames, codec/quality settings, checkpoint, precision, model version, driver, warmup and wall-clock runtime. The existing `scripts.compare_results` also requires a **clean uncensored ground-truth reference**, not the censored source, for PSNR/SSIM. Different trained architectures are not visually equivalent by default; a speed ratio alone is insufficient.

## Primary external references

- [Torch-TensorRT compilation and strict coverage](https://docs.pytorch.org/TensorRT/user_guide/compilation/compilation_settings.html)
- [Torch-TensorRT FP8 + ModelOpt PTQ](https://docs.pytorch.org/TensorRT/user_guide/shapes_precision/quantization.html)
- [PyTorch CUDA Graph semantics](https://docs.pytorch.org/docs/main/notes/cuda.html#cuda-graphs)
- [PyNvVideoCodec API and DLPack](https://docs.nvidia.com/video-technologies/pynvvideocodec/pynvc-api-prog-guide/using_pynvvideocodec_apis.html)

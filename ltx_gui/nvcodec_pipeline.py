"""Opt-in NVDEC -> GPU inference -> NVENC pipeline (NVIDIA PyNvVideoCodec).

Decoded frames are imported through DLPack without a host transfer. GPU
stacking, normalization, inpainting and ABGR format conversion allocate device
buffers; *not* literally zero-allocation/zero-copy throughout. Encoded H.264
bitstream is necessarily returned to host for muxing. Audio uses FFmpeg remux.

Scope: CFR, 8-bit RGB source, static rectangle/image mask. Automatic ONNX
tracking uses the existing CPU path and is explicitly rejected here.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from .engine import Cancelled, Progress, make_mask, read_mask
from .native_engine import clip_batches


class CudaABGRFrame:
    """NVENC GPU-frame adapter preserving the backing tensor's lifetime."""

    def __init__(self, rgba):
        if (not rgba.is_cuda or rgba.dtype != __import__("torch").uint8 or
                rgba.ndim != 3 or rgba.shape[-1] != 4 or not rgba.is_contiguous()):
            raise ValueError("Expected contiguous CUDA uint8 ABGR [H,W,4]")
        self.tensor = rgba

    def cuda(self):
        # PyNvVideoCodec GPU input contract: one CUDA Array Interface per plane.
        x = self.tensor
        return [{"shape": tuple(x.shape), "strides": None, "typestr": "|u1",
                 "data": (x.data_ptr(), False), "version": 3}]


def rgb_to_abgr(rgb):
    """GPU-only conversion of CHW uint8 RGB into NVENC ABGR (alpha=255)."""
    import torch
    if rgb.ndim != 3 or rgb.shape[0] != 3 or rgb.dtype != torch.uint8 or not rgb.is_cuda:
        raise ValueError("Expected device RGB [3,H,W]")
    r, g, b = rgb.unbind(0)
    return CudaABGRFrame(torch.stack((torch.full_like(r, 255), b, g, r), dim=-1).contiguous())


def import_decoded_frame(frame):
    """DLPack shares the GPU buffer; verify no unexpected CPU copy occurs."""
    import torch
    tensor = torch.from_dlpack(frame)
    if not tensor.is_cuda or tensor.dtype != torch.uint8 or tensor.ndim != 3 or tensor.shape[0] != 3:
        raise RuntimeError("Decoder must return RGBP CUDA uint8 frame [3,H,W]")
    return tensor


def _probe_video(path: Path) -> tuple[int, int, float, int]:
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError("Cannot read source video metadata")
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
    if w <= 0 or h <= 0 or fps <= 0 or fps > 240 or n < 1:
        raise ValueError("Invalid video metadata for NVDEC/NVENC")
    if w % 2 or h % 2:
        raise ValueError("NVENC RGB source requires even frame dimensions")
    return w, h, fps, n


def _mux_command(ffmpeg, elementary_path, source_path, output_path, fps):
    return [ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-r", str(fps), "-f", "h264", "-i", str(elementary_path),
            "-i", str(source_path), "-map", "0:v:0", "-map", "1:a?",
            "-c:v", "copy", "-c:a", "copy", "-shortest", str(output_path)]


def restore_nvcodec_video(planned, options, vae, flow, *, stop=None, progress=None) -> Path:
    """GPU-resident static-mask path. Reject unsupported combinations before output."""
    import torch
    from .neural.pipeline import accelerated_inpaint_tensor
    stop = stop or threading.Event()
    if planned.job.auto_detect:
        raise ValueError("nvcodec backend does not support CPU ONNX automatic detection")
    if options.encoder != "h264_nvenc":
        raise ValueError("NVCodec path requires encoder h264_nvenc")
    if not options.neural_device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("NVCodec requires NVIDIA GPU and CUDA PyTorch")
    if planned.target.exists():
        raise FileExistsError(str(planned.target))
    if not shutil.which(options.ffmpeg) and not Path(options.ffmpeg).is_file():
        raise FileNotFoundError("FFmpeg required to mux encoded video and source audio")
    try:
        import PyNvVideoCodec as nvc
    except ImportError as exc:
        raise RuntimeError("Install a compatible NVIDIA PyNvVideoCodec wheel") from exc
    width, height, fps, total = _probe_video(planned.source)
    idx = torch.device(options.neural_device).index or 0
    detector_mask = read_mask(planned.mask_path) if planned.mask_path else None
    mask_np = make_mask((height, width), rectangle=planned.job.rectangle, image=detector_mask)
    static_mask = torch.from_numpy(np.ascontiguousarray(mask_np)).to(options.neural_device).float().div_(255).clamp_(0, 1)
    if mask_np.max() == 0:
        raise ValueError("Empty restoration mask")
    decoder = nvc.SimpleDecoder(str(planned.source), gpu_id=idx, use_device_memory=True,
                                output_color_type=nvc.OutputColorType.RGBP)
    # NVENC requires a GPU input buffer; never allow transparent CPU fallback.
    encoder = nvc.CreateEncoder(width, height, "ABGR", False, gpu_id=idx,
                                codec="h264", fps=max(1, round(fps)))
    completed = 0
    started = time.monotonic()
    runtime_cache: dict = {}
    # Raw H264 is temporary; FFmpeg remuxes original audio into the final container.
    planned.target.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=".ltx_nvcodec_", suffix=".h264", dir=planned.target.parent)
    os.close(fd)
    elementary_path = Path(temp_name)
    mux_path = None
    try:
        def read_frame():
            if stop.is_set():
                raise Cancelled("NVDEC operation cancelled")
            try:
                return import_decoded_frame(next(frame_iter))
            except StopIteration:
                return None

        frame_iter = iter(decoder)
        with elementary_path.open("wb") as output:
            for batch, skip_first in clip_batches(read_frame, frames_per_clip=options.neural_clip_frames):
                # GPU allocations only; no .cpu() or raw-frame host readback.
                video = torch.stack(batch, dim=1).unsqueeze(0).float().div_(255)
                mask = static_mask.unsqueeze(0).unsqueeze(0).unsqueeze(2).expand(1, 1, len(batch), height, width)
                restored = accelerated_inpaint_tensor(
                    video, mask, vae, flow, steps=options.neural_steps,
                    gpu_backend=options.neural_gpu_backend, amp=options.neural_amp,
                    compile_dit=options.neural_compile, runtime_cache=runtime_cache,
                    engine=options.neural_accelerator,
                    precision=options.neural_precision, cuda_graph=options.neural_cuda_graph,
                    fp8_calibration=options.neural_fp8_calibration, assume_nonempty=True)
                restored_uint8 = restored.mul(255).round().clamp_(0, 255).to(torch.uint8)[0]
                for f in range(1 if skip_first else 0, restored_uint8.shape[1]):
                    if stop.is_set():
                        raise Cancelled("NVENC operation cancelled")
                    # RGBP [C,H,W] -> ABGR; encoded bitstream crosses to host.
                    abgr_frame = rgb_to_abgr(restored_uint8[:, f])
                    # GPU producer stream must complete before handing its pointer to NVENC.
                    torch.cuda.current_stream(abgr_frame.tensor.device).synchronize()
                    encoded = encoder.Encode(abgr_frame)
                    output.write(bytearray(encoded))
                    completed += 1
                if progress:
                    progress(Progress(completed, total, completed / max(time.monotonic() - started, 1e-9)))
            output.write(bytearray(encoder.EndEncode()))
        if completed == 0 or (total > 1 and completed < total-1):
            raise ValueError(f"Unexpected decoded frame count: {completed}/{total}")
        # Generate temp output in final directory for atomic publication.
        with tempfile.NamedTemporaryFile(prefix=".ltx_nvcodec_mux_", suffix=planned.target.suffix,
                                         dir=planned.target.parent, delete=False) as fd_out:
            mux_path = Path(fd_out.name)
        proc = subprocess.run(_mux_command(options.ffmpeg, elementary_path, planned.source, mux_path, fps),
                              capture_output=True, text=True, timeout=3600)
        if proc.returncode:
            raise RuntimeError(f"NVCodec FFmpeg remux failed: {proc.stderr[-1200:]}")
        if stop.is_set():
            raise Cancelled("NVCodec operation cancelled")
        os.link(mux_path, planned.target)  # no silent overwrite
        return planned.target
    finally:
        elementary_path.unlink(missing_ok=True)
        if mux_path is not None:
            mux_path.unlink(missing_ok=True)

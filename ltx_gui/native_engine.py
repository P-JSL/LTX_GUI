"""End-to-end independent learned video inpainting (no Jasna/LTX packages).

Streams overlapping 8k+1 frame clips to a local PyTorch DiT/VAE checkpoint.
Maintains exact input pixels outside the mask BEFORE lossy video encoding.
Only the generated masked area is synthesized; audio is remuxed by FFmpeg.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .core import Options, PlannedJob
from .engine import Cancelled, Progress, ffmpeg_args, make_mask, read_mask


def clip_batches(read_frame, *, frames_per_clip: int):
    """Yield (frames, skip_first) with exactly one context frame reused per clip.

    Processing cost is linear in total frames for a fixed clip size. The very
    first frame of each subsequent clip is context, not written twice.
    """
    if frames_per_clip < 9 or (frames_per_clip - 1) % 8:
        raise ValueError("Clip must be 8k+1 (minimum 9 frames)")
    previous = None
    while True:
        batch = [] if previous is None else [previous]
        while len(batch) < frames_per_clip:
            item = read_frame()
            if item is None:
                break
            batch.append(item)
        if not batch or (previous is not None and len(batch) == 1):
            return
        yield batch, (previous is not None)
        previous = batch[-1]


def restore_native_video(planned: PlannedJob, options: Options,
                         stop: threading.Event | None = None,
                         progress: Callable[[Progress], None] | None = None) -> Path:
    """Restore a masked video from a *trained local* native model checkpoint.

    No external LTX runtime or weights are loaded. RAM is O(clip_frames*H*W),
    with bounded attention O(tokens*window_tokens). The model's actual visual
    quality depends on dataset/training; none is supplied by this software.
    """
    from .neural.checkpoint import load_trained
    from .neural.pipeline import inpaint_arrays

    stop = stop or threading.Event()
    options.validate()
    if options.method != "neural":
        raise ValueError("Native neural processing requires method='neural'")
    if planned.target.exists():
        raise FileExistsError(f"출력 파일이 이미 존재합니다: {planned.target}")
    if not shutil.which(options.ffmpeg) and not Path(options.ffmpeg).is_file():
        raise FileNotFoundError(f"FFmpeg 실행 파일을 찾을 수 없습니다: {options.ffmpeg}")
    if options.neural_compile and options.neural_device == "cpu":
        raise ValueError("torch.compile 활성화는 CUDA 모드에서만 지원합니다")
    if stop.is_set():
        raise Cancelled("복원이 시작되기 전에 취소했습니다")
    vae, flow, metadata = load_trained(options.checkpoint_path, device=options.neural_device)
    del metadata
    if options.neural_device.startswith("cuda"):
        if not __import__("torch").cuda.is_available():
            raise RuntimeError("선택한 CUDA 장치에서 GPU를 사용할 수 없습니다.")
        # Enable inference-only kernel fusion; leave training behavior intact.
        from .neural.model import RMSNorm
        for module in flow.modules():
            if isinstance(module, RMSNorm):
                # With torch.compile the compiler can already fuse RMSNorm;
                # avoid custom kernels inside a compiled transformer graph.
                module.gpu_backend = ("torch" if options.neural_compile else
                                      "triton" if options.neural_gpu_backend == "triton" else
                                      "auto" if options.neural_gpu_backend == "auto" else "torch")
    if options.neural_video_io == "nvcodec":
        from .nvcodec_pipeline import restore_nvcodec_video
        return restore_nvcodec_video(planned, options, vae, flow, stop=stop, progress=progress)
    runtime_cache: dict = {}
    source = cv2.VideoCapture(str(planned.source))
    if not source.isOpened():
        raise ValueError(f"입력 영상을 열 수 없습니다: {planned.source}")
    temp_path = log_path = None
    process = None
    try:
        width = int(source.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(source.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(source.get(cv2.CAP_PROP_FPS))
        total = int(source.get(cv2.CAP_PROP_FRAME_COUNT))
        if not (0 < fps <= 1000 and width > 0 and height > 0):
            raise ValueError("영상 해상도 또는 FPS가 올바르지 않습니다")
        detector = tracker = None
        if planned.job.auto_detect:
            from .detection import OnnxConfig, OnnxMosaicDetector
            from .tracking import MosaicTracker
            detector = OnnxMosaicDetector(OnnxConfig(
                model=options.detector_model, confidence=options.detector_confidence,
                cuda=options.detector_cuda, format=options.detector_format,
                mosaic_class=options.detector_class, input_size=options.detector_input_size))
            tracker = MosaicTracker(min_iou=options.tracker_iou,
                                    max_gap=options.tracker_gap, padding=options.tracker_padding)
            static_mask = None
        else:
            image = read_mask(planned.mask_path) if planned.mask_path else None
            static_mask = make_mask((height, width), rectangle=planned.job.rectangle, image=image)
        planned.target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=planned.target.parent, prefix=".ltx_gui_partial_",
                                         suffix=planned.target.suffix, delete=False) as f:
            temp_path = Path(f.name)
        with tempfile.NamedTemporaryFile(dir=planned.target.parent, prefix=".ltx_gui_ffmpeg_",
                                         suffix=".log", delete=False) as f:
            log_path = Path(f.name)

        with log_path.open("wb") as errors:
            kwargs = {"stdin": subprocess.PIPE, "stdout": subprocess.DEVNULL,
                      "stderr": errors}
            if os.name == "nt":
                kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
            process = subprocess.Popen(ffmpeg_args(options.ffmpeg, width, height, fps,
                                                  planned.source, temp_path, options.encoder), **kwargs)
            count = 0
            started = time.monotonic()

            def read_frame():
                if stop.is_set():
                    raise Cancelled("복원이 중지되었습니다")
                ok, frame = source.read()
                if not ok:
                    return None
                mask = (tracker.update(frame, detector.detect(frame))
                        if detector is not None else static_mask)
                return frame, mask

            for batch, skip_first in clip_batches(read_frame, frames_per_clip=options.neural_clip_frames):
                if stop.is_set():
                    raise Cancelled("복원이 중지되었습니다")
                frames = np.stack([pair[0] for pair in batch])
                masks = np.stack([pair[1] for pair in batch])
                if masks.any():
                    # OpenCV/FFmpeg operate on BGR; neural model expects RGB.
                    restored_rgb = inpaint_arrays(np.ascontiguousarray(frames[..., ::-1]), masks,
                                                  vae, flow, steps=options.neural_steps,
                                                  compile_dit=options.neural_compile,
                                                  gpu_backend=options.neural_gpu_backend,
                                                  amp=options.neural_amp,
                                                  runtime_cache=runtime_cache,
                                                  engine=options.neural_accelerator,
                                                  precision=options.neural_precision,
                                                  cuda_graph=options.neural_cuda_graph,
                                                  fp8_calibration=options.neural_fp8_calibration)
                    restored = np.ascontiguousarray(restored_rgb[..., ::-1])
                else:
                    restored = frames  # zero-mask fast path: no GPU launches
                frames_to_write = restored[1:] if skip_first else restored
                assert process.stdin is not None
                for frame in frames_to_write:
                    if stop.is_set():
                        raise Cancelled("복원이 중지되었습니다")
                    process.stdin.write(frame.tobytes())
                    count += 1
                if progress:
                    progress(Progress(count, total, count / max(time.monotonic() - started, 1e-9)))
            if count == 0:
                raise ValueError("입력 영상에 프레임이 없습니다")
            if total > 1 and count < total - 1:
                raise ValueError(f"영상 조기 종료 {count}/{total} frames")
            assert process.stdin is not None
            process.stdin.close()
            result = process.wait()
            if result:
                raise RuntimeError(f"FFmpeg encoding failed ({result}): {log_path.read_text(errors='replace')[-1500:]}")
            if stop.is_set():
                raise Cancelled("복원이 중지되었습니다")
            # Atomic publish without overwriting concurrently-created output.
            os.link(temp_path, planned.target)
            return planned.target
    finally:
        source.release()
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        if log_path is not None:
            log_path.unlink(missing_ok=True)

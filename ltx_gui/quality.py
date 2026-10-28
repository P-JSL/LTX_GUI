"""Reference-grounded image/video metrics for paired inpainting experiments.

A censored source is never ground truth. Quality scores require a genuine clean
reference for each frame; otherwise only temporal/no-reference diagnostics exist.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class FrameMetrics:
    psnr: float
    masked_psnr: float | None
    ssim: float
    outside_mae: float | None

    def json(self) -> dict:
        return asdict(self)


def psnr(reference: np.ndarray, candidate: np.ndarray, mask: np.ndarray | None = None) -> float | None:
    """8-bit RGB/BGR MSE->PSNR, weighted by the number of valid channel samples."""
    if reference.shape != candidate.shape or reference.ndim != 3 or reference.shape[2] != 3:
        raise ValueError("같은 크기의 3채널 영상만 비교 가능합니다.")
    if reference.dtype != np.uint8 or candidate.dtype != np.uint8:
        raise ValueError("uint8 프레임만 비교 가능합니다.")
    diff = reference.astype(np.float64) - candidate.astype(np.float64)
    if mask is not None:
        if mask.shape != reference.shape[:2]:
            raise ValueError("마스크가 참조 영상과 다릅니다.")
        valid = mask != 0
        if not valid.any():
            return None
        diff = diff[valid]
    mse = np.mean(diff * diff)
    return float("inf") if mse == 0 else float(10 * np.log10(255.0**2/mse))


def ssim(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Mean per-channel SSIM map (11x11 Gaussian), same resolution and uint8."""
    if reference.shape != candidate.shape or reference.ndim != 3 or reference.shape[2] != 3:
        raise ValueError("동일한 영상 해상도가 필요합니다.")
    if reference.dtype != np.uint8 or candidate.dtype != np.uint8:
        raise ValueError("uint8 프레임만 비교 가능합니다.")
    x, y = reference.astype(np.float64), candidate.astype(np.float64)
    def blur(v): return cv2.GaussianBlur(v, (11,11), 1.5)
    u, v = blur(x), blur(y)
    ux, vy = blur(x*x) - u*u, blur(y*y) - v*v
    cov = blur(x*y) - u*v
    c1, c2 = (0.01*255)**2, (0.03*255)**2
    result = ((2*u*v+c1)*(2*cov+c2))/((u*u+v*v+c1)*(ux+vy+c2))
    return float(np.mean(result))


def evaluate_frame(reference: np.ndarray, candidate: np.ndarray,
                   mask: np.ndarray | None = None) -> FrameMetrics:
    p = psnr(reference, candidate)
    m = psnr(reference, candidate, mask) if mask is not None else None
    outside = None
    if mask is not None:
        original = mask == 0
        if original.any():
            outside = float(np.abs(reference.astype(np.float64)-candidate.astype(np.float64))[original].mean())
    return FrameMetrics(p, m, ssim(reference, candidate), outside)


def compare_video(reference_path: str | Path, result_path: str | Path,
                  *, mask: np.ndarray | None = None, masks: list[np.ndarray] | None = None,
                  max_frames: int | None = None) -> dict:
    """Deterministic paired-frame comparison; rejects mismatched fps/size/length."""
    if max_frames is not None and max_frames < 1:
        raise ValueError("최대 프레임은 1 이상이어야 합니다.")
    first = cv2.VideoCapture(str(reference_path)); second = cv2.VideoCapture(str(result_path))
    try:
        if not first.isOpened() or not second.isOpened():
            raise ValueError("영상 파일을 읽을 수 없습니다.")
        shape_a = (int(first.get(cv2.CAP_PROP_FRAME_WIDTH)), int(first.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        shape_b = (int(second.get(cv2.CAP_PROP_FRAME_WIDTH)), int(second.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fps_a, fps_b = float(first.get(cv2.CAP_PROP_FPS)), float(second.get(cv2.CAP_PROP_FPS))
        if shape_a != shape_b or abs(fps_a - fps_b) > 0.01:
            raise ValueError("영상 해상도 또는 FPS가 같지 않습니다.")
        totals=[]
        index=0
        while max_frames is None or index < max_frames:
            yes_a, a = first.read(); yes_b, b = second.read()
            if yes_a != yes_b:
                raise ValueError(f"길이가 다른 영상: {index} 프레임 이후")
            if not yes_a: break
            if masks is not None and index >= len(masks):
                raise ValueError("정답 마스크의 수가 부족합니다.")
            frame_mask = masks[index] if masks is not None else mask
            totals.append(evaluate_frame(a,b,frame_mask))
            index+=1
        if index == 0: raise ValueError("비어 있는 영상입니다.")
        out={"frames": index, "fps": fps_a}
        for field in ("psnr", "masked_psnr", "ssim", "outside_mae"):
            values=[getattr(m,field) for m in totals if getattr(m,field) is not None]
            out[field]=float(np.mean(values)) if values else None
        return out
    finally:
        first.release(); second.release()

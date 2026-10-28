"""GPU compositing kernel: functional reference, optional Inductor fusion.

Only the masked compositing op is provided here. This is NOT a substitute for
DiT denoising or video VAE kernels. PyTorch's CUDA backend is used when present.
"""
from __future__ import annotations

from functools import lru_cache


def _compose(original, regenerated, mask):
    # Predicate-based selection avoids unnecessary linear combinations outside ROI.
    return original + (regenerated - original) * mask


@lru_cache(maxsize=1)
def _cuda_fused_compose():
    import torch
    return torch.compile(_compose, fullgraph=True)


def mask_composite(original, regenerated, mask, *, compile_cuda: bool = False):
    """Blend tensors in [0,1] (NCHW or BCTHW) with a broadcastable mask.

    This operation is O(output elements), which is the lower bound for producing
    a dense output. torch.compile may fuse arithmetic but speed is unproven.
    """
    import torch
    if original.shape != regenerated.shape:
        raise ValueError("원본과 생성 영상 Tensor shape가 다릅니다.")
    if original.dtype != regenerated.dtype or original.device != regenerated.device:
        raise ValueError("원본과 생성 영상 dtype/device가 일치해야 합니다.")
    if not torch.is_floating_point(original):
        raise ValueError("합성 입력은 floating tensor여야 합니다.")
    if not torch.is_floating_point(mask) or mask.device != original.device:
        raise ValueError("마스크 dtype/device가 일치해야 합니다.")
    if not torch.isfinite(mask).all().item() or mask.min().item() < 0 or mask.max().item() > 1:
        raise ValueError("마스크 값은 0부터 1까지입니다.")
    if compile_cuda and original.device.type != "cuda":
        raise RuntimeError("CUDA 커널 융합은 CUDA tensor에서만 활성화됩니다.")
    fn = _cuda_fused_compose() if compile_cuda else _compose
    return fn(original, regenerated, mask)

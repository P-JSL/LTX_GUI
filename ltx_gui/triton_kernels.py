"""Original inference-only Triton kernels; no third-party source copied.

Triton operates on contiguous [B,C,T,H,W] and float32 [B,1,T,H,W].
These are independent GPU pointwise/reduction kernels, not an LTX 22B model.
"""
from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


def is_available() -> bool:
    return triton is not None and torch.cuda.is_available()


if triton is not None:
    @triton.jit
    def _update_kernel(z, vel, known, mask, out, N: tl.constexpr,
                       C: tl.constexpr, HW: tl.constexpr, STEP: tl.constexpr, BLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        ok = i < N
        # Dense BCTHW index -> broadcastable B1THW mask index
        b = i // (C * HW)
        m = b * HW + i % HW
        zv = tl.load(z + i, ok, other=0).to(tl.float32)
        vv = tl.load(vel + i, ok, other=0).to(tl.float32)
        kv = tl.load(known + i, ok, other=0).to(tl.float32)
        mv = tl.load(mask + m, ok, other=0)
        tl.store(out + i, (zv - vv * STEP) * mv + kv, ok)

    @triton.jit
    def _blend_kernel(original, restored, mask, out, N: tl.constexpr,
                      C: tl.constexpr, HW: tl.constexpr, BLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        ok = i < N
        b = i // (C * HW)
        m = b * HW + i % HW
        ov = tl.load(original + i, ok, other=0).to(tl.float32)
        rv = tl.load(restored + i, ok, other=0).to(tl.float32)
        mv = tl.load(mask + m, ok, other=0)
        tl.store(out + i, ov + (rv - ov) * mv, ok)

    @triton.jit
    def _rmsnorm_kernel(x, gain, out, ROWS: tl.constexpr,
                        D: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        col = tl.arange(0, BLOCK)
        val = tl.load(x + row * D + col, col < D, other=0).to(tl.float32)
        sq = tl.sum(val * val, 0) / D
        inv = tl.rsqrt(sq + 1e-6)
        g = tl.load(gain + col, col < D, other=0).to(tl.float32)
        tl.store(out + row * D + col, val * inv * g, col < D)

    @triton.jit
    def _unpack_kernel(frames, masks, video, mask, PIXELS: tl.constexpr, BLOCK: tl.constexpr):
        p = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        ok = p < PIXELS
        r = tl.load(frames + 3*p, ok, other=0).to(tl.float32) * (1.0/255.0)
        g = tl.load(frames + 3*p + 1, ok, other=0).to(tl.float32) * (1.0/255.0)
        b = tl.load(frames + 3*p + 2, ok, other=0).to(tl.float32) * (1.0/255.0)
        m = tl.load(masks + p, ok, other=0)
        tl.store(video + p, r, ok)
        tl.store(video + PIXELS + p, g, ok)
        tl.store(video + 2*PIXELS + p, b, ok)
        tl.store(mask + p, tl.where(m >= 128, 1.0, 0.0), ok)


def masked_update(z, velocity, known, mask, step: float):
    if not is_available():
        raise RuntimeError("Triton/CUDA not installed")
    out = torch.empty_like(z)
    b, c, t, h, w = z.shape
    n, hw = z.numel(), t * h * w
    _update_kernel[(triton.cdiv(n, 512),)](z, velocity, known, mask, out, n, c, hw, step, 512)
    return out


def masked_composite(original, restored, mask):
    if not is_available():
        raise RuntimeError("Triton/CUDA not installed")
    out = torch.empty_like(original)
    _, c, t, h, w = original.shape
    n, hw = original.numel(), t * h * w
    _blend_kernel[(triton.cdiv(n, 512),)](original, restored, mask, out, n, c, hw, 512)
    return out


def rmsnorm(x, gain):
    if not is_available():
        raise RuntimeError("Triton/CUDA not installed")
    d = x.shape[-1]
    out = torch.empty_like(x)
    _rmsnorm_kernel[(x.numel() // d,)](x, gain, out, x.numel() // d, d, triton.next_power_of_2(d))
    return out


def unpack_video(frames, masks):
    """Fused uint8 NHWC->float32 BCTHW, binary mask conversion on GPU.

    Input: GPU contiguous uint8 frames [T,H,W,3] and masks [T,H,W].
    Output: normalized 5D tensors. No extra transpose/threshold launches.
    """
    if not is_available() or not (frames.is_cuda and masks.is_cuda and frames.is_contiguous() and masks.is_contiguous()):
        raise ValueError("Fused unpack requires contiguous CUDA inputs")
    if frames.dtype != torch.uint8 or masks.dtype != torch.uint8 or frames.ndim != 4 or masks.shape != frames.shape[:3] or frames.shape[-1] != 3:
        raise ValueError("Expected uint8 frames [T,H,W,3], masks [T,H,W]")
    t,h,w,_ = frames.shape
    pix = t*h*w
    video = torch.empty((1,3,t,h,w),device=frames.device,dtype=torch.float32)
    mask = torch.empty((1,1,t,h,w),device=frames.device,dtype=torch.float32)
    _unpack_kernel[(triton.cdiv(pix,512),)](frames,masks,video,mask,pix,512)
    return video,mask

"""Optional CUDA/Triton inference kernels with a numerically safe torch fallback.

GPU paths run only when explicitly requested or on CUDA tensors (backend=auto).
No JIT compiler is required by the Python package; CUDA C++ is opt-in build.
The fused kernels are inference-only: training uses PyTorch autograd operations.
"""
from __future__ import annotations

from importlib import import_module
from functools import lru_cache
from typing import Literal

import torch
from torch import Tensor

Backend = Literal["auto", "torch", "triton", "cuda"]
BACKENDS = ("auto", "torch", "triton", "cuda")


def _check_backend(backend: str) -> None:
    if backend not in BACKENDS:
        raise ValueError(f"Unknown GPU backend: {backend}")


def _mask_shape(x: Tensor, mask: Tensor) -> bool:
    return (x.ndim == 5 and mask.shape == (x.shape[0], 1, *x.shape[2:]))


def _supports_fused(x: Tensor, mask: Tensor, *others: Tensor) -> bool:
    return (x.is_cuda and x.is_contiguous() and mask.is_contiguous()
            and _mask_shape(x, mask) and mask.dtype == torch.float32
            and x.dtype in (torch.float32, torch.float16, torch.bfloat16)
            and all(t.is_contiguous() and t.shape == x.shape and t.dtype == x.dtype
                    and t.device == x.device for t in others)
            and mask.device == x.device)


@lru_cache(maxsize=1)
def _import_native():
    try:
        return import_module("ltx_gui._cuda_kernels")
    except ImportError:
        return None


@lru_cache(maxsize=1)
def _import_triton():
    try:
        return import_module("ltx_gui.triton_kernels")
    except ImportError:
        return None


def _select(backend: str, supported: bool):
    _check_backend(backend)
    if backend == "torch":
        return None
    if not supported:
        if backend != "auto":
            raise ValueError("CUDA backend requires contiguous CUDA [B,C,T,H,W] tensors and float32 [B,1,T,H,W] mask")
        return None
    if backend in ("cuda", "auto"):
        native = _import_native()
        if native is not None:
            return native
        if backend == "cuda":
            raise RuntimeError("CUDA C++ extension unavailable: run python setup_cuda.py build_ext --inplace")
    if backend in ("triton", "auto"):
        module = _import_triton()
        if module is not None and module.is_available():
            return module
        if backend == "triton":
            raise RuntimeError("Triton GPU kernels unavailable on this system")
    return None


def masked_update(z: Tensor, velocity: Tensor, known: Tensor, mask: Tensor,
                  step: float, *, backend: Backend = "auto") -> Tensor:
    """Fused Euler step and latent preservation, preserving original torch semantics.

    Each output element is read/written once in CUDA/Triton mode; O(B*C*T*H*W).
    Automatic dispatch never moves tensors between CPU and GPU implicitly.
    """
    if z.shape != velocity.shape or z.shape != known.shape or not _mask_shape(z, mask):
        raise ValueError("Expected z, velocity, known [B,C,T,H,W] and mask [B,1,T,H,W]")
    if not all(t.device == z.device for t in (velocity, known, mask)):
        raise ValueError("All inputs must be on the same device")
    _check_backend(backend)
    # One contiguous model-output conversion is cheaper than letting multiple
    # unfused pointwise ops launch for each inference step on a GPU.
    if z.is_cuda and backend != "torch" and _select(backend, True) is not None:
        z, velocity, known, mask = (t.contiguous() for t in (z, velocity, known, mask))
    kernel = _select(backend, _supports_fused(z, mask, velocity, known))
    if kernel is not None:
        return kernel.masked_update(z, velocity, known, mask, float(step))
    return (z - velocity * step) * mask + known


def masked_composite(original: Tensor, restored: Tensor, mask: Tensor,
                     *, backend: Backend = "auto") -> Tensor:
    """Mask blend. Original pixels with mask=0 remain unchanged before encoding."""
    if original.shape != restored.shape or not _mask_shape(original, mask):
        raise ValueError("Expected original/restored [B,C,T,H,W] and mask [B,1,T,H,W]")
    if not all(t.device == original.device for t in (restored, mask)):
        raise ValueError("All inputs must be on the same device")
    _check_backend(backend)
    if original.is_cuda and backend != "torch" and _select(backend, True) is not None:
        original, restored, mask = (t.contiguous() for t in (original, restored, mask))
    kernel = _select(backend, _supports_fused(original, mask, restored))
    if kernel is not None:
        return kernel.masked_composite(original, restored, mask)
    return original + (restored - original) * mask


def fused_rmsnorm(x: Tensor, gain: Tensor, *, backend: Backend = "auto") -> Tensor:
    """Inference-only last-dimension normalization, unbiased float32 reduction."""
    if x.shape[-1] != gain.numel() or x.device != gain.device:
        raise ValueError("RMSNorm parameters mismatch")
    # Native extension handles masked 5D ops; Triton handles RMSNorm.
    _check_backend(backend)
    if backend == "cuda":
        raise ValueError("RMSNorm is implemented via Triton, not CUDA C++")
    if backend in ("auto", "triton") and not torch.is_grad_enabled() and x.is_cuda and x.is_contiguous() and gain.is_contiguous() and x.shape[-1] <= 8192:
        kernel = _import_triton()
        if kernel is not None and kernel.is_available():
            return kernel.rmsnorm(x, gain)
        if backend == "triton":
            raise RuntimeError("Triton GPU kernel unavailable")
    elif backend == "triton":
        raise ValueError("Triton RMSNorm requires contiguous CUDA tensors in inference mode")
    return (x * torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + 1e-6).to(x.dtype)) * gain


def backend_status() -> dict[str, object]:
    """Report availability without initializing CUDA or compiling extensions."""
    return {"torch_cuda": torch.cuda.is_available(),
            "native_extension": _import_native() is not None,
            "triton": _import_triton() is not None and _import_triton().is_available()}

"""Independent input preparation, mask-guided diffusion, pixel-exact compositing."""
from __future__ import annotations

from contextlib import nullcontext

import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import functional as F

from .model import ModelConfig, VideoVAE, LatentFlow


def pad_video(video: Tensor, mask: Tensor) -> tuple[Tensor, Tensor, tuple[int, int, int]]:
    """Pad a [B,3,T,H,W] clip to 8k+1 and 32x32, replicating edge pixels.

    Zero padding of the mask ensures artificial padding is never edited.
    """
    if video.ndim != 5 or video.shape[1] != 3 or mask.shape != (video.shape[0], 1, *video.shape[2:]):
        raise ValueError("Expected video [B,3,T,H,W] and mask [B,1,T,H,W]")
    t, h, w = video.shape[2:]
    if t < 1 or h < 1 or w < 1:
        raise ValueError("Empty video dimensions")
    extra_t = (-(t - 1)) % 8
    extra_h = (-h) % 32
    extra_w = (-w) % 32
    # F.pad with replicate is not implemented for all 5-D modes/dtypes on CPU.
    video = F.pad(video, (0, extra_w, 0, extra_h, 0, 0), mode="replicate")
    mask = F.pad(mask, (0, extra_w, 0, extra_h, 0, 0), value=0)
    if extra_t:
        video = torch.cat((video, video[:, :, -1:].expand(-1, -1, extra_t, -1, -1)), dim=2)
        mask = torch.cat((mask, mask.new_zeros((mask.shape[0], 1, extra_t, mask.shape[3], mask.shape[4]))), dim=2)
    return video, mask, (t, h, w)


def latent_mask_from_pixels(mask: Tensor, latent_shape: tuple[int, int, int]) -> Tensor:
    """Conservative pooling: latent token is editable when any contributing pixel is masked."""
    if mask.ndim != 5 or mask.shape[1] != 1:
        raise ValueError("Expected [B,1,T,H,W] mask")
    lt, lh, lw = latent_shape
    if (mask.shape[2]-1) != 8 * (lt-1) or mask.shape[3] != 32 * lh or mask.shape[4] != 32 * lw:
        raise ValueError("Mask and VAE latent compression factors disagree")
    # Spatial max reduction with no 3-D materialization.
    spatial = F.max_pool3d(mask.float(), kernel_size=(1, 32, 32), stride=(1, 32, 32))
    if lt == 1:
        return spatial
    b = spatial.shape[0]
    first = spatial[:, :, :1]
    rest = spatial[:, :, 1:].reshape(b, 1, lt-1, 8, lh, lw).amax(3)
    return torch.cat((first, rest), dim=2)


@torch.no_grad()
def inpaint_tensor(video: Tensor, mask: Tensor, vae: VideoVAE, flow: LatentFlow,
                   *, steps: int = 8, seed: int = 42, compile_dit: bool = False,
                   gpu_backend: str = "auto", amp: str = "none") -> Tensor:
    """Inpaint normalized [0,1] pixels and preserve nonmasked pixels exactly.

    PyTorch SDPA chooses a fused GPU attention kernel when supported. Compiling
    for every clip shape may hurt throughput; enable only for stable shapes.
    """
    if not torch.is_floating_point(video) or torch.any(video < 0) or torch.any(video > 1):
        raise ValueError("Video pixels must be floats in [0,1]")
    if seed < 0 or seed >= 2**63:
        raise ValueError("Seed outside allowed range")
    if gpu_backend not in ("auto", "torch", "triton", "cuda"):
        raise ValueError("Unknown GPU backend")
    if amp not in ("none", "bf16", "fp16"):
        raise ValueError("Unknown mixed precision mode")
    if amp != "none" and video.device.type != "cuda":
        raise ValueError("Mixed precision requires CUDA")
    if amp == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("The selected CUDA device does not support BF16")
    if torch.count_nonzero(mask) == 0:
        return video.clone()
    padded, padded_mask, original_shape = pad_video(video, mask)
    generator = torch.Generator(device=padded.device).manual_seed(seed)
    precision_context = (torch.autocast("cuda", dtype=torch.bfloat16 if amp == "bf16" else torch.float16)
                         if amp != "none" else nullcontext())
    with precision_context:
        mu, _ = vae.encode(padded)
    latent_mask = latent_mask_from_pixels(padded_mask, mu.shape[2:])
    predictor = None
    if compile_dit:
        if padded.device.type != "cuda":
            raise ValueError("torch.compile requires CUDA in this application")
        # Compile the model's actual forward (not flow.sample, which is not
        # nn.Module.forward). Cache the callable without changing state_dict.
        predictor = flow.__dict__.get("_ltx_compiled_predictor")
        if predictor is None:
            predictor = torch.compile(flow.net, mode="reduce-overhead")
            flow.__dict__["_ltx_compiled_predictor"] = predictor
    with precision_context:
        z = flow.sample(mu, latent_mask, steps=steps, generator=generator,
                        gpu_backend=gpu_backend,
                        assume_nonempty=True, predictor=predictor)
        pixels = vae.decode(z)
    t, h, w = original_shape
    # Original pixels outside mask are bytewise preserved when returned as uint8.
    from ..cuda_backend import masked_composite
    return masked_composite(video, pixels[:, :, :t, :h, :w].to(video.dtype), mask,
                            backend=gpu_backend)


def clip_arrays_to_tensor(frames: np.ndarray, masks: np.ndarray, device: str,
                          *, gpu_backend: str = "auto") -> tuple[Tensor, Tensor]:
    if frames.ndim != 4 or frames.shape[-1] != 3 or frames.dtype != np.uint8:
        raise ValueError("Expected RGB frames [T,H,W,3], uint8")
    if masks.shape != frames.shape[:3]:
        raise ValueError("Expected masks [T,H,W]")
    from ..cuda_backend import _import_triton
    if gpu_backend in ("auto", "triton") and str(device).startswith("cuda") and masks.dtype == np.uint8:
        module = _import_triton()
        if module is not None and module.is_available():
            # Transfer uint8 first (~4x less H2D bandwidth than float32).
            # Pinned memory is used only for larger clips to amortize allocation.
            f = torch.from_numpy(np.ascontiguousarray(frames))
            m = torch.from_numpy(np.ascontiguousarray(masks))
            pin = f.numel() + m.numel() >= 1024 * 1024
            if pin:
                f, m = f.pin_memory(), m.pin_memory()
            f = f.to(device, non_blocking=pin)
            m = m.to(device, non_blocking=pin)
            return module.unpack_video(f, m)
        if gpu_backend == "triton":
            raise RuntimeError("Triton preprocessing unavailable")
    video = torch.from_numpy(np.ascontiguousarray(frames)).permute(3, 0, 1, 2).unsqueeze(0).contiguous().to(device=device, dtype=torch.float32).div_(255)
    mask = torch.from_numpy(np.ascontiguousarray(masks)).unsqueeze(0).unsqueeze(0).to(device=device, dtype=torch.float32)
    mask = (mask >= 128).float() if masks.dtype == np.uint8 else (mask > 0).float()
    return video, mask


@torch.no_grad()
def inpaint_arrays(frames: np.ndarray, masks: np.ndarray,
                   vae: VideoVAE, flow: LatentFlow, *, steps: int = 8,
                   seed: int = 42, compile_dit: bool = False,
                   gpu_backend: str = "auto", amp: str = "none") -> np.ndarray:
    try:
        device = next(vae.parameters()).device
    except StopIteration:
        raise ValueError("VAE has no parameters")
    video, mask = clip_arrays_to_tensor(frames, masks, str(device), gpu_backend=gpu_backend)
    output = inpaint_tensor(video, mask, vae, flow, steps=steps,
                           seed=seed, compile_dit=compile_dit,
                           gpu_backend=gpu_backend, amp=amp)
    result = output.mul(255).round().clamp_(0,255).to(torch.uint8).squeeze(0).permute(1,2,3,0).cpu().numpy()
    # Direct replacement guarantees exact original uint8 in nonmasked regions.
    return np.where((masks > 0)[..., None], result, frames)

"""Opt-in NVIDIA inference acceleration for independently trained LTX_GUI models.

No implicit fallback: a requested full TensorRT conversion must succeed for
both VAE halves and DiT, or the job fails before any output is published.
FP8 *requires* a representative calibration callback and Hopper+ hardware.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class AccelerationSettings:
    engine: str = "torch"         # torch | tensorrt
    precision: str = "fp32"       # fp32 | fp16 | bf16 | fp8
    cuda_graph: bool = False
    require_full_compilation: bool = True
    workspace_gib: int = 2

    def validate(self, *, device: str = "cuda", capability: tuple[int, int] | None = None) -> None:
        if self.engine not in ("torch", "tensorrt") or self.precision not in ("fp32", "fp16", "bf16", "fp8"):
            raise ValueError("Unknown accelerator engine or precision")
        if not 1 <= self.workspace_gib <= 64:
            raise ValueError("TensorRT workspace must be 1..64 GiB")
        if (self.engine == "tensorrt" or self.cuda_graph or self.precision != "fp32") and not device.startswith("cuda"):
            raise ValueError("NVIDIA acceleration requires CUDA")
        if self.precision == "fp8" and capability is not None and capability[0] < 9:
            raise ValueError("FP8 TensorRT execution requires Hopper (compute capability 9.0) or newer")
        if self.precision == "fp8" and self.engine != "tensorrt":
            raise ValueError("FP8 is supported only with a calibrated TensorRT path")


def require_cuda(settings: AccelerationSettings, device: str):
    """Validate actual device rather than trusting a requested GPU index."""
    import torch
    if not device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("CUDA runtime and a compatible NVIDIA GPU are required")
    idx = torch.device(device).index
    if idx is None:
        idx = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(idx)
    settings.validate(device=device, capability=(props.major, props.minor))
    if settings.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("This GPU does not support BF16")
    return idx


def validate_calibration(samples) -> None:
    """Never silently replace calibration data with random test input."""
    if not samples:
        raise ValueError("FP8 requires representative, user-supplied calibration samples")
    for item in samples:
        if not isinstance(item, tuple) or not item:
            raise ValueError("Calibration samples must be nonempty input tuples")


def compile_tensorrt_components(vae, flow, video, mask, settings: AccelerationSettings,
                                  *, calibration: dict[str, list[tuple]] | None = None):
    """Compile encode, DiT and decode independently using a strict TRT coverage gate.

    Does not mutate the original trained modules.  Compilation may reject
    custom ops / dynamic shapes; that is a useful error, not a success.
    Returns (encode, predict, decode), each called with PyTorch tensors.
    """
    import torch
    from torch import nn
    from .neural.pipeline import latent_mask_from_pixels

    if settings.engine != "tensorrt":
        raise ValueError("Use this builder only for TensorRT")
    if not settings.require_full_compilation:
        raise ValueError("Full TRT coverage is mandatory for this builder")
    require_cuda(settings, str(video.device))
    if video.ndim != 5 or mask.shape != (video.shape[0], 1, *video.shape[2:]):
        raise ValueError("Expected video [B,3,T,H,W] and matching mask")
    try:
        import torch_tensorrt
    except ImportError as exc:
        raise RuntimeError("Install a compatible torch-tensorrt / TensorRT for your CUDA stack") from exc

    class Encoder(nn.Module):
        def __init__(self, parent):
            super().__init__()
            self.parent = parent
        def forward(self, x):
            return self.parent.encode(x)

    class Decoder(nn.Module):
        def __init__(self, parent):
            super().__init__()
            self.parent = parent
        def forward(self, x):
            return self.parent.decode(x)

    dtype = {"fp32": torch.float32, "fp16": torch.float16,
             "bf16": torch.bfloat16, "fp8": torch.bfloat16}[settings.precision]
    # Compile separate *copies*, never change the original checkpoint model.
    import copy
    safe_vae, safe_flow = copy.deepcopy(vae).eval().to(dtype=dtype), copy.deepcopy(flow).eval().to(dtype=dtype)
    # Custom inference kernels may not be exportable. Disable them while building TRT.
    from .neural.model import RMSNorm
    for module in (*safe_vae.modules(), *safe_flow.modules()):
        if isinstance(module, RMSNorm):
            module.gpu_backend = "torch"
    enc = Encoder(copy.deepcopy(safe_vae)).eval()
    dec = Decoder(copy.deepcopy(safe_vae)).eval()
    model = safe_flow.net.eval()
    example = video.to(dtype=dtype)
    with torch.inference_mode():
        mu, _ = enc(example)
        latent_mask = latent_mask_from_pixels(mask, mu.shape[2:]).to(dtype)
        known = mu * (1 - latent_mask)
        sigma = torch.ones((mu.shape[0],), device=mu.device, dtype=dtype)
    inputs = {"encoder": (example,), "dit": (mu, known, latent_mask, sigma), "decoder": (mu,)}
    parts = {"encoder": enc, "dit": model, "decoder": dec}
    if settings.precision == "fp8":
        if not calibration:
            raise ValueError("FP8 requires representative input batches for encoder, DiT and decoder")
        try:
            import modelopt.torch.quantization as mtq
        except ImportError as exc:
            raise RuntimeError("Install nvidia-modelopt for FP8 calibration") from exc
        for name, module in parts.items():
            samples = calibration.get(name)
            validate_calibration(samples)
            with torch.inference_mode():
                mtq.quantize(module, mtq.FP8_DEFAULT_CFG,
                             forward_loop=lambda m, values=samples: [m(*(item.to(dtype=dtype) if item.is_floating_point() else item for item in value)) for value in values])

    compiled = {}
    for name, module in parts.items():
        try:
            with torch.inference_mode():
                compiled[name] = torch_tensorrt.compile(
                    module, ir="dynamo", arg_inputs=inputs[name],
                    require_full_compilation=True, min_block_size=1,
                    workspace_size=settings.workspace_gib << 30,
                    use_explicit_typing=True)
        except Exception as exc:
            raise RuntimeError(f"TensorRT full-graph conversion rejected {name}: {exc}") from exc
    # Runtime tensors start in fp32 for pixel compositing; do explicit boundary
    # casts rather than relying on implicit TensorRT input conversions.
    def encode_runtime(video):
        return compiled["encoder"](video.to(dtype))
    def predict_runtime(noisy, known, mask, sigma):
        return compiled["dit"](noisy.to(dtype), known.to(dtype), mask.to(dtype), sigma.to(dtype))
    def decode_runtime(latent):
        return compiled["decoder"](latent.to(dtype))
    return encode_runtime, predict_runtime, decode_runtime


def load_fp8_calibration(path: str, vae, example_video, example_mask) -> dict[str, list[tuple]]:
    """Derive representative module calls from real RGB video samples.

    .npz schema: frames[N,T,H,W,3] uint8, masks[N,T,H,W] uint8.
    Shape must match the *padded* clip being compiled; maximum 16 samples.
    """
    from pathlib import Path
    import numpy as np
    import torch
    from .neural.pipeline import latent_mask_from_pixels
    if not path or not Path(path).is_file():
        raise FileNotFoundError("FP8 requires existing calibration .npz")
    with np.load(path, allow_pickle=False) as data:
        if "frames" not in data or "masks" not in data:
            raise ValueError("FP8 calibration requires frames and masks arrays")
        f, m = data["frames"], data["masks"]
    expected = (example_video.shape[2], example_video.shape[3], example_video.shape[4])
    if (f.ndim != 5 or m.shape != f.shape[:-1] or f.shape[-1] != 3 or
        f.shape[1:4] != expected or f.dtype != np.uint8 or m.dtype != np.uint8 or
        not 1 <= f.shape[0] <= 16):
        raise ValueError("Calibration batch must contain 1..16 matching RGB/mask clips")
    examples = {"encoder": [], "dit": [], "decoder": []}
    device = example_video.device
    with torch.inference_mode():
        for frames, masks in zip(f, m):
            video = torch.from_numpy(frames.copy()).to(device=device).permute(3, 0, 1, 2).unsqueeze(0).float().div_(255)
            mask = torch.from_numpy(masks.copy()).to(device=device).unsqueeze(0).unsqueeze(0).float().div_(255).clamp_(0, 1)
            mu, _ = vae.encode(video)
            latent_mask = latent_mask_from_pixels(mask, mu.shape[2:])
            known = mu * (1 - latent_mask)
            sigma = torch.full((1,), 0.5, device=device, dtype=mu.dtype)
            examples["encoder"].append((video,))
            examples["dit"].append((mu, known, latent_mask, sigma))
            examples["decoder"].append((mu,))
    return examples

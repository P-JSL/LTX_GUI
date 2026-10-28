"""Original compact latent-video rectified-flow architecture and 3D attention.

This is a from-scratch *research-sized* DiT/VAE, not an implementation of the
22B LTX-2.5 dual-stream/Gemma architecture and not compatible with its weights.
The first video frame is retained in the 8k+1 temporal compression convention.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from collections import OrderedDict

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from .window_ops import partition_3d, unpartition_3d, valid_window_keys, partition_rope


@dataclass(frozen=True)
class ModelConfig:
    latent_channels: int = 16
    base_channels: int = 16
    dim: int = 96
    heads: int = 3
    depth: int = 3
    temporal_scale: int = 8
    spatial_scale: int = 32
    window_frames: int = 4
    window_side: int = 8
    causal_temporal: bool = False

    def validate(self) -> None:
        if not 2 <= self.latent_channels <= 512 or not 4 <= self.base_channels <= 512:
            raise ValueError("Invalid VAE channel count")
        if not 16 <= self.dim <= 8192 or not 1 <= self.heads <= 128 or self.dim % self.heads:
            raise ValueError("Attention dimension must divide by head count")
        if self.dim // self.heads < 6 or not 1 <= self.depth <= 128:
            raise ValueError("Invalid attention head dimension/depth")
        if self.temporal_scale != 8 or self.spatial_scale != 32:
            raise ValueError("Research VAE supports 8x time and 32x spatial compression only")
        if not 1 <= self.window_frames <= 64 or not 1 <= self.window_side <= 64:
            raise ValueError("Window limits must be between 1 and 64")
        if self.window_frames * self.window_side**2 > 4096:
            raise ValueError("Attention window exceeds 4096 tokens")

    def to_dict(self) -> dict[str, int | bool]:
        return asdict(self)


class SpatialConv(nn.Module):
    """Spatial conv per video frame. Avoids a 3D kernel over static pixels."""
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)

    def forward(self, video: Tensor) -> Tensor:
        b, c, t, h, w = video.shape
        pixels = video.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        output = self.conv(pixels)
        _, nc, nh, nw = output.shape
        return output.reshape(b, t, nc, nh, nw).permute(0, 2, 1, 3, 4)


class CausalTemporalDepthwise(nn.Module):
    """Lightweight causal temporal mixing at 1/32 spatial resolution.

    Each channel uses the current frame and two past frames. Initialize to an
    identity map to avoid disturbing an untrained model's spatial behavior.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv3d(channels, channels, (3, 1, 1), groups=channels, bias=False)
        with torch.no_grad():
            self.conv.weight.zero_()
            self.conv.weight[:, 0, 2, 0, 0] = 1.0

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(F.pad(x, (0, 0, 0, 0, 2, 0)))


class VideoVAE(nn.Module):
    """Independent 8x/32x compressed VAE with learned spatial encoder/decoder.

    The temporal pool/expand stages are simplified (not the learned causal
    temporal model of the official LTX VAE). They require 8k+1 source frames.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        config.validate()
        self.config = config
        channels = [3, config.base_channels, config.base_channels * 2,
                    config.base_channels * 2, config.base_channels * 2,
                    config.base_channels * 2]
        self.encoder = nn.ModuleList(SpatialConv(channels[i], channels[i + 1], 2) for i in range(5))
        hidden = channels[-1]
        self.encoder_temporal = CausalTemporalDepthwise(hidden) if config.causal_temporal else nn.Identity()
        self.decoder_temporal = CausalTemporalDepthwise(hidden) if config.causal_temporal else nn.Identity()
        self.mean = nn.Conv3d(hidden, config.latent_channels, 1)
        self.logvar = nn.Conv3d(hidden, config.latent_channels, 1)
        self.from_latent = nn.Conv3d(config.latent_channels, hidden, 1)
        self.decoder = nn.ModuleList(SpatialConv(hidden, hidden) for _ in range(4))
        self.rgb = SpatialConv(hidden, 3)

    @staticmethod
    def time_compress(x: Tensor, step: int = 8) -> Tensor:
        b, c, t, h, w = x.shape
        if (t - 1) % step:
            raise ValueError("Video frames must be 8k+1, e.g. 9, 17, 25")
        if t == 1:
            return x
        return torch.cat((x[:, :, :1], x[:, :, 1:].reshape(b, c, (t - 1)//step, step, h, w).mean(3)), dim=2)

    @staticmethod
    def time_expand(z: Tensor, step: int = 8) -> Tensor:
        if z.shape[2] == 1:
            return z
        return torch.cat((z[:, :, :1], z[:, :, 1:].repeat_interleave(step, dim=2)), dim=2)

    def encode(self, video: Tensor) -> tuple[Tensor, Tensor]:
        if video.ndim != 5 or video.shape[1] != 3 or video.shape[3] % 32 or video.shape[4] % 32:
            raise ValueError("Expected RGB [B,3,T,H,W], with H/W divisible by 32")
        x = video
        for layer in self.encoder:
            x = F.silu(layer(x))
        x = self.time_compress(self.encoder_temporal(x), self.config.temporal_scale)
        return self.mean(x), self.logvar(x).clamp(-20, 10)

    def decode(self, z: Tensor) -> Tensor:
        x = self.decoder_temporal(self.time_expand(self.from_latent(z), self.config.temporal_scale))
        for layer in self.decoder:
            x = F.interpolate(x, scale_factor=(1, 2, 2), mode="nearest")
            x = F.silu(layer(x))
        x = F.interpolate(x, scale_factor=(1, 2, 2), mode="nearest")
        return torch.sigmoid(self.rgb(x))

    def forward(self, video: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        mu, logvar = self.encode(video)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar) if self.training else mu
        return self.decode(z), mu, logvar


def timestep_embedding(t: Tensor, dim: int) -> Tensor:
    """Sinusoidal scalar timestep embedding, in float32 for numerical safety."""
    half = dim // 2
    freq = torch.exp(-math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / max(half-1, 1))
    angles = t.float()[:, None] * freq[None] * 1000
    result = torch.cat((torch.cos(angles), torch.sin(angles)), dim=-1)
    return F.pad(result, (0, dim-result.shape[-1]))


def rope_positions(frames: int, height: int, width: int, per_head: int, device: torch.device) -> tuple[Tensor, Tensor, int]:
    """Precompute 3D rotary angles once per forward (reused across blocks).

    Each axis gets the same even-dimensional share of the head channels.
    The leftover channels (if any) remain unchanged.
    """
    axis_dim = 2 * (per_head // 6)
    if axis_dim < 2:
        raise ValueError("Head size too small for 3D RoPE")
    coords = torch.stack(torch.meshgrid(torch.arange(frames, device=device),
                                       torch.arange(height, device=device),
                                       torch.arange(width, device=device), indexing="ij"), dim=-1).reshape(-1, 3)
    base = torch.exp(-math.log(10000) * torch.arange(0, axis_dim, 2, dtype=torch.float32, device=device) / axis_dim)
    angles = torch.cat([coords[:, axis:axis+1].float() * base for axis in range(3)], dim=-1)
    return angles.cos()[None, None], angles.sin()[None, None], 3 * axis_dim


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor, rotated: int) -> Tensor:
    """Rotate an even channel prefix of [B,H,N,D] using interleaved pairs."""
    a, tail = x[..., :rotated], x[..., rotated:]
    pairs = a.float().reshape(*a.shape[:-1], -1, 2)
    even, odd = pairs[..., 0], pairs[..., 1]
    out = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2).to(x.dtype)
    return torch.cat((out, tail), dim=-1) if tail.shape[-1] else out


class RMSNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gain = nn.Parameter(torch.ones(dim))
        # Runtime-only toggle: never changes checkpoint parameter names/shapes.
        self.gpu_backend = "torch"

    def forward(self, x: Tensor) -> Tensor:
        if not self.training and not torch.is_grad_enabled() and self.gpu_backend in ("auto", "triton"):
            from ..cuda_backend import fused_rmsnorm
            # Preserve PyTorch mixed-precision promotion when gain is fp32.
            if x.dtype == self.gain.dtype and x.is_contiguous() and self.gain.is_contiguous():
                return fused_rmsnorm(x, self.gain, backend=self.gpu_backend)
        return (x * torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + 1e-6).to(x.dtype)) * self.gain


class AttentionBlock(nn.Module):
    """AdaLN + RoPE + fused SDPA + gated MLP, without materializing N^2 logits."""
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.norm1 = RMSNorm(dim)
        self.qkv = nn.Linear(dim, 3*dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.norm2 = RMSNorm(dim)
        self.gate = nn.Linear(dim, 2 * (dim * 4))
        self.mlp_out = nn.Linear(dim * 4, dim)
        self.modulation = nn.Linear(dim, dim * 6)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def forward(self, x: Tensor, conditioning: Tensor,
                pos: tuple[Tensor, Tensor, int],
                key_mask: Tensor | None = None) -> Tensor:
        b, n, d = x.shape
        shift_a, scale_a, gate_a, shift_b, scale_b, gate_b = self.modulation(conditioning).unsqueeze(1).chunk(6, dim=-1)
        normed = self.norm1(x) * (1 + scale_a) + shift_a
        q, k, v = self.qkv(normed).reshape(b, n, 3, self.heads, self.head_dim).unbind(2)
        q, k, v = (z.transpose(1, 2) for z in (q, k, v))
        cos, sin, rotated = pos
        q = apply_rope(q, cos, sin, rotated)
        k = apply_rope(k, cos, sin, rotated)
        # PyTorch dispatches to fused Flash/memory-efficient kernels where available.
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=key_mask, dropout_p=0.0, is_causal=False)
        attn = attn.transpose(1, 2).reshape(b, n, d)
        x = x + gate_a * self.out(attn)
        ff = self.norm2(x) * (1 + scale_b) + shift_b
        u, v = self.gate(ff).chunk(2, dim=-1)
        return x + gate_b * self.mlp_out(F.silu(u) * v)


class VideoDiffusionTransformer(nn.Module):
    """Spatiotemporal latent denoiser, mask/known-latent conditioned."""
    def __init__(self, config: ModelConfig):
        super().__init__()
        config.validate()
        self.config = config
        c, d = config.latent_channels, config.dim
        self.input = nn.Conv3d(c * 2 + 1, d, 1)
        self.time_proj = nn.Sequential(nn.Linear(d, d*4), nn.SiLU(), nn.Linear(d*4, d))
        self.blocks = nn.ModuleList(AttentionBlock(d, config.heads) for _ in range(config.depth))
        self.norm = RMSNorm(d)
        self.velocity = nn.Linear(d, c)
        # Small bounded cache for inference-only absolute 3D RoPE across steps.
        # Never register as a parameter/buffer; checkpoint format is unchanged.
        self._rope_cache = OrderedDict()
        # At initialization, velocity is zero, not arbitrary displacement.
        nn.init.zeros_(self.velocity.weight)
        nn.init.zeros_(self.velocity.bias)

    def forward(self, noisy: Tensor, known: Tensor, mask: Tensor, sigma: Tensor) -> Tensor:
        if noisy.ndim != 5 or known.shape != noisy.shape or mask.shape != noisy.shape[:1] + (1,) + noisy.shape[2:]:
            raise ValueError("Expected noisy/known [B,C,T,H,W], mask [B,1,T,H,W]")
        if sigma.shape != (noisy.shape[0],):
            raise ValueError("Sigma is one scalar per batch entry")
        b, _, t, h, w = noisy.shape
        z = self.input(torch.cat((noisy, known, mask.to(noisy.dtype)), dim=1))
        conditioning = self.time_proj(timestep_embedding(sigma, self.config.dim).to(z.dtype))
        key = (t, h, w, self.config.dim // self.config.heads, noisy.device)
        compiling = getattr(torch.compiler, "is_compiling", lambda: False)()
        if self.training or compiling:
            cos, sin, rotated = rope_positions(t, h, w, key[3], noisy.device)
        else:
            cached = self._rope_cache.get(key)
            if cached is None:
                cached = rope_positions(t, h, w, key[3], noisy.device)
                self._rope_cache[key] = cached
                if len(self._rope_cache) > 4:
                    self._rope_cache.popitem(last=False)
            else:
                self._rope_cache.move_to_end(key)
            cos, sin, rotated = cached
        limits = (self.config.window_frames, self.config.window_side, self.config.window_side)
        if t <= limits[0] and h <= limits[1] and w <= limits[2]:
            # Exact global-attention path for tiny clips: no padding/copy overhead.
            x = z.flatten(2).transpose(1, 2)
            for block in self.blocks:
                x = block(x, conditioning, (cos, sin, rotated))
            return self.velocity(self.norm(x)).transpose(1, 2).reshape_as(noisy)

        x, layout = partition_3d(z.permute(0, 2, 3, 4, 1), limits)
        local_cos, local_sin = partition_rope(cos, sin, layout)
        valid = valid_window_keys(layout, noisy.device)
        local_conditioning = conditioning.repeat_interleave(layout.count_per_batch, dim=0)
        for block in self.blocks:
            x = block(x, local_conditioning, (local_cos, local_sin, rotated), valid)
        y = unpartition_3d(self.velocity(self.norm(x)), layout)
        return y.permute(0, 4, 1, 2, 3)


class LatentFlow(nn.Module):
    """Rectified-flow denoiser with hard preservation of known latent regions."""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.net = VideoDiffusionTransformer(config)
        self.config = config

    def training_loss(self, clean: Tensor, mask: Tensor, *, generator: torch.Generator | None = None) -> Tensor:
        if mask.shape != clean.shape[:1] + (1,) + clean.shape[2:]:
            raise ValueError("Mask does not match latents")
        if not torch.any(mask > 0):
            raise ValueError("Training mask must contain positive pixels")
        sigma = torch.rand((clean.shape[0],), device=clean.device, generator=generator).clamp(0.01, 0.99)
        eps = torch.randn(clean.shape, device=clean.device, dtype=clean.dtype, generator=generator)
        s = sigma.view(-1, 1, 1, 1, 1)
        masked_noisy = ((1-s) * clean + s * eps) * mask + clean * (1-mask)
        predicted = self.net(masked_noisy, clean * (1-mask), mask, sigma)
        err = (predicted - (eps-clean)).square() * mask
        return err.sum() / mask.sum().clamp(min=1).mul(clean.shape[1])

    @torch.no_grad()
    def sample(self, clean: Tensor, mask: Tensor, *, steps: int = 8,
               generator: torch.Generator | None = None,
               gpu_backend: str = "auto", assume_nonempty: bool = False,
               predictor=None) -> Tensor:
        if steps < 1 or steps > 1000:
            raise ValueError("Inference steps must be between 1 and 1000")
        if mask.shape != clean.shape[:1] + (1,) + clean.shape[2:]:
            raise ValueError("Latent mask size mismatch")
        if not assume_nonempty and torch.count_nonzero(mask) == 0:
            return clean.clone()
        known = clean * (1-mask)
        noise = torch.randn(clean.shape, device=clean.device, dtype=clean.dtype, generator=generator)
        z = noise * mask + known
        from ..cuda_backend import masked_update
        sigma_tensor = clean.new_empty((clean.shape[0],))
        for i in range(steps):
            sigma = 1-i/steps
            velocity = (self.net if predictor is None else predictor)(z, known, mask, sigma_tensor.fill_(sigma))
            # Under CUDA AMP, linear/conv kernels may return bf16/fp16 while
            # mask arithmetic promotes the latent to fp32. Match the output
            # dtype before a fused kernel (the pure torch path did this implicitly).
            if velocity.dtype != z.dtype:
                velocity = velocity.to(z.dtype)
            z = masked_update(z, velocity, known, mask, 1.0 / steps, backend=gpu_backend)
        return z

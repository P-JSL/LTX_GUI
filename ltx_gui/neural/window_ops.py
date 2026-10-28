"""3D non-overlapping token windows; pure tensor permutations with lossless inverse.

A fixed window bounds local attention memory independently of full clip length.
This is a *local* approximation, not an equivalent replacement for global attention.
"""
from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True)
class WindowLayout:
    batch: int
    original: tuple[int, int, int]
    padded: tuple[int, int, int]
    window: tuple[int, int, int]

    @property
    def count_per_batch(self) -> int:
        return (self.padded[0] // self.window[0]
                * self.padded[1] // self.window[1]
                * self.padded[2] // self.window[2])

    @property
    def tokens_per_window(self) -> int:
        return self.window[0] * self.window[1] * self.window[2]


def partition_3d(x: Tensor, limits: tuple[int, int, int]) -> tuple[Tensor, WindowLayout]:
    """[B,T,H,W,C] -> [B*num_windows, window_tokens, C].

    `limits` are upper bounds; a smaller input never expands to a larger window.
    """
    if x.ndim != 5 or any(d < 1 for d in x.shape) or len(limits) != 3 or any(v < 1 for v in limits):
        raise ValueError("Expected nonempty [B,T,H,W,C] and positive window limits")
    b, t, h, w, c = x.shape
    wt, wh, ww = (min(limit, size) for limit, size in zip(limits, (t, h, w)))
    pt = -t % wt
    ph = -h % wh
    pw = -w % ww
    y = F.pad(x, (0, 0, 0, pw, 0, ph, 0, pt))
    tt, hh, wx = t + pt, h + ph, w + pw
    nt, nh, nw = tt // wt, hh // wh, wx // ww
    windows = (y.reshape(b, nt, wt, nh, wh, nw, ww, c)
               .permute(0, 1, 3, 5, 2, 4, 6, 7)
               .reshape(b * nt * nh * nw, wt * wh * ww, c))
    return windows, WindowLayout(b, (t, h, w), (tt, hh, wx), (wt, wh, ww))


def unpartition_3d(windows: Tensor, layout: WindowLayout) -> Tensor:
    """Reconstruct original [B,T,H,W,C], dropping any padded border tokens."""
    if windows.ndim != 3 or windows.shape[:2] != (layout.batch * layout.count_per_batch, layout.tokens_per_window):
        raise ValueError("Window tensor does not match layout")
    b = layout.batch
    tt, hh, wx = layout.padded
    wt, wh, ww = layout.window
    nt, nh, nw = tt // wt, hh // wh, wx // ww
    channels = windows.shape[-1]
    y = (windows.reshape(b, nt, nh, nw, wt, wh, ww, channels)
         .permute(0, 1, 4, 2, 5, 3, 6, 7)
         .reshape(b, tt, hh, wx, channels))
    t, h, w = layout.original
    return y[:, :t, :h, :w, :]


def valid_window_keys(layout: WindowLayout, device: torch.device) -> Tensor | None:
    """Boolean [B*num_windows,1,1,tokens] attention key mask; None when no padding."""
    if layout.original == layout.padded:
        return None
    t, h, w = layout.original
    mask = torch.ones((1, t, h, w, 1), dtype=torch.bool, device=device)
    keys, _ = partition_3d(mask, layout.window)
    return keys[..., 0].repeat(layout.batch, 1)[:, None, None, :]


def partition_rope(cos: Tensor, sin: Tensor, layout: WindowLayout) -> tuple[Tensor, Tensor]:
    """Remap global absolute 3D RoPE to local windows without resetting positions."""
    t, h, w = layout.original
    half = cos.shape[-1]
    if cos.shape != (1, 1, t * h * w, half) or sin.shape != cos.shape:
        raise ValueError("Rotary tensors are inconsistent with input geometry")
    cp, _ = partition_3d(cos.reshape(1, t, h, w, half), layout.window)
    sp, _ = partition_3d(sin.reshape(1, t, h, w, half), layout.window)
    return (cp.repeat(layout.batch, 1, 1)[:, None, :, :],
            sp.repeat(layout.batch, 1, 1)[:, None, :, :])

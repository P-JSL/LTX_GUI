"""Explicit training entry point; never silently substitutes random weights."""
from __future__ import annotations

import argparse
import sys

from .model import ModelConfig
from .training import fit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train native compact masked video flow on owned/licensed NPZ data")
    parser.add_argument("dataset", help="NPZ file or directory with videos.npy and masks.npy; uint8 arrays")
    parser.add_argument("--output", required=True, help="New local .pt checkpoint; never overwritten")
    parser.add_argument("--vae-steps", type=int, default=100)
    parser.add_argument("--flow-steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dim", type=int, default=96)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--heads", type=int, default=3)
    parser.add_argument("--window-frames", type=int, default=4)
    parser.add_argument("--window-side", type=int, default=8)
    parser.add_argument("--causal-temporal", action="store_true", help="Train learned causal temporal filters")
    args = parser.parse_args(argv)
    cfg = ModelConfig(dim=args.dim, depth=args.depth, heads=args.heads,
                      window_frames=args.window_frames, window_side=args.window_side,
                      causal_temporal=args.causal_temporal)
    try:
        target = fit(args.dataset, args.output, cfg, vae_steps=args.vae_steps,
                     flow_steps=args.flow_steps, lr=args.lr, device=args.device)
    except (ValueError, OSError, RuntimeError, FileExistsError) as exc:
        print(f"학습 오류: {exc}", file=sys.stderr)
        return 1
    print(f"Checkpoint written: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

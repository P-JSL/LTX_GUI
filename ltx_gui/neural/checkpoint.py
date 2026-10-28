"""Strict, atomic, local training checkpoint handling (no arbitrary class import)."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from dataclasses import asdict

from .model import ModelConfig, VideoVAE, LatentFlow

CHECKPOINT_VERSION = 1


def save_trained(path: str | Path, config: ModelConfig, vae: VideoVAE,
                 flow: LatentFlow, *, vae_steps: int, flow_steps: int) -> Path:
    import torch
    config.validate()
    if vae_steps < 0 or flow_steps < 0:
        raise ValueError("Negative step count")
    target = Path(path).expanduser().resolve()
    if target.exists():
        raise FileExistsError(f"Checkpoint already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    obj = {"version": CHECKPOINT_VERSION, "config": asdict(config),
           "vae": vae.state_dict(), "flow": flow.state_dict(),
           "vae_steps": int(vae_steps), "flow_steps": int(flow_steps)}
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, suffix=".pt", prefix=".incomplete_", delete=False) as f:
            temporary = Path(f.name)
        torch.save(obj, temporary)
        # Hardlink is atomic, preserves any competing user's output.
        os.link(temporary, target)
        return target
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_trained(path: str | Path, device: str = "cpu", *,
                 require_trained: bool = True) -> tuple[VideoVAE, LatentFlow, dict]:
    import torch
    file = Path(path)
    # Weights-only mode rejects arbitrary pickle code execution.
    data = torch.load(file, map_location="cpu", weights_only=True)
    if not isinstance(data, dict) or data.get("version") != CHECKPOINT_VERSION:
        raise ValueError("Unsupported or invalid native LTX_GUI checkpoint")
    config = ModelConfig(**data["config"])
    config.validate()
    if require_trained and (data.get("vae_steps", 0) < 1 or data.get("flow_steps", 0) < 1):
        raise ValueError("Model has not been trained in both VAE and diffusion phases")
    vae, flow = VideoVAE(config), LatentFlow(config)
    # strict=True rejects missing/unexpected tensors and incorrect shapes.
    vae.load_state_dict(data["vae"], strict=True)
    flow.load_state_dict(data["flow"], strict=True)
    vae.to(device).eval()
    flow.to(device).eval()
    return vae, flow, {"config": config, "vae_steps": data["vae_steps"], "flow_steps": data["flow_steps"]}

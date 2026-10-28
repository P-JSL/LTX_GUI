"""Independent experimental latent video diffusion model. No Jasna or LTX imports.

Architecture is LTX-*inspired*, NOT checkpoint-compatible with official LTX-2.5.
Actual generated quality depends on training and validation, not code shape alone.
"""
from .model import ModelConfig, VideoVAE, VideoDiffusionTransformer, LatentFlow
from .checkpoint import load_trained, save_trained

__all__ = ["ModelConfig", "VideoVAE", "VideoDiffusionTransformer", "LatentFlow", "load_trained", "save_trained"]

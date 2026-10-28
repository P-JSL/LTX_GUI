"""Train the independent VAE and rectified-flow model on licensed video data.

For large datasets, provide videos.npy and masks.npy inside a directory. NumPy
memmaps avoid allocating the entire corpus, and each training step transfers
only the selected video to GPU. Single-file NPZ remains supported for small sets.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .model import ModelConfig, VideoVAE, LatentFlow
from .pipeline import latent_mask_from_pixels, pad_video
from .checkpoint import save_trained


class VideoTrainingDataset:
    """Read-only dataset with optional memory mapped NPY arrays."""
    def __init__(self, path: str | Path):
        self.root = Path(path)
        self._npz = None
        if self.root.is_dir():
            self.videos = np.load(self.root / "videos.npy", mmap_mode="r", allow_pickle=False)
            self.masks = np.load(self.root / "masks.npy", mmap_mode="r", allow_pickle=False)
        else:
            self._npz = np.load(self.root, allow_pickle=False)
            self.videos = self._npz["videos"]
            self.masks = self._npz["masks"]
        if self.videos.dtype != np.uint8 or self.masks.dtype != np.uint8:
            raise ValueError("Dataset must use uint8 arrays")
        if (self.videos.ndim != 5 or self.videos.shape[-1] != 3
                or self.masks.shape != self.videos.shape[:4]):
            raise ValueError("Videos [N,T,H,W,3] and masks [N,T,H,W] shape mismatch")
        if not self.videos.shape[0] or min(self.videos.shape[1:4]) < 1:
            raise ValueError("Dataset is empty")
        # Memmap mode scans masks once but never materializes all frames on GPU.
        self.nonempty = [i for i in range(len(self.videos)) if np.any(self.masks[i])]
        if not self.nonempty:
            raise ValueError("Dataset has no restoration masks")

    def __len__(self):
        return self.videos.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Small contiguous copy avoids read-only NumPy/memmap shared-storage hazards.
        video = torch.from_numpy(np.array(self.videos[index], copy=True))
        mask = torch.from_numpy(np.array(self.masks[index], copy=True))
        video = video.permute(3, 0, 1, 2).unsqueeze(0).float().div_(255)
        mask = (mask.unsqueeze(0).unsqueeze(0) > 0).float()
        return pad_video(video, mask)[:2]

    def close(self):
        if self._npz is not None:
            self._npz.close()
            self._npz = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def read_dataset(path: str | Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Backwards-compatible small-dataset helper (materializes all samples)."""
    with VideoTrainingDataset(path) as data:
        samples = [data[i] for i in range(len(data))]
        # Existing API returns unpadded tensors; only `fit` pads one sample.
        _, t, h, w, _ = data.videos.shape
        return (torch.cat([v[:, :, :t, :h, :w] for v, _ in samples]),
                torch.cat([m[:, :, :t, :h, :w] for _, m in samples]))


def fit(dataset: str | Path, output: str | Path, config: ModelConfig,
        *, vae_steps: int = 1, flow_steps: int = 1, lr: float = 1e-4,
        device: str = "cpu", seed: int = 42) -> Path:
    config.validate()
    if vae_steps < 1 or flow_steps < 1 or not 0 < lr < 1:
        raise ValueError("Training requires positive steps and learning rate")
    torch.manual_seed(seed)
    vae, flow = VideoVAE(config).to(device), LatentFlow(config).to(device)
    with VideoTrainingDataset(dataset) as data:
        optimizer = torch.optim.AdamW(vae.parameters(), lr=lr)
        vae.train()
        for step in range(vae_steps):
            batch, _ = data[step % len(data)]
            batch = batch.to(device)
            restored, mu, logvar = vae(batch)
            recon_loss = F.l1_loss(restored, batch)
            kl = .5 * (mu.square() + logvar.exp() - logvar - 1).mean()
            loss = recon_loss + 1e-4 * kl
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        vae.eval().requires_grad_(False)
        optimizer = torch.optim.AdamW(flow.parameters(), lr=lr)
        flow.train()
        for step in range(flow_steps):
            video, mask = data[data.nonempty[step % len(data.nonempty)]]
            video, mask = video.to(device), mask.to(device)
            with torch.no_grad():
                z, _ = vae.encode(video)
            latent_mask = latent_mask_from_pixels(mask, z.shape[2:])
            if not torch.any(latent_mask):
                # Should be unreachable when spatial pooling preserves masks.
                raise ValueError("Training mask disappeared during latent pooling")
            loss = flow.training_loss(z, latent_mask)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return save_trained(output, config, vae, flow,
                        vae_steps=vae_steps, flow_steps=flow_steps)

"""Single-shape full native inference CUDA Graph: encode -> sample -> decode -> composite.

NVDEC/NVENC are separate hardware engines and cannot be captured by a PyTorch
CUDA Graph. Graph replays the GPU inference operations *only*.
"""
from __future__ import annotations

from contextlib import nullcontext


class NativeCudaGraph:
    """Owns static buffers and one captured graph; not thread-safe.

    Reuse with the *same padded* [B,3,T,H,W] input shape and model instances.
    Every call copies current input, mask and new seeded noise to fixed buffers.
    Output is cloned to avoid invalidation by the next replay.
    """
    def __init__(self, vae, flow, example_video, example_mask, *, steps: int = 8,
                 amp: str = "none", predictor=None, encoder=None, decoder=None):
        import torch
        from .neural.pipeline import latent_mask_from_pixels
        from .cuda_backend import masked_update, masked_composite

        if example_video.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("CUDA Graph requires an available CUDA device")
        if not 1 <= steps <= 1000 or amp not in ("none", "bf16", "fp16"):
            raise ValueError("Invalid graph configuration")
        if example_mask.shape != (example_video.shape[0], 1, *example_video.shape[2:]):
            raise ValueError("Invalid graph mask")
        if vae.training or flow.training:
            raise ValueError("CUDA Graph inference requires eval() models")
        # Runtime-generated Triton launches may compile on the capture stream.
        # Use PyTorch (CUDA SDPA/Inductor) kernels during graph capture/replay.
        from .neural.model import RMSNorm
        for module in (*vae.modules(), *flow.modules()):
            if isinstance(module, RMSNorm):
                module.gpu_backend = "torch"
        self._vae, self._flow = vae, flow  # Keep parameters alive for replay.
        self.shape = tuple(example_video.shape)
        self.steps = steps
        self.amp = amp
        self.device = example_video.device
        self.video = torch.empty_like(example_video).copy_(example_video)
        self.mask = torch.empty_like(example_mask).copy_(example_mask)
        self.encoder = encoder or vae.encode
        self.predictor = predictor or flow.net
        self.decoder = decoder or vae.decode
        autocast = (torch.autocast(device_type="cuda", dtype=torch.bfloat16 if amp == "bf16" else torch.float16)
                    if amp != "none" else nullcontext())
        with torch.inference_mode(), autocast:
            probe, _ = self.encoder(self.video)
        self.noise = torch.empty_like(probe)
        self.noise.zero_()
        b = self.video.shape[0]
        self.sigmas = torch.empty((b,), device=self.device, dtype=self.video.dtype)

        def compute():
            with torch.inference_mode(), autocast:
                clean, _ = self.encoder(self.video)
                latent_mask = latent_mask_from_pixels(self.mask, clean.shape[2:])
                known = clean * (1 - latent_mask)
                z = self.noise * latent_mask + known
                for index in range(steps):
                    velocity = self.predictor(z, known, latent_mask, self.sigmas.fill_(1 - index / steps))
                    z = masked_update(z, velocity.to(z.dtype), known, latent_mask,
                                      1.0 / steps, backend="torch")
                rgb = self.decoder(z).to(self.video.dtype)
                # Inputs have already been padded to the VAE contract.
                return masked_composite(self.video, rgb, self.mask, backend="torch")

        # Isolate warmup allocations from graph capture; never capture first-run JIT.
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            for _ in range(3):
                _ = compute()
        torch.cuda.current_stream(self.device).wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.result = compute()

    def run(self, video, mask, *, seed: int = 42):
        import torch
        if tuple(video.shape) != self.shape or mask.shape != self.mask.shape:
            raise ValueError("CUDA Graph has fixed input shapes; create another graph for this clip")
        if video.device != self.device or mask.device != self.device:
            raise ValueError("CUDA Graph inputs must remain on the same GPU")
        if seed < 0 or seed >= 2**63:
            raise ValueError("Invalid seed")
        self.video.copy_(video)
        self.mask.copy_(mask)
        generator = torch.Generator(device=self.device).manual_seed(seed)
        self.noise.copy_(torch.randn(self.noise.shape, generator=generator,
                                     device=self.device, dtype=self.noise.dtype))
        self.graph.replay()
        return self.result.clone()

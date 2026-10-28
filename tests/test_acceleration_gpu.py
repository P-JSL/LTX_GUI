"""Real NVIDIA GPU checks; unconditionally skipped when no CUDA is present."""
import pytest
import torch

from ltx_gui.neural.model import LatentFlow, ModelConfig, VideoVAE
from ltx_gui.neural.pipeline import accelerated_inpaint_tensor


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires real NVIDIA CUDA GPU")
def test_full_inference_graph_matches_fixed_seed_eager():
    config = ModelConfig(base_channels=4, dim=24, heads=2, depth=1)
    vae = VideoVAE(config).eval().cuda()
    flow = LatentFlow(config).eval().cuda()
    video = torch.rand((1, 3, 9, 32, 32), device="cuda")
    mask = torch.zeros((1, 1, 9, 32, 32), device="cuda")
    mask[:, :, :, 8:16, 8:16] = 1
    baseline = accelerated_inpaint_tensor(video, mask, vae, flow, steps=2,
                                           gpu_backend="torch", seed=123)
    cache = {}
    actual = accelerated_inpaint_tensor(video, mask, vae, flow, steps=2,
                                       gpu_backend="torch", seed=123,
                                       cuda_graph=True, runtime_cache=cache)
    assert torch.allclose(actual, baseline, atol=2e-4, rtol=2e-4)
    torch.testing.assert_close(actual[mask.expand_as(actual) == 0],
                               video[mask.expand_as(video) == 0], atol=0, rtol=0)
    assert next(iter(cache))[1] is True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires real NVIDIA CUDA GPU")
def test_graph_replay_different_inputs_refreshes_static_buffers():
    config = ModelConfig(base_channels=4, dim=24, heads=2, depth=1)
    vae, flow = VideoVAE(config).eval().cuda(), LatentFlow(config).eval().cuda()
    video = torch.rand((1, 3, 9, 32, 32), device="cuda")
    mask = torch.zeros((1, 1, 9, 32, 32), device="cuda")
    mask[:, :, :, 3:13, 3:13] = 1
    cache = {}
    first = accelerated_inpaint_tensor(video, mask, vae, flow, steps=1,
                                       cuda_graph=True, runtime_cache=cache, seed=1)
    second = accelerated_inpaint_tensor(video * 0.5, mask, vae, flow, steps=1,
                                        cuda_graph=True, runtime_cache=cache, seed=2)
    assert torch.equal(first[mask.expand_as(first) == 0], video[mask.expand_as(video) == 0])
    assert torch.equal(second[mask.expand_as(second) == 0],
                       (video * 0.5)[mask.expand_as(video) == 0])

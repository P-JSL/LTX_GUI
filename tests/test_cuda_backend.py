"""CPU reference + optional GPU equivalence tests (never silently fake a CUDA run)."""
import pytest
import torch

from ltx_gui.cuda_backend import masked_update, masked_composite, backend_status, fused_rmsnorm
from ltx_gui.core import Options
from ltx_gui.neural.model import ModelConfig, LatentFlow, RMSNorm
from ltx_gui.neural.pipeline import clip_arrays_to_tensor, inpaint_tensor


def _inputs(device="cpu", dtype=torch.float32):
    torch.manual_seed(19)
    shape = (2, 3, 3, 4, 5)
    z = torch.rand(shape, device=device, dtype=dtype)
    vel = torch.rand_like(z)
    known = torch.rand_like(z)
    mask = torch.randint(0, 2, (2, 1, 3, 4, 5), device=device).float()
    return z, vel, known, mask


def test_cpu_reference_update_and_composite():
    z, vel, known, mask = _inputs()
    actual = masked_update(z, vel, known, mask, .125)
    assert torch.equal(actual, (z - vel * .125) * mask + known)
    combined = masked_composite(z, known, mask)
    assert torch.allclose(combined, z + (known-z)*mask)
    assert torch.equal(combined[mask.expand_as(z)==0], z[mask.expand_as(z)==0])


def test_dtype_and_shape_validation():
    z, vel, known, mask = _inputs()
    with pytest.raises(ValueError, match='mask'):
        masked_update(z, vel, known, mask[:, :, :2], 1.)
    with pytest.raises(ValueError, match='Unknown'):
        masked_update(z, vel, known, mask, 1., backend='fake')
    with pytest.raises(ValueError, match='CUDA'):
        masked_composite(z, known, mask, backend='cuda')
    with pytest.raises(ValueError, match='CUDA'):
        fused_rmsnorm(z, torch.ones(z.shape[-1]), backend='triton')
    assert isinstance(backend_status()['native_extension'], bool)


def test_settings_validate_gpu_modes():
    Options(neural_gpu_backend='auto').validate()
    with pytest.raises(ValueError, match='GPU'):
        Options(neural_gpu_backend='invalid').validate()
    with pytest.raises(ValueError, match='CUDA'):
        Options(neural_amp='bf16', neural_device='cpu').validate()


def test_sampling_uses_fused_update_with_torch_backend():
    cfg = ModelConfig(latent_channels=4, base_channels=4, dim=24, heads=2,
                      depth=1, window_frames=2, window_side=2)
    flow = LatentFlow(cfg).eval()
    clean = torch.rand(1, 4, 2, 2, 2)
    mask = torch.ones(1, 1, 2, 2, 2)
    a = flow.sample(clean, mask, steps=2, generator=torch.Generator().manual_seed(9), gpu_backend='torch')
    b = flow.sample(clean, mask, steps=2, generator=torch.Generator().manual_seed(9), gpu_backend='auto', assume_nonempty=True)
    assert torch.equal(a, b)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Requires real CUDA GPU')
@pytest.mark.parametrize('backend', ['auto', 'triton', 'cuda'])
def test_real_cuda_matches_reference(backend):
    from ltx_gui.cuda_backend import _import_triton, _import_native
    if backend == 'triton' and (_import_triton() is None or not _import_triton().is_available()):
        pytest.skip('Triton not installed')
    if backend == 'cuda' and _import_native() is None:
        pytest.skip('CUDA extension not built')
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        z, v, known, mask = _inputs('cuda', dtype)
        expected = masked_update(z, v, known, mask, .125, backend='torch')
        actual = masked_update(z, v, known, mask, .125, backend=backend)
        assert torch.allclose(actual.float(), expected.float(), atol=2e-3, rtol=2e-3)
        expected = masked_composite(z, known, mask, backend='torch')
        actual = masked_composite(z, known, mask, backend=backend)
        assert torch.allclose(actual.float(), expected.float(), atol=2e-3, rtol=2e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Requires real CUDA GPU')
def test_triton_unpack_matches_cpu():
    import numpy as np
    from ltx_gui.cuda_backend import _import_triton
    if _import_triton() is None or not _import_triton().is_available():
        pytest.skip('Triton unavailable')
    rng = np.random.default_rng(19)
    frames = rng.integers(0, 256, (3, 7, 6, 3), dtype=np.uint8)
    masks = rng.integers(0, 256, (3, 7, 6), dtype=np.uint8)
    expected = clip_arrays_to_tensor(frames, masks, 'cpu', gpu_backend='torch')
    actual = clip_arrays_to_tensor(frames, masks, 'cuda', gpu_backend='triton')
    assert torch.allclose(actual[0].cpu(), expected[0], atol=2e-7)
    assert torch.equal(actual[1].cpu(), expected[1])


def test_cpu_pipeline_pixel_preservation_and_rope_cache():
    from ltx_gui.neural.model import VideoVAE
    torch.set_num_threads(min(2, torch.get_num_threads()))
    cfg = ModelConfig(latent_channels=4, base_channels=4, dim=24, heads=2,
                      depth=1, window_frames=2, window_side=2)
    vae, flow = VideoVAE(cfg).eval(), LatentFlow(cfg).eval()
    frame = torch.rand(1, 3, 9, 32, 32)
    mask = torch.zeros(1, 1, 9, 32, 32)
    mask[:, :, :, 8:12, 8:12] = 1
    output = inpaint_tensor(frame, mask, vae, flow, steps=1, gpu_backend='torch')
    assert output.shape == frame.shape
    assert torch.equal(output[~mask.bool().expand_as(frame)], frame[~mask.bool().expand_as(frame)])
    assert len(flow.net._rope_cache) == 1
    inpaint_tensor(frame, mask, vae, flow, steps=1, gpu_backend='torch')
    assert len(flow.net._rope_cache) == 1


def test_sample_injected_compiled_predictor():
    cfg = ModelConfig(latent_channels=4, base_channels=4, dim=24, heads=2,
                      depth=1, window_frames=2, window_side=2)
    model = LatentFlow(cfg).eval()
    z, mask = torch.randn(1, 4, 1, 2, 2), torch.ones(1, 1, 1, 2, 2)
    calls = []
    def predictor(*args):
        calls.append(len(args))
        return torch.zeros_like(args[0])
    output = model.sample(z, mask, steps=3, gpu_backend='torch', predictor=predictor)
    assert calls == [4, 4, 4]
    assert output.shape == z.shape


def test_sample_predictor_mixed_precision_output_cast():
    cfg = ModelConfig(latent_channels=4, base_channels=4, dim=24, heads=2,
                      depth=1, window_frames=2, window_side=2)
    model = LatentFlow(cfg).eval()
    clean = torch.ones(1, 4, 1, 2, 2)
    mask = torch.ones(1, 1, 1, 2, 2)
    def mixed_predictor(*args):
        return torch.zeros_like(args[0], dtype=torch.float16)
    result = model.sample(clean, mask, steps=2, gpu_backend='torch', predictor=mixed_predictor)
    assert result.dtype == torch.float32
    assert torch.isfinite(result).all()

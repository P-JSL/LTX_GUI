"""Hardware-independent gates. GPU coverage is in scripts/validate_acceleration.py."""
import numpy as np
import pytest
import torch

from ltx_gui.acceleration import AccelerationSettings, require_cuda, validate_calibration, load_fp8_calibration
from ltx_gui.core import Options
from ltx_gui.cuda_graph import NativeCudaGraph
from ltx_gui.neural.model import ModelConfig, VideoVAE
from ltx_gui.neural.pipeline import accelerated_inpaint_tensor
from ltx_gui.nvcodec_pipeline import CudaABGRFrame, import_decoded_frame, _mux_command


def test_settings_reject_cpu_tensorrt():
    with pytest.raises(ValueError, match="CUDA"):
        Options(neural_accelerator="tensorrt", neural_device="cpu").validate()


def test_settings_reject_cpu_graph():
    with pytest.raises(ValueError, match="CUDA"):
        Options(neural_cuda_graph=True, neural_device="cpu").validate()


def test_fp8_requires_real_calibration():
    with pytest.raises(ValueError, match="npz"):
        Options(neural_accelerator="tensorrt", neural_device="cuda", neural_precision="fp8").validate()


def test_fp8_requires_hopper_when_capability_known():
    config = AccelerationSettings(engine="tensorrt", precision="fp8")
    with pytest.raises(ValueError, match="Hopper"):
        config.validate(capability=(8, 9))
    config.validate(capability=(9, 0))


def test_fp8_reject_wrong_engine():
    with pytest.raises(ValueError, match="calibrated TensorRT"):
        AccelerationSettings(precision="fp8").validate()


def test_unknown_engine_fails():
    with pytest.raises(ValueError):
        AccelerationSettings(engine="magic").validate()


def test_workspace_bounds():
    with pytest.raises(ValueError):
        AccelerationSettings(workspace_gib=0).validate()


def test_gpu_required_for_requested_acceleration():
    if torch.cuda.is_available():
        pytest.skip("CPU-only guard test")
    with pytest.raises(RuntimeError, match="CUDA"):
        require_cuda(AccelerationSettings(engine="tensorrt"), "cuda:0")


def test_graph_requires_real_gpu():
    if torch.cuda.is_available():
        pytest.skip("CPU-only guard test")
    from ltx_gui.neural.model import LatentFlow
    conf = ModelConfig(base_channels=4, dim=24, heads=2, depth=1)
    model, flow = VideoVAE(conf).eval(), LatentFlow(conf).eval()
    video = torch.rand(1, 3, 9, 32, 32)
    mask = torch.ones(1, 1, 9, 32, 32)
    with pytest.raises(RuntimeError, match="CUDA"):
        NativeCudaGraph(model, flow, video, mask)


def test_invalid_calibration_no_random_fallback():
    with pytest.raises(ValueError, match="representative"):
        validate_calibration([])
    with pytest.raises(ValueError, match="tuples"):
        validate_calibration(["random"])


def test_load_calibration_shape_validation(tmp_path):
    file = tmp_path / "calibration.npz"
    np.savez(file, frames=np.zeros((1, 9, 32, 32, 3), dtype=np.uint8),
             masks=np.zeros((1, 8, 32, 32), dtype=np.uint8))
    vae = VideoVAE(ModelConfig(base_channels=4, dim=24, heads=2, depth=1)).eval()
    with pytest.raises(ValueError, match="Calibration batch"):
        load_fp8_calibration(str(file), vae, torch.zeros(1, 3, 9, 32, 32),
                             torch.zeros(1, 1, 9, 32, 32))


def test_load_calibration_valid(tmp_path):
    file = tmp_path / "valid.npz"
    np.savez(file, frames=np.ones((1, 9, 32, 32, 3), dtype=np.uint8),
             masks=np.ones((1, 9, 32, 32), dtype=np.uint8))
    vae = VideoVAE(ModelConfig(base_channels=4, dim=24, heads=2, depth=1)).eval()
    examples = load_fp8_calibration(str(file), vae, torch.zeros(1, 3, 9, 32, 32),
                                    torch.zeros(1, 1, 9, 32, 32))
    assert set(examples) == {"encoder", "dit", "decoder"}
    assert examples["dit"][0][0].shape == (1, 16, 2, 1, 1)


def test_device_encoder_rejects_host_tensor():
    with pytest.raises(ValueError, match="CUDA"):
        CudaABGRFrame(torch.zeros(32, 32, 4, dtype=torch.uint8))


def test_dlpack_cpu_rejected():
    host = torch.zeros(3, 32, 32, dtype=torch.uint8)
    with pytest.raises(RuntimeError, match="CUDA"):
        import_decoded_frame(host)


def test_nvcodec_options_restrictions():
    with pytest.raises(ValueError, match="CUDA"):
        Options(neural_video_io="nvcodec").validate()
    Options(neural_video_io="nvcodec", method="neural", neural_device="cuda:0").validate()


def test_mux_uses_separate_audio_mapping():
    cmd = _mux_command('ffmpeg', 'raw.h264', 'source.mp4', 'output.mkv', 30)
    assert cmd[cmd.index('-map') + 1] == '0:v:0'
    assert '1:a?' in cmd
    assert 'copy' in cmd


def test_accelerated_torch_matches_standard_path():
    from ltx_gui.neural.model import LatentFlow
    conf = ModelConfig(base_channels=4, dim=24, heads=2, depth=1)
    vae, flow = VideoVAE(conf).eval(), LatentFlow(conf).eval()
    video = torch.rand(1, 3, 9, 32, 32)
    mask = torch.zeros(1, 1, 9, 32, 32)
    mask[:, :, :, 8:16, 8:16] = 1
    result = accelerated_inpaint_tensor(video, mask, vae, flow, steps=1, seed=12,
                                       engine='torch')
    assert result.shape == video.shape
    assert torch.equal(result[mask.expand_as(video)==0], video[mask.expand_as(video)==0])


def test_precision_flag_does_not_silently_ignore_torch():
    with pytest.raises(ValueError, match="AMP"):
        Options(neural_device="cuda", neural_accelerator="torch", neural_precision="fp16").validate()


def test_trt_compilation_rejects_any_partial_component(monkeypatch):
    """Test strict failure handling without pretending to run the NVIDIA backend."""
    import types
    from ltx_gui.acceleration import compile_tensorrt_components
    import ltx_gui.acceleration as accel
    from ltx_gui.neural.model import LatentFlow
    conf = ModelConfig(base_channels=4, dim=24, heads=2, depth=1)
    vae, flow = VideoVAE(conf).eval(), LatentFlow(conf).eval()
    video = torch.rand(1, 3, 9, 32, 32)
    mask = torch.ones(1, 1, 9, 32, 32)
    monkeypatch.setattr(accel, "require_cuda", lambda *args: 0)
    calls = []
    def fake_compile(model, **kwargs):
        calls.append((model, kwargs))
        assert kwargs["require_full_compilation"] is True
        if len(calls) == 2:
            raise RuntimeError("unsupported operator")
        return model
    import sys
    monkeypatch.setitem(sys.modules, "torch_tensorrt", types.SimpleNamespace(compile=fake_compile))
    with pytest.raises(RuntimeError, match="TensorRT full-graph conversion rejected dit"):
        compile_tensorrt_components(vae, flow, video, mask, AccelerationSettings(engine="tensorrt"))
    assert len(calls) == 2


def test_trt_compilation_requires_strict_coverage(monkeypatch):
    from ltx_gui.acceleration import compile_tensorrt_components
    conf = ModelConfig(base_channels=4, dim=24, heads=2, depth=1)
    from ltx_gui.neural.model import LatentFlow
    with pytest.raises(ValueError, match="mandatory"):
        compile_tensorrt_components(VideoVAE(conf), LatentFlow(conf),
            torch.zeros(1,3,9,32,32), torch.ones(1,1,9,32,32),
            AccelerationSettings(engine="tensorrt", require_full_compilation=False))

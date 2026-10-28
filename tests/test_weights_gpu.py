from pathlib import Path
import pytest
from ltx_gui.weights import LTX25ModelPack,validate_weight_mix
from ltx_gui.gpu_ops import mask_composite
from ltx_gui.ltx_runtime import RetakeRequest


def test_checkpoint_directory_contract(tmp_path):
    p=LTX25ModelPack.from_directory(tmp_path)
    assert p.transformer.name=='ltx-2.5-22b-distilled-transformer-bf16.safetensors'
    assert p.video_vae.name=='ltx-2.5-video-vae-conv-bf16.safetensors'
    with pytest.raises(FileNotFoundError):p.verify()


def test_no_untrained_mean_weights(tmp_path):
    a=tmp_path/'a.safetensors';a.touch()
    with pytest.raises(ValueError):validate_weight_mix([a],[1.0])
    with pytest.raises(FileNotFoundError):validate_weight_mix([a,tmp_path/'missing'],[.5,.5])
    b=tmp_path/'b.safetensors';b.touch()
    with pytest.raises(ValueError):validate_weight_mix([a,b],[.3,.3])
    validate_weight_mix([a,b],[.5,.5])


def test_cuda_blend_cpu_equivalence():
    torch = pytest.importorskip("torch")
    orig=torch.rand(2,3,4,4)
    generated=torch.rand_like(orig)
    mask=torch.zeros((2,1,4,4));mask[:,:,1:3,1:3]=1
    out=mask_composite(orig,generated,mask)
    assert torch.equal(out[mask.expand_as(orig)==0],orig[mask.expand_as(orig)==0])
    assert torch.allclose(out[mask.expand_as(orig)==1],generated[mask.expand_as(orig)==1],atol=1e-7)
    with pytest.raises(RuntimeError):mask_composite(orig,generated,mask,compile_cuda=True)


def test_retake_preflight_does_not_invent_weights(tmp_path):
    video=tmp_path/'src.mp4';video.touch()
    req=RetakeRequest(video,tmp_path/'out.mp4',tmp_path,'a natural scene',0,1)
    req.validate()
    with pytest.raises(FileNotFoundError):
        from ltx_gui.ltx_runtime import run_temporal_retake
        run_temporal_retake(req)

"""Regression for the bounded-memory 3D attention execution path."""
import pytest
import torch

from ltx_gui.neural.model import ModelConfig, VideoDiffusionTransformer
from ltx_gui.neural.window_ops import partition_3d, unpartition_3d, valid_window_keys, partition_rope
from ltx_gui.neural.model import rope_positions


@pytest.mark.parametrize("dims,limits,batch", [((1,1,1),(4,8,8),1),((5,3,4),(2,2,3),2),((8,9,17),(4,8,8),1)])
def test_partition_inverse(dims,limits,batch):
    x=torch.randn(batch,*dims,7)
    windows, layout=partition_3d(x, limits)
    assert torch.equal(x,unpartition_3d(windows,layout))
    mask=valid_window_keys(layout, x.device)
    if any(d % min(d,l) for d,l in zip(dims,limits)):
        assert mask is not None and not torch.all(mask)
    else:
        assert mask is None


def test_absolute_rope_windows():
    dims=(3,5,7)
    cos,sin,k=rope_positions(*dims, 12, torch.device('cpu'))
    packed,layout=partition_3d(cos.reshape(1,*dims,-1),(2,4,4))
    a,b=partition_rope(cos,sin,layout)
    assert torch.equal(a[:,0],packed)
    assert k>0 and b.shape==a.shape


def test_local_forward_backward_and_masked_padding():
    config=ModelConfig(latent_channels=4,dim=24,heads=2,depth=1,window_frames=2,window_side=2)
    net=VideoDiffusionTransformer(config)
    z=torch.randn(2,4,3,3,4,requires_grad=True)
    known=torch.randn_like(z)
    mask=torch.ones(2,1,3,3,4)
    output=net(z,known,mask,torch.tensor([0.2,0.8]))
    assert output.shape==z.shape and torch.isfinite(output).all()
    output.sum().backward()
    assert z.grad is not None and torch.isfinite(z.grad).all()


def test_full_window_same_as_global_numerically():
    config=ModelConfig(latent_channels=4,dim=24,heads=2,depth=1,window_frames=2,window_side=2)
    a=VideoDiffusionTransformer(config)
    b=VideoDiffusionTransformer(ModelConfig(latent_channels=4,dim=24,heads=2,depth=1,window_frames=4,window_side=4))
    b.load_state_dict(a.state_dict())
    z=torch.randn(1,4,2,2,2)
    sigma=torch.tensor([.42])
    assert torch.allclose(a(z,z,torch.ones(1,1,2,2,2),sigma), b(z,z,torch.ones(1,1,2,2,2),sigma),atol=1e-6)


def test_reject_unbounded_attention_window():
    with pytest.raises(ValueError,match="4096"):
        ModelConfig(window_frames=64,window_side=64).validate()


def test_causal_mixer_excludes_future_and_initializes_as_identity():
    from ltx_gui.neural.model import CausalTemporalDepthwise
    layer=CausalTemporalDepthwise(3)
    x=torch.randn(1,3,4,2,2)
    assert torch.allclose(layer(x),x,atol=1e-6)
    with torch.no_grad():
        layer.conv.weight[:,0,0,0,0]=0.2
        layer.conv.weight[:,0,1,0,0]=0.3
    y=layer(x)
    x2=x.clone();x2[:,:,3]+=100
    assert torch.equal(y[:,:,:3],layer(x2)[:,:,:3])

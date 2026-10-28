import numpy as np
import pytest
import torch

from ltx_gui.neural.training import VideoTrainingDataset, fit
from ltx_gui.neural.model import ModelConfig
from ltx_gui.neural.checkpoint import load_trained


def tiny_config():
    return ModelConfig(latent_channels=4, base_channels=4, dim=24, heads=2, depth=1,
                       window_frames=2, window_side=2, causal_temporal=True)


def test_memmap_dataset_trains_and_saves(tmp_path):
    torch.set_num_threads(min(2,torch.get_num_threads()))
    path=tmp_path/'dataset';path.mkdir()
    rng=np.random.default_rng(0)
    video=rng.integers(0,256,size=(2,3,32,32,3),dtype=np.uint8)
    mask=np.zeros((2,3,32,32),np.uint8);mask[1,:,12:20,12:20]=255
    np.save(path/'videos.npy',video);np.save(path/'masks.npy',mask)
    with VideoTrainingDataset(path) as data:
        assert isinstance(data.videos,np.memmap)
        assert data.nonempty==[1]
        v,m=data[0]
        assert v.shape==(1,3,9,32,32) and m.sum()==0
    target=fit(path,tmp_path/'trained.pt',tiny_config(),vae_steps=1,flow_steps=1,lr=1e-4)
    vae,flow,meta=load_trained(target)
    assert meta['vae_steps']==meta['flow_steps']==1
    assert vae.config.causal_temporal
    with pytest.raises(FileExistsError):
        fit(path,target,tiny_config(),vae_steps=1,flow_steps=1)


def test_dataset_rejects_missing_masks(tmp_path):
    data=tmp_path/'bad';data.mkdir()
    np.save(data/'videos.npy',np.zeros((1,9,32,32,3),np.uint8))
    np.save(data/'masks.npy',np.zeros((1,9,32,32),np.uint8))
    with pytest.raises(ValueError,match='no restoration masks'):
        VideoTrainingDataset(data)


def test_read_dataset_preserves_old_unpadded_shapes(tmp_path):
    from ltx_gui.neural.training import read_dataset
    folder=tmp_path/'old';folder.mkdir()
    videos=np.zeros((1,3,32,32,3),np.uint8)
    masks=np.zeros((1,3,32,32),np.uint8);masks[0,0,1,1]=255
    np.save(folder/'videos.npy',videos);np.save(folder/'masks.npy',masks)
    a,b=read_dataset(folder)
    assert a.shape==(1,3,3,32,32) and b.shape==(1,1,3,32,32)

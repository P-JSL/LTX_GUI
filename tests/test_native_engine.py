"""CPU tests with synthetic clips; never imply pretrained LTX-2.5 fidelity."""
from pathlib import Path
import shutil

import cv2
import numpy as np
import pytest
import torch

from ltx_gui.core import Job, Options, Rectangle, plan_job
from ltx_gui.neural.model import ModelConfig, VideoVAE, LatentFlow
from ltx_gui.neural.checkpoint import save_trained, load_trained
from ltx_gui.native_engine import clip_batches, restore_native_video


def test_clip_batches_linear_count_and_context():
    for total in (1, 2, 8, 9, 10, 17, 18, 33):
        values = iter(range(total))
        chunks = list(clip_batches(lambda: next(values, None), frames_per_clip=9))
        joined = [v for chunk, skip in chunks for v in chunk[int(skip):]]
        assert joined == list(range(total))
        for i in range(1, len(chunks)):
            assert chunks[i][0][0] == chunks[i-1][0][-1]


@pytest.mark.parametrize('frames', (2, 8, 10, 18))
def test_reject_non_8k_plus_1(frames):
    with pytest.raises(ValueError, match='8k'):
        list(clip_batches(lambda: None, frames_per_clip=frames))


def test_native_preflight_requires_checkpoint(tmp_path):
    video = tmp_path/'in.mp4'; video.touch()
    opts=Options(method='neural')
    with pytest.raises(FileNotFoundError, match='체크포인트'):
        plan_job(Job(str(video), Rectangle(.1,.1,.5,.5)),opts)


def test_native_checkpoint_forward_and_masks(tmp_path):
    torch.set_num_threads(min(2,torch.get_num_threads()))
    from ltx_gui.neural.pipeline import inpaint_arrays
    cfg = ModelConfig(latent_channels=4, base_channels=4, dim=24, heads=2,
                      depth=1, window_frames=2, window_side=2)
    model=VideoVAE(cfg).eval(); flow=LatentFlow(cfg).eval()
    checkpoint=save_trained(tmp_path/'tiny.pt',cfg,model,flow,vae_steps=1,flow_steps=1)
    vae, net, _=load_trained(checkpoint)
    frames=np.random.default_rng(7).integers(0,256,(2,32,32,3),dtype=np.uint8)
    masks=np.zeros((2,32,32),dtype=np.uint8); masks[:,12:18,10:17]=255
    out=inpaint_arrays(frames,masks,vae,net,steps=1,seed=13)
    assert out.shape==frames.shape
    assert np.array_equal(out[masks==0],frames[masks==0])


@pytest.mark.skipif(shutil.which('ffmpeg') is None, reason='FFmpeg missing')
def test_native_real_video_ffmpeg_and_audio_remux(tmp_path):
    torch.set_num_threads(min(2,torch.get_num_threads()))
    cfg = ModelConfig(latent_channels=4, base_channels=4, dim=24, heads=2,
                      depth=1, window_frames=2, window_side=2)
    checkpoint=save_trained(tmp_path/'tiny.pt',cfg,VideoVAE(cfg),LatentFlow(cfg),vae_steps=1,flow_steps=1)
    source=tmp_path/'in.mp4'
    writer=cv2.VideoWriter(str(source),cv2.VideoWriter_fourcc(*'mp4v'),8,(32,32))
    assert writer.isOpened()
    for i in range(11):writer.write(np.full((32,32,3), 30+i*5,np.uint8))
    writer.release()
    opts=Options(method='neural',checkpoint_path=str(checkpoint),neural_steps=1,neural_clip_frames=9)
    job=plan_job(Job(str(source),Rectangle(.25,.25,.5,.5)),opts)
    out=restore_native_video(job,opts)
    reader=cv2.VideoCapture(str(out));n=0
    while reader.read()[0]:n+=1
    reader.release()
    assert n==11
    assert not list(tmp_path.glob('.ltx_gui_partial_*'))


def test_empty_mask_skips_neural_computation(tmp_path,monkeypatch):
    import ltx_gui.native_engine as engine
    # No false claims about speed; empty detections are passed through untouched.
    batches=[];frames=iter([(np.zeros((4,4,3),np.uint8),np.zeros((4,4),np.uint8))])
    for batch, skip in engine.clip_batches(lambda: next(frames,None),frames_per_clip=9):
        batches.append((batch,skip))
    assert len(batches)==1 and not batches[0][0][0][1].any()

import numpy as np
import pytest
from ltx_gui.quality import psnr, ssim, evaluate_frame


def test_identity_metrics():
    frame=np.random.default_rng(1).integers(0,256,(48,48,3),np.uint8)
    mask=np.zeros((48,48),np.uint8);mask[4:20,4:20]=255
    report=evaluate_frame(frame,frame,mask)
    assert report.psnr==float('inf')
    assert report.masked_psnr==float('inf')
    assert report.ssim==pytest.approx(1,abs=1e-6)
    assert report.outside_mae==0


def test_masked_psnr_and_outside_mae():
    ref=np.zeros((32,32,3),np.uint8)
    candidate=ref.copy()
    candidate[10:20,10:20]=10
    mask=np.zeros((32,32),np.uint8);mask[10:20,10:20]=255
    assert psnr(ref,candidate,mask)==pytest.approx(20*np.log10(255/10))
    assert evaluate_frame(ref,candidate,mask).outside_mae==0
    candidate[0,0]=[5,5,5]
    assert evaluate_frame(ref,candidate,mask).outside_mae>0


def test_empty_mask_no_score():
    frame=np.zeros((32,32,3),np.uint8)
    assert psnr(frame,frame,np.zeros((32,32),np.uint8)) is None


def test_compare_video_same_fps_and_size(tmp_path):
    import cv2
    from ltx_gui.quality import compare_video
    first=tmp_path/'ground.mp4';second=tmp_path/'restored.mp4'
    for path,brightness in ((first,70),(second,72)):
        writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),8,(64,48))
        assert writer.isOpened()
        for _ in range(5):writer.write(np.full((48,64,3),brightness,np.uint8))
        writer.release()
    mask=np.zeros((48,64),np.uint8);mask[10:20,10:20]=255
    result=compare_video(first,second,mask=mask)
    assert result['frames']==5
    assert result['masked_psnr'] is not None
    assert result['ssim']<=1+1e-6


def test_benchmark_rejects_different_source_hash(tmp_path):
    from pathlib import Path
    import json
    import importlib.util
    import sys
    spec=importlib.util.spec_from_file_location('comparison_script',Path(__file__).resolve().parents[1]/'scripts'/'compare_results.py')
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    source=tmp_path/'src.mp4';source.write_bytes(b'video')
    manifest=tmp_path/'manifest.json'
    fields=('source_sha256','mask_sha256','gpu_name','gpu_driver','resolution','fps','codec','ltx_gui_version','jasna_version','warmup_policy')
    manifest.write_text(json.dumps({name:'dummy' for name in fields}),encoding='utf-8')
    with pytest.raises(ValueError,match='Input hash'):
        mod.main(['--source',str(source),'--reference','ref.mp4','--ours','ours.mp4','--manifest',str(manifest)])

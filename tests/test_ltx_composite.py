import shutil
import cv2
import numpy as np
import pytest
from ltx_gui.core import Job, Rectangle, Options, plan_job
from ltx_gui.engine import restore_video


@pytest.mark.skipif(shutil.which('ffmpeg') is None,reason='FFmpeg unavailable')
def test_ltx_generated_masked_composite(tmp_path):
    source=tmp_path/'original.mp4';generated=tmp_path/'generated.mp4'
    for path,color in ((source,(60,30,20)),(generated,(200,160,140))):
        writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),8,(64,48))
        assert writer.isOpened()
        for _ in range(9):writer.write(np.full((48,64,3),color,np.uint8))
        writer.release()
    plan=plan_job(Job(str(source),Rectangle(.25,.25,.75,.75)),Options())
    target=restore_video(plan,Options(),generated_video=generated)
    cap=cv2.VideoCapture(str(target));ok,result=cap.read();cap.release()
    assert ok
    assert np.mean(result[24,32])>110
    assert np.mean(result[2,2])<90


@pytest.mark.skipif(shutil.which('ffmpeg') is None,reason='FFmpeg unavailable')
def test_generated_frame_size_rejected(tmp_path):
    source=tmp_path/'src.mp4';generated=tmp_path/'wrong.mp4'
    for path,sz in ((source,(64,48)),(generated,(32,32))):
        writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),8,sz)
        for _ in range(9):writer.write(np.zeros((sz[1],sz[0],3),np.uint8))
        writer.release()
    plan=plan_job(Job(str(source),Rectangle(.1,.1,.3,.3)),Options())
    with pytest.raises(ValueError,match='해상도'):
        restore_video(plan,Options(),generated_video=generated)
    assert not plan.target.exists()

@pytest.mark.skipif(shutil.which('ffmpeg') is None,reason='FFmpeg unavailable')
def test_auto_detection_real_video_with_synthetic_detections(tmp_path,monkeypatch):
    from ltx_gui import detection
    source=tmp_path/'moving.mp4'; model=tmp_path/'mosaic.onnx';model.touch()
    writer=cv2.VideoWriter(str(source),cv2.VideoWriter_fourcc(*'mp4v'),8,(64,48))
    for i in range(9):
        frame=np.full((48,64,3),55,np.uint8)
        frame[12:24,10+i:22+i]=(70,190,170)
        writer.write(frame)
    writer.release()
    class MockDetector:
        def __init__(self,config):self.count=0
        def detect(self,frame):
            from ltx_gui.detection import Detection
            i=self.count;self.count+=1
            return [Detection(10+i,12,22+i,24,.9)]
    monkeypatch.setattr(detection,'OnnxMosaicDetector',MockDetector)
    opts=Options(detector_model=str(model))
    plan=plan_job(Job(str(source),auto_detect=True),opts)
    output=restore_video(plan,opts)
    cap=cv2.VideoCapture(str(output));count=0
    while cap.read()[0]:count+=1
    cap.release()
    assert count==9

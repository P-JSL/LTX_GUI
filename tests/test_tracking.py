import numpy as np
from ltx_gui.detection import Detection
from ltx_gui.tracking import MosaicTracker, iou
from ltx_gui.core import Options
from ltx_gui.engine import SpatialTemporalInpaint


def test_iou_bounds():
    assert iou((0,0,10,10),(0,0,10,10))==1.0
    assert iou((0,0,10,10),(11,11,20,20))==0.0


def test_dynamic_track_and_mask():
    tracker=MosaicTracker(max_gap=1,padding=0)
    frame=np.zeros((32,64,3),np.uint8)
    first=tracker.update(frame,[Detection(10,10,20,20,.95)])
    second=tracker.update(frame,[Detection(12,10,22,20,.95)])
    assert first.sum()==10*10*255
    assert second.sum()==10*10*255
    assert second[15,21]==255
    assert tracker.tracks[0].identifier==1
    tracker.update(frame,[])
    assert tracker.tracks
    tracker.update(frame,[])
    assert not tracker.tracks


def test_scene_cut_resets_track_ids():
    tracker=MosaicTracker(max_gap=5,padding=0)
    black=np.zeros((32,64,3),np.uint8)
    white=np.full_like(black,255)
    tracker.update(black,[Detection(5,5,10,10,.9)])
    trackmask=tracker.update(white,[])
    assert not np.any(trackmask)


def test_dynamic_inpaint_preserves_outside_and_empty():
    frame=np.random.default_rng(4).integers(0,256,(40,60,3),dtype=np.uint8)
    zero=np.zeros((40,60),np.uint8)
    processor=SpatialTemporalInpaint(zero,Options(method='temporal'))
    assert np.array_equal(processor.process(frame,zero),frame)
    mask=zero.copy();mask[10:20,15:25]=255
    out=processor.process(frame,mask)
    assert np.array_equal(out[mask==0],frame[mask==0])
    assert out.shape==frame.shape

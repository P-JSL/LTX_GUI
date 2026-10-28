from pathlib import Path
import numpy as np
import cv2
import pytest
from ltx_gui.detection import OnnxConfig, OnnxMosaicDetector, Detection, decode_predictions, letterbox


def test_letterbox_projection_non_square():
    frame=np.zeros((60,120,3),np.uint8)
    padded,scale,left,top=letterbox(frame,128)
    assert padded.shape==(128,128,3)
    assert 0 < scale <= 2
    assert top > 0
    xyxy=np.array([[10*scale+left,10*scale+top,50*scale+left,30*scale+top,.9,0]],np.float32)
    boxes=decode_predictions(xyxy,OnnxConfig('model.onnx'),(60,120),scale,left,top)
    assert len(boxes)==1
    assert all(abs(a-b)<=1 for a,b in zip(boxes[0].box,(10,10,50,30)))


def test_class_filter_and_nms():
    predictions=np.array([[10,10,40,40,.91,0],[11,11,41,41,.89,0],
                          [50,50,70,70,.98,1],[70,70,90,90,.10,0]],np.float32)
    boxes=decode_predictions(predictions,OnnxConfig('model.onnx'),(100,100),1,0,0)
    assert len(boxes)==1
    assert boxes[0].confidence==pytest.approx(.91,abs=1e-6)


def test_yolo_detect_layout():
    pred=np.array([[20,20,10,10,.8],[40,40,10,10,.9]],np.float32)
    boxes=decode_predictions(pred,OnnxConfig('model.onnx',format='yolo_cls'),(100,100),1,0,0)
    assert len(boxes)==2


def test_onnx_model_missing_fails():
    with pytest.raises(FileNotFoundError):
        OnnxConfig('/no-such-model-mosaic.onnx').validate()


def test_mocked_onnx_session(tmp_path):
    model=tmp_path/'mosaic.onnx'; model.write_bytes(b'fake model for injected session')
    class Input: name='images'
    class Session:
        def get_inputs(self):return [Input()]
        def run(self,outputs,data):
            assert data['images'].shape==(1,3,640,640)
            return [np.array([[[10,10,60,60,.9,0]]],np.float32)]
    detector=OnnxMosaicDetector(OnnxConfig(str(model)),session=Session())
    assert len(detector.detect(np.zeros((640,640,3),np.uint8)))==1

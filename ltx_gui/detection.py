"""Independent ONNX mosaic-region inference; no Jasna imports or model weights.

A user-supplied ONNX model must have a compatible detection head. This module
implements preprocessing, decoding, confidence threshold and NMS. It does not
claim that an arbitrary object-detection model recognizes pixelation.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np


@dataclass(frozen=True)
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float

    def __post_init__(self):
        if self.x1 < 0 or self.y1 < 0 or self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError('Invalid detection bounds')
        if not 0 <= self.confidence <= 1 or not np.isfinite(self.confidence):
            raise ValueError('Invalid confidence')

    @property
    def box(self) -> tuple[int, int, int, int]:
        return self.x1, self.y1, self.x2, self.y2


class Detector(Protocol):
    def detect(self, frame: np.ndarray) -> list[Detection]: ...


@dataclass(frozen=True)
class OnnxConfig:
    model: str
    input_size: int = 640
    confidence: float = 0.35
    nms_iou: float = 0.45
    # Supported outputs: [N,6] xyxy+conf+class, or [4+C,N] xywh+class probabilities.
    format: str = 'xyxy6'
    mosaic_class: int = 0
    cuda: bool = False

    def validate(self) -> None:
        if not Path(self.model).is_file():
            raise FileNotFoundError(self.model)
        if self.input_size < 32 or self.input_size % 32:
            raise ValueError('Input size must be divisible by 32')
        if not 0 < self.confidence <= 1 or not 0 < self.nms_iou < 1:
            raise ValueError('Invalid detection thresholds')
        if self.format not in {'xyxy6', 'yolo_cls'} or self.mosaic_class < 0:
            raise ValueError('Invalid ONNX output format/class')


def letterbox(frame: np.ndarray, size: int) -> tuple[np.ndarray, float, int, int]:
    h, w = frame.shape[:2]
    if not h or not w:
        raise ValueError('Empty image')
    scale = min(size / w, size / h)
    rw, rh = min(size, round(w * scale)), min(size, round(h * scale))
    left, top = (size-rw)//2, (size-rh)//2
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[top:top+rh, left:left+rw] = cv2.resize(frame, (rw, rh), interpolation=cv2.INTER_LINEAR)
    return canvas, scale, left, top


def decode_predictions(pred: np.ndarray, config: OnnxConfig,
                       image_shape: tuple[int, int], scale: float, left: int, top: int) -> list[Detection]:
    """Decode a documented tensor layout only; fail instead of guessing model semantics."""
    array = np.asarray(pred)
    if array.ndim == 3:
        if array.shape[0] != 1:
            raise ValueError('Only batch size 1 is supported')
        array = array[0]
    if config.format == 'xyxy6':
        if array.ndim != 2 or array.shape[1] != 6:
            raise ValueError('xyxy6 expects [N,6]')
        coords, scores, classes = array[:, :4], array[:, 4], array[:, 5]
    else:
        if array.ndim != 2:
            raise ValueError('yolo_cls expects [4+C,N]')
        if (array.shape[1] < 5 + config.mosaic_class and array.shape[0] >= 5 + config.mosaic_class) or (5 <= array.shape[0] <= 256 and array.shape[0] < array.shape[1] and array.shape[1] > 256):
            array = array.T
        if array.shape[1] < 5 + config.mosaic_class:
            raise ValueError('yolo_cls requires 4 boxes and C class probabilities')
        centers = array[:, :4]
        coords = np.column_stack((centers[:,0]-centers[:,2]/2, centers[:,1]-centers[:,3]/2,
                                  centers[:,0]+centers[:,2]/2, centers[:,1]+centers[:,3]/2))
        scores = array[:, 4 + config.mosaic_class]
        classes = np.full(len(scores), config.mosaic_class)
    if len(scores) > 100000:
        raise ValueError('Oversized detector output')
    h,w = image_shape
    keep = np.isfinite(scores) & (scores >= config.confidence) & (classes == config.mosaic_class)
    candidates = []
    for c, score in zip(coords[keep], scores[keep]):
        if not np.isfinite(c).all() or not 0 <= float(score) <= 1:
            continue
        x1, y1, x2, y2 = (float(c[0]-left)/scale, float(c[1]-top)/scale,
                           float(c[2]-left)/scale, float(c[3]-top)/scale)
        x1, y1 = max(0, min(w, x1)), max(0, min(h, y1))
        x2, y2 = max(0, min(w, x2)), max(0, min(h, y2))
        if x2-x1 < 2 or y2-y1 < 2:
            continue
        candidates.append((int(x1), int(y1), int(np.ceil(x2)), int(np.ceil(y2)), float(score)))
    if not candidates:
        return []
    rectangles = [[x1,y1,x2-x1,y2-y1] for x1,y1,x2,y2,_ in candidates]
    selected = cv2.dnn.NMSBoxes(rectangles, [c[4] for c in candidates], config.confidence, config.nms_iou)
    return [Detection(*candidates[int(i)]) for i in np.asarray(selected).reshape(-1)]


class OnnxMosaicDetector:
    def __init__(self, config: OnnxConfig, session=None):
        config.validate()
        self.config = config
        if session is None:
            try:
                import onnxruntime as ort
            except ImportError as exc:
                raise RuntimeError('Install onnxruntime or onnxruntime-gpu') from exc
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if config.cuda else ['CPUExecutionProvider']
            if config.cuda and 'CUDAExecutionProvider' not in ort.get_available_providers():
                raise RuntimeError('CUDAExecutionProvider not available; refusing silent CPU fallback')
            session = ort.InferenceSession(config.model, providers=providers)
        self.session = session
        self.input_name = session.get_inputs()[0].name

    def detect(self, frame: np.ndarray) -> list[Detection]:
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
            raise ValueError('Detector expects HWC BGR uint8')
        config = self.config
        padded, scale, left, top = letterbox(frame, config.input_size)
        inp = np.ascontiguousarray(padded[:,:,::-1].transpose(2,0,1), np.float32)[None]/255.0
        prediction = self.session.run(None, {self.input_name: inp})[0]
        return decode_predictions(prediction, config, frame.shape[:2], scale, left, top)

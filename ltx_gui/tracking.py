"""Short-gap track association and per-frame dynamic masks (original implementation)."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .detection import Detection


def iou(a: tuple[int,int,int,int], b: tuple[int,int,int,int]) -> float:
    x1, y1 = max(a[0],b[0]), max(a[1],b[1])
    x2, y2 = min(a[2],b[2]), min(a[3],b[3])
    intersection = max(0,x2-x1)*max(0,y2-y1)
    area_a=(a[2]-a[0])*(a[3]-a[1]); area_b=(b[2]-b[0])*(b[3]-b[1])
    union=area_a+area_b-intersection
    return intersection/union if union else 0.0


@dataclass
class Track:
    identifier: int
    box: tuple[int,int,int,int]
    score: float
    lost: int = 0


class MosaicTracker:
    """Greedy IoU assignment; unmatched tracks survive a bounded gap.

    Dense pair matching is O(tracks*observations); spatial gating is a later
    optimization when benchmarked crowded videos demonstrate necessity.
    """
    def __init__(self, *, min_iou: float=0.2, max_gap: int=2, padding: int=3,
                 scene_cut_threshold: float=50.0):
        if not 0 <= min_iou < 1 or max_gap < 0 or padding < 0:
            raise ValueError('Invalid tracker parameters')
        self.min_iou, self.max_gap, self.padding = min_iou, max_gap, padding
        self.scene_cut_threshold = scene_cut_threshold
        self.tracks: list[Track] = []
        self._next_id = 1
        self._previous: np.ndarray | None = None

    def update(self, frame: np.ndarray, observations: list[Detection]) -> np.ndarray:
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
            raise ValueError('Expected BGR uint8 frame')
        h,w=frame.shape[:2]
        small = cv2.resize(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),(64,36))
        if self._previous is not None:
            if np.mean(cv2.absdiff(small,self._previous)) > self.scene_cut_threshold:
                self.tracks.clear()
        self._previous = small
        # Highest IoU first, each detection and track used at most once.
        pairs=[]
        for ti, track in enumerate(self.tracks):
            for di, obs in enumerate(observations):
                overlap=iou(track.box,obs.box)
                if overlap >= self.min_iou:
                    pairs.append((overlap,ti,di))
        pairs.sort(reverse=True)
        used_tracks=set(); used_detections=set()
        for _, ti, di in pairs:
            if ti in used_tracks or di in used_detections:
                continue
            track = self.tracks[ti]; obs=observations[di]
            track.box=obs.box; track.score=obs.confidence; track.lost=0
            used_tracks.add(ti); used_detections.add(di)
        for ti,track in enumerate(self.tracks):
            if ti not in used_tracks:
                track.lost+=1
        self.tracks=[t for t in self.tracks if t.lost<=self.max_gap]
        for di,obs in enumerate(observations):
            if di not in used_detections:
                self.tracks.append(Track(self._next_id,obs.box,obs.confidence))
                self._next_id+=1
        mask=np.zeros((h,w),dtype=np.uint8)
        for track in self.tracks:
            x1,y1,x2,y2=track.box
            mask[max(0,y1-self.padding):min(h,y2+self.padding),
                 max(0,x1-self.padding):min(w,x2+self.padding)]=255
        return mask

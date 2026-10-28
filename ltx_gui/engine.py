"""Original, independent mask-guided spatial/temporal inpainting engine.

Uses OpenCV algorithms and FFmpeg as libraries/tools; imports no Jasna/LTX code.
This is classical video inpainting, not an LTX neural diffusion model.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .core import Options, PlannedJob, Rectangle


@dataclass(frozen=True)
class Progress:
    frames: int
    total: int
    fps: float


class Cancelled(Exception):
    pass


def make_mask(shape: tuple[int, int], *, rectangle: Rectangle | None = None,
              image: np.ndarray | None = None) -> np.ndarray:
    """Create 8-bit mask: 255 means synthesize, 0 means preserve original."""
    h, w = shape
    if h <= 0 or w <= 0:
        raise ValueError("영상 해상도가 올바르지 않습니다.")
    if (rectangle is None) == (image is None):
        raise ValueError("영역 또는 이미지를 하나만 지정해야 합니다.")
    mask = np.zeros((h, w), np.uint8)
    if rectangle is not None:
        x0, y0, x1, y1 = rectangle.pixels(w, h)
        mask[y0:y1, x0:x1] = 255
    else:
        if image is None or image.ndim not in (2, 3):
            raise ValueError("마스크 이미지를 읽을 수 없습니다.")
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if image.shape != (h, w):
            # INTER_NEAREST prevents unintentional grayscale interpolation near edges.
            image = cv2.resize(image, (w, h), interpolation=cv2.INTER_NEAREST)
        mask = np.where(image >= 128, 255, 0).astype(np.uint8)
    if not cv2.countNonZero(mask):
        raise ValueError("선택된 복원 영역이 없습니다.")
    if cv2.countNonZero(mask) == w * h:
        raise ValueError("전체 프레임을 가리면 인페인팅 참조 픽셀이 없습니다.")
    return mask


def mask_bbox(mask: np.ndarray, padding: int) -> tuple[int, int, int, int]:
    nz = cv2.findNonZero(mask)
    if nz is None:
        return (0, 0, 0, 0)
    x, y, w, h = cv2.boundingRect(nz)
    return max(0, x-padding), max(0, y-padding), min(mask.shape[1], x+w+padding), min(mask.shape[0], y+h+padding)


class SpatialTemporalInpaint:
    """Stateful streaming processor. Memory O(W*H), no full-video buffering."""
    def __init__(self, mask: np.ndarray, options: Options) -> None:
        options.validate()
        if mask.ndim != 2 or mask.dtype != np.uint8:
            raise ValueError("8-bit 단일 채널 마스크가 필요합니다.")
        self.mask = mask
        self.options = options
        self.box = mask_bbox(mask, max(12, options.radius*5))
        self.previous_original: np.ndarray | None = None
        self.previous_result: np.ndarray | None = None
        if options.method == "temporal":
            x0, y0, x1, y1 = self.box
            self.grid_x, self.grid_y = np.meshgrid(
                np.arange(x1-x0, dtype=np.float32), np.arange(y1-y0, dtype=np.float32))

    def process(self, frame: np.ndarray) -> np.ndarray:
        if frame.ndim != 3 or frame.shape[:2] != self.mask.shape or frame.shape[2] != 3 or frame.dtype != np.uint8:
            raise ValueError("영상 프레임 해상도/형식이 마스크와 일치하지 않습니다.")
        x0, y0, x1, y1 = self.box
        input_roi = frame[y0:y1, x0:x1]
        mask_roi = self.mask[y0:y1, x0:x1]
        restored_roi = cv2.inpaint(input_roi, mask_roi, self.options.radius, cv2.INPAINT_TELEA)
        if self.options.method == "temporal" and self.previous_original is not None:
            previous_roi = self.previous_original[y0:y1, x0:x1]
            previous_result_roi = self.previous_result[y0:y1, x0:x1]
            # Reject an apparent scene cut rather than blending unrelated frames.
            # Compare only pixels outside the obscured region; threshold is heuristic.
            cur_gray = cv2.cvtColor(input_roi, cv2.COLOR_BGR2GRAY)
            old_gray = cv2.cvtColor(previous_roi, cv2.COLOR_BGR2GRAY)
            cut_difference = cv2.mean(cv2.absdiff(cur_gray, old_gray),
                                      cv2.bitwise_not(mask_roi))[0]
            if cut_difference >= 30.0:
                result = frame.copy()
                cv2.copyTo(restored_roi, mask_roi, result[y0:y1, x0:x1])
                self.previous_original = frame.copy()
                self.previous_result = result.copy()
                return result
            # Flow from current to previous; remap previous restoration into current coords.
            # Compute on reduced image, then restore vectors to full-resolution units.
            factor = self.options.flow_scale
            if factor < 1:
                small_size = (max(16, round(cur_gray.shape[1]*factor)), max(16, round(cur_gray.shape[0]*factor)))
                cur_small = cv2.resize(cur_gray, small_size, interpolation=cv2.INTER_AREA)
                old_small = cv2.resize(old_gray, small_size, interpolation=cv2.INTER_AREA)
            else:
                cur_small, old_small = cur_gray, old_gray
            flow = cv2.calcOpticalFlowFarneback(cur_small, old_small, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            rh, rw = cur_gray.shape
            sh, sw = cur_small.shape
            if (sw, sh) != (rw, rh):
                flow = cv2.resize(flow, (rw, rh), interpolation=cv2.INTER_LINEAR)
                flow[..., 0] *= rw/sw
                flow[..., 1] *= rh/sh
            map_x, map_y = self.grid_x+flow[..., 0], self.grid_y+flow[..., 1]
            warped = cv2.remap(previous_result_roi, map_x, map_y, cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)
            # Only blend inside the target mask. Source pixels remain identical.
            weight = self.options.temporal_weight
            if weight:
                blend = cv2.addWeighted(restored_roi, 1-weight, warped, weight, 0)
                restored_roi = np.where(mask_roi[..., None] != 0, blend, restored_roi)
        result = frame.copy()
        target = result[y0:y1, x0:x1]
        cv2.copyTo(restored_roi, mask_roi, target)
        self.previous_original = frame.copy() if self.options.method == "temporal" else None
        self.previous_result = result.copy() if self.options.method == "temporal" else None
        return result


def read_mask(mask_path: Path) -> np.ndarray:
    # np.fromfile + imdecode supports Unicode file names on Windows.
    pixels = np.fromfile(str(mask_path), dtype=np.uint8)
    image = cv2.imdecode(pixels, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"마스크 PNG를 디코딩할 수 없습니다: {mask_path}")
    return image


def ffmpeg_args(binary: str, width: int, height: int, fps: float, source: Path,
                temp_target: Path, encoder: str) -> list[str]:
    if fps <= 0 or fps > 1000 or width <= 0 or height <= 0:
        raise ValueError("프레임 형식이 올바르지 않습니다.")
    if encoder not in ("libx264", "h264_nvenc"):
        raise ValueError("허용되지 않은 인코더입니다.")
    return [binary, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s:v", f"{width}x{height}",
            "-r", f"{fps:.8f}", "-i", "pipe:0", "-i", str(source),
            "-map", "0:v:0", "-map", "1:a?", "-c:v", encoder,
            "-preset", "medium" if encoder == "libx264" else "p4", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", "-f", "matroska" if temp_target.suffix == ".mkv" else "mp4",
            str(temp_target)]


def restore_video(planned: PlannedJob, options: Options, stop: threading.Event | None = None,
                  progress: Callable[[Progress], None] | None = None) -> Path:
    """Decode -> independent inpaint -> encode/remux. Partial output is never published.

    No shell, no Jasna. Caller owns worker thread. Video is assumed constant FPS.
    """
    stop = stop or threading.Event()
    options.validate()
    if planned.target.exists():
        raise FileExistsError(f"출력 파일이 이미 존재합니다: {planned.target}")
    if not shutil.which(options.ffmpeg) and not Path(options.ffmpeg).is_file():
        raise FileNotFoundError(f"FFmpeg 실행 파일을 찾지 못했습니다: {options.ffmpeg}")
    video = cv2.VideoCapture(str(planned.source))
    if not video.isOpened():
        raise ValueError(f"비디오를 열 수 없습니다: {planned.source}")
    out_file: Path | None = None
    proc: subprocess.Popen | None = None
    log_path: Path | None = None
    try:
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(video.get(cv2.CAP_PROP_FPS))
        total = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        if not 0 < fps <= 1000 or width <= 0 or height <= 0:
            raise ValueError("영상 FPS 또는 해상도가 올바르지 않습니다.")
        mask_image = read_mask(planned.mask_path) if planned.mask_path else None
        mask = make_mask((height, width), rectangle=planned.job.rectangle, image=mask_image)
        processor = SpatialTemporalInpaint(mask, options)
        planned.target.parent.mkdir(parents=True, exist_ok=True)
        # Unique staging file + exclusive creation avoids collisions among processes.
        with tempfile.NamedTemporaryFile(dir=str(planned.target.parent), suffix=planned.target.suffix,
                                         prefix=".ltx_gui_partial_", delete=False) as f:
            out_file = Path(f.name)
        with tempfile.NamedTemporaryFile(dir=str(planned.target.parent), suffix=".ffmpeg.log", delete=False) as f:
            log_path = Path(f.name)
        with open(log_path, "wb") as log:
            args = ffmpeg_args(options.ffmpeg, width, height, fps, planned.source, out_file, options.encoder)
            startup = {"stdin": subprocess.PIPE, "stdout": subprocess.DEVNULL, "stderr": log}
            if os.name == "nt":
                startup["creationflags"] = subprocess.CREATE_NO_WINDOW
            proc = subprocess.Popen(args, **startup)
            count = 0
            t0 = time.perf_counter()
            while True:
                if stop.is_set():
                    raise Cancelled("사용자가 처리를 취소했습니다.")
                ok, frame = video.read()
                if not ok:
                    break
                result = processor.process(frame)
                assert proc.stdin is not None
                proc.stdin.write(result.tobytes())
                count += 1
                if progress and (count == 1 or count % 5 == 0 or (total and count == total)):
                    elapsed = max(time.perf_counter()-t0, 1e-8)
                    progress(Progress(count, total, count/elapsed))
            if count == 0:
                raise ValueError("입력 영상에 유효한 프레임이 없습니다.")
            # CAP_PROP_FRAME_COUNT can be estimated for some containers, hence 1 frame tolerance.
            if total > 1 and count < total - 1:
                raise ValueError(f"영상 디코딩이 조기에 중단되었습니다: {count}/{total} 프레임")
            assert proc.stdin is not None
            proc.stdin.close()
            ret = proc.wait()
            if ret:
                raise RuntimeError(f"FFmpeg 인코딩 실패({ret}): {log_path.read_text(errors='replace')[-1600:]}")
            if stop.is_set():
                raise Cancelled("사용자가 처리를 취소했습니다.")
        # Atomic no-overwrite publication on the same volume (NTFS / supported local FS).
        # Unlike exists()+replace(), os.link() will not destroy a file created concurrently.
        try:
            os.link(out_file, planned.target)
        except FileExistsError:
            raise FileExistsError(f"영상 처리 중 동일한 출력 파일이 생성되었습니다: {planned.target}")
        return planned.target
    except FileExistsError:
        raise
    except OSError as exc:
        detail = log_path.read_text(errors="replace")[-1600:] if log_path and log_path.exists() else ""
        raise RuntimeError(f"영상 파이프라인 실패: {exc}. FFmpeg: {detail}") from exc
    finally:
        video.release()
        if proc and proc.poll() is None:
            proc.kill()
            proc.wait()
        if out_file:
            out_file.unlink(missing_ok=True)
        if log_path:
            log_path.unlink(missing_ok=True)

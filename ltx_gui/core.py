"""Pure, independently authored job planning for an in-process video restoration engine."""
from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

VIDEO_EXTENSIONS = frozenset({".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v"})
METHODS = ("fast", "temporal", "neural")
ENCODERS = ("libx264", "h264_nvenc")


@dataclass(frozen=True)
class Rectangle:
    """Normalized rectangle: 0 <= left < right <= 1, likewise top/bottom."""
    left: float
    top: float
    right: float
    bottom: float

    def validate(self) -> None:
        if not (0 <= self.left < self.right <= 1 and 0 <= self.top < self.bottom <= 1):
            raise ValueError("영역 좌표는 0~1 범위에서 좌측<우측, 상단<하단이어야 합니다.")

    def pixels(self, width: int, height: int) -> tuple[int, int, int, int]:
        self.validate()
        import math
        return (math.floor(self.left * width), math.floor(self.top * height),
                math.ceil(self.right * width), math.ceil(self.bottom * height))

    @classmethod
    def parse(cls, value: str) -> Rectangle:
        values = [part.strip() for part in value.split(",")]
        if len(values) != 4:
            raise ValueError("영역은 x1,y1,x2,y2 형식이며 값은 0~1입니다.")
        try:
            obj = cls(*(float(item) for item in values))
        except ValueError as exc:
            raise ValueError("영역 좌표는 숫자 네 개여야 합니다.") from exc
        obj.validate()
        return obj

    def as_text(self) -> str:
        return ",".join(f"{v:.5f}" for v in (self.left, self.top, self.right, self.bottom))


@dataclass(frozen=True)
class Options:
    output_dir: str = ""
    method: str = "fast"
    encoder: str = "libx264"
    radius: int = 3
    temporal_weight: float = 0.35
    flow_scale: float = 0.5
    ffmpeg: str = "ffmpeg"
    detector_model: str = ""
    detector_confidence: float = 0.35
    detector_cuda: bool = False
    detector_format: str = "xyxy6"
    detector_class: int = 0
    detector_input_size: int = 640
    tracker_iou: float = 0.20
    tracker_gap: int = 2
    tracker_padding: int = 3
    checkpoint_path: str = ""
    neural_device: str = "cpu"
    neural_steps: int = 8
    neural_clip_frames: int = 9
    neural_compile: bool = False
    neural_gpu_backend: str = "auto"
    neural_amp: str = "none"
    neural_accelerator: str = "torch"
    neural_precision: str = "fp32"
    neural_cuda_graph: bool = False
    neural_fp8_calibration: str = ""
    neural_video_io: str = "opencv"

    def validate(self) -> None:
        if self.method not in METHODS or self.encoder not in ENCODERS:
            raise ValueError("지원하지 않는 복원 모드 또는 인코더입니다.")
        if not 1 <= self.radius <= 20:
            raise ValueError("복원 반경은 1~20이어야 합니다.")
        if not 0 <= self.temporal_weight <= 0.8:
            raise ValueError("시간축 합성 강도는 0~0.8이어야 합니다.")
        if not 0.125 <= self.flow_scale <= 1:
            raise ValueError("Optical Flow 해상도 배율은 0.125~1이어야 합니다.")
        if not 0 < self.detector_confidence <= 1:
            raise ValueError("자동 탐지 신뢰도 범위는 (0,1]입니다.")
        if self.detector_format not in ("xyxy6", "yolo_cls") or self.detector_class < 0:
            raise ValueError("지원하지 않는 탐지 출력 형식 또는 클래스입니다.")
        if self.detector_input_size < 32 or self.detector_input_size % 32:
            raise ValueError("탐지 해상도는 32의 배수여야 합니다.")
        if not 0 <= self.tracker_iou < 1 or not 0 <= self.tracker_gap <= 120 or not 0 <= self.tracker_padding <= 256:
            raise ValueError("추적기 설정 범위가 올바르지 않습니다.")
        if self.neural_device != "cpu" and not (self.neural_device == "cuda" or (self.neural_device.startswith("cuda:") and self.neural_device[5:].isdigit())):
            raise ValueError("GPU 장치는 cpu 또는 cuda[:index] 형식이어야 합니다.")
        if not 1 <= self.neural_steps <= 1000:
            raise ValueError("확산 단계는 1~1000 사이여야 합니다.")
        if not 9 <= self.neural_clip_frames <= 129 or (self.neural_clip_frames-1) % 8:
            raise ValueError("신경망 클립은 9,17,...,129 프레임만 지원합니다.")
        if self.neural_gpu_backend not in ("auto", "torch", "triton", "cuda"):
            raise ValueError("GPU 커널 종류는 auto, torch, triton, cuda 중 하나입니다.")
        if self.neural_amp not in ("none", "bf16", "fp16"):
            raise ValueError("GPU 혼합 정밀도는 none, bf16, fp16 중 하나입니다.")
        if self.neural_amp != "none" and self.neural_device == "cpu":
            raise ValueError("혼합 정밀도는 CUDA 장치에서만 사용 가능합니다.")
        from .acceleration import AccelerationSettings
        AccelerationSettings(engine=self.neural_accelerator, precision=self.neural_precision,
                             cuda_graph=self.neural_cuda_graph).validate(device=self.neural_device)
        if self.neural_accelerator == "torch" and self.neural_precision != "fp32":
            raise ValueError("PyTorch 정밀도 변경은 GPU 혼합 정밀도(AMP) 설정을 사용하세요.")
        if self.neural_precision == "fp8" and not self.neural_fp8_calibration:
            raise ValueError("FP8 양자화에는 실측 대표 입력 .npz 데이터가 필요합니다.")
        if self.neural_video_io not in ("opencv", "nvcodec"):
            raise ValueError("영상 IO는 opencv 또는 nvcodec만 지원합니다.")
        if self.neural_video_io == "nvcodec" and (self.method != "neural" or not self.neural_device.startswith("cuda")):
            raise ValueError("NVDEC/NVENC 영상 IO는 CUDA 신경망 모드에서만 사용합니다.")
        if self.neural_compile and self.neural_accelerator != "torch":
            raise ValueError("torch.compile과 TensorRT는 동시에 선택할 수 없습니다.")
        if not self.ffmpeg.strip():
            raise ValueError("FFmpeg 실행 파일 경로가 필요합니다.")


@dataclass(frozen=True)
class Job:
    input_path: str
    rectangle: Rectangle | None = None
    mask_path: str = ""
    output_path: str = ""
    auto_detect: bool = False


@dataclass(frozen=True)
class PlannedJob:
    job: Job
    source: Path
    target: Path
    mask_path: Path | None


def normalize_path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def path_key(path: Path) -> str:
    # Windows is case-insensitive regardless of the OS running CI tests.
    value = str(path.resolve())
    return value.casefold() if os.name == "nt" else value


def plan_job(job: Job, options: Options) -> PlannedJob:
    options.validate()
    source = normalize_path(job.input_path)
    if not source.is_file() or source.suffix.lower() not in VIDEO_EXTENSIONS:
        raise ValueError(f"영상 파일이 없거나 확장자를 지원하지 않습니다: {source}")
    selected_modes = int(job.rectangle is not None) + int(bool(job.mask_path.strip())) + int(job.auto_detect)
    if selected_modes != 1:
        raise ValueError("마스크 이미지, 사각형 또는 자동 탐지 중 하나를 지정해야 합니다.")
    if options.neural_video_io == "nvcodec" and job.auto_detect:
        raise ValueError("NVCodec mode currently supports static masks only; use OpenCV mode for automatic detection")
    if options.neural_video_io == "nvcodec" and options.encoder != "h264_nvenc":
        raise ValueError("NVCodec mode requires h264_nvenc")
    if options.method == "neural" and not Path(options.checkpoint_path).is_file():
        raise FileNotFoundError("독립 신경망의 학습된 .pt 체크포인트를 지정해야 합니다.")
    if job.auto_detect and not Path(options.detector_model).is_file():
        raise FileNotFoundError("자동 탐지를 위한 학습된 ONNX 모델 파일을 지정해야 합니다.")
    if job.rectangle:
        job.rectangle.validate()
    mask = normalize_path(job.mask_path) if job.mask_path.strip() else None
    if mask and (not mask.is_file() or mask.suffix.lower() not in (".png", ".bmp", ".jpg", ".jpeg")):
        raise ValueError(f"마스크 이미지 파일을 확인하세요: {mask}")
    folder = normalize_path(options.output_dir) if options.output_dir.strip() else source.parent
    target = normalize_path(job.output_path) if job.output_path.strip() else folder / f"{source.stem}_restored.mkv"
    if target.suffix.lower() not in (".mkv", ".mp4"):
        raise ValueError("출력 파일은 .mkv 또는 .mp4만 지원합니다.")
    if path_key(target) == path_key(source):
        raise ValueError("입력 파일은 덮어쓸 수 없습니다.")
    if target.exists():
        raise FileExistsError(f"출력 파일이 이미 존재합니다: {target}")
    return PlannedJob(job, source, target, mask)


def plan_queue(jobs: Iterable[Job], options: Options) -> list[PlannedJob]:
    planned: list[PlannedJob] = []
    targets: set[str] = set()
    for job in jobs:
        entry = plan_job(job, options)
        key = path_key(entry.target)
        if key in targets:
            raise ValueError(f"출력 대상 중복: {entry.target}")
        targets.add(key)
        planned.append(entry)
    return planned


class JobQueue:
    """Order-preserving list + hash membership (amortized O(1) add/lookup)."""
    def __init__(self, jobs: Iterable[Job] = ()) -> None:
        self._items: list[Job] = []
        self._keys: set[str] = set()
        for job in jobs:
            self.add(job)

    def add(self, job: Job) -> bool:
        key = path_key(normalize_path(job.input_path))
        if key in self._keys:
            return False
        self._keys.add(key)
        self._items.append(job)
        return True

    def remove_at(self, index: int) -> Job:
        item = self._items.pop(index)
        self._keys.remove(path_key(normalize_path(item.input_path)))
        return item

    def update_at(self, index: int, **fields: object) -> None:
        self._items[index] = replace(self._items[index], **fields)

    def clear(self) -> None:
        self._items.clear()
        self._keys.clear()

    def items(self) -> list[Job]:
        return self._items.copy()

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> Job:
        return self._items[index]

    def __iter__(self):
        return iter(self._items)

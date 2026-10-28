"""Pure, independently authored job planning for an in-process video restoration engine."""
from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

VIDEO_EXTENSIONS = frozenset({".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v"})
METHODS = ("fast", "temporal")
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

    def validate(self) -> None:
        if self.method not in METHODS or self.encoder not in ENCODERS:
            raise ValueError("지원하지 않는 복원 모드 또는 인코더입니다.")
        if not 1 <= self.radius <= 20:
            raise ValueError("복원 반경은 1~20이어야 합니다.")
        if not 0 <= self.temporal_weight <= 0.8:
            raise ValueError("시간축 합성 강도는 0~0.8이어야 합니다.")
        if not 0.125 <= self.flow_scale <= 1:
            raise ValueError("Optical Flow 해상도 배율은 0.125~1이어야 합니다.")
        if not self.ffmpeg.strip():
            raise ValueError("FFmpeg 실행 파일 경로가 필요합니다.")


@dataclass(frozen=True)
class Job:
    input_path: str
    rectangle: Rectangle | None = None
    mask_path: str = ""
    output_path: str = ""


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
    if (job.rectangle is None) == (not bool(job.mask_path.strip())):
        raise ValueError("마스크 이미지 또는 사각형 중 정확히 하나를 지정해야 합니다.")
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

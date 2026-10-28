"""Independent job planning and validation; never imports a GPU framework.

Jasna is a separately installed, licensed runtime. No Jasna code or models are
included in this application.
"""
from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

VIDEO_EXTENSIONS = frozenset({".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".ts"})
QUALITY_MODELS = {
    "Standard": "ltx", "Good": "ltx-4step",
    "High": "ltx-8step", "Top": "ltx-undistilled",
    "BasicVSR++": "basicvsrpp",
}
TIME_PATTERN = re.compile(r"^(?:(\d+):)?(\d+):(\d{2}(?:\.\d{1,3})?)$|^(\d+(?:\.\d{1,3})?)$")
SEGMENT_PATTERN = re.compile(r"^([^,\-]+)-([^,\-]+)$")


@dataclass(frozen=True)
class Options:
    executable: str = "jasna"
    output_dir: str = ""
    quality: str = "Standard"
    gpu_index: int = 0
    codec: str = "hevc"
    seed: int = 20260923
    fast: bool = False
    large_canvas: bool = False
    less_overlap: bool = True
    trial: bool = False
    auto_split_percent: int = 0
    basic_clip_size: int = 90
    basic_overlap: int = 8
    cq: int | None = None


@dataclass(frozen=True)
class Job:
    input_path: str
    segments: str = ""
    output_path: str = ""


@dataclass(frozen=True)
class PlannedJob:
    job: Job
    source: Path
    target: Path
    command: tuple[str, ...]


def parse_time(raw: str) -> float:
    """Parse s, mm:ss, hh:mm:ss. Returns seconds; rejects negative/NaN."""
    raw = raw.strip()
    match = TIME_PATTERN.fullmatch(raw)
    if not match:
        raise ValueError(f"잘못된 시간 형식: {raw!r}")
    if match.group(4) is not None:
        return float(match.group(4))
    if match.group(1) is None:
        minutes, seconds = int(match.group(2)), float(match.group(3))
        if seconds >= 60:
            raise ValueError("초는 60 미만이어야 합니다.")
        return minutes * 60 + seconds
    hours, minutes, seconds = int(match.group(1)), int(match.group(2)), float(match.group(3))
    if minutes >= 60 or seconds >= 60:
        raise ValueError("분과 초는 60 미만이어야 합니다.")
    return hours * 3600 + minutes * 60 + seconds


def validate_segments(segments: str) -> str:
    """Validate non-overlapping ascending intervals without changing Jasna syntax."""
    if not segments.strip():
        return ""
    normalized = []
    previous_end = -1.0
    for entry in segments.split(","):
        match = SEGMENT_PATTERN.fullmatch(entry.strip())
        if not match:
            raise ValueError("구간은 '시작-종료,시작-종료' 형태로 입력하세요.")
        start, end = parse_time(match.group(1)), parse_time(match.group(2))
        if start >= end:
            raise ValueError("복원 구간 종료 시간은 시작 시간보다 커야 합니다.")
        if start < previous_end:
            raise ValueError("복원 구간은 시작 시간순이며 서로 겹칠 수 없습니다.")
        normalized.append(f"{match.group(1).strip()}-{match.group(2).strip()}")
        previous_end = end
    return ",".join(normalized)


def resolve_path(path: str) -> Path:
    return Path(path).expanduser().resolve()


def target_path(job: Job, options: Options) -> Path:
    src = resolve_path(job.input_path)
    if job.output_path.strip():
        return resolve_path(job.output_path)
    directory = resolve_path(options.output_dir) if options.output_dir.strip() else src.parent
    return directory / f"{src.stem}_ltx.mkv"


def validate_options(options: Options) -> None:
    if not options.executable.strip():
        raise ValueError("Jasna 실행 파일 경로가 비어 있습니다.")
    if options.quality not in QUALITY_MODELS:
        raise ValueError(f"지원하지 않는 품질: {options.quality}")
    if options.gpu_index < 0 or options.gpu_index > 63:
        raise ValueError("GPU 번호는 0~63 범위여야 합니다.")
    if options.codec not in ("h264", "hevc", "av1"):
        raise ValueError("인코딩 코덱을 확인하세요.")
    if not 0 <= options.seed < 2**63:
        raise ValueError("시드는 0 이상, 2^63 미만이어야 합니다.")
    if options.auto_split_percent and not 1 <= options.auto_split_percent <= 100:
        raise ValueError("자동 분할 임계값은 1~100%입니다.")
    if not 2 <= options.basic_clip_size <= 10000:
        raise ValueError("클립 길이는 2~10000 프레임이어야 합니다.")
    if not 0 <= options.basic_overlap * 2 < options.basic_clip_size:
        raise ValueError("겹치는 프레임은 클립 길이의 절반보다 작아야 합니다.")
    if options.cq is not None and not 0 <= options.cq <= 63:
        raise ValueError("CQ는 0~63 범위여야 합니다. (코덱/하드웨어별 제한 별도 확인)")


def plan_job(job: Job, options: Options) -> PlannedJob:
    validate_options(options)
    src = resolve_path(job.input_path)
    if not src.is_file():
        raise ValueError(f"입력 파일이 존재하지 않습니다: {src}")
    if src.suffix.lower() not in VIDEO_EXTENSIONS:
        raise ValueError(f"지원하지 않는 영상 확장자: {src.suffix}")
    dst = target_path(job, options)
    if dst == src:
        raise ValueError("입력 파일과 출력 파일이 같습니다.")
    if dst.suffix.lower() not in VIDEO_EXTENSIONS:
        raise ValueError("출력 파일은 영상 확장자를 가져야 합니다.")
    if dst.exists():
        raise FileExistsError(f"기존 출력 파일을 덮어쓸 수 없습니다: {dst}")
    segments = validate_segments(job.segments)
    if segments and options.auto_split_percent and options.quality != "BasicVSR++":
        raise ValueError("개별 구간과 LTX 자동 분할 기능은 동시에 사용할 수 없습니다.")
    if segments and options.codec != "hevc":
        # With --segments Jasna always preserves the input codec, not an explicitly
        # chosen output codec. We therefore omit --codec in this mode altogether.
        pass
    # One folder per input/target pair: avoids unrelated jobs sharing resume state.
    tag = hashlib.sha256((str(src) + "\0" + str(dst)).encode("utf-8")).hexdigest()[:16]
    working = dst.parent / ".ltx_gui_work" / tag
    command = [options.executable, "--input", str(src), "--output", str(dst),
               "--device", f"cuda:{options.gpu_index}",
               "--restoration-model-name", QUALITY_MODELS[options.quality],
               "--working-directory", str(working), "--log-level", "info"]
    if not segments:
        command += ["--codec", options.codec]
        if options.cq is not None:
            command += ["--cq", str(options.cq)]
    if segments:
        command += ["--segments", segments]
    if options.quality == "BasicVSR++":
        command += ["--max-clip-size", str(options.basic_clip_size),
                    "--temporal-overlap", str(options.basic_overlap)]
    else:
        command += ["--ltx-seed", str(options.seed)]
        if options.fast:
            command.append("--ltx-fast")
        if options.large_canvas:
            command.append("--ltx-large-canvas")
        if not options.less_overlap:
            command.append("--no-experimental-ltx-less-overlap")
        if options.trial:
            command.append("--ltx-trial")
        if options.auto_split_percent:
            command += ["--ltx-auto-split", str(options.auto_split_percent)]
    return PlannedJob(job, src, dst, tuple(command))


def plan_queue(jobs: Iterable[Job], options: Options) -> list[PlannedJob]:
    """O(n) planning, O(1) average-case duplicate-output detection."""
    result = []
    occupied: set[str] = set()
    for job in jobs:
        planned = plan_job(job, options)
        key = os.path.normcase(str(planned.target))
        if key in occupied:
            raise ValueError(f"출력 파일명이 중복되었습니다: {planned.target.name}")
        occupied.add(key)
        result.append(planned)
    return result


class JobQueue:
    """Insertion-ordered queue: O(1) random read, amortized O(1) append and membership.

    Arbitrary deletion remains O(n), which is optimal for compact array storage.
    """

    def __init__(self, jobs: Iterable[Job] = ()) -> None:
        self._queue: list[Job] = []
        self._known: set[str] = set()
        for job in jobs:
            self.add(job)

    def add(self, job: Job) -> bool:
        key = os.path.normcase(str(resolve_path(job.input_path)))
        if key in self._known:
            return False
        self._known.add(key)
        self._queue.append(job)
        return True

    def contains(self, input_path: str) -> bool:
        return os.path.normcase(str(resolve_path(input_path))) in self._known

    def __getitem__(self, index: int) -> Job:
        return self._queue[index]

    def update_at(self, index: int, *, segments: str) -> None:
        self._queue[index] = Job(self._queue[index].input_path, segments, self._queue[index].output_path)

    def remove_at(self, index: int) -> Job:
        # Arbitrary removal O(n), but constant-time membership and FIFO operations.
        if index < 0 or index >= len(self._queue):
            raise IndexError(index)
        job = self._queue[index]
        del self._queue[index]
        self._known.remove(os.path.normcase(str(resolve_path(job.input_path))))
        return job

    def clear(self) -> None:
        self._queue.clear()
        self._known.clear()

    def __len__(self) -> int:
        return len(self._queue)

    def __iter__(self):
        return iter(self._queue)

    def items(self) -> list[Job]:
        return list(self._queue)


def estimated_windows(frames: int, window: int = 121, hop: int = 104) -> int:
    """Minimal fixed-size overlapping windows to cover N frames, O(1) arithmetic.

    This models *one* continuous track only; actual tracking can yield more.
    """
    if frames < 0 or window <= 0 or not 0 < hop <= window:
        raise ValueError("잘못된 프레임/윈도 파라미터입니다.")
    if frames == 0:
        return 0
    if frames <= window:
        return 1
    return 1 + (frames - window + hop - 1) // hop

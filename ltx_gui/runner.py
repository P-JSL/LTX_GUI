"""Single worker thread executes the self-contained restoration implementation."""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Callable, Iterable

from .core import Options, PlannedJob
from .engine import Cancelled, Progress, restore_video


@dataclass(frozen=True)
class Update:
    index: int
    kind: str
    message: str = ""
    percent: float = -1.0
    fps: float = 0.0


class BatchRunner:
    def __init__(self, on_update: Callable[[Update], None], options: Options) -> None:
        self._on_update = on_update
        self._options = options
        self._stop = threading.Event()

    def request_stop(self) -> None:
        self._stop.set()

    def run(self, jobs: Iterable[PlannedJob]) -> None:
        for index, job in enumerate(jobs):
            if self._stop.is_set():
                break
            self._on_update(Update(index, "started", job.source.name))
            try:
                def report(p: Progress) -> None:
                    percent = (100 * p.frames / p.total) if p.total else -1
                    self._on_update(Update(index, "progress", f"{p.frames}/{p.total} frame", percent, p.fps))
                result = restore_video(job, self._options, self._stop, report)
                self._on_update(Update(index, "completed", str(result), 100))
            except Cancelled as exc:
                self._on_update(Update(index, "stopped", str(exc)))
                break
            except Exception as exc:
                self._on_update(Update(index, "failed", str(exc)))
        self._on_update(Update(-1, "all_done", "중지" if self._stop.is_set() else "대기열 완료"))

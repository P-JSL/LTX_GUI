"""Non-shell subprocess runner: sequential jobs, cooperative cancellation.

UI updates are marshalled to the Qt main thread by signals, never made here.
"""
from __future__ import annotations

import os
import queue
import re
import signal
import subprocess
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from .core import PlannedJob

_PERCENT = re.compile(r"(?<!\d)(100|\d{1,2})(?:\.\d+)?\s*%")
_STAGE_WORDS = ("detect", "prepar", "restor", "sav", "find", "scan", "encode")


@dataclass(frozen=True)
class Update:
    index: int
    kind: str  # started, log, progress, completed, failed, stopped, all_done
    message: str = ""
    percent: float = -1.0
    elapsed: float = 0.0


class BatchRunner:
    """Owns one process at a time; processing thread is controlled by caller."""

    def __init__(self, on_update: Callable[[Update], None]) -> None:
        self._on_update = on_update
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._process: subprocess.Popen | None = None

    def request_stop(self) -> None:
        self._stop.set()
        with self._lock:
            process = self._process
        if process is not None and process.poll() is None:
            self._terminate(process)

    @staticmethod
    def _terminate(process: subprocess.Popen) -> None:
        try:
            if os.name == "nt":
                # CREATE_NEW_PROCESS_GROUP lets CTRL_BREAK_EVENT interrupt a CLI
                # that supports checkpoint-on-Ctrl+C, without shell invocation.
                process.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                os.killpg(process.pid, signal.SIGINT)
        except (OSError, ValueError):
            try:
                process.terminate()
            except OSError:
                pass

    @staticmethod
    def _kill(process: subprocess.Popen) -> None:
        try:
            if os.name == "nt":
                process.kill()
            else:
                os.killpg(process.pid, signal.SIGKILL)
        except (OSError, ValueError):
            pass

    def run(self, jobs: Iterable[PlannedJob]) -> None:
        # This runner is single-use; preserve a stop requested before start.
        batch = list(jobs)
        for index, item in enumerate(batch):
            if self._stop.is_set():
                break
            self._run_one(index, item)
            if self._stop.is_set():
                break
        self._on_update(Update(-1, "all_done", "작업 중지" if self._stop.is_set() else "대기열 종료"))

    def _run_one(self, index: int, item: PlannedJob) -> None:
        # Check immediately before launching: output may have appeared after planning.
        if item.target.exists():
            self._on_update(Update(index, "failed", f"출력 파일이 이미 존재합니다: {item.target}"))
            return
        try:
            item.target.parent.mkdir(parents=True, exist_ok=True)
            started = time.monotonic()
            opts = {"stdout": subprocess.PIPE, "stderr": subprocess.STDOUT,
                    "stdin": subprocess.DEVNULL, "text": True, "errors": "replace",
                    "bufsize": 1}
            if os.name == "nt":
                opts["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
            else:
                opts["start_new_session"] = True
            process = subprocess.Popen(item.command, **opts)
        except (OSError, ValueError) as exc:
            self._on_update(Update(index, "failed", f"실행 실패: {exc}"))
            return
        with self._lock:
            self._process = process
        self._on_update(Update(index, "started", item.source.name))
        # Dedicated reader prevents a stalled child from blocking stop requests.
        pending: queue.SimpleQueue[str | None] = queue.SimpleQueue()

        def read_output() -> None:
            assert process.stdout is not None
            try:
                for line in process.stdout:
                    pending.put(line.rstrip("\r\n"))
            finally:
                pending.put(None)

        reader = threading.Thread(target=read_output, daemon=True)
        reader.start()
        got_eof = False
        escalation_deadline: float | None = None
        try:
            while True:
                try:
                    line = pending.get(timeout=0.1)
                    if line is None:
                        got_eof = True
                    elif line:
                        self._on_update(Update(index, "log", line))
                        if any(word in line.lower() for word in _STAGE_WORDS):
                            match = _PERCENT.search(line)
                            if match:
                                self._on_update(Update(index, "progress", line,
                                                        float(match.group(0).strip().rstrip("%")),
                                                        time.monotonic() - started))
                except queue.Empty:
                    pass
                if self._stop.is_set() and process.poll() is None:
                    if escalation_deadline is None:
                        self._terminate(process)
                        escalation_deadline = time.monotonic() + 8.0
                    elif time.monotonic() >= escalation_deadline:
                        self._kill(process)
                        escalation_deadline = time.monotonic() + 2.0
                if got_eof and process.poll() is not None:
                    break
            result = process.wait()
            elapsed = time.monotonic() - started
            if self._stop.is_set():
                self._on_update(Update(index, "stopped", "중지됨 (Jasna 체크포인트는 보존될 수 있습니다)", elapsed=elapsed))
            elif result != 0:
                self._on_update(Update(index, "failed", f"프로세스 종료 코드: {result}", elapsed=elapsed))
            elif "--ltx-trial" in item.command:
                self._on_update(Update(index, "completed", "LTX 속도 테스트 완료 (복원 파일은 생성하지 않습니다)", 100.0, elapsed))
            elif not item.target.is_file():
                self._on_update(Update(index, "failed", "종료 코드는 0이지만 출력 파일이 없습니다.", elapsed=elapsed))
            else:
                self._on_update(Update(index, "completed", str(item.target), 100.0, elapsed))
        finally:
            with self._lock:
                if self._process is process:
                    self._process = None
            if process.stdout:
                process.stdout.close()

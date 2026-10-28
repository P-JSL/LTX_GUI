"""Bounded-cost resource reads; never on the GUI thread."""
from __future__ import annotations

import csv
import io
import shutil
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class GpuStats:
    index: int
    name: str
    total_mib: int
    used_mib: int
    utilization_percent: int


def parse_nvidia_smi(content: str) -> list[GpuStats]:
    result = []
    for row in csv.reader(io.StringIO(content)):
        if len(row) != 5:
            continue
        try:
            index, name, total, used, utilization = (part.strip() for part in row)
            result.append(GpuStats(int(index), name, int(total), int(used), int(utilization)))
        except ValueError:
            continue
    return result


def gpu_stats() -> list[GpuStats]:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return []
    try:
        result = subprocess.run([executable, "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
                                 "--format=csv,noheader,nounits"],
                                capture_output=True, text=True, timeout=2, check=False)
        return parse_nvidia_smi(result.stdout) if result.returncode == 0 else []
    except (OSError, subprocess.TimeoutExpired):
        return []


def cpu_ram_stats() -> tuple[float | None, float | None]:
    try:
        import psutil  # optional and outside of core computation
        return psutil.cpu_percent(interval=None), psutil.virtual_memory().percent
    except ImportError:
        return None, None

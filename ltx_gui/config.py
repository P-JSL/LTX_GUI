"""JSON preferences, independent of JASNA configuration and models."""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

from .core import Job, Options, Rectangle


def settings_path() -> Path:
    base = Path(os.environ.get("LOCALAPPDATA") or Path.home()/".config")
    return base / "LTX_GUI" / "settings.json"


def save_settings(options: Options, jobs: list[Job], path: Path | None = None) -> None:
    path = path or settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"schema": 2, "options": asdict(options), "jobs": [asdict(j) for j in jobs]}
    pending = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".settings_", suffix=".tmp", delete=False) as f:
            pending = Path(f.name)
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(pending, path)
    finally:
        if pending:
            pending.unlink(missing_ok=True)


def load_settings(path: Path | None = None) -> tuple[Options, list[Job]]:
    path = path or settings_path()
    if not path.exists():
        return Options(), []
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") != 2:
        # Old CLI-specific settings are intentionally not interpreted as native settings.
        return Options(), []
    opts = Options(**{k: v for k, v in data.get("options", {}).items()
                      if k in Options.__dataclass_fields__})
    opts.validate()
    jobs = []
    for item in data.get("jobs", []):
        region = item.get("rectangle")
        rect = Rectangle(**region) if isinstance(region, dict) else None
        jobs.append(Job(input_path=item["input_path"], rectangle=rect,
                        mask_path=item.get("mask_path", ""), output_path=item.get("output_path", ""),
                        auto_detect=bool(item.get("auto_detect", False))))
    return opts, jobs

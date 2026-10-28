"""Settings persistence without code execution or credential storage."""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, fields
from pathlib import Path

from .core import Job, Options


def settings_file() -> Path:
    base = os.environ.get("APPDATA") or os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "LTX_GUI" / "settings.json"


def load_settings(path: Path | None = None) -> tuple[Options, list[Job]]:
    path = path or settings_file()
    if not path.is_file():
        return Options(), []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        allowed = {f.name for f in fields(Options)}
        opts = Options(**{k: v for k, v in data.get("options", {}).items() if k in allowed})
        items = [Job(**{k: v for k, v in j.items() if k in {"input_path", "segments", "output_path"}})
                 for j in data.get("jobs", []) if isinstance(j, dict)]
        return opts, items
    except (OSError, ValueError, TypeError, KeyError):
        # Fail closed; malformed persisted configuration cannot run arbitrary code.
        return Options(), []


def save_settings(options: Options, jobs: list[Job], path: Path | None = None) -> None:
    path = path or settings_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({"schema": 1, "options": asdict(options),
                          "jobs": [asdict(job) for job in jobs]}, ensure_ascii=False, indent=2)
    # Atomic replace prevents a crash from truncating a working configuration.
    handle, temp_name = tempfile.mkstemp(prefix=".settings-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as fp:
            fp.write(payload)
            fp.flush()
            os.fsync(fp.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)

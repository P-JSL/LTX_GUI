from pathlib import Path

from ltx_gui.config import load_settings, save_settings
from ltx_gui.core import Options, Job


def test_persistence(tmp_path):
    path = tmp_path / "settings.json"
    options = Options(quality="Good", gpu_index=3)
    jobs = [Job("테스트.mp4", "00:10-00:20")]
    save_settings(options, jobs, path)
    o, j = load_settings(path)
    assert o == options
    assert j == jobs
    assert list(tmp_path.glob("*.tmp")) == []


def test_corrupt_config_falls_back_to_defaults(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text("invalid json", encoding="utf-8")
    assert load_settings(path) == (Options(), [])

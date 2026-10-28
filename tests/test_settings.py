from ltx_gui.config import load_settings, save_settings
from ltx_gui.core import Options, Job, Rectangle


def test_roundtrip(tmp_path):
    path = tmp_path/"prefs.json"
    options = Options(method="temporal", encoder="h264_nvenc")
    job = Job("C:/movie.mp4", rectangle=Rectangle(0.1,0.2,.3,.4))
    save_settings(options, [job], path)
    actual, jobs = load_settings(path)
    assert actual == options
    assert jobs == [job]


def test_legacy_settings_not_loaded(tmp_path):
    path = tmp_path/"prefs.json"
    path.write_text('{"schema":1,"options":{"executable":"jasna"}}')
    assert load_settings(path) == (Options(), [])

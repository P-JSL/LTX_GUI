import dataclasses
from pathlib import Path

import pytest

from ltx_gui.core import (
    Job, JobQueue, Options, estimated_windows, parse_time, plan_job, plan_queue,
    validate_options, validate_segments,
)


def sample(tmp_path, name="sample.mp4"):
    p = tmp_path / name
    p.write_bytes(b"synthetic video placeholder")
    return p


@pytest.mark.parametrize("input_time,expected", [("12", 12), ("01:02", 62),
                                                ("1:02:03.250", 3723.25), ("0", 0)])
def test_times(input_time, expected):
    assert parse_time(input_time) == expected


@pytest.mark.parametrize("bad", ["-2", "NaN", "1:60", "1:03:60", "", "00:00:60"])
def test_bad_times(bad):
    with pytest.raises(ValueError):
        parse_time(bad)


def test_segments():
    assert validate_segments("00:10-00:25,01:00-01:15") == "00:10-00:25,01:00-01:15"
    for invalid in ("10-10", "20-10", "10-20,15-30", "10,20", "30-31,10-20"):
        with pytest.raises(ValueError):
            validate_segments(invalid)


def test_ltx_options_and_arguments(tmp_path):
    src = sample(tmp_path, '테스트 영상.mp4')
    opts = Options(executable="jasna", output_dir=str(tmp_path / "export"),
                   quality="High", gpu_index=2, seed=77, fast=False,
                   less_overlap=False, large_canvas=True)
    planned = plan_job(Job(str(src)), opts)
    args = planned.command
    assert args[args.index("--restoration-model-name") + 1] == "ltx-8step"
    assert args[args.index("--device") + 1] == "cuda:2"
    assert args[args.index("--ltx-seed") + 1] == "77"
    assert "--ltx-large-canvas" in args
    assert "--no-experimental-ltx-less-overlap" in args
    assert planned.target.name == "테스트 영상_ltx.mkv"
    assert str(src) in args


def test_basic_ignores_ltx_only_settings(tmp_path):
    src = sample(tmp_path)
    opts = Options(quality="BasicVSR++", fast=True, trial=True,
                   auto_split_percent=12, basic_clip_size=180, basic_overlap=12)
    args = plan_job(Job(str(src)), opts).command
    assert "--ltx-trial" not in args
    assert "--ltx-fast" not in args
    assert "--ltx-auto-split" not in args
    assert args[args.index("--max-clip-size") + 1] == "180"


def test_segment_preserves_source_codec(tmp_path):
    src = sample(tmp_path)
    args = plan_job(Job(str(src), segments="1-12"), Options(codec="h264")).command
    assert "--segments" in args
    assert "--codec" not in args
    assert "--cq" not in args


def test_output_safety_and_duplicates(tmp_path):
    src = sample(tmp_path)
    p = plan_job(Job(str(src)), Options()).target
    p.write_bytes(b"existing")
    with pytest.raises(FileExistsError):
        plan_job(Job(str(src)), Options())
    p.unlink()
    with pytest.raises(ValueError, match="중복"):
        plan_queue([Job(str(src)), Job(str(src))], Options())
    with pytest.raises(ValueError, match="입력 파일과 출력 파일"):
        plan_job(Job(str(src), output_path=str(src)), Options())


def test_auto_split_and_segments_incompatible(tmp_path):
    src = sample(tmp_path)
    with pytest.raises(ValueError, match="동시에"):
        plan_job(Job(str(src), segments="1-5"), Options(auto_split_percent=12))


def test_options_invalid():
    for opts in (Options(gpu_index=-1), Options(basic_clip_size=60, basic_overlap=30),
                 Options(quality="bad"), Options(seed=-1)):
        with pytest.raises(ValueError):
            validate_options(opts)


def test_queue_set_and_random_reads(tmp_path):
    q = JobQueue()
    a = str(sample(tmp_path, "a.mp4"))
    b = str(sample(tmp_path, "b.mp4"))
    assert q.add(Job(a)) is True
    assert q.add(Job(a)) is False
    assert q.add(Job(b)) is True
    assert q[1].input_path == b
    q.update_at(0, segments="10-20")
    assert q[0].segments == "10-20"
    assert q.contains(a)
    q.remove_at(0)
    assert not q.contains(a)
    assert q[0].input_path == b


@pytest.mark.parametrize("n,window,hop,expected", [(0, 121, 104, 0), (1, 121, 104, 1),
                                                   (121, 121, 104, 1), (122, 121, 104, 2),
                                                   (225, 121, 104, 2), (226, 121, 104, 3)])
def test_window_count(n, window, hop, expected):
    assert estimated_windows(n, window, hop) == expected


def test_window_coverage_invariant():
    for frames in range(1, 2000):
        count = estimated_windows(frames)
        assert 121 + (count - 1)*104 >= frames
        if count > 1:
            assert 121 + (count - 2)*104 < frames

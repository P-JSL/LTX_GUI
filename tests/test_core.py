from pathlib import Path
import pytest
from ltx_gui.core import Rectangle, Options, Job, JobQueue, plan_job, plan_queue


def test_rectangle_normalization():
    assert Rectangle.parse("0.1,0.2,0.6,0.8").pixels(100, 200) == (10, 40, 60, 160)


@pytest.mark.parametrize("value", ["-0.2,0,1,1", "0,0,1,1.1", "0,0,0,1", "1,0,0,1", "nan,0,1,1", "0,0,0.2"])
def test_reject_bad_rectangle(value):
    with pytest.raises(ValueError):
        Rectangle.parse(value)


def test_plan_without_jasna(tmp_path):
    source = tmp_path / "input.mp4"
    source.write_bytes(b"input")
    rect = Rectangle(0.2, 0.2, 0.5, 0.5)
    planned = plan_job(Job(str(source), rect), Options())
    assert planned.target.name == "input_restored.mkv"
    assert not hasattr(planned, "command")


def test_input_output_protection(tmp_path):
    source = tmp_path / "input.mkv"
    source.touch()
    with pytest.raises(ValueError):
        plan_job(Job(str(source), Rectangle(0,0,.4,.4), output_path=str(source)), Options())
    target = tmp_path / "input_restored.mkv"
    target.touch()
    with pytest.raises(FileExistsError):
        plan_job(Job(str(source), Rectangle(0,0,.4,.4)), Options())


def test_requires_exactly_one_mask(tmp_path):
    source = tmp_path / "input.mkv"
    source.touch()
    with pytest.raises(ValueError):
        plan_job(Job(str(source)), Options())
    mask = tmp_path / "mask.png"
    mask.touch()
    with pytest.raises(ValueError):
        plan_job(Job(str(source), Rectangle(0,0,.4,.4), str(mask)), Options())
    assert plan_job(Job(str(source), mask_path=str(mask)), Options()).mask_path == mask


def test_duplicate_queue_outputs(tmp_path):
    for name in ("a.mp4", "b.mp4"):
        (tmp_path/name).touch()
    target = str(tmp_path/"same.mkv")
    jobs = [Job(str(tmp_path/name), Rectangle(0,0,.4,.4), output_path=target) for name in ("a.mp4","b.mp4")]
    with pytest.raises(ValueError):
        plan_queue(jobs, Options())


def test_fast_membership(tmp_path):
    file = str(tmp_path/"a.mp4")
    queue = JobQueue()
    assert queue.add(Job(file))
    assert not queue.add(Job(file))
    queue.update_at(0, rectangle=Rectangle(0,0,.1,.1))
    assert queue[0].rectangle
    assert queue.remove_at(0).input_path == file
    assert queue.add(Job(file))


@pytest.mark.parametrize("val", [Options(radius=0), Options(flow_scale=0), Options(temporal_weight=0.9), Options(encoder="h265_nvenc")])
def test_bad_options(val):
    with pytest.raises(ValueError):
        val.validate()

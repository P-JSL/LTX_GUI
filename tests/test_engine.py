import shutil
import threading

import cv2
import numpy as np
import pytest

from ltx_gui.core import Job, Options, Rectangle, plan_job
from ltx_gui.engine import make_mask, SpatialTemporalInpaint, ffmpeg_args, restore_video, Cancelled


def test_mask_rect_and_grayscale():
    mask = make_mask((100,200), rectangle=Rectangle(.25,.2,.75,.8))
    assert cv2.countNonZero(mask) == 100*60
    assert np.all(mask[0] == 0)
    img = np.zeros((10,20), dtype=np.uint8)
    img[2:8, 5:15] = 255
    scaled = make_mask((100,200), image=img)
    assert cv2.countNonZero(scaled) == 100*60


def test_reject_full_mask():
    with pytest.raises(ValueError):
        make_mask((12,12), rectangle=Rectangle(0,0,1,1))


def test_mask_outside_frame_identical():
    rng = np.random.default_rng(42)
    frame = rng.integers(0,256,(64,96,3),dtype=np.uint8)
    mask = make_mask((64,96), rectangle=Rectangle(.25,.25,.75,.75))
    for mode in ("fast", "temporal"):
        engine = SpatialTemporalInpaint(mask, Options(method=mode, radius=3))
        for _ in range(3):
            out = engine.process(frame)
            assert np.array_equal(out[mask==0], frame[mask==0])
            assert out.shape == frame.shape


def test_larger_frame_rejected():
    mask = make_mask((10,10), rectangle=Rectangle(.2,.2,.8,.8))
    with pytest.raises(ValueError):
        SpatialTemporalInpaint(mask, Options()).process(np.zeros((11,10,3), dtype=np.uint8))


def test_ffmpeg_args_no_shell_injection(tmp_path):
    args = ffmpeg_args("ffmpeg", 640, 360, 30, tmp_path/"a;rm -rf.mp4", tmp_path/"out.mkv", "libx264")
    assert args[-1].endswith("out.mkv")
    assert "-c:a" in args
    assert all("jasna" not in arg.lower() for arg in args)
    with pytest.raises(ValueError):
        ffmpeg_args("ffmpeg", 640, 360, 30, tmp_path/"x", tmp_path/"x", "danger")


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_video_pipeline_real_ffmpeg(tmp_path):
    source = tmp_path/"test.mp4"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 12, (64, 48))
    assert writer.isOpened()
    for i in range(8):
        frame = np.full((48,64,3), (i*23)%255, np.uint8)
        frame[12:28, 20:36] = (0,200,200)
        writer.write(frame)
    writer.release()
    job = plan_job(Job(str(source), Rectangle(.25,.2,.75,.8)), Options())
    states = []
    final = restore_video(job, Options(), progress=states.append)
    assert final.is_file() and final.stat().st_size > 0
    assert len(states) > 0
    result = cv2.VideoCapture(str(final))
    count = 0
    while result.read()[0]:
        count += 1
    result.release()
    assert count == 8
    assert source.exists()
    with pytest.raises(FileExistsError):
        restore_video(job, Options())


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_cancel_cleans_temporary(tmp_path):
    src = tmp_path/"clip.mp4"
    writer = cv2.VideoWriter(str(src), cv2.VideoWriter_fourcc(*"mp4v"), 8, (32,32))
    for _ in range(2):
        writer.write(np.zeros((32,32,3),np.uint8))
    writer.release()
    job = plan_job(Job(str(src), Rectangle(.1,.1,.5,.5)), Options())
    stopped = threading.Event()
    stopped.set()
    with pytest.raises(Cancelled):
        restore_video(job, Options(), stopped)
    assert not job.target.exists()
    assert not list(tmp_path.glob(".ltx_gui_partial_*"))

@pytest.mark.skipif(shutil.which('ffmpeg') is None or shutil.which('ffprobe') is None, reason='ffmpeg/ffprobe unavailable')
def test_audio_track_retained(tmp_path):
    import subprocess
    source = tmp_path/'with_audio.mkv'
    subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-f', 'lavfi',
                    '-i', 'testsrc2=size=96x64:rate=8:duration=1', '-f', 'lavfi',
                    '-i', 'sine=frequency=440:duration=1', '-c:v', 'mpeg4', '-c:a',
                    'aac', '-shortest', str(source)], check=True)
    plan = plan_job(Job(str(source), Rectangle(.2,.2,.6,.6)), Options())
    result = restore_video(plan, Options())
    probe = subprocess.run(['ffprobe', '-v', 'error', '-show_entries',
                            'stream=codec_type', '-of', 'csv=p=0', str(result)],
                           check=True, capture_output=True, text=True)
    assert sorted(probe.stdout.split()) == ['audio', 'video']


def test_scene_cut_does_not_blend_unrelated_frames():
    mask = make_mask((64,96), rectangle=Rectangle(.2,.2,.8,.8))
    processor = SpatialTemporalInpaint(mask, Options(method='temporal', temporal_weight=.8))
    first = np.zeros((64,96,3), np.uint8)
    second = np.full((64,96,3), 250, np.uint8)
    processor.process(first)
    new = processor.process(second)
    assert np.array_equal(new[mask == 0], second[mask == 0])
    assert np.median(new[mask != 0]) >= 245

import os
import sys
import threading
import time
from dataclasses import replace

from ltx_gui.core import Job, Options, plan_job
from ltx_gui.runner import BatchRunner


def test_fake_backend_success(tmp_path):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"synthetic")
    plan = plan_job(Job(str(source)), Options())
    target = plan.target
    # The subprocess consumes arguments without a shell. It only writes a synthetic result.
    fake = ("import pathlib,sys; print('restoring 50%',flush=True); "
            "pathlib.Path(sys.argv[1]).write_bytes(b'completed')")
    plan = replace(plan, command=(sys.executable, "-c", fake, str(target)))
    events = []
    BatchRunner(events.append).run([plan])
    assert target.read_bytes() == b"completed"
    assert "started" in [e.kind for e in events]
    assert "completed" in [e.kind for e in events]
    assert "progress" in [e.kind for e in events]
    assert events[-1].kind == "all_done"


def test_fake_backend_error_and_no_output(tmp_path):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"synthetic")
    plan = plan_job(Job(str(source)), Options())
    events = []
    failing = replace(plan, command=(sys.executable, "-c", "raise SystemExit(7)"))
    BatchRunner(events.append).run([failing])
    assert any(e.kind == "failed" and "7" in e.message for e in events)
    events.clear()
    no_output = replace(plan, command=(sys.executable, "-c", "print('ok')"))
    BatchRunner(events.append).run([no_output])
    assert any(e.kind == "failed" and "출력 파일" in e.message for e in events)


def test_cancel_slow_backend(tmp_path):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"synthetic")
    plan = plan_job(Job(str(source)), Options())
    slow = replace(plan, command=(sys.executable, "-c", "import time; time.sleep(15)"))
    events = []
    runner = BatchRunner(events.append)
    thread = threading.Thread(target=runner.run, args=([slow],))
    thread.start()
    deadline = time.monotonic() + 3
    while not any(e.kind == "started" for e in events) and time.monotonic() < deadline:
        time.sleep(0.01)
    runner.request_stop()
    thread.join(timeout=6)
    assert not thread.is_alive()
    assert any(e.kind == "stopped" for e in events)


def test_trial_does_not_require_output(tmp_path):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"synthetic")
    plan = plan_job(Job(str(source)), Options(trial=True))
    fake = replace(plan, command=(sys.executable, "-c", "print('trial')", "--ltx-trial"))
    events = []
    BatchRunner(events.append).run([fake])
    assert any(e.kind == "completed" for e in events)
    assert not plan.target.exists()

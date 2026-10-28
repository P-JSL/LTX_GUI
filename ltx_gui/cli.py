"""Headless entrypoint for reproducible CPU/GPU-encoder integration tests."""
from __future__ import annotations

import argparse
import sys

from .core import Job, Options, Rectangle, plan_job
from .engine import Cancelled, restore_video


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Independent, mask-guided video restoration (no Jasna)")
    parser.add_argument("input", help="Input video")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--rectangle", type=Rectangle.parse, metavar="x1,y1,x2,y2")
    group.add_argument("--mask", default="", help="White=restore, black=preserve mask image")
    group.add_argument("--auto-detect", action="store_true", help="ONNX mosaic detector and temporal tracking")
    parser.add_argument("--detector-model", default="", help="Trained mosaic ONNX path")
    parser.add_argument("--detector-confidence", type=float, default=0.35)
    parser.add_argument("--detector-format", choices=("xyxy6", "yolo_cls"), default="xyxy6")
    parser.add_argument("--detector-class", type=int, default=0)
    parser.add_argument("--detector-cuda", action="store_true")
    parser.add_argument("--output", default="", help="Output .mkv/.mp4, defaults next to source")
    parser.add_argument("--method", choices=("fast", "temporal"), default="fast")
    parser.add_argument("--encoder", choices=("libx264", "h264_nvenc"), default="libx264")
    parser.add_argument("--radius", type=int, default=3)
    parser.add_argument("--weight", type=float, default=0.35)
    parser.add_argument("--flow-scale", type=float, default=0.5)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    ns = parser.parse_args(argv)
    options = Options(method=ns.method, encoder=ns.encoder, radius=ns.radius,
                      temporal_weight=ns.weight, flow_scale=ns.flow_scale, ffmpeg=ns.ffmpeg,
                      detector_model=ns.detector_model, detector_confidence=ns.detector_confidence,
                      detector_format=ns.detector_format, detector_class=ns.detector_class,
                      detector_cuda=ns.detector_cuda)
    job = Job(ns.input, rectangle=ns.rectangle, mask_path=ns.mask, output_path=ns.output,
              auto_detect=ns.auto_detect)
    try:
        planned = plan_job(job, options)
        completed = restore_video(planned, options,
                                  progress=lambda p: print(f"\r{p.frames}/{p.total} frames - {p.fps:.1f} fps", end="", flush=True))
    except (ValueError, OSError, RuntimeError, Cancelled) as exc:
        print(f"\n오류: {exc}", file=sys.stderr)
        return 1
    print(f"\n완료: {completed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Local microbenchmark (NOT a comparison against JASNA or neural LTX)."""
from __future__ import annotations

import argparse
import json
import statistics
import time

import cv2
import numpy as np

from ltx_gui.core import Options, Rectangle
from ltx_gui.engine import SpatialTemporalInpaint, make_mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--iterations", type=int, default=30)
    args = parser.parse_args()
    if args.width < 32 or args.height < 32 or args.iterations < 3:
        parser.error("width/height >=32, iterations>=3")
    rng = np.random.default_rng(23)
    frame = rng.integers(0, 256, (args.height, args.width, 3), dtype=np.uint8)
    mask = make_mask((args.height,args.width), rectangle=Rectangle(.35,.3,.65,.7))
    results = {}
    for mode in ("fast", "temporal"):
        processor = SpatialTemporalInpaint(mask, Options(method=mode))
        for _ in range(3):
            processor.process(frame)
        elapsed = []
        for _ in range(args.iterations):
            start = time.perf_counter()
            out = processor.process(frame)
            elapsed.append(time.perf_counter()-start)
            assert np.array_equal(out[mask == 0], frame[mask == 0])
        results[mode] = {"median_ms": round(statistics.median(elapsed)*1000, 3),
                         "p95_ms": round(sorted(elapsed)[int(len(elapsed)*.95)]*1000, 3),
                         "fps_compute_only": round(1/statistics.median(elapsed), 2)}
    print(json.dumps({"frame_size": [args.width,args.height], "iterations": args.iterations,
                      "opencv": cv2.__version__, "performance": results}, indent=2))


if __name__ == "__main__":
    main()

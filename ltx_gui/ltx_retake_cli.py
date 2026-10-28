"""Explicit LTX-2.5 model execution (not Jasna) + optional exact mask compositing."""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

from .core import Job, Options, Rectangle, plan_job
from .engine import restore_video
from .ltx_runtime import RetakeRequest, run_temporal_retake


def main(argv=None) -> int:
    parser=argparse.ArgumentParser(description='LTX-2.5 DiT/VAE retake + optional masked composite')
    parser.add_argument('input')
    parser.add_argument('--output', required=True)
    parser.add_argument('--weights-root', default='models/LTX-2.5', help='Official LTX-2.5 split pack (default models/LTX-2.5)')
    parser.add_argument('--prompt', required=True)
    parser.add_argument('--start',type=float,required=True)
    parser.add_argument('--end',type=float,required=True)
    parser.add_argument('--seed',type=int,default=20260923)
    group=parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--rectangle',type=Rectangle.parse)
    group.add_argument('--mask',default='')
    group.add_argument('--auto-detect',action='store_true')
    parser.add_argument('--detector-model',default='')
    parser.add_argument('--detector-confidence',type=float,default=.35)
    parser.add_argument('--encoder',choices=('libx264','h264_nvenc'),default='libx264')
    parser.add_argument('--ffmpeg',default='ffmpeg')
    ns=parser.parse_args(argv)
    dst=Path(ns.output).expanduser().resolve()
    try:
        if not (ns.rectangle or ns.mask or ns.auto_detect):
            # Time retake without mask; regenerates the whole spatiotemporal region.
            run_temporal_retake(RetakeRequest(Path(ns.input).resolve(),dst,Path(ns.weights_root),
                                              ns.prompt,ns.start,ns.end,ns.seed))
        else:
            job=Job(ns.input, rectangle=ns.rectangle, mask_path=ns.mask,
                    output_path=str(dst),auto_detect=ns.auto_detect)
            opts=Options(detector_model=ns.detector_model,
                         detector_confidence=ns.detector_confidence,
                         encoder=ns.encoder,ffmpeg=ns.ffmpeg)
            planned=plan_job(job,opts)
            # Temporary LTX output is never the published restoration result.
            with tempfile.TemporaryDirectory(prefix='ltx_gui_generate_') as tmp:
                generated=Path(tmp)/'retake.mp4'
                run_temporal_retake(RetakeRequest(Path(ns.input).resolve(),generated,
                                                  Path(ns.weights_root),ns.prompt,ns.start,ns.end,ns.seed))
                restore_video(planned,opts,generated_video=generated)
        print('Output:',dst)
        return 0
    except (ValueError,FileNotFoundError,FileExistsError,RuntimeError,OSError) as exc:
        print(f'LTX-2.5 실행 실패: {exc}',file=sys.stderr)
        return 1

if __name__=='__main__':raise SystemExit(main())

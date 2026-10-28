"""Paired, reproducible quality comparison for existing outputs.

No Jasna dependency: a competitor output is accepted as a file only. Run both
applications separately on the SAME source, mask and hardware, and record
wall-time / peak VRAM using the shared experiment manifest.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ltx_gui.quality import compare_video


def main(argv=None):
    parser=argparse.ArgumentParser(description="Compare LTX GUI and Jasna outputs against *clean ground truth*")
    parser.add_argument('--source',required=True,help='Censored video input shared by both methods')
    parser.add_argument('--reference',required=True,help='Unmodified clean reference video (not censored source)')
    parser.add_argument('--ours',required=True)
    parser.add_argument('--jasna',default='',help='Optional independently created Jasna output')
    parser.add_argument('--mask',default='',help='Same static target mask PNG for masked PSNR / outside MAE')
    parser.add_argument('--max-frames',type=int)
    parser.add_argument('--manifest',required=True,help='JSON with same video/mask/GPU/codec/versions/seed')
    parser.add_argument('--output',default='comparison.json')
    args=parser.parse_args(argv)
    meta=json.loads(Path(args.manifest).read_text(encoding='utf-8'))
    required=('source_sha256','mask_sha256','gpu_name','gpu_driver','resolution','fps',
              'codec','ltx_gui_version','jasna_version','warmup_policy')
    missing=[key for key in required if not meta.get(key)]
    if missing: raise ValueError('Missing experiment controls: '+', '.join(missing))
    import hashlib
    def sha256(path):
        digest=hashlib.sha256()
        with open(path,'rb') as source:
            for block in iter(lambda:source.read(1024*1024),b''):
                digest.update(block)
        return digest.hexdigest()
    if sha256(args.source).lower()!=str(meta['source_sha256']).lower():
        raise ValueError('Input hash differs from experiment manifest')
    mask=None
    if args.mask:
        if sha256(args.mask).lower()!=str(meta['mask_sha256']).lower():
            raise ValueError('Mask hash differs from experiment manifest')
        import cv2
        import numpy as np
        from ltx_gui.engine import make_mask,read_mask
        cap=cv2.VideoCapture(args.reference)
        try:
            if not cap.isOpened():raise ValueError('Unusable clean reference')
            shape=(int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        finally:
            cap.release()
        mask=make_mask(shape,image=read_mask(Path(args.mask)))
    output={'experiment':meta,'ours':compare_video(args.reference,args.ours,mask=mask,max_frames=args.max_frames)}
    if args.jasna:
        output['jasna']=compare_video(args.reference,args.jasna,mask=mask,max_frames=args.max_frames)
    if args.jasna and all(isinstance(meta.get(k),(float,int)) and meta.get(k)>0 for k in ('ours_wall_seconds','jasna_wall_seconds')):
        output['relative_speedup_ours_vs_jasna']=float(meta['jasna_wall_seconds']/meta['ours_wall_seconds'])
    target=Path(args.output)
    if target.exists():raise FileExistsError(target)
    import math
    def json_safe(value):
        if isinstance(value,float) and not math.isfinite(value):
            return "Infinity" if value>0 else "-Infinity" if value<0 else "NaN"
        if isinstance(value,dict):return {key:json_safe(v) for key,v in value.items()}
        if isinstance(value,list):return [json_safe(v) for v in value]
        return value
    rendered=json.dumps(json_safe(output),ensure_ascii=False,indent=2,allow_nan=False)
    target.write_text(rendered,encoding='utf-8')
    print(rendered)


if __name__=='__main__':main()

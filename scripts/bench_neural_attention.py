"""Local/global PyTorch attention benchmark; no speedup claims until measured.

Usage: python -m scripts.bench_neural_attention --device cpu --iterations 15
CUDA benchmarking synchronizes GPU to exclude misleading asynchronous timings.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time


def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--iterations',type=int,default=15)
    parser.add_argument('--warmup',type=int,default=3)
    parser.add_argument('--height',type=int,default=16)
    parser.add_argument('--width',type=int,default=16)
    parser.add_argument('--frames',type=int,default=2)
    parser.add_argument('--threads',type=int,default=1)
    ns=parser.parse_args(argv)
    if min(ns.iterations,ns.height,ns.width,ns.frames)<1 or ns.warmup<0:
        parser.error('positive sizes and iterations are required')
    import torch
    if ns.threads > 0:
        torch.set_num_threads(ns.threads)
    from ltx_gui.neural.model import ModelConfig, VideoDiffusionTransformer
    device=torch.device(ns.device)
    if device.type=='cuda' and not torch.cuda.is_available():
        raise SystemExit('CUDA unavailable; benchmark not run')
    torch.manual_seed(7)
    base=dict(latent_channels=4,base_channels=4,dim=24,heads=2,depth=1)
    local=VideoDiffusionTransformer(ModelConfig(**base,window_frames=2,window_side=4)).eval().to(device)
    global_model=VideoDiffusionTransformer(ModelConfig(**base,window_frames=ns.frames,window_side=max(ns.width,ns.height))).eval().to(device)
    global_model.load_state_dict(local.state_dict())
    x=torch.randn((1,4,ns.frames,ns.height,ns.width),device=device)
    mask=torch.ones((1,1,ns.frames,ns.height,ns.width),device=device)
    sigma=torch.tensor([.5],device=device)
    def sync():
        if device.type=='cuda':torch.cuda.synchronize(device)
    def bench(model):
        timings=[]
        with torch.inference_mode():
            for i in range(ns.iterations+ns.warmup):
                sync();start=time.perf_counter()
                model(x,x,mask,sigma)
                sync();dt=(time.perf_counter()-start)*1000
                if i>=ns.warmup: timings.append(dt)
        return statistics.median(timings)
    global_ms=bench(global_model)
    local_ms=bench(local)
    tokens=ns.frames*ns.height*ns.width
    window_tokens=min(ns.frames,2)*min(ns.height,4)*min(ns.width,4)
    data={'device':str(device),'torch':torch.__version__,'torch_threads':torch.get_num_threads(),'tokens':tokens,
          'local_window_tokens':window_tokens,'theoretical_pairwise_interactions':{
              'global':tokens*tokens,'local_upper_bound':tokens*window_tokens},
          'median_ms':{'global':global_ms,'local':local_ms},'iterations':ns.iterations}
    print(json.dumps(data,indent=2))
    return data

if __name__ == '__main__':main()

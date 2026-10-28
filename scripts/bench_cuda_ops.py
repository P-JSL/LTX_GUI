"""Validate custom CUDA/Triton kernels against the PyTorch reference, then benchmark.

Example: python -m scripts.bench_cuda_ops --device cuda --iterations 100
Timings are GPU-event milliseconds (not wall-clock, includes kernel scheduling).
No GPU -> explicit unavailable status; never claim CUDA speed.
"""
from __future__ import annotations
import argparse
import json
import statistics
import time


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--device', default='cuda')
    p.add_argument('--backend', choices=['auto', 'torch', 'triton', 'cuda'], default='auto')
    p.add_argument('--iterations', type=int, default=50)
    p.add_argument('--warmup', type=int, default=10)
    p.add_argument('--pixels', type=int, default=1024 * 1024)
    ns = p.parse_args(argv)
    if ns.pixels < 1 or ns.iterations < 1 or ns.warmup < 0:
        p.error('pixels and iterations must be positive')
    import torch
    from ltx_gui.cuda_backend import masked_update, masked_composite, backend_status
    if ns.device.startswith('cuda') and not torch.cuda.is_available():
        out = {'status':'CUDA unavailable; benchmark not run','backend':ns.backend,'availability':backend_status()}
        print(json.dumps(out, indent=2)); return out
    device = torch.device(ns.device)
    # 3D BCTHW layout with one broadcast mask across channels.
    z = torch.rand((1, 4, 1, 1, ns.pixels), device=device)
    vel = torch.rand_like(z)
    known = torch.rand_like(z)
    mask = torch.randint(0, 2, (1, 1, 1, 1, ns.pixels), device=device).float()
    reference = masked_update(z, vel, known, mask, 0.125, backend='torch')
    actual = masked_update(z, vel, known, mask, 0.125, backend=ns.backend)
    atol = 5e-6
    correct = bool(torch.allclose(reference, actual, atol=atol))
    if not correct:
        raise AssertionError('CUDA update differs from PyTorch reference')

    def measure(fn):
        for _ in range(ns.warmup): fn()
        if device.type == 'cuda': torch.cuda.synchronize(device)
        values = []
        for _ in range(ns.iterations):
            if device.type == 'cuda':
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record(); fn(); end.record(); end.synchronize()
                values.append(start.elapsed_time(end))
            else:
                start = time.perf_counter(); fn()
                values.append(1000*(time.perf_counter()-start))
        return statistics.median(values)

    result = {'status':'completed', 'device':str(device), 'backend':ns.backend, 'pixels':ns.pixels,
              'atol':atol, 'correct':correct,
              'median_ms':{'torch':measure(lambda:masked_update(z,vel,known,mask,.125,backend='torch')),
                           'selected':measure(lambda:masked_update(z,vel,known,mask,.125,backend=ns.backend))},
              'availability':backend_status()}
    print(json.dumps(result, indent=2)); return result

if __name__ == '__main__': main()

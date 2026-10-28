"""Profile the trained, independent DiT/VAE on a real GPU.

No Jasna dependencies, no fabricated benchmark numbers. Requires a real
trained LTX_GUI checkpoint and a CUDA-capable PyTorch environment.

Usage:
  python -m scripts.profile_cuda --checkpoint checkpoints/native.pt --trace profile.json
"""
from __future__ import annotations
import argparse
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--backend', choices=['auto', 'torch', 'triton', 'cuda'], default='auto')
    parser.add_argument('--amp', choices=['none', 'bf16', 'fp16'], default='none')
    parser.add_argument('--trace', type=Path, default=Path('profile_cuda.json'))
    parser.add_argument('--frames', type=int, default=9)
    parser.add_argument('--size', type=int, default=64)
    parser.add_argument('--steps', type=int, default=2)
    args = parser.parse_args(argv)
    if not args.device.startswith('cuda'):
        parser.error('CUDA profiler requires --device cuda[:index]')
    if args.size < 32 or args.size % 32 or args.frames < 9 or (args.frames-1) % 8:
        parser.error('Frame count must be 8k+1; size must be multiple of 32')
    import torch
    if not torch.cuda.is_available():
        raise SystemExit('CUDA GPU unavailable: profiling not run')
    from torch.profiler import profile, ProfilerActivity
    from ltx_gui.neural.checkpoint import load_trained
    from ltx_gui.neural.pipeline import inpaint_tensor
    vae, flow, _ = load_trained(args.checkpoint, device=args.device)
    video = torch.rand((1, 3, args.frames, args.size, args.size), device=args.device)
    mask = torch.zeros((1, 1, args.frames, args.size, args.size), device=args.device)
    mask[:, :, :, args.size//4:args.size//2, args.size//4:args.size//2] = 1
    def run():
        return inpaint_tensor(video, mask, vae, flow, steps=args.steps,
                              gpu_backend=args.backend, amp=args.amp)
    with torch.inference_mode():
        run()  # warmup to initialize CUDA kernels & caches
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     record_shapes=True, profile_memory=True) as prof:
            run()
            torch.cuda.synchronize()
    args.trace.parent.mkdir(parents=True, exist_ok=True)
    prof.export_chrome_trace(str(args.trace))
    print(prof.key_averages().table(sort_by='self_cuda_time_total', row_limit=30))
    print('Peak CUDA allocated bytes:',torch.cuda.max_memory_allocated())
    print('Trace saved to:',args.trace)

if __name__ == '__main__': main()

"""Measure real NVIDIA inference performance and numerical correctness.

Does not claim JASNA comparison; use compare_results.py with original outputs,
identical video/mask/hardware, ground truth, and a controlled manifest.
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description="Real-GPU Torch / TensorRT / CUDA Graph comparison")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--frames", type=int, default=9)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16", "fp8"), default="fp32")
    parser.add_argument("--calibration", default="")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--output", default="gpu_inference_benchmark.json")
    args = parser.parse_args(argv)
    if args.frames < 9 or (args.frames-1) % 8 or args.width % 32 or args.height % 32:
        parser.error("Expected 8k+1 frames and spatial dimensions divisible by 32")
    if args.warmup < 1 or args.iterations < 3:
        parser.error("At least one warmup and three measured iterations are required")
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("No NVIDIA CUDA GPU detected. No benchmark has been run.")
    from ltx_gui.neural.checkpoint import load_trained
    from ltx_gui.neural.pipeline import accelerated_inpaint_tensor
    vae, flow, _ = load_trained(args.checkpoint, device=args.device)
    video = torch.rand((1, 3, args.frames, args.height, args.width), device=args.device)
    mask = torch.zeros((1, 1, args.frames, args.height, args.width), device=args.device)
    mask[:, :, :, args.height//4:args.height//2, args.width//4:args.width//2] = 1
    modes = [('pytorch', 'torch', False), ('cuda_graph', 'torch', True),
             ('tensorrt', 'tensorrt', False), ('tensorrt_graph', 'tensorrt', True)]
    results = {}
    baseline = None
    for label, engine, graph in modes:
        cache = {}
        def run():
            return accelerated_inpaint_tensor(video, mask, vae, flow, steps=args.steps,
                           engine=engine, cuda_graph=graph, runtime_cache=cache,
                           precision=args.precision if engine == 'tensorrt' else 'fp32',
                           fp8_calibration=args.calibration, seed=9876,
                           gpu_backend='torch')
        try:
            with torch.inference_mode():
                for _ in range(args.warmup): run()
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                timings = []
                for _ in range(args.iterations):
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    last = run()
                    end.record()
                    end.synchronize()
                    timings.append(start.elapsed_time(end))
                if baseline is None:
                    baseline = last.clone()
                    max_diff = 0
                else:
                    max_diff = float((last-baseline).abs().max().item())
                results[label] = {'status':'measured', 'p50_ms':statistics.median(timings),
                                  'p95_ms': sorted(timings)[max(0, int(len(timings)*0.95)-1)],
                                  'peak_vram_bytes':torch.cuda.max_memory_allocated(),
                                  'max_abs_difference_from_pytorch': max_diff}
        except (RuntimeError, ImportError, ValueError) as exc:
            torch.cuda.synchronize()
            results[label] = {'status':'unsupported', 'reason':str(exc)[:900]}
    idx = torch.device(args.device).index
    if idx is None:idx = torch.cuda.current_device()
    gpu = torch.cuda.get_device_properties(idx)
    report = {'checkpoint':str(Path(args.checkpoint).resolve()), 'gpu':gpu.name,
              'compute_capability': [gpu.major,gpu.minor], 'torch':torch.__version__,
              'cuda':torch.version.cuda, 'platform':platform.platform(),
              'input_shape':list(video.shape), 'steps':args.steps,
              'warmup':args.warmup, 'iterations':args.iterations,
              'precision':args.precision, 'results':results,
              'jasna_comparison':'Not measured. Use compare_results.py on separately generated paired outputs.'}
    target=Path(args.output)
    if target.exists():
        raise FileExistsError(target)
    target.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
